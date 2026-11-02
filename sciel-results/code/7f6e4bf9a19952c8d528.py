"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h."""
import time
import numpy as np

N = 100
TIME_LIMIT = 115.0  # safety margin before 120s kill


def _compute_overlap(h):
    """Compute the maximum overlap of h with shifted (1-h) over all shifts.
    
    For each shift k, overlap_k = sum_i h[i] * (1 - h[i+k]) for valid i,
    scaled by step width 2/N to give integral overlap.
    """
    n = len(h)
    max_ov = 0.0
    for k in range(n):
        # overlap at shift k: sum over i where both h[i] and h[i+k] are defined
        m = n - k  # number of overlapping terms
        # sum h[i] * (1 - h[i+k]) for i = 0..m-1
        s = np.sum(h[:m] * (1.0 - h[k:]))
        ov = s * (2.0 / n)
        if ov > max_ov:
            max_ov = ov
    return max_ov


def _project(h):
    """Project h to satisfy: 0 <= h[i] <= 1, sum(h) = N/2."""
    h = np.clip(h, 0.0, 1.0)
    target = n_target = N / 2.0
    # Simple iterative projection
    for _ in range(50):
        diff = h.sum() - target
        if abs(diff) < 1e-10:
            break
        # Adjust: add/subtract evenly but respect bounds
        room_up = 1.0 - h
        room_dn = h
        total_room = room_up.sum() + room_dn.sum()
        if total_room < 1e-12:
            break
        if diff > 0:
            # Need to decrease
            total_dn = room_dn.sum()
            if total_dn < 1e-12:
                break
            h -= diff / total_dn * room_dn
        else:
            # Need to increase
            total_up = room_up.sum()
            if total_up < 1e-12:
                break
            h += (-diff) / total_up * room_up
    h = np.clip(h, 0.0, 1.0)
    # Final correction
    diff = h.sum() - target
    if abs(diff) > 1e-12:
        # Uniform adjustment where possible
        mask_up = h < 1.0
        mask_dn = h > 0.0
        if diff > 0 and mask_dn.sum() > 0:
            h[mask_dn] -= diff / mask_dn.sum()
        elif diff < 0 and mask_up.sum() > 0:
            h[mask_up] += (-diff) / mask_up.sum()
    h = np.clip(h, 0.0, 1.0)
    return h


def _gradient_overlap(h):
    """Compute gradient of max overlap w.r.t. h, using the shift that achieves max."""
    n = len(h)
    best_k = 0
    best_ov = -1.0
    for k in range(n):
        m = n - k
        s = np.sum(h[:m] * (1.0 - h[k:]))
        ov = s * (2.0 / n)
        if ov > best_ov:
            best_ov = ov
            best_k = k
    
    k = best_k
    m = n - k
    grad = np.zeros(n)
    # d/dh[i] of sum_j h[j]*(1-h[j+k]) for j=0..m-1
    # = sum_j [delta_{i,j}*(1-h[j+k]) + h[j]*(-delta_{i,j+k})]
    # Term 1: if i in [0, m-1], contribution (1 - h[i+k])
    # Term 2: if i in [k, n-1], contribution -h[i-k]
    if m > 0:
        grad[:m] += (1.0 - h[k:k+m]) * (2.0 / n)
        if k + m <= n:
            grad[k:k+m] += -h[:m] * (2.0 / n)
    return grad


def _smooth_max_overlap(h, tau=0.1):
    """Soft-max surrogate for the maximum overlap using log-sum-exp."""
    n = len(h)
    ovs = np.empty(n)
    for k in range(n):
        m = n - k
        ovs[k] = np.sum(h[:m] * (1.0 - h[k:])) * (2.0 / n)
    # Log-sum-exp soft max
    max_ov = ovs.max()
    if max_ov <= 0:
        return 0.0, 0
    exps = np.exp((ovs - max_ov) / tau)
    soft_max = max_ov + tau * np.log(np.sum(exps))
    # Return soft max value and the weights
    return soft_max, tau


def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    # Start from the constant 0.5 solution
    h = np.full(N, 0.5)
    best_h = h.copy()
    best_score = _compute_overlap(h)
    
    # Try small perturbations from 0.5 to see if we can do better
    rng = np.random.default_rng(42)
    
    # Gradient-based local optimization with multiple restarts
    lr = 0.01
    n_iter_per_restart = 500
    
    for restart in range(20):
        if time.time() > deadline - 10:
            break
        
        if restart == 0:
            h = np.full(N, 0.5)
        else:
            # Perturb the best solution
            h = best_h.copy()
            noise_scale = 0.1 / (restart + 1)
            h += rng.standard_normal(N) * noise_scale
            h = _project(h)
        
        # Local optimization
        for it in range(n_iter_per_restart):
            if time.time() > deadline - 5:
                break
            
            # Compute gradient of max overlap
            grad = _gradient_overlap(h)
            
            # Take step to minimize max overlap
            h_new = h - lr * grad
            h_new = _project(h_new)
            
            new_score = _compute_overlap(h_new)
            if new_score < best_score:
                best_score = new_score
                best_h = h_new.copy()
            
            h = h_new
        
        # Try a smooth surrogate approach
        if time.time() > deadline - 10:
            break
        
        tau = 0.05
        for it in range(200):
            if time.time() > deadline - 5:
                break
            
            # Compute soft-max overlap and its gradient
            n_local = N
            ovs = np.empty(n_local)
            for k in range(n_local):
                m = n_local - k
                ovs[k] = np.sum(h[:m] * (1.0 - h[k:])) * (2.0 / n_local)
            
            max_ov = ovs.max()
            exps = np.exp((ovs - max_ov) / tau)
            weights = exps / exps.sum()
            
            # Gradient of soft max
            grad = np.zeros(N)
            for k in range(n_local):
                if weights[k] < 1e-10:
                    continue
                m = n_local - k
                # grad_k = (2/N) * [(1-h[k:k+m]) for first m, -h[:m] for last k+m]
                contrib = weights[k] * (2.0 / n_local)
                grad[:m] += contrib * (1.0 - h[k:k+m])
                grad[k:k+m] += contrib * (-h[:m])
            
            h_new = h - lr * grad
            h_new = _project(h_new)
            
            raw_score = _compute_overlap(h_new)
            if raw_score < best_score:
                best_score = raw_score
                best_h = h_new.copy()
            
            h = h_new
    
    # Final polish: try to fine-tune around the best solution
    if time.time() < deadline - 2:
        h = best_h.copy()
        lr_small = 0.001
        for it in range(200):
            if time.time() > deadline - 1:
                break
            grad = _gradient_overlap(h)
            h_new = h - lr_small * grad
            h_new = _project(h_new)
            new_score = _compute_overlap(h_new)
            if new_score < best_score:
                best_score = new_score
                best_h = h_new.copy()
            h = h_new
    
    # Ensure the result is valid
    best_h = np.clip(best_h, 0.0, 1.0)
    # Fix sum exactly
    diff = best_h.sum() - N / 2.0
    if abs(diff) > 1e-12:
        mask = (best_h > 1e-6) & (best_h < 1 - 1e-6)
        if mask.sum() > 0:
            best_h[mask] -= diff / mask.sum()
        best_h = np.clip(best_h, 0.0, 1.0)
    
    return best_h
