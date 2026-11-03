"""Erdos minimum overlap: choose h in [0,1]^N, sum=N/2, minimize max_k sum_i h_i h_{i+k}."""
import numpy as np
import time

N = 100

def _overlap_matrix():
    """Return matrix M where M[k, i] = 1 if h_i and h_{i+k} overlap (cyclic shift)."""
    # Overlap of h with shifted 1-h: for shift k, overlap = sum_i h_i * (1 - h_{(i+k) % N})
    # = sum_i h_i - sum_i h_i h_{(i+k)%N}
    # To minimize max overlap, maximize min_k sum_i h_i h_{(i+k)%N}
    # Since sum h_i = N/2 is fixed, minimizing max overlap = maximizing min cross-correlation
    M = np.zeros((N, N))
    for k in range(N):
        for i in range(N):
            M[k, i] = 1.0
    return M

def _cross_corr(h):
    """Compute sum_i h_i * h_{(i+k)%N} for all shifts k."""
    from numpy.fft import fft, ifft
    H = fft(h)
    # Circular cross-correlation
    cc = np.real(ifft(H * np.conj(H)))
    # cc[k] = sum_i h_i * h_{(i-k)%N}, we want sum_i h_i * h_{(i+k)%N}
    # Reverse: cc_rev[k] = cc[-k mod N]
    cc_rev = cc[np.arange(N) * -1 % N]
    return cc_rev

def _max_overlap(h):
    """Compute max_k sum_i h_i * (1 - h_{(i+k)%N}) = N/2 - min_k cc[k]."""
    cc = _cross_corr(h)
    return N / 2.0 - np.min(cc)

def _soft_min(x, tau):
    """Soft-min approximation: -1/tau * log(sum exp(-tau * x))."""
    # Numerically stable
    x_min = np.min(x)
    return -x_min - np.log(np.sum(np.exp(-(x - x_min) * tau))) / tau

def _soft_max_overlap(h, tau):
    """Smooth surrogate for max overlap using soft-min on cross-correlations."""
    cc = _cross_corr(h)
    return N / 2.0 - _soft_min(cc, tau)

def _fista_step(h, cc, step_size, M):
    """One FISTA iteration step."""
    # Gradient of soft-min surrogate
    weights = np.exp(-(cc - np.min(cc)) * M)
    weights /= weights.sum()
    grad_cc = -weights / M  # d(soft_min)/d(cc_k) = -weights_k
    
    # Gradient w.r.t. h: grad = -2 * sum_k grad_cc[k] * circ_shift(h, -k)
    # Actually: cc[k] = sum_i h_i h_{(i+k)%N}
    # d(cc[k])/d(h_j) = h_{(j-k)%N} + h_{(j+k)%N} (for k != 0)
    # For k=0: d(cc[0])/d(h_j) = 2*h_j
    
    grad_h = np.zeros(N)
    for k in range(N):
        if grad_cc[k] == 0:
            continue
        # d(cc[k])/d(h_j) = h_{(j-k)%N} + (1 if k==0 else 0) * ... 
        # More carefully: cc[k] = sum_i h_i * h_{(i+k)%N}
        # d/dh_j: if j = i, term is h_{(j+k)%N}; if j = i+k, term is h_i = h_{(j-k)%N}
        # So d(cc[k])/d(h_j) = h_{(j+k)%N} + h_{(j-k)%N} for k != 0
        # For k = 0: d(cc[0])/d(h_j) = 2*h_j
        if k == 0:
            grad_h += 2.0 * grad_cc[k] * h
        else:
            grad_h += grad_cc[k] * (np.roll(h, -k) + np.roll(h, k))
    
    # We want to MINIMIZE soft_max_overlap = N/2 - soft_min(cc)
    # d(soft_max_overlap)/d(h_j) = -d(soft_min)/d(h_j) = -sum_k grad_cc[k] * d(cc[k])/d(h_j)
    # But grad_cc[k] = d(soft_min)/d(cc_k) = -weights_k, so:
    # d(soft_max_overlap)/d(h_j) = sum_k weights_k * d(cc[k])/d(h_j)
    
    # Recompute properly
    grad_h = np.zeros(N)
    for k in range(N):
        if weights[k] == 0:
            continue
        if k == 0:
            grad_h += 2.0 * weights[k] * h
        else:
            grad_h += weights[k] * (np.roll(h, -k) + np.roll(h, k))
    
    # Gradient descent step
    new_h = h - step_size * grad_h
    
    # Project onto {h in [0,1]^N : sum h = N/2}
    new_h = np.clip(new_h, 0, 1)
    # Adjust to satisfy sum constraint
    diff = N / 2.0 - np.sum(new_h)
    if diff > 0:
        # Need to increase some elements
        mask = new_h < 1.0
        if mask.any():
            add = diff / mask.sum()
            new_h[mask] = np.minimum(new_h[mask] + add, 1.0)
    elif diff < 0:
        mask = new_h > 0.0
        if mask.any():
            sub = -diff / mask.sum()
            new_h[mask] = np.maximum(new_h[mask] - sub, 0.0)
    
    # Re-project if needed
    for _ in range(5):
        diff = N / 2.0 - np.sum(new_h)
        if abs(diff) < 1e-12:
            break
        if diff > 0:
            mask = new_h < 1.0
            if mask.any():
                add = diff / mask.sum()
                new_h[mask] = np.minimum(new_h[mask] + add, 1.0)
        else:
            mask = new_h > 0.0
            if mask.any():
                sub = -diff / mask.sum()
                new_h[mask] = np.maximum(new_h[mask] - sub, 0.0)
    
    return new_h

def _bang_bang_local_search(h, max_iters=200):
    """Local search: try flipping individual or small groups to bang-bang values."""
    best_h = h.copy()
    best_val = _max_overlap(best_h)
    
    for it in range(max_iters):
        improved = False
        # Try moving each element toward 0 or 1
        for i in range(N):
            # Try h_i -> 0 (if currently > 0)
            if best_h[i] > 1e-9:
                trial = best_h.copy()
                trial[i] = 0.0
                # Fix sum
                diff = N / 2.0 - np.sum(trial)
                if diff > 0:
                    mask = trial < 1.0
                    if mask.any():
                        add = diff / mask.sum()
                        trial[mask] = np.minimum(trial[mask] + add, 1.0)
                val = _max_overlap(trial)
                if val < best_val - 1e-10:
                    best_h = trial
                    best_val = val
                    improved = True
            
            # Try h_i -> 1 (if currently < 1)
            if best_h[i] < 1.0 - 1e-9:
                trial = best_h.copy()
                trial[i] = 1.0
                diff = N / 2.0 - np.sum(trial)
                if diff < 0:
                    mask = trial > 0.0
                    if mask.any():
                        sub = -diff / mask.sum()
                        trial[mask] = np.maximum(trial[mask] - sub, 0.0)
                val = _max_overlap(trial)
                if val < best_val - 1e-10:
                    best_h = trial
                    best_val = val
                    improved = True
        
        if not improved:
            break
    
    return best_h, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s
    
    # Initialize
    h = np.full(N, 0.5)
    
    # FISTA with adaptive tau
    tau = 10.0  # Initial soft-min parameter
    step_size = 1.0 / (2.0 * N)  # Initial step size
    
    # Momentum for FISTA
    y = h.copy()
    t = 1.0
    
    best_h = h.copy()
    best_val = _max_overlap(best_h)
    
    for it in range(2000):
        if time.time() > deadline:
            break
        
        cc = _cross_corr(y)
        
        # Compute gradient
        weights = np.exp(-(cc - np.min(cc)) * tau)
        wsum = weights.sum()
        if wsum > 0:
            weights /= wsum
        else:
            weights = np.full(N, 1.0 / N)
        
        grad_h = np.zeros(N)
        for k in range(N):
            if weights[k] == 0:
                continue
            if k == 0:
                grad_h += 2.0 * weights[k] * y
            else:
                grad_h += weights[k] * (np.roll(y, -k) + np.roll(y, k))
        
        # FISTA step
        new_h = y - step_size * grad_h
        new_h = np.clip(new_h, 0, 1)
        
        # Project onto sum constraint
        for _ in range(10):
            diff = N / 2.0 - np.sum(new_h)
            if abs(diff) < 1e-14:
                break
            if diff > 0:
                mask = new_h < 1.0
                if mask.any():
                    add = diff / mask.sum()
                    new_h[mask] = np.minimum(new_h[mask] + add, 1.0)
            else:
                mask = new_h > 0.0
                if mask.any():
                    sub = -diff / mask.sum()
                    new_h[mask] = np.maximum(new_h[mask] - sub, 0.0)
        
        # Update momentum
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
        y = new_h + ((t - 1.0) / t_new) * (new_h - h)
        y = np.clip(y, 0, 1)
        h = new_h
        t = t_new
        
        # Adaptive step size
        cc_new = _cross_corr(h)
        val_new = _max_overlap(h)
        if val_new < best_val:
            best_val = val_new
            best_h = h.copy()
        
        # Adapt tau: increase as we converge
        if it % 50 == 0 and it > 0:
            tau *= 1.5
            step_size /= 1.2
    
    # Final local search
    if time.time() < deadline:
        best_h, best_val = _bang_bang_local_search(best_h)
    
    # Ensure exact sum
    diff = N / 2.0 - np.sum(best_h)
    if abs(diff) > 1e-12:
        if diff > 0:
            mask = best_h < 1.0
            if mask.any():
                add = diff / mask.sum()
                best_h[mask] = np.minimum(best_h[mask] + add, 1.0)
        else:
            mask = best_h > 0.0
            if mask.any():
                sub = -diff / mask.sum()
                best_h[mask] = np.maximum(best_h[mask] - sub, 0.0)
    
    return best_h
