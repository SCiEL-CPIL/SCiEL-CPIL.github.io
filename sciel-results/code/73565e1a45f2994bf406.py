# EVOLVE-BLOCK-START
import numpy as np
import time

def _get_linear_overlaps(h):
    """
    Compute linear cross-correlation of h with (1-h) for all non-zero shifts.
    Returns an array of overlap values C_k for k != 0.
    C_k = sum_i h_i * (1 - h_{i+k})
    Note: np.correlate(h, g) computes sum h[i+k] g[i].
    So c[k] for k>0 is sum h[i+k] (1-h[i]).
    This matches the definition if we interpret shift appropriately.
    """
    g = 1.0 - h
    c = np.correlate(h, g, mode='full')
    # c is length 2N-1. Center is at N-1.
    # c[0:N-1] are negative shifts (h[i] g[i+k]?)
    # c[N-1] is k=0
    # c[N:] are positive shifts
    # We want all k != 0.
    c_neg = c[:N-1]
    c_pos = c[N:]
    return np.concatenate([c_neg, c_pos])

def _max_overlap(h):
    """Return the maximum linear overlap value."""
    overlaps = _get_linear_overlaps(h)
    if len(overlaps) == 0:
        return 0.0
    return np.max(overlaps)

def _project_to_feasible(h):
    """
    Project h onto the feasible set: 0 <= h_i <= 1 and sum(h) = N/2.
    """
    h = np.clip(h, 0, 1).copy()
    target = TARGET_SUM
    
    def _sum_at_delta(delta):
        return np.sum(np.clip(h + delta, 0, 1))
    
    lo, hi = -1.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        s = _sum_at_delta(mid)
        if s < target:
            lo = mid
        else:
            hi = mid
    
    delta = (lo + hi) / 2
    h_new = np.clip(h + delta, 0, 1)
    
    sum_h = np.sum(h_new)
    diff = target - sum_h
    if abs(diff) > 1e-15:
        if diff > 0:
            candidates = np.where(h_new < 1.0)[0]
            if len(candidates) > 0:
                idx = candidates[0]
                h_new[idx] = min(1.0, h_new[idx] + diff)
        else:
            candidates = np.where(h_new > 0.0)[0]
            if len(candidates) > 0:
                idx = candidates[0]
                h_new[idx] = max(0.0, h_new[idx] + diff)
    
    return h_new

def construct():
    """
    Construct the step heights h to minimize the maximum overlap.
    Uses a greedy local search strategy focusing on reducing the maximum overlap shifts.
    """
    start_time = time.time()
    deadline = start_time + 115.0
    
    rng = np.random.default_rng(42)
    
    # Initialize with a diverse set of candidates
    candidates = []
    
    # 1. Constant 0.5
    candidates.append(np.full(N, 0.5))
    
    # 2. Sine waves
    t = np.arange(N)
    for freq in [1, 2, 3, 5, 7, 11, 13]:
        h = 0.5 + 0.4 * np.sin(2 * np.pi * freq * t / N)
        candidates.append(h)
        
    # 3. Random
    for _ in range(10):
        h = rng.uniform(0, 1, N)
        candidates.append(h)
        
    # 4. Binary Random
    for _ in range(10):
        h = rng.choice([0, 1], N)
        # Adjust to sum N/2
        sum_h = np.sum(h)
        if sum_h < N/2:
            idxs = np.where(h==0)[0]
            need = int(N/2 - sum_h)
            if need > 0 and len(idxs) >= need:
                h[idxs[:need]] = 1
        elif sum_h > N/2:
            idxs = np.where(h==1)[0]
            need = int(sum_h - N/2)
            if need > 0 and len(idxs) >= need:
                h[idxs[:need]] = 0
        candidates.append(h)

    # Project and evaluate
    best_h = None
    best_score = np.inf
    
    for h_init in candidates:
        h_proj = _project_to_feasible(h_init)
        score = _max_overlap(h_proj)
        if score < best_score:
            best_score = score
            best_h = h_proj

    # Local Search: Greedy reduction of max overlaps
    # We will perform iterations where we try to reduce the maximum overlap
    # by adjusting h_i in the direction that reduces the top overlaps.
    
    h = best_h.copy()
    current_score = best_score
    
    # Pre-compute overlaps to find active shifts
    # We will do a coordinate descent-like step
    
    max_iterations = 1000
    step_size = 0.05
    
    for it in range(max_iterations):
        if time.time() > deadline - 10.0:
            break
            
        overlaps = _get_linear_overlaps(h)
        max_ov = np.max(overlaps)
        
        if max_ov < current_score - 1e-6:
            current_score = max_ov
            best_h = h.copy()
        elif max_ov > current_score + 1e-4:
            # If we got worse, revert? No, we only accept improvements in the loop below.
            pass
            
        # Find the shifts that are close to the max
        threshold = max_ov - 0.1
        active_shifts = np.where(overlaps >= threshold)[0]
        
        if len(active_shifts) == 0:
            active_shifts = [np.argmax(overlaps)]
            
        # Try to reduce the overlap for these shifts
        # C_k = sum h[i+k] (1-h[i])
        # Gradient wrt h[j] for a specific shift k:
        # Term where j is i: -h[j+k]
        # Term where j is i+k: (1-h[j-k])
        # Total grad_j = (1 - h[j-k]) - h[j+k]
        # We want to decrease C_k, so we move h_j opposite to grad_j.
        # delta_j = -step * grad_j
        
        improved = False
        # To maintain sum, we need to move mass.
        # Simple approach: pick a random pair (i, j) and swap/move mass.
        # Or: update all h_j based on gradient, then project.
        
        grad_sum = np.zeros(N)
        for k in active_shifts:
            # Map k from overlap index to shift
            # Overlaps array: [c_neg, c_pos]
            # c_neg corresponds to shifts -1, -2, ... -(N-1)
            # c_pos corresponds to shifts 1, 2, ... N-1
            if k < N-1:
                shift = -(N-1 - k)
            else:
                shift = k - (N-1)
                
            # Calculate gradient contribution for this shift
            # C_shift = sum h[i+shift] (1-h[i])
            # d/dh[j] C_shift:
            # If j = i: term is h[j+shift](1-h[j]) -> deriv wrt h[j] is -h[j+shift]
            # If j = i+shift: term is h[j](1-h[j-shift]) -> deriv wrt h[j] is (1-h[j-shift])
            
            # We need to be careful with bounds.
            # For simplicity, we compute the gradient for the whole vector.
            
            # Vectorized gradient for shift:
            # grad = np.zeros(N)
            # For i in 0..N-1:
            #   if i+shift in bounds:
            #     grad[i] -= h[i+shift]
            #     grad[i+shift] += (1-h[i])
            
            # Let's do it explicitly for small N
            for i in range(N):
                i_plus_shift = i + shift
                if 0 <= i_plus_shift < N:
                    # Contribution to C_shift from pair (i, i+shift)
                    # Term: h[i+shift] * (1 - h[i])
                    # d/dh[i]: -h[i+shift]
                    grad_sum[i] -= h[i_plus_shift]
                    # d/dh[i+shift]: (1 - h[i])
                    grad_sum[i_plus_shift] += (1 - h[i])
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad_sum)
        if grad_norm > 1e-6:
            grad_sum = grad_sum / grad_norm
            
            # Move in opposite direction
            h_new = h - step_size * grad_sum
            h_new = _project_to_feasible(h_new)
            
            new_score = _max_overlap(h_new)
            if new_score < current_score - 1e-8:
                h = h_new
                current_score = new_score
                best_h = h.copy()
                improved = True
                # Reduce step size if we improved significantly?
                # step_size = max(0.001, step_size * 0.95)
                
        if not improved:
            # Try random perturbations
            if rng.random() < 0.1:
                h_pert = h + rng.standard_normal(N) * 0.1
                h_pert = _project_to_feasible(h_pert)
                score_pert = _max_overlap(h_pert)
                if score_pert < current_score - 1e-8:
                    h = h_pert
                    current_score = score_pert
                    best_h = h.copy()
                    improved = True
            else:
                # Stagnation, break or reduce step
                step_size *= 0.9
                if step_size < 0.001:
                    break

    return best_h

# EVOLVE-BLOCK-END
