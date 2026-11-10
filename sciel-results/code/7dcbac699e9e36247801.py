import numpy as np
import time

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    
    Returns:
        support: np.ndarray of shape (n,), non-negative support points of the measure.
        weights: np.ndarray of shape (n,), non-negative weights.
    """
    start_time = time.time()
    deadline = start_time + 600.0  # Safety margin: 630s limit, use 600s for computation

    def compute_score(support, weights):
        n = len(support)
        w_sum = np.sum(weights)
        if w_sum <= 1e-9:
            return 0.0
        p = weights / w_sum
        
        # Pairwise sums and weights for Z = X1 + X2
        s_sums = support[:, None] + support[None, :]
        p_prods = p[:, None] * p[None, :]
        
        # Flatten and sort
        s_sums_flat = s_sums.flatten()
        p_prods_flat = p_prods.flatten()
        
        sort_idx = np.argsort(s_sums_flat)
        s_sorted = s_sums_flat[sort_idx]
        p_sorted = p_prods_flat[sort_idx]
        
        # Prefix sums of probabilities for Z = X1 + X2
        p_prefix = np.cumsum(p_sorted)
        
        # Vectorized calculation for X3 and X4
        # We need sum over k, l of p_k * p_l * P(Z < 2*s_l - s_k)
        # Let's compute thresholds T_{kl} = 2*s_l - s_k for all k, l
        # s_k is shape (n,), s_l is shape (n,)
        # T_{kl} = 2*s_l - s_k => shape (n, n) where rows are k, cols are l?
        # Let's define T[k, l] = 2 * s_l - s_k
        # s_k = support[k] (row vector if we think of it that way, but let's use broadcasting)
        # support[:, None] is s_k
        # support[None, :] is s_l
        # T = 2 * support[None, :] - support[:, None]  -> Shape (n, n), T[k, l]
        
        T = 2.0 * support[None, :] - support[:, None]
        
        # For each T[k, l], find idx = number of elements in s_sorted < T[k, l]
        # np.searchsorted(s_sorted, T, side='left') gives index of first element >= T
        # This index is the count of elements < T
        # Since T is 2D and s_sorted is 1D, searchsorted handles it?
        # Yes, if s_sorted is 1D and T is 2D, it returns 2D array of indices.
        
        idx = np.searchsorted(s_sorted, T, side='left')
        
        # Now we need p_prefix[idx - 1] but handle idx=0
        # We can create a shifted prefix sum where index 0 corresponds to 0.
        # p_prefix is length n*n.
        # Let's make a zero-padded prefix sum: p_prefix_pad = [0, p_prefix[0], p_prefix[1], ...]
        p_prefix_pad = np.zeros(n * n + 1)
        p_prefix_pad[1:] = p_prefix
        
        # prob_z_less[k, l] = p_prefix_pad[idx[k, l]]
        prob_z_less = p_prefix_pad[idx]
        
        # Total prob = sum_{k, l} p_k * p_l * prob_z_less[k, l]
        # p_k is shape (n,), p_l is shape (n,)
        # p_k * p_l -> p[:, None] * p[None, :]
        total_prob = np.sum(p[:, None] * p[None, :] * prob_z_less)
        
        return total_prob

    def refine(support, weights, rng, time_limit):
        n = len(support)
        current_support = support.copy()
        current_weights = weights.copy()
        current_score = compute_score(current_support, current_weights)
        
        # Adaptive step sizes
        step_s = 0.1
        step_w = 0.1
        
        # Local search
        # We use a simple hill climbing with random restarts within the local basin
        # Or just pure hill climbing with decreasing step size?
        # Let's try hill climbing with adaptive step size.
        
        last_improve = time.time()
        stagnation = 0
        
        while time.time() < time_limit:
            # Perturb
            new_support = current_support.copy()
            new_weights = current_weights.copy()
            
            # Perturb support
            # Move all points slightly? Or one?
            # Moving all allows for global shifts. Moving one allows for local tweaks.
            # Let's mix: 50% move one, 50% move all slightly.
            if rng.random() < 0.5:
                idx = rng.integers(n)
                new_support[idx] += rng.normal(0, step_s)
            else:
                new_support += rng.normal(0, step_s * 0.1, n)
            
            # Ensure non-negative
            new_support = np.maximum(new_support, 0)
            
            # Perturb weights
            # Add noise and re-normalize
            new_weights += rng.normal(0, step_w, n)
            new_weights = np.maximum(new_weights, 0)
            w_sum = np.sum(new_weights)
            if w_sum < 1e-9:
                continue
            new_weights = new_weights / w_sum
            
            new_score = compute_score(new_support, new_weights)
            
            if new_score > current_score:
                current_support = new_support
                current_weights = new_weights
                current_score = new_score
                last_improve = time.time()
                stagnation = 0
                # Maybe increase step size slightly if we've been improving for a while?
                # Or keep it simple.
            else:
                stagnation += 1
                # Decrease step size if stagnant
                if stagnation > 10:
                    step_s *= 0.9
                    step_w *= 0.9
                    stagnation = 0
                    # If step size is too small, restart with larger step?
                    if step_s < 1e-4:
                        break
                        
        return current_support, current_weights, current_score

    best_score = -1.0
    best_support = None
    best_weights = None
    
    rng = np.random.default_rng(42)
    
    # Try different sizes n
    # Smaller n is faster per eval.
    # n=8, 12, 16
    n_sizes = [8, 12, 16]
    
    # Allocate time to each n?
    # Or just loop through restarts for each n until deadline?
    # Let's do a few restarts for each n.
    
    for n in n_sizes:
        if time.time() > deadline - 10:
            break
            
        # Number of restarts for this n
        # Estimate time per restart.
        # For n=8, compute_score is fast.
        # For n=16, compute_score is slower.
        # Let's just run until deadline is approaching, or a fixed number of restarts.
        # Fixed number of restarts: 5? 10?
        # Let's do 5 restarts per n, each with a time limit.
        
        restarts = 5
        
        for r in range(restarts):
            if time.time() > deadline - 5:
                break
                
            # Initialize support
            # Random sorted points in [0, 1]
            support_init = np.sort(rng.uniform(0, 1, n))
            weights_init = rng.dirichlet(np.ones(n))
            
            # Refine
            # Time limit for this refinement: remaining time / restarts left?
            # Simpler: give each refinement a fixed time slice, e.g., 20s?
            # Total time 600s. 3 n_sizes * 5 restarts = 15 refinements.
            # 600 / 15 = 40s per refinement.
            # But we need to be careful not to exceed deadline.
            
            time_limit = min(deadline - 5, time.time() + 30) # Cap at 30s per refinement for safety
            
            s, w, score = refine(support_init, weights_init, rng, time_limit)
            
            if score > best_score:
                best_score = score
                best_support = s.copy()
                best_weights = w.copy()
                
    if best_support is None:
        # Fallback
        n = 8
        best_support = np.arange(n, dtype=np.float64)
        best_weights = np.ones(n, dtype=np.float64) / n
        best_score = compute_score(best_support, best_weights)

    # Final cleanup
    best_support = np.maximum(best_support, 0)
    best_weights = np.maximum(best_weights, 0)
    
    # Ensure valid format
    # Flatten
    best_support = best_support.flatten()
    best_weights = best_weights.flatten()
    
    return best_support, best_weights
