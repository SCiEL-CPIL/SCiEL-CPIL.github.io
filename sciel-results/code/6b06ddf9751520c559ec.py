"""Low discrepancy on all arithmetic progressions - Quadratic Minimization with Gradient Descent."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _compute_all_ap_prefix_sums(signs):
    """
    Compute prefix sums for all arithmetic progressions.
    Returns a list of (start, step, length, prefix_sums_array)
    """
    n = len(signs)
    s = np.asarray(signs, dtype=np.int16)
    results = []
    
    for d in range(1, n + 1):
        for i in range(min(d, n)):
            seq = s[i:n:d]
            if len(seq) == 0:
                continue
            prefix_sums = np.cumsum(seq)
            results.append((i, d, len(seq), prefix_sums))
    
    return results

def _get_max_ap_discrepancy(signs):
    """
    Compute max absolute signed sum over any arithmetic progression in [1..n].
    Returns (max_disc, worst_start, worst_step, worst_len)
    """
    n = len(signs)
    s = np.asarray(signs, dtype=np.int16)
    
    max_abs = 0
    best_start = 0
    best_step = 1
    best_len = 1
    
    for d in range(1, n + 1):
        for i in range(min(d, n)):
            seq = s[i:n:d]
            if len(seq) == 0:
                continue
            
            cumsum = np.cumsum(seq)
            abs_cumsum = np.abs(cumsum)
            current_max = np.max(abs_cumsum)
            
            if current_max > max_abs:
                max_abs = current_max
                best_start = i
                best_step = d
                best_len = np.argmax(abs_cumsum) + 1
    
    return int(max_abs), best_start, best_step, best_len

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s limit to be safe (90s wall-clock)
    
    n = 256
    rng = np.random.default_rng(42)
    
    # Precompute all AP structures (start, step, length, indices)
    # This is done once and reused
    ap_structures = []
    for d in range(1, n + 1):
        for i in range(min(d, n)):
            indices = np.arange(i, n, d)
            if len(indices) > 0:
                ap_structures.append((i, d, len(indices), indices))
    
    num_aps = len(ap_structures)
    
    # Initial random signs
    signs = rng.choice([-1, 1], size=n).astype(np.int16)
    
    # Evaluate initial
    current_disc, _, _, _ = _get_max_ap_discrepancy(signs)
    best_signs = signs.copy()
    best_disc = current_disc
    
    # Gradient descent approach
    # Objective: minimize sum of (prefix_sum^2) for all APs
    # This is a smooth surrogate for the max absolute discrepancy
    
    learning_rate = 0.05
    momentum = 0.9
    velocity = np.zeros(n, dtype=np.float64)
    
    # Precompute for faster gradient computation
    # For each AP, we need to know which positions contribute to which prefix sums
    
    # Gradient descent loop
    step_count = 0
    last_eval_time = start_time
    
    while time.time() < deadline:
        # Compute gradient of sum of squared prefix sums
        # For each AP with prefix sums p_1, p_2, ..., p_L
        # d/dx_j (sum_k p_k^2) = 2 * sum_{k: j in prefix_k} p_k
        
        # More efficiently: for each AP, the contribution to gradient at position j
        # is 2 * p_k where k is the index in the AP such that the k-th term is at position j
        # and p_k is the prefix sum up to k
        
        grad = np.zeros(n, dtype=np.float64)
        
        for start, step, length, indices in ap_structures:
            if length == 0:
                continue
            seq = signs[indices]
            prefix_sums = np.cumsum(seq)
            # For each position in the AP, add 2 * prefix_sum to gradient
            # Position indices[k] contributes prefix_sums[k] to the sum of squares
            # d/dx (sum_k prefix_sums[k]^2) where x is the sign at indices[k]
            # = 2 * prefix_sums[k] (since flipping sign changes prefix sum by -2*sign)
            # Actually, we're treating signs as continuous for gradient purposes
            # The gradient with respect to x_j (continuous relaxation) is:
            # For each AP where j appears at position k (0-indexed), 
            # the prefix sum at position k depends on x_j
            # d(prefix_sums[k])/dx_j = 1 if j == indices[k], else 0
            # So d(sum prefix_sums^2)/dx_j = 2 * prefix_sums[k] where k is such that indices[k] == j
            
            # For this AP, each position indices[k] gets gradient contribution 2 * prefix_sums[k]
            grad[indices] += 2.0 * prefix_sums
        
        # Update with momentum
        velocity = momentum * velocity + learning_rate * grad
        signs_continuous = signs.astype(np.float64) - velocity
        
        # Project back to {-1, 1} by thresholding
        new_signs = np.where(signs_continuous >= 0, 1, -1).astype(np.int16)
        
        # Check if we improved
        if not np.array_equal(new_signs, signs):
            signs = new_signs
            step_count += 1
            
            # Evaluate periodically
            if step_count % 10 == 0 or time.time() - last_eval_time > 2.0:
                disc, _, _, _ = _get_max_ap_discrepancy(signs)
                if disc < best_disc:
                    best_disc = disc
                    best_signs = signs.copy()
                last_eval_time = time.time()
                
                # If we reach the theoretical minimum, stop
                if best_disc <= 1:
                    break
        
        # Adaptive learning rate
        if step_count % 100 == 0 and step_count > 0:
            learning_rate *= 0.95
    
    # Final evaluation
    final_disc, _, _, _ = _get_max_ap_discrepancy(signs)
    if final_disc < best_disc:
        best_disc = final_disc
        best_signs = signs.copy()
    
    return best_signs.tolist()

# EVOLVE-BLOCK-END
