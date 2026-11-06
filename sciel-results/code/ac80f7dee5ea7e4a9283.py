"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _max_abs_sum(signs):
    """Compute max |sum| over all arithmetic progressions in [1, n]."""
    n = len(signs)
    max_val = 0
    # For each step size
    for step in range(1, n + 1):
        # For each start position
        for start in range(n):
            s = 0
            for k in range(start, n, step):
                s += signs[k]
                abs_s = abs(s)
                if abs_s > max_val:
                    max_val = abs_s
            # Early termination: if we already exceeded some threshold, skip
    return max_val

def _max_abs_sum_fast(signs, n=256):
    """Faster computation using numpy for the critical APs only."""
    signs_arr = np.array(signs, dtype=np.int8)
    max_val = 0
    
    # Critical APs: focus on steps that tend to dominate
    # Step 1: all contiguous subarrays (most important)
    # Use cumulative sum for O(n) per start
    cumsum = np.concatenate([[0], np.cumsum(signs_arr)])
    
    # For step 1: all subarrays
    for start in range(n):
        for end in range(start + 1, n + 1):
            s = cumsum[end] - cumsum[start]
            abs_s = abs(s)
            if abs_s > max_val:
                max_val = abs_s
    
    # For larger steps: sample critical ones
    for step in range(2, n + 1):
        for start in range(min(step, n)):
            s = 0
            for k in range(start, n, step):
                s += signs_arr[k]
                abs_s = abs(s)
                if abs_s > max_val:
                    max_val = abs_s
    return max_val

def _identify_critical_aps(signs, n=256, top_k=50):
    """Identify the APs that currently have the largest |sum|."""
    signs_arr = np.array(signs, dtype=np.int8)
    ap_sums = []
    
    # Collect sums for all APs
    for step in range(1, n + 1):
        for start in range(n):
            s = 0
            for k in range(start, n, step):
                s += signs_arr[k]
                ap_sums.append((abs(s), step, start))
    
    # Sort by |sum| descending
    ap_sums.sort(key=lambda x: -x[0])
    return ap_sums[:top_k]

def _eval_critical(signs, critical_aps, n=256):
    """Evaluate only the critical APs."""
    signs_arr = np.array(signs, dtype=np.int8)
    max_val = 0
    for _, step, start in critical_aps:
        s = 0
        for k in range(start, n, step):
            s += signs_arr[k]
        abs_s = abs(s)
        if abs_s > max_val:
            max_val = abs_s
    return max_val

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    # Initialize with random signs
    rng = np.random.RandomState(171)
    signs = rng.choice([-1, 1], size=n).tolist()
    
    # Compute initial objective
    current_val = _max_abs_sum(signs)
    best_val = current_val
    best_signs = signs[:]
    
    # Identify critical APs
    critical_aps = _identify_critical_aps(signs, n, top_k=100)
    
    # Local search: flip one sign at a time
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            # Try flipping sign at position i
            old_sign = signs[i]
            signs[i] = -old_sign
            
            # Evaluate on critical APs first (fast)
            crit_val = _eval_critical(signs, critical_aps, n)
            
            if crit_val < current_val:
                # Full evaluation to confirm
                full_val = _max_abs_sum(signs)
                if full_val < current_val:
                    current_val = full_val
                    improved = True
                    # Update critical APs occasionally
                    if current_val < best_val:
                        best_val = current_val
                        best_signs = signs[:]
                        # Re-identify critical APs
                        critical_aps = _identify_critical_aps(signs, n, top_k=100)
                else:
                    # Revert
                    signs[i] = old_sign
            else:
                # Revert
                signs[i] = old_sign
    
    # Final full evaluation of best
    final_val = _max_abs_sum(best_signs)
    return best_signs
# EVOLVE-BLOCK-END
