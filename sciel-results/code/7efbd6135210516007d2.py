"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def _verify_distinct_subset_sums(vals):
    """
    Verify that all subset sums of vals are distinct.
    Returns True if valid, False otherwise.
    """
    n = len(vals)
    expected_size = 1 << n
    sums = {0}
    for v in vals:
        new_sums = set()
        for s in sums:
            new_sums.add(s + v)
        # Check for intersection
        if new_sums & sums:
            return False
        sums.update(new_sums)
    return len(sums) == expected_size

def _construct_greedy(max_val):
    """
    Fallback greedy construction if the hardcoded optimal solution fails verification
    (which should never happen, but ensures robustness).
    """
    vals = []
    sums = {0}
    for i in range(12):
        start = vals[-1] + 1 if vals else 1
        if start > max_val:
            return None
        for cand in range(start, max_val + 1):
            # Check collision
            collides = False
            for s in sums:
                if s + cand in sums:
                    collides = True
                    break
            if not collides:
                vals.append(cand)
                for s in list(sums):
                    sums.add(s + cand)
                break
        else:
            return None
    return vals

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    # The optimal solution for n distinct positive integers with distinct subset sums
    # minimizing the largest element is the sequence of powers of 2: 1, 2, 4, ..., 2^(n-1).
    # For n=12, the largest element is 2^11 = 2048.
    
    optimal_vals = [1 << i for i in range(12)]
    
    # Verify the optimal solution
    if _verify_distinct_subset_sums(optimal_vals):
        return optimal_vals
    
    # If verification fails (should not happen), fall back to greedy search
    # The greedy search will also find the powers of 2, but it's slower.
    fallback_vals = _construct_greedy(1 << 20)
    if fallback_vals is not None and _verify_distinct_subset_sums(fallback_vals):
        return fallback_vals
    
    # Last resort: return the optimal vals anyway (they are valid)
    return optimal_vals
# EVOLVE-BLOCK-END
