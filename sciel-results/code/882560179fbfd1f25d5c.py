# EVOLVE-BLOCK-START
import time

def construct():
    """
    Returns a list of 12 distinct positive integers <= 1048576 such that
    all 2^12 subset sums are distinct, minimizing the largest integer.
    
    Uses greedy construction: at each step, pick the smallest positive integer
    not yet used that preserves distinctness of all subset sums.
    """
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin (90s limit)
    
    n = 12
    max_val = 1048576
    
    chosen = []
    used = set()
    subset_sums = {0}
    
    for k in range(n):
        if time.time() > deadline:
            break
        
        found = False
        # Try candidates from 1 upward
        for candidate in range(1, max_val + 1):
            if candidate in used:
                continue
            
            # Check if adding candidate keeps all subset sums distinct
            # Conflict if exists s in subset_sums such that (s + candidate) in subset_sums
            conflict = False
            for s in subset_sums:
                if (s + candidate) in subset_sums:
                    conflict = True
                    break
            
            if not conflict:
                chosen.append(candidate)
                used.add(candidate)
                # Update subset sums
                new_sums = set(subset_sums)
                for s in subset_sums:
                    new_sums.add(s + candidate)
                subset_sums = new_sums
                found = True
                break
        
        if not found:
            break
    
    # Verify we have 12 elements; fallback to powers of 2 if needed
    if len(chosen) < n:
        return [1 << i for i in range(n)]
    
    return chosen
# EVOLVE-BLOCK-END
