import time
import random

START = time.time()
DEADLINE = 55.0

def _validate(vals):
    """
    Validate if the list of values satisfies the constraint:
    Subset sums of different cardinalities must be disjoint.
    """
    n = len(vals)
    if n == 0:
        return True
        
    # sums_by_k[k] is a bitset of all sums achievable by subsets of size k
    # sums_by_k[0] = 1 (bit 0 set, sum 0)
    sums_by_k = [0] * (n + 1)
    sums_by_k[0] = 1
    
    for v in vals:
        # Update in reverse to avoid using the same element twice in one step
        # We are adding v to all subsets of size k to form subsets of size k+1
        for k in range(n - 1, -1, -1):
            if sums_by_k[k]:
                sums_by_k[k + 1] |= (sums_by_k[k] << v)
                
    # Check for collisions between different cardinalities
    for i in range(n + 1):
        for j in range(i + 1, n + 1):
            if sums_by_k[i] & sums_by_k[j]:
                return False
    return True

def _check_candidate(sums_by_k, c):
    """
    Check if candidate c can be added to the current set represented by sums_by_k.
    Returns (is_valid, new_sums_by_k, score)
    """
    n = len(sums_by_k) - 1
    
    # Precompute New_k = sums_by_k[k-1] << c
    # New_k represents the sums formed by adding c to subsets of size k-1
    # These will be the *additional* sums for cardinality k.
    
    # We need to check if (sums_by_k[k] | New_k) is disjoint from (sums_by_k[j] | New_j) for all k != j.
    # Since sums_by_k[i] are disjoint, we only need to check:
    # 1. New_k & sums_by_k[j] == 0 for all j != k
    # 2. New_k & New_j == 0 for all j != k (This is always true if sums_by_k are disjoint, 
    #    because New_k & New_j = (sums_by_k[k-1] << c) & (sums_by_k[j-1] << c) 
    #    = (sums_by_k[k-1] & sums_by_k[j-1]) << c. If k != j, then k-1 != j-1, so intersection is 0.)
    
    # So we just check New_k against all existing sums_by_k[j] for j != k.
    
    new_sums = sums_by_k[:]
    new_sums.append(0) # Placeholder for new max cardinality
    
    total_score = 0
    
    for k in range(n + 1):
        if sums_by_k[k]:
            new_bits = sums_by_k[k] << c
            target_k = k + 1
            
            # Check collision with existing sums of other cardinalities
            # Note: new_bits are added to sums_by_k[target_k].
            # We must ensure new_bits does not intersect with:
            # - sums_by_k[m] for m != target_k
            # - (new_bits from other k's) -> handled by disjointness of sources
            
            for m in range(n + 1):
                if m == target_k:
                    continue
                if new_bits & sums_by_k[m]:
                    return False, None, 0
                    
            # Update new_sums
            new_sums[target_k] |= new_bits
            
    # Calculate score: total number of bits set in all sums_by_k
    # Lower score is better (less dense)
    for s in new_sums:
        total_score += s.bit_count()
        
    return True, new_sums, total_score

def _greedy_solve(seed):
    """
    Greedy solver with a specific seed for candidate ordering.
    """
    rng = random.Random(seed)
    
    # Start with descending order to bias towards large numbers
    candidates = list(range(256, 0, -1))
    
    # Apply a light shuffle to break ties and explore different paths
    # We want to maintain the "large numbers first" bias, so we only do a few swaps
    # or a partial shuffle.
    # Let's swap a random number of pairs.
    num_swaps = rng.randint(0, 5)
    for _ in range(num_swaps):
        i = rng.randint(0, 255)
        j = rng.randint(0, 255)
        candidates[i], candidates[j] = candidates[j], candidates[i]
        
    sums_by_k = [1]
    current_vals = []
    
    while candidates:
        if time.time() - START > DEADLINE - 1:
            break
            
        best_c = None
        best_sums = None
        best_score = float('inf')
        best_idx = -1
        
        # Evaluate all candidates
        # Optimization: If we have many candidates, checking all might be slow if n is large.
        # But n is small (max ~20-30), and bitsets are small. 256 * 30 checks is ~7680 checks.
        # Each check is O(n) bitwise ops. Fast.
        
        for idx, c in enumerate(candidates):
            valid, new_sums, score = _check_candidate(sums_by_k, c)
            if valid:
                if score < best_score:
                    best_score = score
                    best_c = c
                    best_sums = new_sums
                    best_idx = idx
                    
        if best_c is None:
            # No more candidates can be added
            break
            
        # Commit to the best candidate
        sums_by_k = best_sums
        current_vals.append(best_c)
        candidates.pop(best_idx)
        
    return current_vals

def construct():
    global START, DEADLINE
    
    best_result = []
    best_len = 0
    
    # Run multiple greedy instances with different seeds
    # The number of seeds can be adjusted based on time
    num_seeds = 50
    
    for seed in range(num_seeds):
        if time.time() - START > DEADLINE - 5:
            break
            
        res = _greedy_solve(seed)
        
        if len(res) > best_len:
            # Validate before accepting
            if _validate(res):
                best_result = res
                best_len = len(res)
                
        # Early exit if we hit a very high size (e.g., > 20)
        # Though 20 might be low, let's see how far we get.
        # If best_len > 15, we can reduce seeds or time per seed?
        # For now, keep running.
        
    # If we have time, try to improve the best result with local search?
    # The prompt says "Maximize set size". Greedy is usually good for this.
    # Local search might swap elements to allow more to be added.
    
    if best_len < 1 and time.time() - START < DEADLINE - 10:
        # Fallback to a known small set if greedy failed completely
        best_result = [1, 2, 4, 8, 16, 32, 64, 128]
        
    return [int(x) for x in best_result]
