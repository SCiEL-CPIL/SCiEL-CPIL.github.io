"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

P = {'n': 256, 'maximum_size': 60}

def compute_dp(S):
    """
    Compute the set of achievable sums for each cardinality using bitsets.
    Returns a list of integers where the i-th integer is a bitset of sums achievable by subsets of size i.
    """
    m = len(S)
    if m == 0:
        return [1]  # dp[0] = bit 0 set
    
    dp = [0] * (m + 1)
    dp[0] = 1  # bit 0 is set for cardinality 0
    
    for x in S:
        # Iterate backwards to ensure each element is used at most once
        for k in range(m, 0, -1):
            # Shift the bitset for cardinality k-1 by x and OR into cardinality k
            dp[k] |= (dp[k-1] << x)
            
    return dp

def count_collisions(dp):
    """
    Count the total number of sum collisions across different cardinalities.
    A collision is a sum value that is achievable by subsets of two different cardinalities.
    """
    m = len(dp) - 1
    collisions = 0
    for i in range(m + 1):
        for j in range(i + 1, m + 1):
            common = dp[i] & dp[j]
            if common:
                # Use bit_count for efficiency (Python 3.8+)
                collisions += common.bit_count()
    return collisions

def construct():
    """
    Construct a set of distinct integers in 1..256 such that whenever two subsets
    have different cardinalities, their sums must differ.
    
    Strategy:
    1. Iterate target size K from 60 down to 1.
    2. For each K, try to find a valid set using random restarts and local search (remove/replace).
    3. Return the largest valid set found.
    """
    max_size = 60
    max_val = 256
    
    # Try to find the largest valid set
    for K in range(max_size, 0, -1):
        found = _find_valid_set_of_size(K, max_val)
        if found is not None:
            return found
            
    # Fallback: return a single element (always valid)
    return [1]

def _find_valid_set_of_size(K, max_val):
    """
    Attempt to find a valid set of exactly size K.
    Uses random restarts and a local search that tries to replace elements to reduce collisions.
    """
    num_restarts = 50
    max_iters_per_restart = 100
    num_replacements_to_try = 10
    
    for _ in range(num_restarts):
        # Initialize with random distinct integers
        S = random.sample(range(1, max_val + 1), K)
        
        for _iter in range(max_iters_per_restart):
            dp = compute_dp(S)
            colls = count_collisions(dp)
            
            if colls == 0:
                return S
            
            # Find the element whose removal minimizes collisions
            min_colls_remove = colls + 1
            best_x_to_remove = None
            
            for idx, x in enumerate(S):
                S_temp = S[:idx] + S[idx+1:]
                dp_temp = compute_dp(S_temp)
                colls_temp = count_collisions(dp_temp)
                
                if colls_temp < min_colls_remove:
                    min_colls_remove = colls_temp
                    best_x_to_remove = x
                    if colls_temp == 0:
                        # If removing x makes it valid, we found a set of size K-1.
                        # But we are looking for size K. 
                        # However, if we can't maintain K, we should break out and try next K or restart.
                        # Actually, if we are in _find_valid_set_of_size(K), and we drop to K-1, we fail this restart.
                        break
            
            if best_x_to_remove is None:
                break
                
            # Try to replace best_x_to_remove with a new element to maintain size K
            improved = False
            best_new_S = None
            best_new_colls = min_colls_remove
            
            # Generate random candidates for replacement
            available = [v for v in range(1, max_val + 1) if v not in S]
            random.shuffle(available)
            
            for y in available[:num_replacements_to_try]:
                S_new = [x for x in S if x != best_x_to_remove]
                S_new.append(y)
                dp_new = compute_dp(S_new)
                colls_new = count_collisions(dp_new)
                
                if colls_new < best_new_colls:
                    best_new_colls = colls_new
                    best_new_S = S_new
                    if colls_new == 0:
                        return S_new
            
            if best_new_S is not None:
                S = best_new_S
            else:
                # No replacement improved the situation, just remove
                S.remove(best_x_to_remove)
                if len(S) < K:
                    break # Failed to maintain size K
            
    return None

# EVOLVE-BLOCK-END
