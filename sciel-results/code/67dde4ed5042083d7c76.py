import time
import math
from functools import lru_cache

# Fixed inputs
DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
MAX_VAL = 1000000
LCM_VAL = 30030 # Precomputed LCM of DIVISORS

def construct():
    start_time = time.time()
    deadline = start_time + 50.0 # 50s safety margin for 60s limit
    
    n = len(DIVISORS)
    
    # We want to find distinct multiples v_i of d_i such that max(v)-min(v) is minimized.
    # Let the window be [S, S+L].
    # We can iterate L from small to large.
    # For a fixed L, we can iterate S from 0 to LCM_VAL-1.
    # For a fixed (L, S), we need to check if there is a valid assignment.
    
    # Optimization: 
    # Instead of backtracking for every (L, S), we can use a bitset or just fast backtracking.
    # Given n=10, backtracking is very fast if domains are small.
    # Domain size for divisor d in window L is roughly L/d + 1.
    # Max L we might need? 
    # The divisors are 6, 10, 14, 15, 21, 22, 26, 33, 35, 39.
    # The "tightest" constraint is likely due to the largest divisors or conflicts.
    # Let's try L up to 100 first. If not found, increase.
    
    # Pre-compute domain sizes or just compute on the fly.
    # On the fly is fine.
    
    # Heuristic: Sort divisors by size descending for MRV in backtracking?
    # Actually, MRV (Minimum Remaining Values) is dynamic.
    # But we can pre-sort indices by domain size (ascending) for initial order, 
    # or just use dynamic MRV which is robust.
    
    # Let's refine the backtracking to be as fast as possible.
    # Represent used values as a set or a boolean array?
    # Since values are in [S, S+L], and L is small (likely < 100), 
    # we can use a boolean array of size L+1 for the window, or just a set of relative indices.
    # Using a set of relative indices (0 to L) is efficient.
    
    # Let's define the backtracking function inside the loop to capture domains.
    
    best_solution = None
    best_L = float('inf')
    
    # We iterate L from 1 upwards.
    # What is the max L? 
    # A trivial solution is the divisors themselves, range 33.
    # So we only need to check L up to 33? 
    # Wait, the divisors themselves are 6, 10, 14, 15, 21, 22, 26, 33, 35, 39.
    # Min=6, Max=39. Range=33.
    # So L=33 is a valid solution (S=6).
    # Can we do better?
    # Let's check L=1, 2, ..., 33.
    # If we find a solution for L, we can return it immediately?
    # Yes, because we iterate L in increasing order. The first valid L found is optimal.
    
    # However, we must ensure that for a given L, we check ALL S in [0, LCM_VAL).
    # If no S works for L, we increment L.
    
    # Is it possible that a smaller L requires a larger S that we might miss?
    # No, we check all S in one period.
    
    # Let's start L from 1.
    
    # Optimization: 
    # For a given L, if L is very small, some divisors might have no multiples in the window.
    # E.g., if L=1, window size 2.
    # Divisor 39 needs a multiple. In any window of size 2, is there always a multiple of 39?
    # No. e.g., [1, 2] has no multiple of 39.
    # So we can quickly reject many (L, S) pairs if any domain is empty.
    
    for L in range(1, 100): # Upper bound 100 is safe since 33 is a known solution
        if time.time() > deadline:
            break
            
        # Pre-compute "possible" S values?
        # Or just iterate all S.
        # 30030 iterations for L.
        # If L is small, we reject fast.
        
        for S in range(LCM_VAL):
            if time.time() > deadline:
                break
                
            # Check if each divisor has at least one multiple in [S, S+L]
            domains = []
            valid = True
            for d in DIVISORS:
                # First multiple >= S
                # start_mult = ((S + d - 1) // d) * d
                start_mult = ((S + d - 1) // d) * d
                if start_mult > S + L:
                    valid = False
                    break
                # Count or store candidates?
                # For backtracking, we need the list.
                # But for quick rejection, just checking existence is enough.
                # Let's store the list if valid so far.
                candidates = []
                m = start_mult
                while m <= S + L:
                    candidates.append(m)
                    m += d
                domains.append(candidates)
                
            if not valid:
                continue
            
            # Now solve CSP
            # We have n domains.
            # Use backtracking with MRV.
            
            assignment = [None] * n
            used = set()
            
            # To speed up MRV, we can pre-compute domain sizes.
            # But domains are static for this (S, L).
            # So we can sort indices by domain size initially?
            # No, MRV is dynamic because 'used' changes.
            # However, n=10 is small.
            
            # Let's use a recursive function.
            # To avoid recursion overhead, we could use a stack, but recursion is fine for n=10.
            
            # Optimization: Order variables by initial domain size (ascending).
            # This is a static heuristic, but often good.
            # Let's try dynamic MRV.
            
            def backtrack(idx):
                if idx == n:
                    return True
                
                # Find unassigned variable with min available values
                min_avail = float('inf')
                min_idx = -1
                
                # We can iterate over all unassigned variables
                for j in range(n):
                    if assignment[j] is None:
                        # Count available values
                        avail_count = 0
                        for v in domains[j]:
                            if v not in used:
                                avail_count += 1
                                if avail_count >= min_avail:
                                    break
                        if avail_count == 0:
                            return False
                        if avail_count < min_avail:
                            min_avail = avail_count
                            min_idx = j
                            
                if min_idx == -1:
                    return True # Should not happen if idx < n and some unassigned
                
                # Try values for min_idx
                # We can sort values? Not necessary.
                for v in domains[min_idx]:
                    if v in used:
                        continue
                    
                    assignment[min_idx] = v
                    used.add(v)
                    
                    if backtrack(idx + 1):
                        return True
                        
                    used.remove(v)
                    assignment[min_idx] = None
                
                return False
            
            if backtrack(0):
                # Found a solution
                result = [int(assignment[i]) for i in range(n)]
                # Final validation
                if all(result[i] % DIVISORS[i] == 0 for i in range(n)) and len(set(result)) == n and all(1 <= x <= MAX_VAL for x in result):
                    return result
    
    # Fallback: return divisors themselves
    return DIVISORS[:]

# EVOLVE-BLOCK-END
