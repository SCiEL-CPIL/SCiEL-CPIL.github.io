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
    
    # Precompute multiples for each divisor within one LCM period?
    # Not strictly necessary if we calculate on the fly, but might help.
    # Let's just calculate candidates on the fly.
    
    n = len(DIVISORS)
    
    # Lower bound for L
    # The minimum possible range is 0, but distinctness requires at least 1.
    # However, if we have divisors 6 and 10, in a window of length 1, 
    # can we pick distinct multiples? 
    # If S=1, window [1,1]. Multiples of 6: none. 
    # If S=6, window [6,6]. Mult 6: {6}, Mult 10: {}. Fail.
    # So L must be large enough to accommodate the "sparsest" divisor's requirement 
    # combined with distinctness.
    # Let's just start from L=1 and go up.
    
    best_solution = None
    best_L = float('inf')
    
    # We can also randomize S order or start from 0.
    # Starting from 0 is fine due to periodicity.
    
    for L in range(1, 10000): # Upper bound guess
        if time.time() > deadline:
            break
            
        found = False
        # Iterate through all possible start positions in one period
        for S in range(LCM_VAL):
            if time.time() > deadline:
                break
                
            # Generate domains for each divisor
            domains = []
            valid = True
            for i, d in enumerate(DIVISORS):
                # Find multiples of d in [S, S+L]
                # First multiple >= S
                start_mult = math.ceil(S / d) * d
                candidates = []
                m = start_mult
                while m <= S + L:
                    candidates.append(m)
                    m += d
                if not candidates:
                    valid = False
                    break
                domains.append(candidates)
                
            if not valid:
                continue
                
            # Solve CSP
            # Assignment array
            assignment = [None] * n
            
            # Backtracking with MRV
            def backtrack(idx, used_values):
                if idx == n:
                    return True
                
                # Find variable with smallest domain (MRV)
                # Since domains are static for this S,L, we can pre-sort indices by domain size
                # But domains are just lists. Let's pick the unassigned one with smallest len(domains[i])
                min_len = float('inf')
                min_idx = -1
                for j in range(n):
                    if assignment[j] is None:
                        # Filter domain by used_values? 
                        # Actually, we can just check during iteration.
                        # For MRV, we should consider the *current* available values.
                        # available = [v for v in domains[j] if v not in used_values]
                        # To avoid creating lists, we can just estimate or do it.
                        # n=10, domains small. Creating list is fine.
                        avail_count = 0
                        for v in domains[j]:
                            if v not in used_values:
                                avail_count += 1
                                if avail_count >= min_len: # Optimization: stop counting if already larger
                                    break
                        if avail_count == 0:
                            return False # Dead end
                        if avail_count < min_len:
                            min_len = avail_count
                            min_idx = j
                            
                if min_idx == -1:
                    return True # All assigned? Should be caught by idx==n if we increment idx properly. 
                # Wait, the standard recursive backtrack usually picks variable based on index or heuristic.
                # Let's restructure: pick the unassigned variable with min available values.
                
                if min_idx == -1:
                    return True
                
                # Try values for min_idx
                for v in domains[min_idx]:
                    if v in used_values:
                        continue
                    assignment[min_idx] = v
                    used_values.add(v)
                    
                    if backtrack(idx + 1, used_values):
                        return True
                        
                    assignment[min_idx] = None
                    used_values.remove(v)
                
                return False
            
            # Initial call
            if backtrack(0, set()):
                # Found solution
                result = [int(assignment[i]) for i in range(n)]
                # Check validity again just in case
                if all(result[i] % DIVISORS[i] == 0 for i in range(n)) and len(set(result)) == n and all(x <= MAX_VAL for x in result):
                    return result
                else:
                    # Should not happen
                    pass
    
    # If no solution found within time or L limit, return best effort or default
    # But a solution should exist for small L.
    # Fallback: return divisors themselves (might not be optimal but valid? No, must be distinct? Divisors are distinct. Are they multiples? Yes. Range is max(D)-min(D).)
    # Wait, the divisors themselves are valid representatives.
    # 6, 10, 14, 15, 21, 22, 26, 33, 35, 39.
    # Distinct? Yes.
    # Multiples? Yes.
    # Range: 39 - 6 = 33.
    # This is a valid baseline.
    
    return DIVISORS[:]

# EVOLVE-BLOCK-END
