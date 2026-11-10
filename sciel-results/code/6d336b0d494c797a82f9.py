import itertools
import math
import time

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    max_val = 1000000
    
    # LCM of divisors to define the period of the pattern
    lcm_val = 1
    for d in divisors:
        lcm_val = lcm_val * d // math.gcd(lcm_val, d)
        
    start_time = time.time()
    deadline = 50.0 # seconds
    
    # Try increasing window widths
    for W in range(1, 200):
        # For each possible start L in one period
        # We only need to check L such that the window [L, L+W] doesn't wrap around 
        # in a way that changes the set of candidates? 
        # Actually, the candidates for x_i are multiples of d_i.
        # The set of candidates depends on floor(L/d_i) and floor((L+W)/d_i).
        
        # We can iterate L from 1 to lcm_val.
        # To speed up, we can break early if time is tight.
        
        for L in range(1, lcm_val + 1):
            if time.time() > deadline:
                break
                
            # Determine candidates for each divisor
            candidates = []
            valid = True
            for d in divisors:
                # Multiples of d in [L, L+W]
                # k_min = ceil(L/d)
                # k_max = floor((L+W)/d)
                k_min = (L + d - 1) // d
                k_max = (L + W) // d
                
                cand_list = []
                for k in range(k_min, k_max + 1):
                    val = k * d
                    # Ensure positive (k_min >= 1 if L>=1)
                    if val <= max_val:
                        cand_list.append(val)
                
                if not cand_list:
                    valid = False
                    break
                candidates.append(cand_list)
            
            if not valid:
                continue
                
            # Check if we can pick distinct values from candidates
            # Backtracking search
            used = set()
            result = [None] * n
            
            def backtrack(idx):
                if idx == n:
                    return True
                for val in candidates[idx]:
                    if val not in used:
                        used.add(val)
                        result[idx] = val
                        if backtrack(idx + 1):
                            return True
                        used.remove(val)
                        result[idx] = None
                return False
            
            if backtrack(0):
                # Verify distinctness and bounds
                if len(set(result)) == n and all(1 <= v <= max_val for v in result):
                    # Check divisibility
                    ok = True
                    for i, v in enumerate(result):
                        if v % divisors[i] != 0:
                            ok = False
                            break
                    if ok:
                        return result
                        
        # If no solution found for this W in the first period, 
        # it's possible one exists but with larger L?
        # No, if a pattern of residues works, it works for any L with that residue.
        # The smallest L giving that residue will give the smallest values.
        # So if no solution in [1, LCM], no solution exists for this W 
        # (with values <= 1000000, assuming LCM is small enough).
        
    # Fallback: return the divisors themselves (score 1.0)
    return divisors[:]
