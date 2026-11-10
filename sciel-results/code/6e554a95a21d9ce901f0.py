import time
from math import gcd
from functools import reduce
import bisect

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    
    start_time = time.time()
    timeout = 85  # Safety margin for 90s limit
    
    # We need to find min W such that there exist distinct multiples m_i of d_i
    # with max(m_i) - min(m_i) <= W.
    
    # The values m_i are positive integers <= 1000000.
    # Given the divisors are small, the optimal solution will have small values.
    # We can limit the search for m_i to, say, 10000.
    
    # Precompute all multiples for each divisor up to a limit
    # Limit 10000 is safe.
    limit = 10000
    multiples_per_div = []
    for d in divisors:
        mults = list(range(d, limit + 1, d))
        multiples_per_div.append(mults)
        
    # To iterate over possible L (min value), we need a sorted list of all candidate values.
    # But L must be one of the m_i.
    # So L is in the union of all multiples.
    all_vals = set()
    for mults in multiples_per_div:
        all_vals.update(mults)
    all_vals = sorted(list(all_vals))
    
    # We will iterate W from 1 upwards.
    # For each W, we check if there's a valid window.
    # A window is defined by [L, L+W].
    # L must be a value such that all divisors have at least one multiple in [L, L+W].
    # And we can assign distinct values.
    
    # Optimization: 
    # For a fixed W, as L increases, the set of available multiples for each d_i changes.
    # We can iterate L over all_vals.
    
    # To speed up matching, we can use a recursive backtracking with ordering by constraint size.
    
    def try_assign(candidates, n):
        # candidates is a list of lists of available values for each divisor
        # Sort indices by number of candidates
        indices = list(range(n))
        indices.sort(key=lambda i: len(candidates[i]))
        
        used = set()
        assignment = [0] * n
        
        def backtrack(idx):
            if idx == n:
                return True
            i = indices[idx]
            for val in candidates[i]:
                if val not in used:
                    used.add(val)
                    assignment[i] = val
                    if backtrack(idx + 1):
                        return True
                    used.remove(val)
            return False
            
        if backtrack(0):
            return assignment
        return None

    # Iterate W
    # Upper bound for W? 
    # The gap between consecutive multiples of the largest divisor (39) is 39.
    # But we have 10 divisors.
    # A safe upper bound is sum of divisors or something.
    # Let's try up to 100.
    
    for W in range(1, 100):
        if time.time() - start_time > timeout:
            break
            
        # To avoid checking redundant L, we only need to check L where the set of candidates changes.
        # The set of candidates for d_i changes when L crosses a multiple of d_i.
        # So we can just iterate over all_vals as L.
        
        # However, if L is not a multiple of any d_i, the window [L, L+W] is the same as [L', L'+W]
        # where L' is the next multiple? No.
        # If L increases, the lower bound increases, potentially removing small multiples.
        # The upper bound L+W also increases, potentially adding large multiples.
        
        # We iterate L over all_vals.
        for L in all_vals:
            if time.time() - start_time > timeout:
                break
                
            R = L + W
            if R > limit:
                # We might still have solutions if we increase limit, but 10000 is likely enough.
                # If R > limit, we can't generate multiples beyond limit.
                # But if the optimal solution is within 10000, this is fine.
                # If L is large, R is large.
                # We can break if L > limit - W? No, just skip if R > limit and we can't find multiples.
                # Actually, if L > limit, no multiples.
                # If R > limit, some multiples might be missing.
                # For safety, if we reach the end of all_vals, we stop.
                pass
            
            candidates = []
            possible = True
            for i, d in enumerate(divisors):
                # Find multiples of d in [L, R]
                # k_min = ceil(L/d)
                # k_max = floor(R/d)
                k_min = (L + d - 1) // d
                k_max = R // d
                
                if k_min > k_max:
                    possible = False
                    break
                    
                # Generate candidates
                # We only need a few? No, we need all for backtracking.
                # But if the number is huge, it might be slow.
                # However, W is small, so k_max - k_min is small.
                cand = []
                for k in range(k_min, k_max + 1):
                    val = k * d
                    cand.append(val)
                candidates.append(cand)
                
            if not possible:
                continue
                
            # Check if any candidate list is empty (already handled by k_min > k_max)
            # Run backtracking
            res = try_assign(candidates, n)
            if res:
                # Verify
                if len(set(res)) == n:
                    for i in range(n):
                        if res[i] % divisors[i] != 0:
                            return divisors[:]
                    return res
    
    # Fallback
    return divisors[:]
