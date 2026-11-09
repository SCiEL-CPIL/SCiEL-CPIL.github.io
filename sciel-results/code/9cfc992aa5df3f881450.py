"""Rewritten program: DIRECT approach to pack distinct multiples of fixed divisors."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from functools import reduce

P = {
    'divisors': [6, 10, 14, 15, 21, 22, 26, 33, 35, 39],
    'maximum': 1000000
}

def construct():
    divisors = P['divisors']
    n = len(divisors)
    MAX_V = P['maximum']
    deadline = time.time() + 85.0  # safety margin before 90s kill
    
    # Calculate LCM of all divisors to exploit periodicity
    def lcm(a, b):
        return a * b // gcd(a, b)
    lcm_all = reduce(lcm, divisors)
    
    # Simple DFS backtracking for assignment
    def find_assignment_dfs(candidates):
        # Sort by number of candidates to prune early
        # We need to map back to original indices
        order = sorted(range(n), key=lambda i: len(candidates[i]))
        sorted_cands = [candidates[i] for i in order]
        
        assigned = [0] * n
        used = set()
        
        def dfs(idx):
            if idx == n:
                return True
            for v in sorted_cands[idx]:
                if v in used:
                    continue
                used.add(v)
                assigned[idx] = v
                if dfs(idx + 1):
                    return True
                used.remove(v)
            return False
        
        if dfs(0):
            # Reconstruct original order
            result = [0] * n
            for s_idx, o_idx in enumerate(order):
                result[o_idx] = assigned[s_idx]
            return result
        return None

    # Iterate span W from minimum possible (9) upwards
    for W in range(9, 1000):
        if time.time() > deadline:
            break
        
        # Iterate window start L
        for L in range(1, lcm_all + 1):
            if time.time() > deadline:
                break
            
            if L + W > MAX_V:
                continue

            candidates = []
            valid_window = True
            
            for d in divisors:
                first_k = (L + d - 1) // d
                last_k = (L + W) // d
                
                if first_k > last_k:
                    valid_window = False
                    break
                
                cands = [k * d for k in range(first_k, last_k + 1)]
                candidates.append(cands)
            
            if not valid_window:
                continue
            
            # Try to assign distinct values
            result = find_assignment_dfs(candidates)
            
            if result is not None:
                # Verify
                if len(set(result)) != n:
                    continue
                for i, d in enumerate(divisors):
                    if result[i] % d != 0:
                        continue
                if any(v < 1 or v > MAX_V for v in result):
                    continue
                
                return [int(v) for v in result]
    
    # Fallback
    fallback = [int(d) for d in divisors]
    return fallback

# EVOLVE-BLOCK-END
