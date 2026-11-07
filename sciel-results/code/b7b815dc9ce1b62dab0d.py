import time
from math import gcd

# EVOLVE-BLOCK-START
P = {'divisors': [6, 10, 14, 15, 21, 22, 26, 33, 35, 39], 'maximum': 1000000}

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s safety margin (evaluator kills at 90s)
    
    divisors = P['divisors']
    n = len(divisors)
    max_val = P['maximum']
    
    # Precompute LCM of all divisors
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = divisors[0]
    for d in divisors[1:]:
        L = lcm(L, d)
    
    # For each divisor, precompute the "step" which is how many integers apart
    # consecutive multiples are. This is just the divisor itself.
    
    # We search for the minimal width W.
    # For a fixed W, we check all possible starting points S in [1, L].
    # If a solution is found in [S, S+W], the values are <= S+W <= L+W.
    # Since L=30030 and W will be small, values are well within 1000000.
    
    def find_assignment(start, width):
        # Returns list of assigned values or None if impossible
        # Candidates for each divisor i: multiples of divisors[i] in [start, start+width]
        candidates = [[] for _ in range(n)]
        
        for i, d in enumerate(divisors):
            # First multiple >= start
            k = (start + d - 1) // d
            x = k * d
            end = start + width
            while x <= end:
                candidates[i].append(x)
                x += d
        
        # Sort indices by number of candidates (MRV heuristic)
        # We need to keep track of original indices to return correct order
        # order[idx] = original index i
        indices = list(range(n))
        indices.sort(key=lambda i: len(candidates[i]))
        
        used = set()
        assignment = [0] * n
        
        # Check for immediate impossibility: any divisor has 0 candidates
        for i in indices:
            if not candidates[i]:
                return None
        
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

    # Search for minimal width
    # Theoretical lower bound: 1 (but distinct integers required, so range >= 9? No, range is max-min.
    # If values are 1,2,3,4,5,6,7,8,9,10, range is 9.
    # But they must be multiples.
    # Let's just iterate W from 1.
    
    # Optimization: Pre-calculate that for W < 9, it's impossible to have 10 distinct integers.
    # So start W from 9?
    # Actually, distinct integers $x_1 < x_2 < \dots < x_{10}$ implies $x_{10} - x_1 \ge 9$.
    # So min W is 9.
    
    for W in range(9, 200):  # Cap at 200 for safety, likely much smaller
        if time.time() > deadline:
            break
            
        for S in range(1, L + 1):
            if time.time() > deadline:
                break
                
            res = find_assignment(S, W)
            if res is not None:
                # Verify constraints
                # 1. Distinct
                if len(set(res)) != n:
                    continue
                # 2. Divisibility
                valid = True
                for i, val in enumerate(res):
                    if val % divisors[i] != 0:
                        valid = False
                        break
                # 3. Bounds
                if valid:
                    if any(val < 1 or val > max_val for val in res):
                        continue
                    return [int(v) for v in res]
                    
    # Fallback: Just return the divisors themselves (might not be optimal but valid)
    # The divisors are distinct?
    # 6, 10, 14, 15, 21, 22, 26, 33, 35, 39 are all distinct.
    # They are positive and <= 1000000.
    return [int(d) for d in divisors]
# EVOLVE-BLOCK-END
