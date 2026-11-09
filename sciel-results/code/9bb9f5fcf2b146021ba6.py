import time
import math

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    period = 30030  # LCM of divisors
    
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    # Precompute divisors sorted by "difficulty" is done dynamically per window
    # because the number of candidates depends on L and W.

    # We search for W from 1 upwards
    for W in range(1, 1000):
        if time.time() > deadline:
            break
            
        # We only need to check L in [1, period]
        # Because the pattern of multiples repeats every 'period'.
        # If a solution exists for (L, W), it exists for (L + k*period, W) with shifted values.
        # Since we want to minimize the span, the absolute values don't matter for the span,
        # only the relative positions. But we need distinct integers.
        # Shifting by period preserves distinctness.
        # So checking L in 1..period is sufficient.
        
        for L in range(1, period + 1):
            if time.time() > deadline:
                break
                
            # Generate candidates for each divisor in the window [L, L+W]
            # A candidate x must satisfy: L <= x <= L+W and x % d == 0
            # Let x = k*d. Then L <= k*d <= L+W => ceil(L/d) <= k <= floor((L+W)/d)
            
            candidates = []
            valid = True
            for d in divisors:
                k_min = (L + d - 1) // d
                k_max = (L + W) // d
                if k_min > k_max:
                    valid = False
                    break
                # Store the actual values
                # Values are k_min*d, (k_min+1)*d, ...
                # We store the list of values
                vals = list(range(k_min * d, k_max * d + 1, d))
                candidates.append(vals)
                
            if not valid:
                continue
                
            # Check if total candidates >= n
            total_cands = sum(len(c) for c in candidates)
            if total_cands < n:
                continue
                
            # MRV Heuristic: Sort indices by length of candidate list
            order = sorted(range(n), key=lambda i: len(candidates[i]))
            sorted_cands = [(i, candidates[i]) for i in order]
            
            # Backtracking
            # used_mask: bitset or boolean list for values in [L, L+W]
            # Since W is small, we can use a boolean list of size W+1
            # index = value - L
            used = [False] * (W + 1)
            assignment = [None] * n
            
            def backtrack(idx):
                if idx == n:
                    return True
                orig_idx, vals = sorted_cands[idx]
                for val in vals:
                    offset = val - L
                    if not used[offset]:
                        used[offset] = True
                        assignment[orig_idx] = val
                        if backtrack(idx + 1):
                            return True
                        used[offset] = False
                        assignment[orig_idx] = None
                return False

            if backtrack(0):
                # Return the assignment in the original order
                result = [int(assignment[i]) for i in range(n)]
                return result

    # Fallback: return the divisors themselves (valid but not optimal)
    return [int(d) for d in divisors]
