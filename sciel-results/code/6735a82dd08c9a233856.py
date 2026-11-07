import time
from math import gcd
from functools import reduce

# EVOLVE-BLOCK-START
P = {'divisors': [6, 10, 14, 15, 21, 22, 26, 33, 35, 39], 'maximum': 1000000}

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s safety margin within 60s limit
    
    divisors = P['divisors']
    n = len(divisors)
    max_val = P['maximum']
    
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    period = reduce(lcm, divisors)
    
    best_diff = float('inf')
    best_solution = None
    
    # We iterate through all possible starting offsets modulo the period.
    # For each start S, we find the minimal multiples >= S for each divisor.
    # We then try to assign distinct values to minimize the span.
    
    # Pre-calculate first multiple >= S for each divisor is O(n) per S.
    # We can optimize by noting that as S increases by 1, the multiples change predictably.
    # But n=10 and period=30030, so 300k operations is trivial.
    
    for S in range(1, period + 1):
        if time.time() > deadline:
            break
            
        # Find the first multiple of d_i >= S for each i
        # candidates[i] will hold a small list of the first few multiples to handle collisions
        candidates = [[] for _ in range(n)]
        
        for i, d in enumerate(divisors):
            # First multiple >= S
            k = (S + d - 1) // d
            x = k * d
            # Add a few candidates to allow backtracking for distinctness
            for _ in range(5):
                candidates[i].append(x)
                x += d
        
        # Now we need to pick one value from each candidates[i] such that all are distinct
        # and max(picks) - min(picks) is minimized.
        
        # Sort indices by the number of candidates (though they are all 5 here, 
        # sorting by the first candidate might help prune)
        # Actually, let's just use a simple backtracking that tries to keep values close.
        
        # A better heuristic: 
        # The optimal min value will be one of the values in the union of candidates.
        # But since we started at S, the smallest value will be close to S.
        
        # Let's try to find the assignment that minimizes the span.
        # We can iterate through possible min values M from the union of candidates,
        # but that's complex.
        
        # Simpler: Since we only have 10 items and 5 candidates each, 
        # we can use a backtracking search ordered by value.
        
        # Flatten all (value, index) pairs, sort by value.
        all_vals = []
        for i in range(n):
            for v in candidates[i]:
                all_vals.append((v, i))
        
        # We want to pick 10 values, one from each index, distinct, min span.
        # This is equivalent to finding a clique in a specific graph, or just backtracking.
        
        # Optimization: 
        # The span is determined by max - min.
        # We can iterate through the sorted unique values as potential "min".
        # For a fixed min M, we must pick M for some index j where M is in candidates[j].
        # Then for all other indices, we must pick a value >= M.
        # To minimize max, we should pick the smallest available value >= M for each index.
        # If all are distinct, we have a valid solution with span max_picked - M.
        
        # Let's implement this "Fixed Min" strategy.
        
        # Get unique sorted values from the first few candidates
        # Note: candidates are already sorted for each i.
        
        # To be efficient, we only consider M from the set of the first 2-3 candidates of each divisor.
        # Because if M is large, the span will likely be large.
        
        potential_mins = set()
        for i in range(n):
            for v in candidates[i][:3]: # Take first 3 as potential mins
                potential_mins.add(v)
        
        for M in potential_mins:
            # Check if M is a candidate for any divisor
            # If so, try to assign M to that divisor (or one of them) and smallest others.
            
            # To handle the case where M is in multiple lists, we try each possibility?
            # Or just: For each index i, if M is in candidates[i], we can assign M to i.
            # Then for j != i, we pick the smallest v in candidates[j] such that v >= M and v != M (if j!=i).
            # Wait, if j != i, v can be M only if M is in candidates[j] AND we didn't assign M to i?
            # No, values must be distinct. So if we assign M to i, no one else can take M.
            
            # So:
            # 1. Identify indices that have M as a candidate.
            # 2. For each such index i, assume x_i = M.
            # 3. For all j != i, find the smallest v in candidates[j] such that v >= M and v != M.
            # 4. If all such v exist, compute span = max(all v) - M.
            # 5. Update best.
            
            indices_with_M = [i for i in range(n) if M in candidates[i]]
            
            for i in indices_with_M:
                used = {M}
                picks = [0] * n
                picks[i] = M
                valid = True
                max_val_pick = M
                
                for j in range(n):
                    if j == i:
                        continue
                    # Find smallest v in candidates[j] >= M and v not in used
                    # Since candidates[j] is sorted, we can iterate
                    found = False
                    for v in candidates[j]:
                        if v < M:
                            continue
                        if v in used:
                            continue
                        # This v is a candidate
                        # But we need to ensure we can assign the rest?
                        # Greedy: pick the smallest available.
                        # This greedy works for minimizing max if we process in order?
                        # Not necessarily, but it's a good heuristic for "minimal span".
                        # Since we want to minimize the max, picking the smallest available for each 
                        # remaining slot is optimal for that specific fixed Min and fixed assignment of Min.
                        
                        picks[j] = v
                        used.add(v)
                        if v > max_val_pick:
                            max_val_pick = v
                        found = True
                        break
                    
                    if not found:
                        valid = False
                        break
                
                if valid:
                    span = max_val_pick - M
                    if span < best_diff:
                        best_diff = span
                        best_solution = picks[:]
                        if best_diff == 1:
                            return [int(x) for x in best_solution]
    
    if best_solution is not None:
        # Ensure types are int
        return [int(x) for x in best_solution]
    
    # Fallback: Just return the divisors themselves (which are distinct multiples)
    # Span will be max(divisors) - min(divisors)
    return [int(d) for d in divisors]

# EVOLVE-BLOCK-END
