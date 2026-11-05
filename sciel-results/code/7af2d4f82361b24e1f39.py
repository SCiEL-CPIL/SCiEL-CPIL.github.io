import time
import math

def construct():
    start_time = time.time()
    DEADLINE = 115  # Safety margin
    
    # Precomputation
    N = 511
    factors = [set() for _ in range(N + 1)]
    for i in range(2, N + 1):
        if not factors[i]: # i is prime
            for j in range(i, N + 1, i):
                factors[j].add(i)
    
    # We want to maximize sum 1/(512-a).
    # Sort candidates a from 511 down to 1.
    candidates = list(range(N, 0, -1))
    weights = [0.0] * (N + 1)
    for a in candidates:
        weights[a] = 1.0 / (512 - a)
        
    # Precompute suffix sums of weights for a quick loose upper bound
    # suffix_sum[i] = sum(weights[candidates[i]] ... weights[candidates[N-1]])
    suffix_sum = [0.0] * (N + 1)
    for i in range(N - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + weights[candidates[i]]
        
    best_score = 0.0
    best_solution = []
    
    # Branch and Bound
    # We process candidates in order of decreasing weight (which is already 511..1)
    # index: current index in the sorted candidates list (0 to N-1)
    # selected_factors: frozenset of prime factors of chosen numbers
    # current_score: sum of weights of chosen numbers
    
    def bound(idx, selected_factors, current_score):
        # Quick check: if selected_factors is empty, the max possible is suffix_sum[idx]
        if not selected_factors:
            return current_score + suffix_sum[idx]
            
        # Otherwise, sum weights of compatible items
        b = current_score
        # Iterate from idx to N
        # Optimization: break if b + suffix_sum[idx] <= best_score? 
        # No, we are calculating the bound itself.
        for j in range(idx, N):
            a = candidates[j]
            if not (factors[a] & selected_factors):
                b += weights[a]
        return b

    def solve(idx, selected_factors, current_score, chosen):
        nonlocal best_score, best_solution
        
        if time.time() - start_time > DEADLINE:
            return
            
        # Pruning
        b = bound(idx, selected_factors, current_score)
        if b <= best_score:
            return
            
        if idx == N:
            if current_score > best_score:
                best_score = current_score
                best_solution = list(chosen)
            return
            
        a = candidates[idx]
        fac_a = factors[a]
        
        # Branch 1: Include a (if possible)
        # We try include first to get a good solution early
        if not (fac_a & selected_factors):
            new_selected_factors = frozenset(selected_factors | fac_a)
            new_current_score = current_score + weights[a]
            new_chosen = chosen + [a]
            solve(idx + 1, new_selected_factors, new_current_score, new_chosen)
            
        # Branch 2: Exclude a
        solve(idx + 1, selected_factors, current_score, chosen)

    solve(0, frozenset(), 0.0, [])
    
    return best_solution
