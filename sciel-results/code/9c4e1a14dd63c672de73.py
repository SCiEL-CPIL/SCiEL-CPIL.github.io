import time
import math
from functools import reduce

def construct():
    """
    Finds 10 distinct positive integers <= 1000000, each divisible by its corresponding divisor in [6,10,14,15,21,22,26,33,35,39],
    such that max(reps) - min(reps) is minimized.
    
    Strategy: The pattern of multiples repeats every LCM of the divisors.
    We search for the optimal window starting point L within one LCM period.
    For each candidate L (which must be a multiple of at least one divisor to be the minimum),
    we greedily assign the smallest available multiple of each other divisor that is >= L.
    """
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    max_val = 1000000
    deadline = time.time() + 85  # Safety margin under 90s limit
    
    # Compute LCM of all divisors
    lcm_val = reduce(lambda a, b: a * b // math.gcd(a, b), divisors)
    # lcm_val is 3270270
    
    # The search space for L is [1, lcm_val] but we only need to check L that are multiples of some divisor
    # because if L is not a multiple of any divisor, the minimum representative would be > L, 
    # so we could shift the window down.
    # Actually, the minimum representative must be a multiple of its corresponding divisor.
    # So the window [min, max] is determined by the set of representatives.
    # The minimum representative, say r_min, corresponds to some divisor d_k.
    # So r_min is a multiple of d_k.
    # Thus, we can iterate over possible r_min values.
    # Since the pattern repeats every lcm_val, we only need to check r_min in [1, lcm_val].
    
    # However, lcm_val is 3.27 million. Iterating through all multiples of all divisors in this range
    # is feasible if we are careful.
    # Total candidates = sum(lcm_val // d for d in divisors) ~ 3.27M * (1/6 + 1/10 + ...) ~ 3.27M * 0.5 ~ 1.6M
    # This is a bit high for a Python loop with inner logic.
    
    # Optimization:
    # 1. We can limit the search range. If we find a span S, we know that any solution with span < S
    #    must have all representatives within an interval of length S.
    # 2. We can use a smarter search. Instead of iterating all L, we can iterate L in steps of 1? No.
    #    We iterate L which are multiples of divisors.
    
    # Let's refine the greedy assignment.
    # Given a lower bound L (which is the minimum representative), for each divisor d_i,
    # we need to find the smallest multiple of d_i that is >= L and distinct from others.
    # To minimize the maximum, we should assign smaller divisors first? Or larger?
    # If we assign larger divisors first, they have fewer multiples, so they are more constrained.
    # Actually, the standard approach for this type of problem (minimize max-min with divisibility)
    # is to fix the minimum and greedily pick the smallest possible values for the rest,
    # ensuring distinctness. The order of assignment matters for the greedy choice's quality.
    # Assigning in descending order of divisors tends to keep the values tighter because
    # larger divisors have sparser multiples, so picking the smallest available one
    # prevents large gaps later.
    
    indices_desc = sorted(range(n), key=lambda i: divisors[i], reverse=True)
    
    best_span = float('inf')
    best_result = None
    
    # Generate candidate minimum values L
    # L must be a multiple of at least one divisor.
    # To avoid duplicates in the candidate set, we can just iterate L from 1 to lcm_val
    # and check if it's a multiple of any divisor? That's 3.27M iterations.
    # Better: iterate over each divisor's multiples and add to a set.
    # But a set of 1.6M integers might be memory heavy and slow to sort.
    
    # Alternative: Iterate L from 1 to lcm_val.
    # For each L, check if L is a multiple of any divisor.
    # This is 3.27M checks. Each check is 10 modulo operations. ~32M ops.
    # In Python, this might take ~5-10 seconds. Acceptable.
    
    # However, we can prune. If L + best_span - 1 > max_val, we can stop?
    # No, because the representatives can be larger than L + best_span.
    # But if L > max_val, we stop.
    # Also, if we have a current best_span, and for a given L, the greedy assignment
    # produces a span >= best_span, we discard it.
    
    # Let's try a different approach:
    # The optimal span is likely small.
    # We can search for the span S starting from 1 upwards.
    # For a given span S, is there a window of length S containing distinct multiples?
    # This is a feasibility problem.
    # For a window [L, L+S], we need to assign each divisor d_i a multiple m_i in [L, L+S]
    # such that all m_i are distinct.
    # This is a bipartite matching problem or a Hall's theorem check.
    # But S is likely small (around 50-100?).
    # If we iterate S from 1 to 1000, and for each S, iterate L... that's too much.
    
    # Let's stick to the L-iteration but optimize it.
    # We will iterate L from 1 to lcm_val.
    # We will only process L if it is a multiple of at least one divisor.
    
    # To speed up the "is multiple" check, we can precompute a boolean array?
    # lcm_val is 3.27M. A boolean array of that size is ~3MB. Feasible.
    # But initializing it takes time.
    
    # Let's just iterate L and check.
    # We can also break early if L + best_span > max_val? No, reps can go up to max_val.
    # But if L is very large, say > max_val - best_span, it's still possible.
    
    # Let's add a heuristic: if we find a solution with span S, we can stop
    # if we've checked enough L's? No, we need the global minimum.
    
    # One more optimization:
    # If L is not a multiple of any divisor, skip.
    # We can check this quickly.
    
    # Let's implement the greedy assignment with a twist:
    # We maintain a set of used numbers.
    # For each divisor in descending order, find the smallest multiple >= L not in used.
    
    for L in range(1, lcm_val + 1):
        if time.time() > deadline:
            break
        
        # Check if L is a multiple of any divisor
        is_multiple = False
        for d in divisors:
            if L % d == 0:
                is_multiple = True
                break
        
        if not is_multiple:
            continue
        
        # Greedy assignment
        used = set()
        reps = [0] * n
        max_rep = L
        valid = True
        
        # We need to ensure L is assigned to the divisor that divides it.
        # If L is a multiple of multiple divisors, which one gets it?
        # To minimize the span, it's generally better to assign L to the divisor
        # that would otherwise have to take a larger multiple?
        # Or simply: try all assignments of L to its valid divisors?
        # That increases complexity by a factor of 10.
        
        # Let's try a simpler heuristic first:
        # Assign L to the largest divisor that divides L.
        # Why? Because that divisor has the fewest multiples, so it's "tightest".
        # If we give it a larger number, the span grows more.
        
        assigned_to = -1
        max_divisor_at_L = 0
        for i, d in enumerate(divisors):
            if L % d == 0 and d > max_divisor_at_L:
                max_divisor_at_L = d
                assigned_to = i
        
        if assigned_to == -1:
            continue # Should not happen if is_multiple is True
        
        reps[assigned_to] = L
        used.add(L)
        
        for i in indices_desc:
            if i == assigned_to:
                continue
            d = divisors[i]
            # Find smallest multiple of d >= L
            m = L
            rem = m % d
            if rem != 0:
                m += d - rem
            
            # Resolve collision
            while m in used:
                m += d
            
            if m > max_val:
                valid = False
                break
            
            used.add(m)
            reps[i] = m
            if m > max_rep:
                max_rep = m
        
        if valid:
            span = max_rep - L
            if span < best_span:
                best_span = span
                best_result = reps[:]
                if best_span <= 10:
                    # Likely optimal or near-optimal. We can break early if we want.
                    # But let's continue for a bit to be sure.
                    # Actually, if we find span=10, it's hard to beat.
                    # Let's just return if span is very small.
                    if best_span <= 5:
                        return [int(x) for x in best_result]
    
    if best_result is None:
        # Fallback: just return the divisors themselves (if distinct and <= max_val)
        # They are distinct and <= 1000000.
        best_result = divisors[:]
    
    return [int(x) for x in best_result]
