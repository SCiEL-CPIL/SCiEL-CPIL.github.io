import math
import time
from fractions import Fraction

def get_divisors(n):
    divs = []
    for i in range(1, int(math.isqrt(n)) + 1):
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
    return divs

def find_solution_for_L(L, deadline):
    # Get divisors >= 5
    divs = [d for d in get_divisors(L) if d >= 5]
    if not divs:
        return None, 100
    
    # Weights are L // d
    # DP: min count to reach sum s
    # Initialize with infinity
    INF = 10**9
    dp = [INF] * (L + 1)
    dp[0] = 0
    
    # To reconstruct, we can store the choice
    parent = [None] * (L + 1)
    
    for d in divs:
        w = L // d
        # Update dp in reverse to avoid using same divisor twice
        for s in range(L - w, -1, -1):
            if dp[s] + 1 < dp[s + w]:
                dp[s + w] = dp[s] + 1
                parent[s + w] = (s, d)
        
        # Check time
        if time.time() > deadline:
            return None, 100

    if dp[L] < INF:
        # Reconstruct solution
        sol = []
        s = L
        while s > 0:
            if parent[s] is None:
                break
            prev_s, d = parent[s]
            sol.append(d)
            s = prev_s
        if sum(1/x for x in sol) == 1.0:
            pass
        return sol, dp[L]
    return None, INF

def greedy_egyptian(target_frac, min_den, max_den, max_terms, deadline, used=set()):
    """
    Greedy algorithm to find an Egyptian fraction representation of target_frac
    with denominators in [min_den, max_den], at most max_terms terms, distinct,
    not in used.
    Returns a list of denominators or None.
    """
    if max_terms <= 0:
        return None
    if target_frac == 0:
        return []
    if target_frac < 0:
        return None
    
    # Find the largest possible unit fraction 1/d <= target_frac
    # i.e., d >= 1/target_frac
    # d must be >= min_den, not in used, and d <= max_den
    # Also, 1/d <= target_frac => d >= ceil(1/target_frac)
    
    target_num, target_den = target_frac.numerator, target_frac.denominator
    # We need d such that 1/d <= target_frac => d >= target_den / target_num
    min_d_needed = math.ceil(target_den / target_num)
    d_start = max(min_den, min_d_needed)
    
    if d_start > max_den:
        return None
    
    # Try d from d_start upwards
    # But we want to minimize the number of terms, so we should pick the smallest d
    # that allows the remainder to be represented by fewer terms.
    # Greedy: pick the smallest d >= d_start such that d not in used.
    # Then recurse on remainder with max_terms - 1.
    
    for d in range(d_start, max_den + 1):
        if d in used:
            continue
        
        rem = target_frac - Fraction(1, d)
        if rem == 0:
            return [d]
        if rem < 0:
            break
        
        # Check if rem can be represented by max_terms - 1 terms
        # with denominators > d (to ensure distinctness and avoid cycles)
        # Actually, denominators just need to be distinct and >= min_den.
        # But for efficiency, we can require next denominators > d.
        sub_sol = greedy_egyptian(rem, max(d + 1, min_den), max_den, max_terms - 1, deadline, used | {d})
        if sub_sol is not None:
            return [d] + sub_sol
        
        if time.time() > deadline:
            return None
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    # Known solution with 10 terms
    best_sol = [5, 6, 7, 8, 9, 10, 14, 15, 63, 2520]
    best_k = len(best_sol)
    
    # Try greedy algorithm first with various starting points
    # Target is 1
    target = Fraction(1, 1)
    
    # The greedy algorithm might get stuck if it picks a denominator too small
    # or if the remainder becomes hard to represent.
    # Let's try a few different "forced" first steps or just pure greedy.
    
    # Pure greedy:
    # 1/1 -> d=1 (not allowed, min 5)
    # 1/5 = 0.2, rem = 0.8 = 4/5
    # 1/2 <= 4/5 -> d=2 (not allowed)
    # ...
    # Smallest d >= 5 such that 1/d <= 1 is d=5.
    # So greedy picks 5. Rem = 4/5.
    # Next smallest d >= 5 (and > 5 to be distinct? or just not in used?)
    # If we allow d=5 again, sum > 1. So d > 5.
    # 1/6 <= 4/5 -> 5/6 <= 4? Yes.
    # So greedy picks 6. Rem = 4/5 - 1/6 = 24/30 - 5/30 = 19/30.
    # Next d > 6. 1/7 <= 19/30? 30/19 ~ 1.57 -> d >= 2? No, d >= ceil(30/19)=2.
    # But d > 6. So d=7. 1/7 = 30/210. 19/30 = 133/210. Rem = 103/210.
    # This is just the standard greedy algorithm.
    
    # Let's run greedy with max_terms=10 first to see if it finds something better or same.
    # Actually, the known solution is 10 terms. Greedy might find 10 or more.
    # We want to MINIMIZE k.
    
    # Let's try to find a solution with k < 10 using a smarter search.
    # The DP approach in find_solution_for_L is good but limited by L.
    
    # Let's try to improve the DP approach by trying smaller L values more aggressively
    # or using the greedy result as a heuristic for the DP search space.
    
    # Actually, let's just try the greedy algorithm with a higher max_terms to see if it
    # finds a valid solution quickly, and then use DP to see if we can do better.
    
    # Try greedy with max_terms=9 to see if a 9-term solution exists easily.
    sol_greedy_9 = greedy_egyptian(target, 5, 10000000, 9, deadline)
    if sol_greedy_9 is not None:
        best_sol = sol_greedy_9
        best_k = 9
        print(f"Greedy found 9-term solution: {sol_greedy_9}")
        
    sol_greedy_8 = greedy_egyptian(target, 5, 10000000, 8, deadline)
    if sol_greedy_8 is not None:
        best_sol = sol_greedy_8
        best_k = 8
        print(f"Greedy found 8-term solution: {sol_greedy_8}")
        
    # Now try DP for small L values to see if we can beat the greedy result.
    lcms = []
    current_lcm = 1
    for m in range(5, 25):
        current_lcm = math.lcm(current_lcm, m)
        lcms.append(current_lcm)
        if current_lcm > 100000:
            break
            
    candidates = sorted(set(lcms + [2520, 5040, 10080, 25200, 50400, 100800]))
    
    for L in candidates:
        if time.time() > deadline:
            break
        if L > 100000:
            continue
            
        sol, k = find_solution_for_L(L, deadline)
        if sol is not None and k < best_k:
            best_k = k
            best_sol = sol
            print(f"DP found {k}-term solution with L={L}: {sol}")
            
    return best_sol
