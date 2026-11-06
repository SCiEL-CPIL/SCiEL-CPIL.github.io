"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 80.0  # 10s safety margin before 90s kill
    
    n = 5040
    target = 4979
    
    # Get all divisors of 5040 efficiently (O(sqrt(n)))
    divisors = []
    i = 1
    while i * i <= n:
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
        i += 1
    divisors.sort(reverse=True)  # largest first
    
    # Filter: only divisors <= target can be used
    divisors = [d for d in divisors if d <= target]
    
    # Quick check: is target itself a divisor? (k=1 solution)
    if target in divisors:
        return [target]
    
    # Try k=2: check all pairs
    div_set = set(divisors)
    for i in range(len(divisors)):
        a = divisors[i]
        b = target - a
        if b > a and b in div_set:
            return [a, b]
    
    # Try k=3: check all triples
    for i in range(len(divisors)):
        a = divisors[i]
        for j in range(i+1, len(divisors)):
            b = divisors[j]
            c = target - a - b
            if c > b and c in div_set:
                return [a, b, c]
        if time.time() > DEADLINE:
            break
    
    # Try k=4: use pair sums
    # Build dict of pair sum -> (i,j) for i<j
    pair_sum = {}
    m = len(divisors)
    for i in range(m):
        a = divisors[i]
        for j in range(i+1, m):
            b = divisors[j]
            s = a + b
            if s > target:
                break  # divisors sorted descending, further j smaller... wait
                # Actually divisors are sorted descending, so a+b decreases as j increases
                # This break condition is wrong for descending sort. Let me fix.
                # Actually if a+b > target, we need smaller b, so continue (j increases means b decreases)
                # So we should NOT break, just skip. But if a is large and even the smallest b makes a+b > target,
                # then no pair with this a works. Let me re-think.
                # divisors sorted descending: divisors[0] >= divisors[1] >= ...
                # For fixed i, a = divisors[i]. As j increases, b = divisors[j] decreases.
                # So a+b decreases as j increases.
                # If a+b > target at some j, then for larger j, a+b might be <= target.
                # So we should NOT break on s > target. We should just skip.
                # But if a + divisors[m-1] > target, then no pair with this a works.
                # Let me just not break and let the condition handle it.
                pass
            if s not in pair_sum:
                pair_sum[s] = (i, j)
        if time.time() > DEADLINE:
            break
    
    # Now find two disjoint pairs with sum to target
    for i in range(m):
        a = divisors[i]
        for j in range(i+1, m):
            b = divisors[j]
            s1 = a + b
            need = target - s1
            if need in pair_sum:
                k, l = pair_sum[need]
                # Ensure disjoint and consistent ordering
                if i != k and i != l and j != k and j != l:
                    # Also need to ensure k,l are not i,j
                    # The check above handles that
                    return [divisors[i], divisors[j], divisors[k], divisors[l]]
        if time.time() > DEADLINE:
            break
    
    # DP: dp[s] = minimum number of distinct divisors to achieve sum s
    INF = float('inf')
    dp = [INF] * (target + 1)
    dp[0] = 0
    # parent tracking: for each sum, store which divisor was last added
    parent = [None] * (target + 1)  # parent[s] = (prev_sum, divisor)
    
    for d in divisors:
        if time.time() > DEADLINE:
            break
        # 0/1 knapsack: iterate backwards
        for s in range(target, d - 1, -1):
            if dp[s - d] + 1 < dp[s]:
                dp[s] = dp[s - d] + 1
                parent[s] = (s - d, d)
    
    if dp[target] != INF:
        # Reconstruct the solution
        chosen = []
        s = target
        while s > 0:
            prev_s, d = parent[s]
            chosen.append(d)
            s = prev_s
        return chosen
    
    # Fallback to greedy (should not happen)
    remainder = target
    chosen = []
    for d in divisors:
        if d <= remainder:
            chosen.append(d)
            remainder -= d
            if remainder == 0:
                break
    return chosen
# EVOLVE-BLOCK-END
