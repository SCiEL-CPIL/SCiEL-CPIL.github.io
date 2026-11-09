import time
from itertools import combinations

def construct():
    t0 = time.time()
    DEADLINE = 55.0  # safety margin under 60s

    n = 5040
    target = 4979

    # Generate all divisors of 5040
    divisors = []
    for i in range(1, int(n ** 0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)

    # Separate odd and even divisors
    odd_divs = [d for d in divisors if d % 2 == 1]
    even_divs = [d for d in divisors if d % 2 == 0]

    # Sort even divisors in descending order for DP
    even_divs.sort(reverse=True)
    m_even = len(even_divs)

    # DP over even divisors: dp[s] = min number of even divisors to sum to s
    INF = float('inf')
    dp_count = [INF] * (target + 1)
    dp_parent = [(-1, -1)] * (target + 1)  # (prev_sum, divisor_value)
    dp_count[0] = 0

    for d in even_divs:
        if d > target:
            continue
        for s in range(target, d - 1, -1):
            if dp_count[s - d] + 1 < dp_count[s]:
                dp_count[s] = dp_count[s - d] + 1
                dp_parent[s] = (s - d, d)

        if time.time() - t0 > DEADLINE:
            break

    # Helper to reconstruct even subset for a given sum
    def reconstruct_even(s):
        chosen = []
        while s > 0:
            prev_s, d = dp_parent[s]
            chosen.append(d)
            s = prev_s
        return chosen

    # Since target is odd, we need an odd number of odd divisors.
    # Try 1 odd divisor first, then 3.
    best_count = INF
    best_odd = None
    best_even_sum = None

    # Try single odd divisors
    for d in odd_divs:
        if d > target:
            continue
        rem = target - d
        if rem >= 0 and dp_count[rem] < INF:
            c = 1 + dp_count[rem]
            if c < best_count:
                best_count = c
                best_odd = d
                best_even_sum = rem

    if time.time() - t0 > DEADLINE:
        # Fallback: return best found so far
        if best_count < INF:
            chosen = [best_odd] + reconstruct_even(best_even_sum)
            assert len(set(chosen)) == len(chosen)
            assert sum(chosen) == target
            assert all(n % x == 0 for x in chosen)
            assert len(chosen) <= 256
            return chosen
        return []

    # Try triples of odd divisors
    for comb in combinations(odd_divs, 3):
        os = sum(comb)
        if os > target:
            continue
        rem = target - os
        if rem >= 0 and dp_count[rem] < INF:
            c = 3 + dp_count[rem]
            if c < best_count:
                best_count = c
                best_odd = comb
                best_even_sum = rem

        if time.time() - t0 > DEADLINE:
            break

    if best_count == INF:
        # Should not happen for this instance, but fallback
        rem = target
        chosen = []
        for d in sorted(divisors, reverse=True):
            if d <= rem:
                chosen.append(d)
                rem -= d
        if rem != 0:
            return []
        return chosen

    # Reconstruct
    if isinstance(best_odd, tuple):
        odd_part = list(best_odd)
    else:
        odd_part = [best_odd]

    even_part = reconstruct_even(best_even_sum)
    result = odd_part + even_part

    # Verify
    assert len(set(result)) == len(result), "divisors not distinct"
    assert sum(result) == target, f"sum {sum(result)} != {target}"
    assert all(n % d == 0 for d in result), "not all divisors of 5040"
    assert len(result) <= 256, "too many divisors"

    return result
