"""
Few-term Egyptian fractions with denominators greater than four.

Returns 1..256 distinct integer denominators in [5, 10000000] whose reciprocal sum is exactly 1,
minimizing their count. Uses a depth-(k-1) DFS with exact last-denominator computation.
"""
import time
from fractions import Fraction
from math import gcd


def construct():
    """
    Find the minimum number of distinct denominators d_i in [5, 10^7] such that sum(1/d_i) = 1.
    Returns the list of denominators.
    """
    start_time = time.time()
    deadline = 80.0  # safety margin before 90s kill

    def _gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    def search_k(k):
        """
        Find k distinct denominators >= 5 summing reciprocals to 1.
        DFS for first k-1 denominators, then compute the last one exactly.
        """
        found = [None]

        def dfs(remaining_num, remaining_den, min_den, current, depth):
            """
            remaining = remaining_num / remaining_den (a positive fraction)
            min_den: minimum allowed denominator for next choice
            current: list of chosen denominators so far
            depth: number of denominators chosen so far
            """
            if time.time() > deadline:
                return

            # If we've chosen k-1 denominators, compute the last one
            if depth == k - 1:
                if remaining_num == 0:
                    return
                # Last denominator d must satisfy: 1/d = remaining_num/remaining_den
                # So d = remaining_den / remaining_num
                if remaining_den % remaining_num == 0:
                    d_last = remaining_den // remaining_num
                    if d_last > min_den - 1 and d_last <= 10000000:
                        # Verify it's distinct (it should be since d_last > min_den - 1 >= last chosen)
                        if d_last not in current:
                            found[0] = list(current) + [d_last]
                            return
                return

            remaining_slots = k - depth

            # Upper bound check: max sum of remaining_slots terms with denominators >= min_den
            # is remaining_slots / min_den. If remaining > that, prune.
            # remaining_num/remaining_den > remaining_slots/min_den
            if remaining_num * min_den > remaining_slots * remaining_den:
                return

            # Lower bound check: min sum of remaining_slots distinct denominators >= min_den
            # is sum_{i=0}^{remaining_slots-1} 1/(min_den + i)
            # If remaining < that, prune.
            if remaining_slots <= 25:
                min_sum_num = 0
                min_sum_den = 1
                for i in range(remaining_slots):
                    d = min_den + i
                    # Add 1/d to min_sum
                    num = min_sum_num * d + min_sum_den
                    den = min_sum_den * d
                    g = _gcd(abs(num), den)
                    if g > 1:
                        num //= g
                        den //= g
                    min_sum_num, min_sum_den = num, den
                # remaining_num/remaining_den < min_sum_num/min_sum_den
                if remaining_num * min_sum_den < min_sum_num * remaining_den:
                    return

            # Determine range for next denominator d
            # 1/d <= remaining => d >= ceil(remaining_den / remaining_num)
            # Also d >= min_den
            d_low = max(min_den, (remaining_den + remaining_num - 1) // remaining_num)

            # Upper bound on d: after choosing d, remaining' = remaining - 1/d
            # Need (remaining_slots - 1) terms with denominators >= d+1 summing to remaining'
            # Max sum = (remaining_slots-1)/(d+1)
            # So remaining - 1/d <= (remaining_slots-1)/(d+1)
            # => remaining <= 1/d + (remaining_slots-1)/(d+1) < remaining_slots/d
            # => d < remaining_slots / remaining = remaining_slots * remaining_den / remaining_num
            d_high = min(10000000, int(remaining_slots * remaining_den / remaining_num) + 1)
            d_high = max(d_high, d_low)

            # Cap for performance
            if k <= 8:
                d_high = min(d_high, 50000)
            elif k <= 12:
                d_high = min(d_high, 500000)

            for d in range(d_low, d_high + 1):
                if time.time() > deadline:
                    return

                # new_remaining = remaining - 1/d = (remaining_num*d - remaining_den) / (remaining_den*d)
                new_num = remaining_num * d - remaining_den
                if new_num < 0:
                    break
                new_den = remaining_den * d

                # Simplify
                g = _gcd(new_num, new_den)
                if g > 1:
                    new_num //= g
                    new_den //= g

                current.append(d)
                dfs(new_num, new_den, d + 1, current, depth + 1)
                current.pop()

                if found[0] is not None:
                    return

        dfs(1, 1, 5, [], 0)
        return found[0]

    # Try k from 5 upward
    for k in range(5, 16):
        if time.time() > deadline - 3:
            break
        result = search_k(k)
        if result is not None:
            # Verify
            s = sum(Fraction(1, d) for d in result)
            if s == 1 and all(5 <= d <= 10000000 for d in result) and len(set(result)) == len(result):
                return result

    # Fallback: 12-term solution
    fallback = [5, 6, 7, 10, 14, 21, 35, 42, 70, 105, 140, 210]
    s = sum(Fraction(1, d) for d in fallback)
    if s == 1:
        return fallback

    # Final fallback: DP
    target = 2520
    reached = {0: []}
    for d in range(5, target + 1):
        if target % d:
            continue
        w = target // d
        for total, values in list(reached.items()):
            if total + w <= target and total + w not in reached:
                reached[total + w] = values + [d]
        if target in reached:
            return reached[target]

    return fallback


# EVOLVE-BLOCK-END
