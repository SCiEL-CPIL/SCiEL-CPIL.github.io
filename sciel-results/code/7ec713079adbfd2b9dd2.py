"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from fractions import Fraction

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s kill

    # We want the minimum number k of distinct denominators d_i in [5, 10^7]
    # such that sum(1/d_i) = 1.
    #
    # Strategy: For small k (5, 6, 7, ...), do a bounded DFS with strong pruning.
    # Use integer arithmetic via common denominators to avoid Fraction overhead
    # where possible, but Fraction is fine for the small depths we need.
    #
    # Key insight: for k terms summing to 1 with denominators >= 5,
    # the first denominator d1 must satisfy 1/d1 <= 1, i.e. d1 >= 1,
    # and we need (k-1) more terms summing to 1 - 1/d1 = (d1-1)/d1.
    #
    # We'll search k from 5 upward, trying to find ANY valid solution.
    # The first k for which we find a solution is the minimum (given our search
    # is complete for that k within the search space).

    def search_k(k, time_limit):
        """Find k distinct denominators >= 5 summing reciprocals to 1."""
        found = [None]

        def dfs(remaining, min_den, current, depth):
            if time.time() > time_limit:
                return
            if depth == k:
                if remaining == 0:
                    found[0] = list(current)
                return
            if remaining <= 0:
                return

            # We need (k - depth) more terms, each with denominator >= min_den
            # The maximum sum of (k-depth) terms with denominators >= min_den
            # is (k-depth)/min_den. If remaining > that, prune.
            remaining_slots = k - depth
            if remaining > Fraction(remaining_slots, min_den):
                return

            # The minimum sum of (k-depth) terms with denominators >= min_den
            # (using the smallest possible distinct denominators min_den, min_den+1, ...)
            # is a lower bound. If remaining < that minimum, prune.
            # Actually, the minimum sum with (k-depth) distinct denominators >= min_den
            # is sum_{i=0}^{remaining_slots-1} 1/(min_den + i)
            # This is a valid lower bound (smallest possible sum).
            min_sum = Fraction(0, 1)
            for i in range(remaining_slots):
                min_sum += Fraction(1, min_den + i)
            if remaining < min_sum:
                return

            # d must satisfy: 1/d <= remaining => d >= ceil(1/remaining)
            # and d >= min_den
            num, den = remaining.numerator, remaining.denominator
            d_low = max(min_den, (den + num - 1) // num)

            # Upper bound: d can't be so large that even with the best remaining
            # (k-depth-1) terms we can't reach the remainder.
            # After choosing d, remaining' = remaining - 1/d
            # We need (k-depth-1) terms with denominators >= d+1 summing to remaining'.
            # Max sum of (k-depth-1) terms with denominators >= d+1 is (k-depth-1)/(d+1)
            # So we need remaining - 1/d <= (k-depth-1)/(d+1)
            # => remaining <= 1/d + (k-depth-1)/(d+1)
            # This gives an upper bound on d.
            # Also, d can't exceed 10^7.
            d_high = 10000000

            # Tighter upper bound: 1/d must be at least remaining - (k-depth-1)/(d+1)
            # Let's just iterate and let the recursion prune.
            # But cap d_high to something reasonable for performance.
            # For small k, d values won't be huge.
            # A safe cap: d <= den * 2 for the first level, but let's use a
            # heuristic: d_high = max(d_low, min(10000000, den * k + 100))
            d_high = min(10000000, max(d_low, den * k + 200))

            for d in range(d_low, d_high + 1):
                if time.time() > time_limit:
                    return
                new_remaining = remaining - Fraction(1, d)
                if new_remaining < 0:
                    break
                if new_remaining == 0:
                    if depth + 1 == k:
                        current.append(d)
                        found[0] = list(current)
                        current.pop()
                        return
                    else:
                        # Need more terms but sum is already 0 - impossible with positive terms
                        continue
                current.append(d)
                dfs(new_remaining, d + 1, current, depth + 1)
                current.pop()
                if found[0] is not None:
                    return

        dfs(Fraction(1, 1), 5, [], 0)
        return found[0]

    # Try k from 5 upward
    for k in range(5, 257):
        if time.time() > deadline - 5:
            break
        result = search_k(k, deadline - 5)
        if result is not None:
            # Verify
            s = sum(Fraction(1, d) for d in result)
            assert s == 1, f"Verification failed: sum = {s}"
            assert all(5 <= d <= 10000000 for d in result)
            assert len(set(result)) == len(result)
            return result

    # Fallback: use a known valid solution (hardcoded)
    # 1 = 1/5 + 1/6 + 1/7 + 1/10 + 1/14 + 1/21 + 1/35 + 1/42 + 1/70 + 1/105 + 1/140 + 1/210
    # Let's just use a DP-based fallback with a small target
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

    raise RuntimeError("no representation found")

# EVOLVE-BLOCK-END
