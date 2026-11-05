"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit

    # Generate smooth numbers (2^i * 3^j) as primary candidates
    smooth = set()
    for i in range(10):
        for j in range(7):
            v = (1 << i) * (3 ** j)
            if 1 <= v <= 512:
                smooth.add(v)

    all_candidates = list(range(1, 513))

    def try_greedy(order):
        selected = []
        selected_set = set()

        for x in order:
            if time.time() > deadline:
                break
            if x in selected_set:
                continue

            valid = True
            n = len(selected)

            # Check 1: Is x the mean of any subset of >=2 elements from selected?
            # Need: sum(S) = k * x for some S subset of selected, |S| = k >= 2
            if n >= 2:
                max_sum = sum(selected)
                # dp[k] is a bitset of sums achievable with exactly k elements
                dp = [0] * (n + 1)
                dp[0] = 1
                for elem in selected:
                    for k in range(n, 0, -1):
                        dp[k] |= (dp[k - 1] << elem)
                for k in range(2, n + 1):
                    target = k * x
                    if target > max_sum:
                        break
                    if (dp[k] >> target) & 1:
                        valid = False
                        break

            # Check 2: Does adding x make any a in selected the mean of a subset
            #         of >=2 elements from (selected \ {a}) ∪ {x}?
            if valid:
                for a in selected:
                    others = [y for y in selected if y != a]
                    m = len(others)
                    if m == 0:
                        continue

                    # We need: sum(S) + x = (j+1) * a for some S subset of others, |S| = j >= 1
                    # => sum(S) = (j+1)*a - x
                    max_others = sum(others)
                    dp2 = [0] * (m + 1)
                    dp2[0] = 1
                    for elem in others:
                        for k in range(m, 0, -1):
                            dp2[k] |= (dp2[k - 1] << elem)

                    for j in range(1, m + 1):
                        target = (j + 1) * a - x
                        if target < 0:
                            continue
                        if target > max_others:
                            break
                        if (dp2[j] >> target) & 1:
                            valid = False
                            break
                    if not valid:
                        break

            if valid:
                selected.append(x)
                selected_set.add(x)
                if len(selected) >= 48:
                    break

        return selected

    # Order 1: Smooth numbers first (sorted), then rest sorted
    smooth_list = sorted(smooth)
    rest = [x for x in all_candidates if x not in smooth]
    order1 = smooth_list + sorted(rest)

    best = try_greedy(order1)

    # If we haven't reached 48, try a different ordering
    if len(best) < 48 and time.time() < deadline - 10:
        # Try descending order (larger numbers first)
        order2 = sorted(all_candidates, reverse=True)
        result2 = try_greedy(order2)
        if len(result2) > len(best):
            best = result2

    # If still not 48, try smooth numbers in reverse order then rest
    if len(best) < 48 and time.time() < deadline - 10:
        order3 = sorted(smooth, reverse=True) + sorted(rest, reverse=True)
        result3 = try_greedy(order3)
        if len(result3) > len(best):
            best = result3

    # Final attempt: try interleaving smooth and non-smooth
    if len(best) < 48 and time.time() < deadline - 10:
        # Create an order that alternates between smooth and non-smooth
        order4 = []
        smooth_sorted = sorted(smooth)
        rest_sorted = sorted(rest)
        i, j = 0, 0
        while i < len(smooth_sorted) or j < len(rest_sorted):
            if i < len(smooth_sorted):
                order4.append(smooth_sorted[i])
                i += 1
            if j < len(rest_sorted):
                order4.append(rest_sorted[j])
                j += 1
        result4 = try_greedy(order4)
        if len(result4) > len(best):
            best = result4

    return best

# EVOLVE-BLOCK-END
