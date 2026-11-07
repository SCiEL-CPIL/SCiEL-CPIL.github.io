import time
from math import gcd

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    
    # LCM of all divisors: the pattern of "which multiples fall in a window" repeats
    # every LCM, so the optimal span is found within one period.
    LCM = 1
    for d in divisors:
        LCM = _lcm(LCM, d)
    # LCM = 2*3*5*7*11*13 = 30030

    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill

    best_span = float('inf')
    best_reps = [0] * n

    # Sort divisors descending so the largest (rarest multiples) are placed first,
    # which prunes the search tree much earlier.
    indexed_divs = sorted(enumerate(divisors), key=lambda x: x[1], reverse=True)
    orig_indices = [d[0] for d in indexed_divs]
    sorted_divs = [d[1] for d in indexed_divs]

    def dfs(idx, chosen_vals, current_max, L):
        nonlocal best_span, best_reps

        if time.time() > deadline:
            return

        if idx == n:
            span = current_max - L
            if span < best_span:
                best_span = span
                temp_reps = [0] * n
                for i, (oi, val) in enumerate(zip(orig_indices, chosen_vals)):
                    temp_reps[oi] = val
                best_reps = temp_reps[:]
            return

        d = sorted_divs[idx]

        # Start from the smallest multiple >= L
        k = (L + d - 1) // d
        val = k * d

        while True:
            # Pruning: if this value already exceeds the best span, no need to continue
            if val - L >= best_span:
                break

            if val not in chosen_vals:
                chosen_vals.append(val)
                new_max = current_max if current_max > val else val
                dfs(idx + 1, chosen_vals, new_max, L)
                chosen_vals.pop()

            val += d

    # Iterate over the anchor L within one LCM period.
    # The optimal window's minimum is a multiple of some divisor, but the window
    # itself can start anywhere. Since the relative pattern repeats every LCM,
    # searching L in [1, LCM] covers all distinct "phases".
    for L in range(1, LCM + 1):
        if time.time() > deadline:
            break
        dfs(0, [], 0, L)
        # 10 distinct integers => minimum possible span is 9
        if best_span <= 9:
            break

    if best_span == float('inf'):
        # Fallback: shouldn't happen, but return something valid
        return divisors[:]

    return [int(x) for x in best_reps]
