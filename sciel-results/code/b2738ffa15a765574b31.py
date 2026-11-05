"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50  # safety margin before 90s kill

    n_val = 5040
    target = 4979

    # Get all divisors of 5040 (60 divisors)
    divisors = [d for d in range(1, n_val + 1) if n_val % d == 0]
    divisor_set = set(divisors)
    divisors.sort()
    n_divs = len(divisors)

    # Precompute prefix sums for O(1) range sum queries
    prefix = [0] * (n_divs + 1)
    for i in range(n_divs):
        prefix[i + 1] = prefix[i] + divisors[i]
    total_sum = prefix[n_divs]

    # k=1: check if target itself is a divisor
    if target in divisor_set:
        return [target]

    # k=2: O(n) using set
    for d in divisors:
        complement = target - d
        if complement in divisor_set and complement != d and complement > 0:
            return [d, complement]

    # k=3: O(n^2) using set
    for i in range(n_divs):
        a = divisors[i]
        if a >= target:
            break
        for j in range(i + 1, n_divs):
            b = divisors[j]
            if a + b >= target:
                break
            complement = target - a - b
            if complement in divisor_set and complement > b:
                return [a, b, complement]

    # For k >= 4, use meet-in-the-middle for exact subset-sum
    # Split divisors into two halves, enumerate all subset sums per half
    # Then for a given k, combine sums from both halves to hit target

    # Split point
    mid = n_divs // 2
    left_divs = divisors[:mid]
    right_divs = divisors[mid:]
    n_left = len(left_divs)
    n_right = len(right_divs)

    # For each half, enumerate all subsets with their (sum, count, bitmask)
    # Store by count -> dict of sum -> bitmask
    # left_subsets[c] = {sum: bitmask}
    left_subsets = [dict() for _ in range(n_left + 1)]
    right_subsets = [dict() for _ in range(n_right + 1)]

    # Enumerate all subsets of left half
    for mask in range(1 << n_left):
        s = 0
        c = 0
        for i in range(n_left):
            if mask & (1 << i):
                s += left_divs[i]
                c += 1
        # Keep only the first mask for each (c, s) - any one works
        if s not in left_subsets[c]:
            left_subsets[c][s] = mask

    # Enumerate all subsets of right half
    for mask in range(1 << n_right):
        s = 0
        c = 0
        for i in range(n_right):
            if mask & (1 << i):
                s += right_divs[i]
                c += 1
        if s not in right_subsets[c]:
            right_subsets[c][s] = mask

    # Now for each k from 4 upward, try all splits c_left + c_right = k
    for k in range(4, 257):
        if time.time() > deadline:
            break

        # Check feasibility
        if k > n_divs:
            break
        min_sum_k = prefix[k]
        max_sum_k = total_sum - prefix[n_divs - k]
        if target < min_sum_k or target > max_sum_k:
            continue

        found = False
        result = None
        # Try all splits of k between left and right
        for c_left in range(0, k + 1):
            c_right = k - c_left
            if c_left > n_left or c_right > n_right:
                continue
            left_dict = left_subsets[c_left]
            right_dict = right_subsets[c_right]
            if not left_dict or not right_dict:
                continue
            # For each sum in smaller dict, check complement in larger
            if len(left_dict) < len(right_dict):
                for s_l, mask_l in left_dict.items():
                    s_r = target - s_l
                    if s_r in right_dict:
                        mask_r = right_dict[s_r]
                        # Reconstruct
                        chosen = []
                        for i in range(n_left):
                            if mask_l & (1 << i):
                                chosen.append(left_divs[i])
                        for i in range(n_right):
                            if mask_r & (1 << i):
                                chosen.append(right_divs[i])
                        if sum(chosen) == target and len(chosen) == k:
                            found = True
                            result = chosen
                            break
                if found:
                    break
            else:
                for s_r, mask_r in right_dict.items():
                    s_l = target - s_r
                    if s_l in left_dict:
                        mask_l = left_dict[s_l]
                        chosen = []
                        for i in range(n_left):
                            if mask_l & (1 << i):
                                chosen.append(left_divs[i])
                        for i in range(n_right):
                            if mask_r & (1 << i):
                                chosen.append(right_divs[i])
                        if sum(chosen) == target and len(chosen) == k:
                            found = True
                            result = chosen
                            break
                if found:
                    break

        if found:
            return result

    # Fallback: greedy (should not reach here if solution exists and time is sufficient)
    remainder = target
    chosen = []
    for d in sorted(divisors, reverse=True):
        if d <= remainder:
            chosen.append(d)
            remainder -= d
            if remainder == 0:
                return chosen

    return chosen
# EVOLVE-BLOCK-END
