import time
from math import gcd
from functools import reduce
from bisect import bisect_left

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill

    denominators = list(range(2, 49))
    n = len(denominators)

    def lcm(a, b):
        return a * b // gcd(a, b)

    L = reduce(lcm, denominators)
    weights = [L // d for d in denominators]

    # Split items into two halves
    mid = n // 2
    half1 = weights[:mid]
    half2 = weights[mid:]

    # Generate all feasible subset sums (sum < L) and masks
    def gen_feasible_sums(weights):
        sums = [0]
        masks = [0]
        for i, w in enumerate(weights):
            current_len = len(sums)
            for j in range(current_len):
                new_sum = sums[j] + w
                if new_sum < L:  # Only keep feasible subsets
                    sums.append(new_sum)
                    masks.append(masks[j] | (1 << i))
        return sums, masks

    # Generate for half1
    sums1, masks1 = gen_feasible_sums(half1)

    # Generate for half2 with deduplication
    sums2_raw, masks2_raw = gen_feasible_sums(half2)

    # Deduplicate half2: for each unique sum, keep one mask
    sum_to_mask = {}
    for s, m in zip(sums2_raw, masks2_raw):
        if s not in sum_to_mask:
            sum_to_mask[s] = m

    # Create sorted list of unique sums
    sorted_sums2 = sorted(sum_to_mask.keys())

    # We want to find max s1 + s2 < L
    best_deficit = L
    best_mask1 = 0
    best_mask2 = 0

    # Initial lower bound: try single largest weight
    max_single = max(weights)
    if max_single < L:
        best_deficit = L - max_single

    # Sort sums1 in descending order to find good candidates first
    # This helps with early termination
    indexed_sums1 = list(zip(sums1, masks1))
    indexed_sums1.sort(key=lambda x: x[0], reverse=True)

    for i in range(len(indexed_sums1)):
        if time.time() > deadline:
            break

        s1, m1 = indexed_sums1[i]
        if s1 >= L:
            continue

        target = L - s1
        # We want the largest s2 such that s2 < target
        idx = bisect_left(sorted_sums2, target)

        if idx > 0:
            s2 = sorted_sums2[idx - 1]
            total_sum = s1 + s2
            deficit = L - total_sum

            if deficit < best_deficit:
                best_deficit = deficit
                best_mask1 = m1
                best_mask2 = sum_to_mask[s2]

            # Early exit if deficit is 1
            if best_deficit == 1:
                break

    # Reconstruct indices
    final_indices = []
    for i in range(mid):
        if best_mask1 & (1 << i):
            final_indices.append(i)

    for i in range(len(half2)):
        if best_mask2 & (1 << i):
            final_indices.append(mid + i)

    # Fallback if no valid nonempty subset found
    if not final_indices:
        # Pick the largest single item
        max_idx = weights.index(max_single)
        final_indices = [max_idx]

    return final_indices
