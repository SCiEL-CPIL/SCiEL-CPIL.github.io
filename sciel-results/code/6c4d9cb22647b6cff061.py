import time
from math import lcm
from fractions import Fraction
import bisect
import random

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # safety margin before 90s kill

    denominators = list(range(2, 49))
    n = len(denominators)

    # Calculate LCM of all denominators
    L = 1
    for d in denominators:
        L = lcm(L, d)

    # weights[i] = L / denominators[i]
    weights = [L // d for d in denominators]

    # Meet-in-the-middle: split into two halves
    mid = n // 2
    weights_1 = weights[:mid]
    weights_2 = weights[mid:]

    def gen_sums(weights_list, limit):
        """Generate all subset sums < limit as list of (sum, mask)."""
        sums = [(0, 0)]
        for i, w in enumerate(weights_list):
            bit = 1 << i
            new_sums = []
            for s, m in sums:
                ns = s + w
                if ns < limit:
                    new_sums.append((ns, m | bit))
            sums.extend(new_sums)
        return sums

    # Generate all subset sums for each half that are < L
    sums_1 = gen_sums(weights_1, L)
    sums_2 = gen_sums(weights_2, L)

    # Sort sums_2 by sum value for binary search
    sums_2.sort(key=lambda x: x[0])
    sums_2_values = [s[0] for s in sums_2]

    best_sum = 0
    best_mask = 0

    # For each sum in half 1, find best matching sum in half 2
    for s1, m1 in sums_1:
        if time.time() > deadline:
            break
        # Need s1 + s2 < L => s2 <= L - s1 - 1
        target = L - s1 - 1
        if target < 0:
            continue
        # Find rightmost index in sums_2_values <= target
        idx = bisect.bisect_right(sums_2_values, target)
        if idx > 0:
            s2, m2 = sums_2[idx - 1]
            total_sum = s1 + s2
            if total_sum > best_sum:
                best_sum = total_sum
                best_mask = m1 | (m2 << mid)

    # Convert best_mask to indices
    if best_mask == 0:
        return [0]

    result_indices = []
    for i in range(n):
        if best_mask & (1 << i):
            result_indices.append(i)

    # Local improvement: try to add or remove single elements to improve
    current_sum = best_sum
    current_indices = set(result_indices)

    improved = True
    while improved and time.time() < deadline - 2.0:
        improved = False

        # Try adding each element not in current set
        for i in range(n):
            if i not in current_indices:
                new_sum = current_sum + weights[i]
                if new_sum < L:
                    current_sum = new_sum
                    current_indices.add(i)
                    improved = True
                    break

        # Try removing each element in current set (to free up room for better additions)
        if not improved:
            for i in list(current_indices):
                new_sum = current_sum - weights[i]
                if new_sum < current_sum:  # always true
                    # Check if we can do better by removing this
                    # Actually we want to keep sum as close to L-1 as possible
                    # Try removing and see if we can add something better
                    removed = i
                    temp_sum = current_sum - weights[removed]
                    temp_indices = current_indices - {removed}

                    # Try adding elements to the temp set
                    best_addition = None
                    best_add_sum = temp_sum
                    for j in range(n):
                        if j not in temp_indices:
                            add_sum = temp_sum + weights[j]
                            if add_sum < L and add_sum > best_add_sum:
                                best_add_sum = add_sum
                                best_addition = j

                    if best_addition is not None and best_add_sum > current_sum:
                        current_sum = best_add_sum
                        temp_indices.add(best_addition)
                        current_indices = temp_indices
                        improved = True
                        break

    result_indices = sorted(list(current_indices))
    return result_indices
