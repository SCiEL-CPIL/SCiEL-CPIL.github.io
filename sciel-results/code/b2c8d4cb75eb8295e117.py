import time
import bisect
from math import gcd
from functools import reduce

def construct():
    start_time = time.time()
    DEADLINE = 82.0

    denominators = list(range(2, 49))
    n = len(denominators)

    def lcm(a, b):
        return a * b // gcd(a, b)

    L = reduce(lcm, range(1, 49))
    weights = [L // d for d in denominators]

    mid = n // 2
    w1 = weights[:mid]
    w2 = weights[mid:]

    def gen_sums(weights):
        sums = [(0, 0)]
        for i, w in enumerate(weights):
            bit = 1 << i
            current_len = len(sums)
            for j in range(current_len):
                s, m = sums[j]
                sums.append((s + w, m | bit))
        return sums

    sums1 = gen_sums(w1)
    sums2 = gen_sums(w2)

    sums1 = [(s, m) for s, m in sums1 if s < L]
    sums2 = [(s, m) for s, m in sums2 if s < L]

    sums2.sort(key=lambda x: x[0])
    sums2_vals = [s for s, m in sums2]

    best_deficit = L
    best_mask1 = 0
    best_mask2 = 0

    for s1, m1 in sums1:
        target = L - 1 - s1
        if target < 0:
            continue
        idx = bisect.bisect_right(sums2_vals, target) - 1
        if idx >= 0:
            s2, m2 = sums2[idx]
            current_sum = s1 + s2
            current_deficit = L - current_sum
            if current_deficit < best_deficit:
                best_deficit = current_deficit
                best_mask1 = m1
                best_mask2 = m2
                if best_deficit == 1:
                    break

    indices = []
    for i in range(mid):
        if best_mask1 & (1 << i):
            indices.append(i)
    for i in range(len(w2)):
        if best_mask2 & (1 << i):
            indices.append(mid + i)

    if not indices:
        indices = [0]

    # Local refinement: try to improve by swapping elements
    if time.time() - start_time < DEADLINE - 5:
        current_sum_num = sum(L // denominators[i] for i in indices)
        current_deficit = L - current_sum_num

        improved = True
        while improved and time.time() - start_time < DEADLINE - 5:
            improved = False
            for i in indices[:]:
                for j in range(n):
                    if j not in indices:
                        new_indices = [x for x in indices if x != i] + [j]
                        new_sum_num = sum(L // denominators[k] for k in new_indices)
                        new_deficit = L - new_sum_num
                        if 0 < new_deficit < current_deficit:
                            indices = new_indices
                            current_deficit = new_deficit
                            improved = True
                            break
                if improved:
                    break

        if time.time() - start_time < DEADLINE - 5:
            for j in range(n):
                if j not in indices:
                    new_indices = indices + [j]
                    new_sum_num = sum(L // denominators[k] for k in new_indices)
                    new_deficit = L - new_sum_num
                    if 0 < new_deficit < current_deficit:
                        indices = new_indices
                        current_deficit = new_deficit

        if time.time() - start_time < DEADLINE - 5:
            for i in indices[:]:
                if len(indices) > 1:
                    new_indices = [x for x in indices if x != i]
                    if not new_indices:
                        continue
                    new_sum_num = sum(L // denominators[k] for k in new_indices)
                    new_deficit = L - new_sum_num
                    if 0 < new_deficit < current_deficit:
                        indices = new_indices
                        current_deficit = new_deficit

    return indices
