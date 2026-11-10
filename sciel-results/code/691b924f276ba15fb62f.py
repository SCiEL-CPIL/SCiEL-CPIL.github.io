import time
from math import gcd
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 85.0

    def elapsed():
        return time.time() - start_time

    # 1. Calculate LCM of 2..48
    L = 1
    for i in range(2, 49):
        L = L * i // gcd(L, i)

    # 2. Weights: index i corresponds to denominator i+2
    weights = [L // (i + 2) for i in range(47)]

    # 3. Meet-in-the-middle with mask tracking
    mid = 23
    weights_A = weights[:mid]
    weights_B = weights[mid:]

    def gen_sums_with_masks(weights_list):
        """Generate (sum, mask) pairs for all subsets with sum < L."""
        results = [(0, 0)]  # (sum, bitmask)
        for idx, w in enumerate(weights_list):
            bit = 1 << idx
            new_results = []
            for s, m in results:
                ns = s + w
                if ns < L:
                    new_results.append((ns, m | bit))
            results.extend(new_results)
        return results

    # Generate all valid (sum, mask) pairs for both halves
    pairs_A = gen_sums_with_masks(weights_A)
    pairs_B = gen_sums_with_masks(weights_B)

    # Sort by sum for two-pointer
    pairs_A.sort(key=lambda x: x[0])
    pairs_B.sort(key=lambda x: x[0])

    sums_A = [p[0] for p in pairs_A]
    sums_B = [p[0] for p in pairs_B]

    # 4. Two-pointer search for max sum < L
    best_sum = -1
    best_A = 0
    best_B = 0
    best_mask_A = 0
    best_mask_B = 0

    i = 0
    j = len(sums_B) - 1

    while i < len(sums_A) and j >= 0:
        current_sum = sums_A[i] + sums_B[j]
        if current_sum < L:
            if current_sum > best_sum:
                best_sum = current_sum
                best_A = sums_A[i]
                best_B = sums_B[j]
                best_mask_A = pairs_A[i][1]
                best_mask_B = pairs_B[j][1]
            i += 1
        else:
            j -= 1

    # 5. Construct initial indices from masks
    indices = []
    for k in range(len(weights_A)):
        if best_mask_A & (1 << k):
            indices.append(k)
    for k in range(len(weights_B)):
        if best_mask_B & (1 << k):
            indices.append(mid + k)

    if not indices:
        # Fallback: use greedy
        selected, total = [], 0
        for i_idx, w in enumerate(weights):
            if total + w < L:
                selected.append(i_idx)
                total += w
        indices = selected

    # 6. Local search refinement using exact integer arithmetic
    current_set = set(indices)
    current_sum = sum(weights[i] for i in current_set)

    def try_improve(current_set, current_sum):
        """Try 1-swap and 1-add moves to improve the sum (keep < L)."""
        improved = True
        while improved and elapsed() < DEADLINE:
            improved = False
            # Try removing one and adding another (swap)
            for i in list(current_set):
                base_sum = current_sum - weights[i]
                for j in range(47):
                    if j in current_set:
                        continue
                    new_sum = base_sum + weights[j]
                    if new_sum < L and new_sum > current_sum:
                        current_set.discard(i)
                        current_set.add(j)
                        current_sum = new_sum
                        improved = True
                        break
                if elapsed() > DEADLINE:
                    break
            if elapsed() > DEADLINE:
                break
            # Try adding one more element
            for j in range(47):
                if j in current_set:
                    continue
                new_sum = current_sum + weights[j]
                if new_sum < L and new_sum > current_sum:
                    current_set.add(j)
                    current_sum = new_sum
                    improved = True
                    break
        return current_set, current_sum

    current_set, current_sum = try_improve(current_set, current_sum)

    # 7. Try 2-swap moves (remove two, add two)
    if elapsed() < DEADLINE:
        improved = True
        while improved and elapsed() < DEADLINE:
            improved = False
            elems = list(current_set)
            for ii in range(len(elems)):
                for jj in range(ii + 1, len(elems)):
                    i, j = elems[ii], elems[jj]
                    base_sum = current_sum - weights[i] - weights[j]
                    for k in range(47):
                        if k in current_set:
                            continue
                        s1 = base_sum + weights[k]
                        if s1 >= L:
                            continue
                        for l in range(k + 1, 47):
                            if l in current_set or l == k:
                                continue
                            new_sum = s1 + weights[l]
                            if new_sum < L and new_sum > current_sum:
                                current_set.discard(i)
                                current_set.discard(j)
                                current_set.add(k)
                                current_set.add(l)
                                current_sum = new_sum
                                improved = True
                                break
                        if improved or elapsed() > DEADLINE:
                            break
                    if improved or elapsed() > DEADLINE:
                        break
                if improved or elapsed() > DEADLINE:
                    break
            if elapsed() > DEADLINE:
                break

    # 8. Final check: ensure nonempty and sum < L
    if not current_set:
        # Fallback to a known valid set
        current_set = {0, 1, 5, 41}  # 1/2+1/3+1/7+1/43
        current_sum = sum(weights[i] for i in current_set)

    if current_sum >= L:
        # Remove elements until sum < L
        for i in sorted(current_set, key=lambda x: -weights[x]):
            if current_sum - weights[i] < L:
                current_set.discard(i)
                current_sum -= weights[i]
                break
        if current_sum >= L:
            # Nuclear fallback
            current_set = {0}
            current_sum = weights[0]

    return sorted(current_set)
