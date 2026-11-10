import time
from math import gcd
from fractions import Fraction
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 87.0  # Safety margin for 90s limit

    N = 256
    values = list(range(2, N + 1))
    size = len(values)  # 255

    # Precompute pairwise conflict: i and j are incompatible if lcm(values[i], values[j]) <= 256
    # Use bitmasks for fast compatibility checks
    conflict_mask = [0] * size
    for i in range(size):
        vi = values[i]
        for j in range(i + 1, size):
            vj = values[j]
            g = gcd(vi, vj)
            if vi * vj // g <= 256:
                conflict_mask[i] |= (1 << j)
                conflict_mask[j] |= (1 << i)

    # Precompute reciprocal weights as Fractions for exact scoring
    weights = [Fraction(1, values[i]) for i in range(size)]
    # Float weights for fast comparisons during search
    weight_list = [1.0 / values[i] for i in range(size)]

    def score_mask(mask):
        total = Fraction(0, 1)
        i = 0
        m = mask
        while m:
            if m & 1:
                total += weights[i]
            i += 1
            m >>= 1
        return total

    def is_valid_mask(mask):
        i = 0
        m = mask
        while m:
            if m & 1:
                if conflict_mask[i] & mask:
                    return False
            i += 1
            m >>= 1
        return True

    def greedy_construct(order):
        S_mask = 0
        for i in order:
            if not (conflict_mask[i] & S_mask):
                S_mask |= (1 << i)
        return S_mask

    # Conflict counts for diversity in initial solutions
    conflict_counts = [bin(cm).count('1') for cm in conflict_mask]

    # Pre-sorted orders for diverse initial solutions
    order_asc = list(range(size))
    order_desc = list(range(size - 1, -1, -1))
    order_by_conflict = sorted(range(size), key=lambda x: (conflict_counts[x], x))
    order_by_ratio = sorted(range(size), key=lambda x: (weight_list[x] / (conflict_counts[x] + 1)), reverse=True)
    order_by_value_desc = sorted(range(size), key=lambda x: values[x], reverse=True)
    order_by_ratio2 = sorted(range(size), key=lambda x: (weight_list[x] / ((conflict_counts[x] + 1) ** 2)), reverse=True)

    # Weight-sorted indices for fast best-v lookup
    weight_sorted = sorted(range(size), key=lambda x: weight_list[x], reverse=True)

    def local_search(S_mask, dl):
        current_score = score_mask(S_mask)
        in_set = [False] * size
        for i in range(size):
            if S_mask & (1 << i):
                in_set[i] = True

        improved = True
        while improved:
            improved = False
            elapsed = time.time() - start_time
            if elapsed > dl:
                break

            # Try swaps: remove u, add v (only best gain per u)
            for u in range(size):
                if not in_set[u]:
                    continue
                S_without_u = S_mask & ~(1 << u)
                best_v = -1
                best_gain = Fraction(0, 1)
                # Iterate in weight-descending order; first compatible v with positive gain is best
                for v in weight_sorted:
                    if in_set[v]:
                        continue
                    if weight_list[v] <= weight_list[u]:
                        break  # no further v can have positive gain
                    if conflict_mask[v] & S_without_u:
                        continue
                    best_v = v
                    best_gain = weights[v] - weights[u]
                    break  # first compatible in weight order is optimal
                if best_v >= 0 and best_gain > 0:
                    S_mask = S_without_u | (1 << best_v)
                    in_set[u] = False
                    in_set[best_v] = True
                    current_score += best_gain
                    improved = True
                    if time.time() - start_time > dl:
                        return S_mask, current_score

            # Try additions
            for v in weight_sorted:
                if in_set[v]:
                    continue
                if not (conflict_mask[v] & S_mask):
                    S_mask |= (1 << v)
                    in_set[v] = True
                    current_score += weights[v]
                    improved = True
                    if time.time() - start_time > dl:
                        return S_mask, current_score

            # Try 1-for-2: remove u, add v1, v2 (if net gain positive)
            if time.time() - start_time < dl - 3:
                in_set_indices = [i for i in range(size) if in_set[i]]
                for u in in_set_indices:
                    if time.time() - start_time > dl:
                        break
                    S_without_u = S_mask & ~(1 << u)
                    base_loss_float = weight_list[u]

                    # Find best pair (v1, v2) not in S, compatible with S_without_u and each other
                    best_pair = (-1, -1)
                    best_pair_gain_float = 0.0
                    best_pair_gain = Fraction(0, 1)

                    # Build candidate list
                    candidates = []
                    for v in weight_sorted:
                        if in_set[v] or v == u:
                            continue
                        if conflict_mask[v] & S_without_u:
                            continue
                        candidates.append(v)
                        if weight_list[v] * 2 <= base_loss_float:
                            break

                    n_cand = len(candidates)
                    for i1 in range(n_cand):
                        v1 = candidates[i1]
                        if weight_list[v1] * 2 <= base_loss_float:
                            break
                        for i2 in range(i1 + 1, n_cand):
                            v2 = candidates[i2]
                            if weight_list[v1] + weight_list[v2] <= base_loss_float:
                                break
                            if conflict_mask[v1] & (1 << v2):
                                continue
                            gain_float = weight_list[v1] + weight_list[v2] - base_loss_float
                            if gain_float > best_pair_gain_float:
                                best_pair_gain = weights[v1] + weights[v2] - weights[u]
                                best_pair_gain_float = gain_float
                                best_pair = (v1, v2)

                    if best_pair_gain > 0:
                        v1, v2 = best_pair
                        S_mask = S_without_u | (1 << v1) | (1 << v2)
                        in_set[u] = False
                        in_set[v1] = True
                        in_set[v2] = True
                        current_score += best_pair_gain
                        improved = True
                        if time.time() - start_time > dl:
                            return S_mask, current_score

            # Try 2-for-1: remove u1, u2; add v (if net gain positive)
            if time.time() - start_time < dl - 2:
                in_set_indices = [i for i in range(size) if in_set[i]]
                n_in = len(in_set_indices)
                if n_in >= 2:
                    for idx1 in range(n_in):
                        if time.time() - start_time > dl:
                            break
                        u1 = in_set_indices[idx1]
                        for idx2 in range(idx1 + 1, n_in):
                            u2 = in_set_indices[idx2]
                            S_without = S_mask & ~(1 << u1) & ~(1 << u2)
                            base_loss_float = weight_list[u1] + weight_list[u2]

                            best_v = -1
                            best_gain_float = 0.0
                            best_gain = Fraction(0, 1)

                            for v in weight_sorted:
                                if in_set[v]:
                                    continue
                                if weight_list[v] <= base_loss_float:
                                    break
                                if conflict_mask[v] & S_without:
                                    continue
                                best_v = v
                                best_gain = weights[v] - weights[u1] - weights[u2]
                                best_gain_float = weight_list[v] - base_loss_float
                                break

                            if best_v >= 0 and best_gain > 0:
                                S_mask = S_without | (1 << best_v)
                                in_set[u1] = False
                                in_set[u2] = False
                                in_set[best_v] = True
                                current_score += best_gain
                                improved = True
                                if time.time() - start_time > dl:
                                    return S_mask, current_score

            # Try 2-for-2: remove u1, u2; add v1, v2 (if net gain positive)
            if time.time() - start_time < dl - 5:
                in_set_indices = [i for i in range(size) if in_set[i]]
                n_in = len(in_set_indices)
                if n_in >= 2:
                    for idx1 in range(n_in):
                        if time.time() - start_time > dl:
                            break
                        u1 = in_set_indices[idx1]
                        for idx2 in range(idx1 + 1, n_in):
                            if time.time() - start_time > dl:
                                break
                            u2 = in_set_indices[idx2]
                            S_without = S_mask & ~(1 << u1) & ~(1 << u2)
                            base_loss_float = weight_list[u1] + weight_list[u2]

                            best_pair = (-1, -1)
                            best_pair_gain_float = 0.0
                            best_pair_gain = Fraction(0, 1)

                            candidates = []
                            for v in weight_sorted:
                                if in_set[v]:
                                    continue
                                if conflict_mask[v] & S_without:
                                    continue
                                candidates.append(v)
                                if weight_list[v] * 2 <= base_loss_float:
                                    break

                            n_cand = len(candidates)
                            for i1 in range(n_cand):
                                v1 = candidates[i1]
                                if weight_list[v1] * 2 <= base_loss_float:
                                    break
                                for i2 in range(i1 + 1, n_cand):
                                    v2 = candidates[i2]
                                    if weight_list[v1] + weight_list[v2] <= base_loss_float:
                                        break
                                    if conflict_mask[v1] & (1 << v2):
                                        continue
                                    gain_float = weight_list[v1] + weight_list[v2] - base_loss_float
                                    if gain_float > best_pair_gain_float:
                                        best_pair_gain = weights[v1] + weights[v2] - weights[u1] - weights[u2]
                                        best_pair_gain_float = gain_float
                                        best_pair = (v1, v2)

                            if best_pair_gain > 0:
                                v1, v2 = best_pair
                                S_mask = S_without | (1 << v1) | (1 << v2)
                                in_set[u1] = False
                                in_set[u2] = False
                                in_set[v1] = True
                                in_set[v2] = True
                                current_score += best_pair_gain
                                improved = True
                                if time.time() - start_time > dl:
                                    return S_mask, current_score

        return S_mask, current_score

    def exhaustive_polish(S_mask, dl):
        """Final polish: ensure no single add or swap improvement is missed."""
        current_score = score_mask(S_mask)
        in_set = [False] * size
        for i in range(size):
            if S_mask & (1 << i):
                in_set[i] = True

        improved = True
        while improved:
            improved = False
            if time.time() - start_time > dl:
                break

            # Try additions
            for v in weight_sorted:
                if in_set[v]:
                    continue
                if not (conflict_mask[v] & S_mask):
                    S_mask |= (1 << v)
                    in_set[v] = True
                    current_score += weights[v]
                    improved = True
                    if time.time() - start_time > dl:
                        break

            # Try swaps
            for u in range(size):
                if not in_set[u]:
                    continue
                S_without_u = S_mask & ~(1 << u)
                for v in weight_sorted:
                    if in_set[v]:
                        continue
                    if weight_list[v] <= weight_list[u]:
                        break
                    if conflict_mask[v] & S_without_u:
                        continue
                    gain = weights[v] - weights[u]
                    if gain > 0:
                        S_mask = S_without_u | (1 << v)
                        in_set[u] = False
                        in_set[v] = True
                        current_score += gain
                        improved = True
                        break

        return S_mask, current_score

    # Generate diverse initial solutions
    solutions = []

    solutions.append(greedy_construct(order_asc))
    solutions.append(greedy_construct(order_desc))
    solutions.append(greedy_construct(order_by_conflict))
    solutions.append(greedy_construct(order_by_ratio))
    solutions.append(greedy_construct(order_by_value_desc))
    solutions.append(greedy_construct(order_by_ratio2))

    # Random restarts with bias towards high weight
    for _ in range(30):
        if time.time() - start_time > deadline - 10:
            break
        order = list(range(size))
        random.shuffle(order)
        solutions.append(greedy_construct(order))

    # Local search on each
    best_mask = max(solutions, key=lambda m: score_mask(m))
    best_val = score_mask(best_mask)

    for S in solutions:
        if time.time() - start_time > deadline - 5:
            break
        S_opt, val = local_search(S, deadline - 3)
        if val > best_val:
            best_val = val
            best_mask = S_opt

    # Additional local search iterations on best solution
    if time.time() - start_time < deadline - 5:
        best_mask, best_val = local_search(best_mask, deadline - 2)

    # Final exhaustive polish pass
    if time.time() - start_time < deadline - 1:
        best_mask, best_val = exhaustive_polish(best_mask, deadline - 0.5)

    # Convert back to values
    result = []
    i = 0
    m = best_mask
    while m:
        if m & 1:
            result.append(values[i])
        i += 1
        m >>= 1

    return result
# EVOLVE-BLOCK-END
