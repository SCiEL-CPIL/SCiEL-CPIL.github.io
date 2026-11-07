import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s kill

    n = 256

    # Sieve of Eratosthenes
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.sqrt(n)) + 1):
        if is_prime[i]:
            for j in range(i * i, n + 1, i):
                is_prime[j] = False

    primes = [i for i in range(2, n + 1) if is_prime[i]]
    large_primes = [p for p in primes if p > n // 2]

    fixed = [1] + large_primes
    fixed_set = set(fixed)

    # Candidates are all integers 2..256 not in fixed
    candidates = [i for i in range(2, n + 1) if i not in fixed_set]
    num_cands = len(candidates)

    # Map value to index
    val_to_idx = {v: i for i, v in enumerate(candidates)}

    # Precompute conflict bitmasks using prime factors
    # For each candidate, compute set of prime factors
    prime_factors = []
    for v in candidates:
        pf = set()
        temp = v
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                pf.add(p)
                while temp % p == 0:
                    temp //= p
        if temp > 1:
            pf.add(temp)
        prime_factors.append(pf)

    # Conflict mask: bit j is 1 if candidates[i] and candidates[j] share a prime factor
    conflict_mask = [0] * num_cands
    for i in range(num_cands):
        for j in range(i + 1, num_cands):
            if prime_factors[i] & prime_factors[j]:  # non-empty intersection
                conflict_mask[i] |= (1 << j)
                conflict_mask[j] |= (1 << i)

    def get_sum(mask):
        s = 0
        for i in range(num_cands):
            if (mask >> i) & 1:
                s += candidates[i]
        return s

    def greedy_initial(order):
        mask = 0
        for v in order:
            idx = val_to_idx[v]
            if (conflict_mask[idx] & mask) == 0:
                mask |= (1 << idx)
        return mask

    def local_search(initial_mask, max_iters=500):
        current_mask = initial_mask
        current_sum = get_sum(current_mask)
        improved = True
        iters = 0

        # Precompute sorted indices by value descending
        indices_sorted_desc = sorted(range(num_cands), key=lambda i: candidates[i], reverse=True)

        while improved and iters < max_iters and (time.time() - start_time) < deadline:
            improved = False
            iters += 1

            # --- 1-for-1 Swap ---
            best_gain = 0
            best_move_11 = None
            for j in range(num_cands):
                if not ((current_mask >> j) & 1):
                    continue
                val_j = candidates[j]
                mask_without_j = current_mask & ~(1 << j)
                for i in range(num_cands):
                    if ((current_mask >> i) & 1):
                        continue
                    val_i = candidates[i]
                    if val_i <= val_j:
                        continue
                    if (conflict_mask[i] & mask_without_j) == 0:
                        gain = val_i - val_j
                        if gain > best_gain:
                            best_gain = gain
                            best_move_11 = (i, j)

            if best_move_11:
                i, j = best_move_11
                current_mask &= ~(1 << j)
                current_mask |= (1 << i)
                current_sum += best_gain
                improved = True
                continue

            # --- 1-for-2 Swap ---
            best_gain_12 = 0
            best_move_12 = None
            K = 25
            unselected_indices = [i for i in indices_sorted_desc if not ((current_mask >> i) & 1)]
            top_unselected = unselected_indices[:K]

            for j in range(num_cands):
                if not ((current_mask >> j) & 1):
                    continue
                val_j = candidates[j]
                mask_without_j = current_mask & ~(1 << j)
                comp_unselected = []
                for i in top_unselected:
                    if (conflict_mask[i] & mask_without_j) == 0:
                        comp_unselected.append(i)
                for a in range(len(comp_unselected)):
                    i1 = comp_unselected[a]
                    for b in range(a + 1, len(comp_unselected)):
                        i2 = comp_unselected[b]
                        if not (prime_factors[i1] & prime_factors[i2]):
                            gain = candidates[i1] + candidates[i2] - val_j
                            if gain > best_gain_12:
                                best_gain_12 = gain
                                best_move_12 = (i1, i2, j)

            if best_move_12:
                i1, i2, j = best_move_12
                current_mask &= ~(1 << j)
                current_mask |= (1 << i1)
                current_mask |= (1 << i2)
                current_sum += best_gain_12
                improved = True
                continue

            # --- 2-for-1 Swap ---
            best_gain_21 = 0
            best_move_21 = None
            top_unselected_1 = top_unselected[:8]

            for j1 in range(num_cands):
                if not ((current_mask >> j1) & 1):
                    continue
                val_j1 = candidates[j1]
                for j2 in range(j1 + 1, num_cands):
                    if not ((current_mask >> j2) & 1):
                        continue
                    val_j2 = candidates[j2]
                    mask_without_j1j2 = current_mask & ~(1 << j1) & ~(1 << j2)
                    for i in top_unselected_1:
                        val_i = candidates[i]
                        if (conflict_mask[i] & mask_without_j1j2) == 0:
                            gain = val_i - (val_j1 + val_j2)
                            if gain > best_gain_21:
                                best_gain_21 = gain
                                best_move_21 = (i, j1, j2)

            if best_move_21:
                i, j1, j2 = best_move_21
                current_mask &= ~(1 << j1)
                current_mask &= ~(1 << j2)
                current_mask |= (1 << i)
                current_sum += best_gain_21
                improved = True
                continue

            # --- 2-for-2 Swap ---
            best_gain_22 = 0
            best_move_22 = None
            top_unselected_2 = top_unselected[:8]

            valid_unselected_pairs = []
            for a in range(len(top_unselected_2)):
                i1 = top_unselected_2[a]
                for b in range(a + 1, len(top_unselected_2)):
                    i2 = top_unselected_2[b]
                    if not (prime_factors[i1] & prime_factors[i2]):
                        valid_unselected_pairs.append((i1, i2))

            for j1 in range(num_cands):
                if not ((current_mask >> j1) & 1):
                    continue
                val_j1 = candidates[j1]
                for j2 in range(j1 + 1, num_cands):
                    if not ((current_mask >> j2) & 1):
                        continue
                    val_j2 = candidates[j2]
                    mask_without_j1j2 = current_mask & ~(1 << j1) & ~(1 << j2)
                    for i1, i2 in valid_unselected_pairs:
                        if (conflict_mask[i1] & mask_without_j1j2) == 0 and \
                           (conflict_mask[i2] & mask_without_j1j2) == 0:
                            gain = candidates[i1] + candidates[i2] - val_j1 - val_j2
                            if gain > best_gain_22:
                                best_gain_22 = gain
                                best_move_22 = (i1, i2, j1, j2)

            if best_move_22:
                i1, i2, j1, j2 = best_move_22
                current_mask &= ~(1 << j1)
                current_mask &= ~(1 << j2)
                current_mask |= (1 << i1)
                current_mask |= (1 << i2)
                current_sum += best_gain_22
                improved = True
                continue

        return current_mask, current_sum

    # Main Loop: Multiple Restarts with diverse initialization strategies
    best_mask = 0
    best_total_sum = sum(fixed)

    # Generate diverse initial orders
    rng = random.Random(12345)
    orders = []

    # 1. Descending value (greedy picks large first)
    orders.append(sorted(candidates, reverse=True))

    # 2. Ascending value
    orders.append(sorted(candidates))

    # 3. Conflict-count ascending (least conflicted first), ties broken by value desc
    conflict_counts = [bin(conflict_mask[i]).count('1') for i in range(num_cands)]
    orders.append(sorted(candidates, key=lambda v: (conflict_counts[val_to_idx[v]], -v)))

    # 4. Value/conflict ratio
    orders.append(sorted(candidates, key=lambda v: candidates[v] / (conflict_counts[val_to_idx[v]] + 1), reverse=True))

    # 5. Random shuffles
    for _ in range(20):
        rand_order = candidates.copy()
        rng.shuffle(rand_order)
        orders.append(rand_order)

    # 6. Weighted random: prioritize high-value, low-conflict nodes
    for _ in range(10):
        weights = []
        for v in candidates:
            idx = val_to_idx[v]
            w = (v ** 1.5) / (conflict_counts[idx] + 1)
            weights.append(w)
        total_w = sum(weights)
        weighted_order = []
        remaining = list(candidates)
        remaining_weights = weights.copy()
        for _ in range(num_cands):
            r = rng.random() * sum(remaining_weights)
            cum = 0
            for k, w in enumerate(remaining_weights):
                cum += w
                if r <= cum:
                    weighted_order.append(remaining[k])
                    remaining_weights.pop(k)
                    remaining.pop(k)
                    break
        orders.append(weighted_order)

    for order in orders:
        if time.time() > deadline:
            break
        init_mask = greedy_initial(order)
        mask, sum_cands = local_search(init_mask)
        total = sum_cands + sum(fixed)
        if total > best_total_sum:
            best_total_sum = total
            best_mask = mask

    # Final polish: run local search from the best solution with more iterations
    if (time.time() - start_time) < deadline:
        mask, sum_cands = local_search(best_mask, max_iters=1000)
        total = sum_cands + sum(fixed)
        if total > best_total_sum:
            best_total_sum = total
            best_mask = mask

    # Construct Result
    result = list(fixed)
    for i in range(num_cands):
        if (best_mask >> i) & 1:
            result.append(candidates[i])

    result = [int(x) for x in result]
    return result
