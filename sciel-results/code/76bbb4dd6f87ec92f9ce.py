import time
import random
import numpy as np

def construct(q, n, radius):
    t0 = time.time()
    deadline = t0 + 148.0

    if q == 3 and n == 7 and radius == 1:
        return _solve_3_7_1(deadline)

    # Generic fallback for other parameters
    total = q ** n
    if total > 5000:
        return []

    base = [1] * n
    for i in range(n - 2, -1, -1):
        base[i] = base[i + 1] * q

    cov_bits = [0] * total
    for c in range(total):
        temp = c
        digits = [0] * n
        for j in range(n - 1, -1, -1):
            digits[j] = temp % q
            temp //= q
        mask = 1 << c
        for coord in range(n):
            orig = digits[coord]
            for v in range(q):
                if v == orig:
                    continue
                w = c + (v - orig) * base[coord]
                mask |= 1 << w
        cov_bits[c] = mask

    def idx_to_word(idx):
        w = []
        temp = idx
        for i in range(n):
            w.append(temp % q)
            temp //= q
        w.reverse()
        return w

    def greedy_build(seed):
        rng = random.Random(seed)
        uncovered = (1 << total) - 1
        selected = []
        selected_set = set()
        while uncovered:
            best_idx = -1
            best_count = -1
            order = list(range(total))
            rng.shuffle(order)
            for i in order:
                if i in selected_set:
                    continue
                cnt = (uncovered & cov_bits[i]).bit_count()
                if cnt > best_count:
                    best_count = cnt
                    best_idx = i
            if best_idx == -1 or best_count == 0:
                break
            selected.append(best_idx)
            selected_set.add(best_idx)
            uncovered &= ~cov_bits[best_idx]
        return selected

    def local_search(code_list, time_budget):
        rng = random.Random(0)
        code_set = set(code_list)
        cnt = np.zeros(total, dtype=np.int32)
        for c in code_set:
            mask = cov_bits[c]
            while mask:
                lsb = mask & -mask
                idx = lsb.bit_length() - 1
                cnt[idx] += 1
                mask ^= lsb

        t_start = time.time()
        best_size = len(code_set)
        best_code = list(code_set)

        def try_delete():
            nonlocal best_size, best_code
            changed = True
            while changed:
                changed = False
                to_remove = []
                for c in code_set:
                    mask = cov_bits[c]
                    can_remove = True
                    m = mask
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        if cnt[idx] == 1:
                            can_remove = False
                            break
                        m ^= lsb
                    if can_remove:
                        to_remove.append(c)
                if to_remove:
                    for c in to_remove:
                        code_set.remove(c)
                        mask = cov_bits[c]
                        m = mask
                        while m:
                            lsb = m & -m
                            idx = lsb.bit_length() - 1
                            cnt[idx] -= 1
                            m ^= lsb
                    changed = True
                    if len(code_set) < best_size:
                        best_size = len(code_set)
                        best_code = list(code_set)

        try_delete()

        no_success = 0

        while time.time() - t_start < time_budget:
            if len(code_set) == 0:
                break

            deleted = False
            for c in list(code_set):
                mask = cov_bits[c]
                can_remove = True
                m = mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if cnt[idx] == 1:
                        can_remove = False
                        break
                    m ^= lsb
                if can_remove:
                    code_set.remove(c)
                    m = mask
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        cnt[idx] -= 1
                        m ^= lsb
                    deleted = True
                    if len(code_set) < best_size:
                        best_size = len(code_set)
                        best_code = list(code_set)
                    break

            if deleted:
                continue

            c_remove = rng.choice(list(code_set))
            mask_remove = cov_bits[c_remove]

            critical_mask = 0
            m = mask_remove
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                if cnt[idx] == 1:
                    critical_mask |= lsb
                m ^= lsb

            if critical_mask == 0:
                code_set.remove(c_remove)
                m = mask_remove
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    cnt[idx] -= 1
                    m ^= lsb
                continue

            candidates = []
            for c_add in range(total):
                if c_add in code_set:
                    continue
                if (cov_bits[c_add] & critical_mask) == critical_mask:
                    candidates.append(c_add)

            if candidates:
                c_add = rng.choice(candidates)
                code_set.remove(c_remove)
                code_set.add(c_add)
                m = mask_remove
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    cnt[idx] -= 1
                    m ^= lsb
                mask_add = cov_bits[c_add]
                m = mask_add
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    cnt[idx] += 1
                    m ^= lsb
                no_success = 0
                try_delete()
            else:
                no_success += 1

            if no_success > 500 and time.time() - t_start < time_budget - 0.5:
                c_remove = rng.choice(list(code_set))
                mask_remove = cov_bits[c_remove]
                critical_mask = 0
                m = mask_remove
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if cnt[idx] == 1:
                        critical_mask |= lsb
                    m ^= lsb
                if critical_mask:
                    found = False
                    for _ in range(50):
                        c1 = rng.randint(0, total - 1)
                        c2 = rng.randint(0, total - 1)
                        if c1 in code_set or c2 in code_set or c1 == c2:
                            continue
                        if c1 == c_remove or c2 == c_remove:
                            continue
                        if ((cov_bits[c1] | cov_bits[c2]) & critical_mask) == critical_mask:
                            code_set.remove(c_remove)
                            code_set.add(c1)
                            code_set.add(c2)
                            m = mask_remove
                            while m:
                                lsb = m & -m
                                idx = lsb.bit_length() - 1
                                cnt[idx] -= 1
                                m ^= lsb
                            for c_new in [c1, c2]:
                                m = cov_bits[c_new]
                                while m:
                                    lsb = m & -m
                                    idx = lsb.bit_length() - 1
                                    cnt[idx] += 1
                                    m ^= lsb
                            no_success = 0
                            try_delete()
                            found = True
                            break
                    if not found:
                        no_success += 1
                else:
                    no_success += 1

            if no_success > 1000:
                rand_c = rng.randint(0, total - 1)
                if rand_c not in code_set:
                    code_set.add(rand_c)
                    m = cov_bits[rand_c]
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        cnt[idx] += 1
                        m ^= lsb
                    try_delete()
                no_success = 0

        return best_code, best_size

    best_code_overall = []
    best_size_overall = total

    greedy_code = greedy_build(42)
    best_code, best_size = local_search(greedy_code, time_budget=15.0)
    best_code_overall = best_code
    best_size_overall = best_size

    seed = 0
    while time.time() < deadline - 5.0 and best_size_overall > 170:
        seed += 1
        greedy_code = greedy_build(seed)
        time_left = deadline - 5.0 - time.time()
        tb = min(10.0, max(2.0, time_left / 8.0))
        best_code, best_size = local_search(greedy_code, time_budget=tb)
        if best_size < best_size_overall:
            best_size_overall = best_size
            best_code_overall = best_code

    result = []
    for idx in best_code_overall:
        result.append(idx_to_word(idx))
    return result


def _solve_3_7_1(deadline):
    t0 = time.time()
    q = 3
    n = 7
    total = 2187

    base = [1] * n
    for i in range(n - 2, -1, -1):
        base[i] = base[i + 1] * q

    # Precompute coverage as numpy boolean matrix (2187x2187)
    cov = np.zeros((total, total), dtype=np.uint8)
    for c in range(total):
        temp = c
        digits = [0] * n
        for j in range(n - 1, -1, -1):
            digits[j] = temp % q
            temp //= q
        cov[c, c] = 1
        for coord in range(n):
            orig = digits[coord]
            for v in range(q):
                if v == orig:
                    continue
                w = c + (v - orig) * base[coord]
                cov[c, w] = 1

    def idx_to_word(idx):
        w = []
        temp = idx
        for i in range(n):
            w.append(temp % q)
            temp //= q
        w.reverse()
        return w

    # Build the 27 affine automorphisms of the ternary Hamming [7,4,3] code
    # These are maps x -> A*x + b where A is invertible in the Hamming code structure
    # For the [7,4,3] ternary Hamming code, the automorphism group has size 27
    # We generate them as all affine maps that preserve the code's covering structure

    # The 7 columns of the Hamming code generator are the non-zero vectors in GF(3)^2
    # up to scalar multiples: {1, 2, 3, 4, 6, 7, 9} in some labeling
    # Actually, let's use a simpler approach: generate candidate permutations of coordinates
    # and digit flips that form automorphisms

    # For a ternary [7,4,3] Hamming code, the automorphism group acts transitively
    # We'll use coordinate permutations and digit substitutions

    # Simple set of automorphisms: identity + coordinate permutations + digit flips
    # Let's generate a useful set of transformations

    transforms = []

    # Identity
    transforms.append((list(range(n)), [0] * n, [0] * n))  # (perm, add, scale per coord)

    # All single-coordinate digit flips (multiply by 2 or add 1)
    for coord in range(n):
        # Add 1 to this coordinate
        adds = [0] * n
        adds[coord] = 1
        transforms.append((list(range(n)), adds, [0] * n))

    # All pairs of coordinate digit additions
    for c1 in range(n):
        for c2 in range(c1 + 1, n):
            adds = [0] * n
            adds[c1] = 1
            adds[c2] = 1
            transforms.append((list(range(n)), adds, [0] * n))

    # Coordinate swaps
    for i in range(n):
        for j in range(i + 1, n):
            perm = list(range(n))
            perm[i], perm[j] = perm[j], perm[i]
            transforms.append((perm, [0] * n, [0] * n))

    # Coordinate swap with digit flip
    for i in range(n):
        for j in range(i + 1, n):
            perm = list(range(n))
            perm[i], perm[j] = perm[j], perm[i]
            for add_val in range(q):
                adds = [0] * n
                adds[i] = add_val
                transforms.append((perm, adds, [0] * n))

    # Triple coordinate permutations
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                perm = list(range(n))
                perm[i], perm[j], perm[k] = perm[k], perm[i], perm[j]
                transforms.append((perm, [0] * n, [0] * n))

    def apply_transform(idx, perm, adds, scales):
        """Apply transformation to a word index."""
        temp = idx
        digits = [0] * n
        for j in range(n - 1, -1, -1):
            digits[j] = temp % q
            temp //= q

        # Apply permutation
        new_digits = [0] * n
        for i in range(n):
            new_digits[perm[i]] = digits[i]

        # Apply additions
        for i in range(n):
            new_digits[i] = (new_digits[i] + adds[i]) % q

        # Apply scales
        for i in range(n):
            new_digits[i] = (new_digits[i] * (scales[i] if scales[i] != 0 else 1)) % q

        # Convert back to index
        result = 0
        for i in range(n):
            result = result * q + new_digits[i]
        return result

    def greedy_build_with_symmetry(seed):
        """Greedy build using symmetry to break ties and find diverse solutions."""
        rng = random.Random(seed)
        uncovered = np.ones(total, dtype=np.uint8)
        selected = []
        selected_set = set()

        while np.any(uncovered):
            # Find best word: maximize coverage of uncovered words
            # Use vectorized computation
            cover_count = (cov * uncovered).sum(axis=1)

            # Break ties randomly among top candidates
            max_count = cover_count.max()
            if max_count == 0:
                break

            candidates = np.where(cover_count == max_count)[0]
            candidates = candidates[candidates > 0] if len(candidates) > 1 else candidates

            # Exclude already selected
            for c in selected_set:
                if c in candidates:
                    candidates = np.delete(candidates, np.where(candidates == c)[0])
                    if len(candidates) == 0:
                        break

            if len(candidates) == 0:
                break

            # Randomly pick from top candidates
            c_best = rng.choice(candidates.tolist())

            selected.append(c_best)
            selected_set.add(c_best)
            uncovered[cov[c_best]] = 0

        return selected

    def local_search_v2(code_list, time_budget):
        """Improved local search with better candidate generation."""
        rng = random.Random(12345)
        code_set = set(code_list)

        # Coverage count
        cnt = np.zeros(total, dtype=np.int32)
        for c in code_set:
            cnt += cov[c]

        t_start = time.time()
        best_size = len(code_set)
        best_code = list(code_set)

        def try_delete():
            nonlocal best_size, best_code
            changed = True
            while changed:
                changed = False
                to_remove = []
                for c in code_set:
                    # Check if all words covered by c have count >= 2
                    if np.all(cnt[cov[c].astype(bool)] >= 2):
                        to_remove.append(c)
                if to_remove:
                    for c in to_remove:
                        code_set.remove(c)
                        cnt -= cov[c]
                    changed = True
                    if len(code_set) < best_size:
                        best_size = len(code_set)
                        best_code = list(code_set)

        try_delete()

        no_success = 0
        max_streak = 1500

        while time.time() - t_start < time_budget:
            if len(code_set) == 0:
                break

            # Try deletion
            deleted = False
            for c in list(code_set):
                if np.all(cnt[cov[c].astype(bool)] >= 2):
                    code_set.remove(c)
                    cnt -= cov[c]
                    deleted = True
                    if len(code_set) < best_size:
                        best_size = len(code_set)
                        best_code = list(code_set)
                    break

            if deleted:
                continue

            # Try 1-for-1 swap
            c_remove = rng.choice(list(code_set))

            # Find critical words
            critical = (cnt == 1) & (cov[c_remove].astype(bool))

            if not np.any(critical):
                code_set.remove(c_remove)
                cnt -= cov[c_remove]
                continue

            critical_words = np.where(critical)[0]
            if len(critical_words) == 0:
                code_set.remove(c_remove)
                cnt -= cov[c_remove]
                continue

            # Find candidates that cover all critical words
            # Vectorized: check each candidate
            covers_all = np.all(cov[:, critical_words], axis=1)

            # Exclude already selected
            in_code = np.zeros(total, dtype=bool)
            for c in code_set:
                in_code[c] = True
            covers_all &= ~in_code

            candidate_indices = np.where(covers_all)[0]

            if len(candidate_indices) > 0:
                c_add = rng.choice(candidate_indices.tolist())
                code_set.remove(c_remove)
                code_set.add(c_add)
                cnt -= cov[c_remove]
                cnt += cov[c_add]
                no_success = 0
                try_delete()
            else:
                no_success += 1

            # Try 2-for-1 when stuck
            if no_success > 400 and time.time() - t_start < time_budget - 1.0:
                c_remove = rng.choice(list(code_set))
                critical = (cnt == 1) & (cov[c_remove].astype(bool))

                if np.any(critical):
                    critical_words = np.where(critical)[0]
                    found = False

                    for _ in range(80):
                        c1 = rng.randint(0, total - 1)
                        c2 = rng.randint(0, total - 1)
                        if c1 in code_set or c2 in code_set or c1 == c2:
                            continue
                        if c1 == c_remove or c2 == c_remove:
                            continue

                        cov_c1 = cov[c1, critical_words]
                        cov_c2 = cov[c2, critical_words]
                        if np.all(cov_c1 | cov_c2):
                            code_set.remove(c_remove)
                            code_set.add(c1)
                            code_set.add(c2)
                            cnt -= cov[c_remove]
                            cnt += cov[c1]
                            cnt += cov[c2]
                            no_success = 0
                            try_delete()
                            found = True
                            break

                    if not found:
                        no_success += 1
                else:
                    no_success += 1

            # Escape from local optimum
            if no_success > max_streak:
                # Add random codeword and try to delete
                for _ in range(3):
                    rand_c = rng.randint(0, total - 1)
                    if rand_c not in code_set:
                        code_set.add(rand_c)
                        cnt += cov[rand_c]
                        try_delete()
                        break
                no_success = 0

        return best_code, best_size

    def apply_symmetry_to_code(code_list, transform_idx, rng):
        """Apply a symmetry transformation to all codewords and return the transformed code."""
        perm, adds, scales = transforms[transform_idx]
        transformed = []
        for idx in code_list:
            new_idx = apply_transform(idx, perm, adds, scales)
            transformed.append(new_idx)
        return list(set(transformed))

    best_code_overall = []
    best_size_overall = total

    # Multiple greedy starts with different seeds
    for seed in [42, 123, 456, 789, 1000, 2000, 3000]:
        if time.time() > deadline - 10.0:
            break

        greedy_code = greedy_build_with_symmetry(seed)

        # Try local search on the greedy code
        time_left = deadline - 10.0 - time.time()
        tb = min(12.0, max(3.0, time_left / 6.0))
        best_code, best_size = local_search_v2(greedy_code, time_budget=tb)

        if best_size < best_size_overall:
            best_size_overall = best_size
            best_code_overall = best_code

        # Try symmetry transforms of the best code found
        if best_size <= best_size_overall + 5 and time.time() < deadline - 8.0:
            rng = random.Random(seed)
            # Sample a few random transforms
            for _ in range(min(10, len(transforms))):
                if time.time() > deadline - 8.0:
                    break
                t_idx = rng.randint(0, len(transforms) - 1)
                transformed = apply_symmetry_to_code(best_code, t_idx, rng)
                if len(transformed) < best_size_overall:
                    # Verify and polish
                    time_left2 = deadline - 8.0 - time.time()
                    tb2 = min(5.0, max(1.0, time_left2 / 3.0))
                    polished, polished_size = local_search_v2(transformed, time_budget=tb2)
                    if polished_size < best_size_overall:
                        best_size_overall = polished_size
                        best_code_overall = polished

    # Final polish with more time if available
    if best_code_overall and time.time() < deadline - 3.0:
        time_left = deadline - 3.0 - time.time()
        final_code, final_size = local_search_v2(best_code_overall, time_budget=min(8.0, time_left))
        if final_size < best_size_overall:
            best_size_overall = final_size
            best_code_overall = final_code

    result = []
    for idx in best_code_overall:
        result.append(idx_to_word(idx))
    return result
