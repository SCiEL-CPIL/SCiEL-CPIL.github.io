import time
import random
import numpy as np

def construct(q, n, radius):
    t0 = time.time()
    deadline = t0 + 118.0

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
        max_streak = 1000

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

            if no_success > max_streak:
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
    while time.time() < deadline - 10.0 and best_size_overall > 170:
        seed += 1
        greedy_code = greedy_build(seed)
        time_left = deadline - 10.0 - time.time()
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

    # Precompute coverage masks as bitsets
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

    # Also build a numpy coverage matrix for vectorized operations
    # cov_mat[i] is a boolean array of length total, True if word i is covered by codeword j
    cov_mat = np.zeros((total, total), dtype=np.bool_)
    for c in range(total):
        mask = cov_bits[c]
        while mask:
            lsb = mask & -mask
            idx = lsb.bit_length() - 1
            cov_mat[c, idx] = True
            mask ^= lsb

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
        
        # Coverage count array
        cnt = np.zeros(total, dtype=np.int32)
        for c in code_set:
            cnt[cov_mat[c]] += 1

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
                    # Check if all words covered by c have cnt >= 2
                    if np.all(cnt[cov_mat[c]] >= 2):
                        to_remove.append(c)
                if to_remove:
                    for c in to_remove:
                        code_set.remove(c)
                        cnt[cov_mat[c]] -= 1
                    changed = True
                    if len(code_set) < best_size:
                        best_size = len(code_set)
                        best_code = list(code_set)

        try_delete()

        no_success = 0
        max_streak = 1000

        while time.time() - t_start < time_budget:
            if len(code_set) == 0:
                break

            # Try deletion
            deleted = False
            for c in list(code_set):
                if np.all(cnt[cov_mat[c]] >= 2):
                    code_set.remove(c)
                    cnt[cov_mat[c]] -= 1
                    deleted = True
                    if len(code_set) < best_size:
                        best_size = len(code_set)
                        best_code = list(code_set)
                    break

            if deleted:
                continue

            # Try 1-for-1 swap
            c_remove = rng.choice(list(code_set))
            
            # Find critical words: covered by c_remove and no one else
            critical_mask = (cnt == 1) & cov_mat[c_remove]
            
            if not np.any(critical_mask):
                # Can just delete
                code_set.remove(c_remove)
                cnt[cov_mat[c_remove]] -= 1
                continue

            # Find candidates that cover all critical words using vectorized ops
            critical_words = np.where(critical_mask)[0]
            if len(critical_words) == 0:
                code_set.remove(c_remove)
                cnt[cov_mat[c_remove]] -= 1
                continue
            
            # Vectorized: for each candidate, check if it covers all critical words
            # cov_mat[:, critical_words] has shape (total, len(critical_words))
            # We need all True for each row
            covers_all = np.all(cov_mat[:, critical_words], axis=1)
            
            # Exclude words already in code_set
            in_code = np.zeros(total, dtype=np.bool_)
            for c in code_set:
                in_code[c] = True
            covers_all &= ~in_code
            
            candidate_indices = np.where(covers_all)[0]
            
            if len(candidate_indices) > 0:
                c_add = rng.choice(candidate_indices.tolist())
                code_set.remove(c_remove)
                code_set.add(c_add)
                cnt[cov_mat[c_remove]] -= 1
                cnt[cov_mat[c_add]] += 1
                no_success = 0
                try_delete()
            else:
                no_success += 1

            # Try 2-for-1 improvement
            if no_success > 500 and time.time() - t_start < time_budget - 0.5:
                c_remove = rng.choice(list(code_set))
                critical_mask = (cnt == 1) & cov_mat[c_remove]
                
                if np.any(critical_mask):
                    critical_words = np.where(critical_mask)[0]
                    found = False
                    for _ in range(50):
                        c1 = rng.randint(0, total - 1)
                        c2 = rng.randint(0, total - 1)
                        if c1 in code_set or c2 in code_set or c1 == c2:
                            continue
                        if c1 == c_remove or c2 == c_remove:
                            continue
                        # Check if c1 and c2 together cover all critical words
                        covers_c1 = cov_mat[c1, critical_words]
                        covers_c2 = cov_mat[c2, critical_words]
                        if np.all(covers_c1 | covers_c2):
                            code_set.remove(c_remove)
                            code_set.add(c1)
                            code_set.add(c2)
                            cnt[cov_mat[c_remove]] -= 1
                            cnt[cov_mat[c1]] += 1
                            cnt[cov_mat[c2]] += 1
                            no_success = 0
                            try_delete()
                            found = True
                            break
                    if not found:
                        no_success += 1
                else:
                    no_success += 1

            if no_success > max_streak:
                rand_c = rng.randint(0, total - 1)
                if rand_c not in code_set:
                    code_set.add(rand_c)
                    cnt[cov_mat[rand_c]] += 1
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
    while time.time() < deadline - 10.0 and best_size_overall > 170:
        seed += 1
        greedy_code = greedy_build(seed)
        time_left = deadline - 10.0 - time.time()
        tb = min(10.0, max(2.0, time_left / 8.0))
        best_code, best_size = local_search(greedy_code, time_budget=tb)
        if best_size < best_size_overall:
            best_size_overall = best_size
            best_code_overall = best_code

    result = []
    for idx in best_code_overall:
        result.append(idx_to_word(idx))
    return result
