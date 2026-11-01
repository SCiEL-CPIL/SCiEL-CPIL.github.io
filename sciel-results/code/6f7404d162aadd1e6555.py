import time
import random
import numpy as np


def construct(q, n, radius):
    t0 = time.time()
    deadline = t0 + 118.0

    if q == 3 and n == 7 and radius == 1:
        return _solve_3_7_1(deadline)

    # Fallback for other parameters: simple greedy + deletion polish
    total = q ** n
    ball_size = 1 + n * (q - 1)
    base = [1] * n
    for i in range(n - 2, -1, -1):
        base[i] = base[i + 1] * q

    # Build coverage masks as bitsets (only feasible when total is small)
    if total <= 4096:
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
    else:
        cov_bits = None

    def idx_to_word(idx):
        w = []
        temp = idx
        for i in range(n):
            w.append(temp % q)
            temp //= q
        w.reverse()
        return w

    if cov_bits is None:
        # Dense numpy fallback
        cov = np.zeros((total, total), dtype=np.bool_)
        for c in range(total):
            temp = c
            digits = [0] * n
            for j in range(n - 1, -1, -1):
                digits[j] = temp % q
                temp //= q
            cov[c, c] = True
            for coord in range(n):
                orig = digits[coord]
                for v in range(q):
                    if v == orig:
                        continue
                    w = c + (v - orig) * base[coord]
                    cov[c, w] = True

        def greedy_build(seed):
            rng = random.Random(seed)
            covered = np.zeros(total, dtype=np.uint8)
            code = []
            while True:
                gains = cov @ (1 - covered)
                max_gain = np.max(gains)
                if max_gain == 0:
                    break
                cands = np.where(gains == max_gain)[0]
                best_c = rng.choice(cands.tolist()) if len(cands) > 1 else cands[0]
                code.append(best_c)
                covered[cov[best_c]] = 1
            return code

        def local_search(code_list, time_budget):
            in_code = np.zeros(total, dtype=np.bool_)
            for c in code_list:
                in_code[c] = True
            cnt = np.zeros(total, dtype=np.int32)
            for c in code_list:
                cnt[cov[c]] += 1
            t_start = time.time()
            best_size = len(code_list)
            best_code = list(code_list)
            current_code = list(code_list)

            def try_delete():
                nonlocal best_size, best_code
                changed = True
                while changed:
                    changed = False
                    to_remove = [c for c in current_code if np.min(cnt[cov[c]]) >= 2]
                    if to_remove:
                        for c in to_remove:
                            current_code.remove(c)
                            in_code[c] = False
                            cnt[cov[c]] -= 1
                            changed = True
                        if len(current_code) < best_size:
                            best_size = len(current_code)
                            best_code = list(current_code)

            try_delete()
            rng = random.Random(0)
            no_success = 0
            while time.time() - t_start < time_budget and current_code:
                deleted = False
                for c in current_code:
                    if np.min(cnt[cov[c]]) >= 2:
                        current_code.remove(c)
                        in_code[c] = False
                        cnt[cov[c]] -= 1
                        deleted = True
                        if len(current_code) < best_size:
                            best_size = len(current_code)
                            best_code = list(current_code)
                        break
                if deleted:
                    continue
                c_remove = rng.choice(current_code)
                critical_mask = (cnt == 1) & cov[c_remove]
                if not np.any(critical_mask):
                    current_code.remove(c_remove)
                    in_code[c_remove] = False
                    cnt[cov[c_remove]] -= 1
                    continue
                critical_words = np.where(critical_mask)[0]
                candidates_mask = np.all(cov[:, critical_words], axis=1) & ~in_code
                candidate_indices = np.where(candidates_mask)[0]
                if len(candidate_indices) > 0:
                    c_add = rng.choice(candidate_indices.tolist())
                    current_code.remove(c_remove)
                    in_code[c_remove] = False
                    current_code.append(c_add)
                    in_code[c_add] = True
                    cnt[cov[c_remove]] -= 1
                    cnt[cov[c_add]] += 1
                    no_success = 0
                    try_delete()
                else:
                    no_success += 1
                if no_success > 800:
                    rand_c = rng.randint(0, total - 1)
                    if not in_code[rand_c]:
                        current_code.append(rand_c)
                        in_code[rand_c] = True
                        cnt[cov[rand_c]] += 1
                        try_delete()
                    no_success = 0
            return best_code, best_size

        best_code_overall, best_size_overall = [], total
        greedy_code = greedy_build(42)
        best_code, best_size = local_search(greedy_code, 20.0)
        best_code_overall, best_size_overall = best_code, best_size
        seed = 0
        while time.time() < deadline - 15.0 and best_size_overall > 165:
            seed += 1
            greedy_code = greedy_build(seed)
            tb = min(12.0, max(3.0, (deadline - 15.0 - time.time()) / 6.0))
            best_code, best_size = local_search(greedy_code, tb)
            if best_size < best_size_overall:
                best_size_overall, best_code_overall = best_size, best_code
        if time.time() < deadline - 5.0 and best_code_overall:
            tl = deadline - 5.0 - time.time()
            if tl > 2.0:
                best_code, best_size = local_search(best_code_overall, tl)
                if best_size < best_size_overall:
                    best_size_overall, best_code_overall = best_size, best_code
        return [idx_to_word(i) for i in best_code_overall]

    # Bit-parallel path for small total (the q=3,n=7 case)
    N = total
    cov_bits = cov_bits

    def mask_bits(m):
        bits = []
        while m:
            lsb = m & -m
            bits.append(lsb.bit_length() - 1)
            m ^= lsb
        return bits

    def greedy_build(seed):
        rng = random.Random(seed)
        uncovered = (1 << N) - 1
        selected = []
        selected_set = set()
        while uncovered:
            best_idx = -1
            best_count = -1
            order = list(range(N))
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
            if time.time() > deadline - 5.0:
                break
        return selected

    def local_search(code, seed):
        rng = random.Random(seed)
        code_set = set(code)
        cnt = [0] * N
        for c in code:
            for w in mask_bits(cov_bits[c]):
                cnt[w] += 1
        best_code = code.copy()
        best_size = len(code)
        no_improve = 0
        t_start = time.time()

        def do_deletions():
            nonlocal best_size, best_code
            changed = True
            while changed:
                changed = False
                to_remove = []
                for c in code:
                    m = cov_bits[c]
                    essential = False
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        if cnt[idx] == 1:
                            essential = True
                            break
                        m ^= lsb
                    if not essential:
                        to_remove.append(c)
                if to_remove:
                    for c in to_remove:
                        code.remove(c)
                        code_set.remove(c)
                        m = cov_bits[c]
                        while m:
                            lsb = m & -m
                            idx = lsb.bit_length() - 1
                            cnt[idx] -= 1
                            m ^= lsb
                        changed = True
                    if len(code) < best_size:
                        best_size = len(code)
                        best_code = code.copy()

        def try_swap():
            if not code:
                return False
            c = rng.choice(code)
            m = cov_bits[c]
            unique = []
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                if cnt[idx] == 1:
                    unique.append(idx)
                m ^= lsb
            if not unique:
                code.remove(c)
                code_set.remove(c)
                m = cov_bits[c]
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    cnt[idx] -= 1
                    m ^= lsb
                return True
            cand = cov_bits[unique[0]]
            for w in unique[1:]:
                cand &= cov_bits[w]
                if not cand:
                    break
            if not cand:
                return False
            cands = []
            mm = cand
            while mm:
                lsb = mm & -mm
                idx = lsb.bit_length() - 1
                if idx not in code_set:
                    cands.append(idx)
                mm ^= lsb
            if not cands:
                return False
            cp = rng.choice(cands)
            code.remove(c)
            code_set.remove(c)
            m = cov_bits[c]
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                cnt[idx] -= 1
                m ^= lsb
            code.append(cp)
            code_set.add(cp)
            m = cov_bits[cp]
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                cnt[idx] += 1
                m ^= lsb
            return True

        def try_two_for_one():
            if not code:
                return False
            c = rng.choice(code)
            m = cov_bits[c]
            unique = []
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                if cnt[idx] == 1:
                    unique.append(idx)
                m ^= lsb
            if not unique:
                return False
            any_mask = 0
            for w in unique:
                any_mask |= cov_bits[w]
            any_mask &= ~0  # all
            # candidates covering at least one unique word, not in code
            pool = []
            mm = any_mask
            while mm:
                lsb = mm & -mm
                idx = lsb.bit_length() - 1
                if idx not in code_set:
                    pool.append(idx)
                mm ^= lsb
            if not pool:
                return False
            sample = rng.sample(pool, min(60, len(pool)))
            for c1 in sample:
                covered1 = cov_bits[c1]
                remaining = [w for w in unique if not ((covered1 >> w) & 1)]
                if not remaining:
                    continue
                cand2 = cov_bits[remaining[0]]
                for w in remaining[1:]:
                    cand2 &= cov_bits[w]
                    if not cand2:
                        break
                if not cand2:
                    continue
                mm = cand2
                found2 = False
                while mm:
                    lsb = mm & -mm
                    idx = lsb.bit_length() - 1
                    if idx not in code_set and idx != c1:
                        found2 = True
                        c2 = idx
                        break
                    mm ^= lsb
                if not found2:
                    continue
                code.remove(c)
                code_set.remove(c)
                m = cov_bits[c]
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    cnt[idx] -= 1
                    m ^= lsb
                for nx in (c1, c2):
                    code.append(nx)
                    code_set.add(nx)
                    m = cov_bits[nx]
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        cnt[idx] += 1
                        m ^= lsb
                return True
            return False

        do_deletions()
        while time.time() - t_start < (deadline - 2.0) - t_start and time.time() < deadline - 2.0:
            improved = False
            do_deletions()
            if len(code) < best_size:
                best_size = len(code)
                best_code = code.copy()
                improved = True
            if not code:
                break
            if try_swap():
                improved = True
                no_improve = 0
                do_deletions()
            else:
                no_improve += 1
                if no_improve > 40 and try_two_for_one():
                    no_improve = 0
                    do_deletions()
                elif no_improve > 250:
                    rand_c = rng.randint(0, N - 1)
                    if rand_c not in code_set:
                        code.append(rand_c)
                        code_set.add(rand_c)
                        m = cov_bits[rand_c]
                        while m:
                            lsb = m & -m
                            idx = lsb.bit_length() - 1
                            cnt[idx] += 1
                            m ^= lsb
                        do_deletions()
                    no_improve = 0
            if improved:
                no_improve = 0
        return best_code, best_size

    best_code_overall, best_size_overall = [], N
    greedy_code = greedy_build(42)
    best_code, best_size = local_search(greedy_code, 0)
    best_code_overall, best_size_overall = best_code, best_size
    seed = 0
    while time.time() < deadline - 15.0 and best_size_overall > 165:
        seed += 1
        greedy_code = greedy_build(seed)
        best_code, best_size = local_search(greedy_code, 0)
        if best_size < best_size_overall:
            best_size_overall, best_code_overall = best_size, best_code
    if time.time() < deadline - 5.0 and best_code_overall:
        best_code, best_size = local_search(best_code_overall, 0)
        if best_size < best_size_overall:
            best_size_overall, best_code_overall = best_size, best_code
    return [idx_to_word(i) for i in best_code_overall]


def _solve_3_7_1(deadline):
    N = 2187
    p3 = [1] * 7
    for i in range(1, 7):
        p3[i] = p3[i - 1] * 3

    cov_bits = [0] * N
    for i in range(N):
        digits = [0] * 7
        tmp = i
        for pos in range(7):
            digits[pos] = tmp % 3
            tmp //= 3
        mask = 1 << i
        for pos in range(7):
            orig = digits[pos]
            for val in range(3):
                if val != orig:
                    new_i = i + (val - orig) * p3[pos]
                    mask |= 1 << new_i
        cov_bits[i] = mask

    def mask_bits(m):
        bits = []
        while m:
            lsb = m & -m
            bits.append(lsb.bit_length() - 1)
            m ^= lsb
        return bits

    def greedy_build(seed):
        rng = random.Random(seed)
        uncovered = (1 << N) - 1
        selected = []
        selected_set = set()
        while uncovered:
            best_idx = -1
            best_count = -1
            order = list(range(N))
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
            if time.time() > deadline - 5.0:
                break
        return selected

    def local_search(code, seed):
        rng = random.Random(seed)
        code_set = set(code)
        cnt = [0] * N
        for c in code:
            for w in mask_bits(cov_bits[c]):
                cnt[w] += 1
        best_code = code.copy()
        best_size = len(code)
        no_improve = 0
        t_start = time.time()

        def do_deletions():
            nonlocal best_size, best_code
            changed = True
            while changed:
                changed = False
                to_remove = []
                for c in code:
                    m = cov_bits[c]
                    essential = False
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        if cnt[idx] == 1:
                            essential = True
                            break
                        m ^= lsb
                    if not essential:
                        to_remove.append(c)
                if to_remove:
                    for c in to_remove:
                        code.remove(c)
                        code_set.remove(c)
                        m = cov_bits[c]
                        while m:
                            lsb = m & -m
                            idx = lsb.bit_length() - 1
                            cnt[idx] -= 1
                            m ^= lsb
                        changed = True
                    if len(code) < best_size:
                        best_size = len(code)
                        best_code = code.copy()

        def try_swap():
            if not code:
                return False
            c = rng.choice(code)
            m = cov_bits[c]
            unique = []
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                if cnt[idx] == 1:
                    unique.append(idx)
                m ^= lsb
            if not unique:
                code.remove(c)
                code_set.remove(c)
                m = cov_bits[c]
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    cnt[idx] -= 1
                    m ^= lsb
                return True
            cand = cov_bits[unique[0]]
            for w in unique[1:]:
                cand &= cov_bits[w]
                if not cand:
                    break
            if not cand:
                return False
            cands = []
            mm = cand
            while mm:
                lsb = mm & -mm
                idx = lsb.bit_length() - 1
                if idx not in code_set:
                    cands.append(idx)
                mm ^= lsb
            if not cands:
                return False
            cp = rng.choice(cands)
            code.remove(c)
            code_set.remove(c)
            m = cov_bits[c]
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                cnt[idx] -= 1
                m ^= lsb
            code.append(cp)
            code_set.add(cp)
            m = cov_bits[cp]
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                cnt[idx] += 1
                m ^= lsb
            return True

        def try_two_for_one():
            if not code:
                return False
            c = rng.choice(code)
            m = cov_bits[c]
            unique = []
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                if cnt[idx] == 1:
                    unique.append(idx)
                m ^= lsb
            if not unique:
                return False
            any_mask = 0
            for w in unique:
                any_mask |= cov_bits[w]
            pool = []
            mm = any_mask
            while mm:
                lsb = mm & -mm
                idx = lsb.bit_length() - 1
                if idx not in code_set:
                    pool.append(idx)
                mm ^= lsb
            if not pool:
                return False
            sample = rng.sample(pool, min(60, len(pool)))
            for c1 in sample:
                covered1 = cov_bits[c1]
                remaining = [w for w in unique if not ((covered1 >> w) & 1)]
                if not remaining:
                    continue
                cand2 = cov_bits[remaining[0]]
                for w in remaining[1:]:
                    cand2 &= cov_bits[w]
                    if not cand2:
                        break
                if not cand2:
                    continue
                mm = cand2
                found2 = False
                while mm:
                    lsb = mm & -mm
                    idx = lsb.bit_length() - 1
                    if idx not in code_set and idx != c1:
                        found2 = True
                        c2 = idx
                        break
                    mm ^= lsb
                if not found2:
                    continue
                code.remove(c)
                code_set.remove(c)
                m = cov_bits[c]
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    cnt[idx] -= 1
                    m ^= lsb
                for nx in (c1, c2):
                    code.append(nx)
                    code_set.add(nx)
                    m = cov_bits[nx]
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        cnt[idx] += 1
                        m ^= lsb
                return True
            return False

        do_deletions()
        while time.time() < deadline - 2.0:
            do_deletions()
            if len(code) < best_size:
                best_size = len(code)
                best_code = code.copy()
            if not code:
                break
            if try_swap():
                no_improve = 0
                do_deletions()
            else:
                no_improve += 1
                if no_improve > 40 and try_two_for_one():
                    no_improve = 0
                    do_deletions()
                elif no_improve > 250:
                    rand_c = rng.randint(0, N - 1)
                    if rand_c not in code_set:
                        code.append(rand_c)
                        code_set.add(rand_c)
                        m = cov_bits[rand_c]
                        while m:
                            lsb = m & -m
                            idx = lsb.bit_length() - 1
                            cnt[idx] += 1
                            m ^= lsb
                        do_deletions()
                    no_improve = 0
        return best_code, best_size

    best_code_overall, best_size_overall = [], N
    greedy_code = greedy_build(42)
    best_code, best_size = local_search(greedy_code, 42)
    best_code_overall, best_size_overall = best_code, best_size
    seed = 0
    while time.time() < deadline - 15.0 and best_size_overall > 165:
        seed += 1
        greedy_code = greedy_build(seed)
        best_code, best_size = local_search(greedy_code, seed)
        if best_size < best_size_overall:
            best_size_overall, best_code_overall = best_size, best_code
    if time.time() < deadline - 5.0 and best_code_overall:
        best_code, best_size = local_search(best_code_overall, 999)
        if best_size < best_size_overall:
            best_size_overall, best_code_overall = best_size, best_code

    def idx_to_word(idx):
        w = []
        temp = idx
        for i in range(7):
            w.append(temp % 3)
            temp //= 3
        w.reverse()
        return w

    return [idx_to_word(i) for i in best_code_overall]
