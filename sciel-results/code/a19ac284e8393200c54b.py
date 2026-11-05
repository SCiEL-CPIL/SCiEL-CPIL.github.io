import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin within 150s wall-clock

    # ---------- Generic fallback ----------
    if q != 3 or n != 7 or radius != 1:
        if n == 0:
            return [[]]
        if n <= 4 and q == 3 and radius == 1:
            return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n >= 4:
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
        else:
            return [list(w) for w in itertools.product(range(q), repeat=n)]

    # ---------- Specific optimization for q=3, n=7, radius=1 ----------
    N = 3 ** n  # 2187
    BALL = 2 * n + 1  # 15

    def idx_to_tuple(idx):
        t = [0] * n
        for i in range(n - 1, -1, -1):
            t[i] = idx % 3
            idx //= 3
        return t

    def tuple_to_idx(t):
        idx = 0
        for x in t:
            idx = idx * 3 + x
        return idx

    # Precompute balls and the ball matrix B[c, w] = 1 iff w in ball(c)
    balls = []
    for i in range(N):
        w = idx_to_tuple(i)
        nb = [i]
        for pos in range(n):
            val = w[pos]
            for nv in (val + 1) % 3, (val + 2) % 3:
                nw = w[:]
                nw[pos] = nv
                nb.append(tuple_to_idx(nw))
        balls.append(nb)

    B = np.zeros((N, N), dtype=np.int8)
    for c in range(N):
        B[c, balls[c]] = 1

    # cover_map[w] = list of codewords whose ball contains w
    cover_map = [[] for _ in range(N)]
    for c in range(N):
        for w in balls[c]:
            cover_map[w].append(c)

    # ---------- Vectorized greedy set-cover ----------
    def greedy_construct(seed_val):
        rng = random.Random(seed_val)
        gain = np.full(N, BALL, dtype=np.int32)
        covered = np.zeros(N, dtype=bool)
        code = []
        uncovered_count = N
        while uncovered_count > 0 and time.time() < deadline:
            max_val = int(gain.max())
            if max_val <= 0:
                break
            cands = np.flatnonzero(gain == max_val)
            c = int(rng.choice(cands))
            code.append(c)
            ball_c = balls[c]
            newly_mask = ~covered[ball_c]
            newly = [w for w in ball_c if newly_mask[w]]
            if newly:
                covered[ball_c] = True
                uncovered_count -= len(newly)
                for w in newly:
                    gain[cover_map[w]] -= 1
        return code

    # ---------- Pure redundancy removal ----------
    def refine(code):
        code_list = list(code)
        if not code_list:
            return code_list
        cover_count = np.zeros(N, dtype=np.int32)
        for c in code_list:
            cover_count[balls[c]] += 1
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for i in range(len(code_list)):
                c = code_list[i]
                if np.all(cover_count[balls[c]] > 1):
                    cover_count[balls[c]] -= 1
                    code_list.pop(i)
                    changed = True
                    break
        return code_list

    # ---------- Local search helpers ----------
    def is_redundant(c, coverage):
        return np.min(coverage[balls[c]]) > 1

    def greedy_repair(uncovered_list, code_set, coverage):
        """Add codewords to cover uncovered_list; returns list of added codewords."""
        if not uncovered_list:
            return []
        added = []
        temp = set(uncovered_list)
        while temp and time.time() < deadline:
            pool = set()
            for w in temp:
                pool.update(cover_map[w])
            best_c = -1
            best_count = 0
            for c in pool:
                if c in code_set:
                    continue
                cnt = 0
                Bc = B[c]
                for w in temp:
                    if Bc[w]:
                        cnt += 1
                if cnt > best_count:
                    best_count = cnt
                    best_c = c
                    if best_count == len(temp):
                        break
            if best_c == -1:
                break
            added.append(best_c)
            temp.difference_update(balls[best_c])
        return added

    def try_kick(k, code, code_set, coverage):
        """Remove k random codewords, repair to cover all; return improved state or None."""
        if len(code) < k:
            return None
        indices = random.sample(range(len(code)), k)
        removed = [code[i] for i in indices]
        temp_cov = coverage.copy()
        for c in removed:
            temp_cov[balls[c]] -= 1
        uncovered = np.where(temp_cov == 0)[0]
        if len(uncovered) == 0:
            new_code = [c for i, c in enumerate(code) if i not in indices]
            new_set = code_set - set(removed)
            return (new_code, new_set, temp_cov)
        unc_set = set(uncovered.tolist())
        # Candidate pool: codewords adjacent to uncovered words, not already in code
        pool = set()
        for w in unc_set:
            pool.update(cover_map[w])
        pool.difference_update(code_set)
        pool = list(pool)
        # Greedy select a minimum-ish set covering unc_set
        chosen = []
        chosen_set = set()
        still = unc_set
        while still and time.time() < deadline:
            best_c = -1
            best_count = 0
            for c in pool:
                if c in chosen_set:
                    continue
                cnt = 0
                Bc = B[c]
                for w in still:
                    if Bc[w]:
                        cnt += 1
                if cnt > best_count:
                    best_count = cnt
                    best_c = c
                    if best_count == len(still):
                        break
            if best_c == -1 or best_count == 0:
                break
            chosen.append(best_c)
            chosen_set.add(best_c)
            still.difference_update(balls[best_c])
        if still:
            return None  # cannot cover everything
        if len(code) - k + len(chosen) >= len(code):
            return None  # no size reduction
        new_code = [c for i, c in enumerate(code) if i not in indices] + chosen
        new_set = (code_set - set(removed)) | chosen_set
        final_cov = temp_cov.copy()
        for c in chosen:
            final_cov[balls[c]] += 1
        return (new_code, new_set, final_cov)

    def try_swap(code, code_set, coverage):
        """Relocate a single random codeword to cover any gap; return improved state or None."""
        if len(code) < 2:
            return None
        idx = random.randint(0, len(code) - 1)
        removed = code[idx]
        temp_cov = coverage.copy()
        temp_cov[balls[removed]] -= 1
        uncovered = np.where(temp_cov == 0)[0]
        if len(uncovered) == 0:
            new_code = [c for i, c in enumerate(code) if i != idx]
            new_set = code_set - {removed}
            return (new_code, new_set, temp_cov)
        unc_set = set(uncovered.tolist())
        pool = set()
        for w in unc_set:
            pool.update(cover_map[w])
        pool.difference_update(code_set)
        pool.discard(removed)
        best_c = -1
        best_gain = 0
        for c in pool:
            cnt = 0
            Bc = B[c]
            for w in unc_set:
                if Bc[w]:
                    cnt += 1
            if cnt > best_gain:
                best_gain = cnt
                best_c = c
                if best_gain >= len(unc_set):
                    break
        if best_c == -1 or best_gain < len(unc_set):
            return None
        new_code = code[:]
        new_code[idx] = best_c
        new_set = (code_set - {removed}) | {best_c}
        final_cov = temp_cov.copy()
        final_cov[balls[best_c]] += 1
        return (new_code, new_set, final_cov)

    def try_swap2(code, code_set, coverage):
        """Relocate two random codewords (2-opt style); return improved state or None."""
        if len(code) < 3:
            return None
        idx1, idx2 = random.sample(range(len(code)), 2)
        r1, r2 = code[idx1], code[idx2]
        if r1 == r2:
            return None
        temp_cov = coverage.copy()
        temp_cov[balls[r1]] -= 1
        temp_cov[balls[r2]] -= 1
        uncovered = np.where(temp_cov == 0)[0]
        if len(uncovered) == 0:
            new_code = [c for i, c in enumerate(code) if i not in (idx1, idx2)]
            new_set = code_set - {r1, r2}
            return (new_code, new_set, temp_cov)
        unc_set = set(uncovered.tolist())
        pool = set()
        for w in unc_set:
            pool.update(cover_map[w])
        pool.difference_update(code_set)
        pool.discard(r1)
        pool.discard(r2)
        # Try to cover unc_set with <=2 new codewords
        pool_list = list(pool)
        found = None
        for c1 in pool_list:
            Bc1 = B[c1]
            still = set()
            for w in unc_set:
                if not Bc1[w]:
                    still.add(w)
            if not still:
                found = [c1]
                break
            # Need one more codeword covering 'still'
            for c2 in pool_list:
                if c2 == c1:
                    continue
                Bc2 = B[c2]
                ok = True
                for w in still:
                    if not Bc2[w]:
                        ok = False
                        break
                if ok:
                    found = [c1, c2]
                    break
            if found:
                break
        if not found:
            return None
        if len(code) - 2 + len(found) >= len(code):
            return None
        new_code = [c for i, c in enumerate(code) if i not in (idx1, idx2)] + found
        new_set = (code_set - {r1, r2}) | set(found)
        final_cov = temp_cov.copy()
        for c in found:
            final_cov[balls[c]] += 1
        return (new_code, new_set, final_cov)

    # ---------- Main construction ----------
    best_code = None
    best_size = float('inf')
    max_restarts = 14
    for restart in range(max_restarts):
        if time.time() > deadline - 50:  # reserve time for local search
            break
        seed = 42 + restart * 137
        cod = greedy_construct(seed)
        refined = refine(cod)
        if len(refined) < best_size:
            best_size = len(refined)
            best_code = refined
        if best_size <= 170:
            break

    if best_code is None:
        best_code = refine(greedy_construct(42))
        best_size = len(best_code)

    code = list(best_code)
    code_set = set(code)
    coverage = np.zeros(N, dtype=np.int32)
    for c in code:
        coverage[balls[c]] += 1

    # ---------- Local search phase ----------
    def removal_pass():
        nonlocal code, code_set, coverage
        changed = True
        while changed and time.time() < deadline - 3:
            changed = False
            for c in list(code):
                if is_redundant(c, coverage):
                    code.remove(c)
                    code_set.remove(c)
                    coverage[balls[c]] -= 1
                    changed = True

    removal_pass()

    no_improve = 0
    max_no_improve = 25
    while time.time() < deadline - 3:
        success = False
        for k in (1, 2, 3):
            if time.time() > deadline - 3:
                break
            res = try_kick(k, code, code_set, coverage)
            if res:
                code, code_set, coverage = res
                success = True
                break
        if not success and time.time() < deadline - 3:
            res = try_swap(code, code_set, coverage)
            if res:
                code, code_set, coverage = res
                success = True
        if not success and time.time() < deadline - 3:
            res = try_swap2(code, code_set, coverage)
            if res:
                code, code_set, coverage = res
                success = True
        if success:
            no_improve = 0
            removal_pass()
        else:
            no_improve += 1
            if no_improve > max_no_improve:
                # Perturb: larger kick to escape local minimum
                if len(code) > 6:
                    k = random.randint(4, min(6, len(code)))
                    res = try_kick(k, code, code_set, coverage)
                    if res:
                        code, code_set, coverage = res
                        no_improve = 0
                        removal_pass()
                    else:
                        no_improve = max_no_improve // 2
                else:
                    break

    # ---------- Final high-precision polish ----------
    removal_pass()
    if time.time() < deadline - 8 and len(code) > 175:
        random.seed(999)
        for _ in range(200):
            if time.time() > deadline - 8:
                break
            improved = False
            for k in (1, 2):
                if time.time() > deadline - 8:
                    break
                res = try_kick(k, code, code_set, coverage)
                if res:
                    code, code_set, coverage = res
                    improved = True
                    break
            if not improved and time.time() < deadline - 8:
                res = try_swap(code, code_set, coverage)
                if res:
                    code, code_set, coverage = res
                    improved = True
            if improved:
                removal_pass()

    final_code = refine(code)

    # ---------- Final exact verification (safety) ----------
    # Verify every word is within radius of some chosen word; if not, repair greedily.
    cov_final = np.zeros(N, dtype=np.int32)
    for c in final_code:
        cov_final[balls[c]] += 1
    if np.any(cov_final == 0):
        # Repair: add codewords to cover missing words
        missing = set(np.where(cov_final == 0)[0].tolist())
        final_set = set(final_code)
        while missing and time.time() < deadline:
            pool = set()
            for w in missing:
                pool.update(cover_map[w])
            pool.difference_update(final_set)
            best_c = -1
            best_count = 0
            for c in pool:
                cnt = 0
                Bc = B[c]
                for w in missing:
                    if Bc[w]:
                        cnt += 1
                if cnt > best_count:
                    best_count = cnt
                    best_c = c
                    if best_count == len(missing):
                        break
            if best_c == -1:
                break
            final_code.append(best_c)
            final_set.add(best_c)
            missing.difference_update(balls[best_c])

    return [idx_to_tuple(c) for c in final_code]
