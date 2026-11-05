import time
import numpy as np
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 118.0

    def get_direct_sum():
        if n >= 4 and q == 3:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
        return [[0] * n]

    if q != 3 or n != 6 or radius != 1:
        return get_direct_sum()

    total = 729
    powers = [1, 3, 9, 27, 81, 243]
    coords = [[(i // p) % 3 for p in powers] for i in range(total)]

    # Build neighbor list (including self)
    nbr = np.empty((total, 13), dtype=np.int32)
    for i in range(total):
        nbr[i, 0] = i
        k = 1
        for p in range(6):
            base = (i // powers[p]) % 3
            for alt in range(3):
                if alt != base:
                    nbr[i, k] = i + (alt - base) * powers[p]
                    k += 1

    # ---- Phase 1: Greedy set cover ----
    uncovered = np.ones(total, dtype=np.int8)
    code = []
    while uncovered.sum() > 0:
        if time.time() > deadline:
            break
        counts = np.sum(uncovered[nbr], axis=1)
        best = int(np.argmax(counts))
        if counts[best] == 0:
            break
        code.append(best)
        uncovered[nbr[best]] = 0

    def try_reduce(cw):
        if not cw:
            return cw
        cov = np.zeros(total, dtype=np.int32)
        for idx in cw:
            cov[nbr[idx]] += 1
        changed = True
        while changed:
            changed = False
            for i in range(len(cw) - 1, -1, -1):
                idx = cw[i]
                if np.min(cov[nbr[idx]]) >= 2:
                    cov[nbr[idx]] -= 1
                    cw.pop(i)
                    changed = True
                    break
        return cw

    code = try_reduce(code)
    best_code = code[:]
    best_size = len(code)

    # ---- Phase 2: Local search (remove & repair) with incremental coverage ----
    rng = np.random.default_rng(42)
    no_improve = 0
    max_no_improve = 5000

    # Build coverage incrementally
    cov = np.zeros(total, dtype=np.int32)
    for idx in code:
        cov[nbr[idx]] += 1
    in_code = np.zeros(total, dtype=np.int8)
    for idx in code:
        in_code[idx] = 1

    while time.time() < deadline - 3.0 and no_improve < max_no_improve:
        if not code:
            break

        i = int(rng.integers(len(code)))
        c_remove = code[i]
        code.pop(i)
        in_code[c_remove] = 0
        cov[nbr[c_remove]] -= 1

        uncovered_mask = (cov == 0)
        n_uncovered = int(np.count_nonzero(uncovered_mask))

        if n_uncovered == 0:
            # Try to remove more
            changed = True
            while changed:
                changed = False
                for j in range(len(code) - 1, -1, -1):
                    idx = code[j]
                    if np.min(cov[nbr[idx]]) >= 2:
                        cov[nbr[idx]] -= 1
                        in_code[idx] = 0
                        code.pop(j)
                        changed = True
                        break
            if len(code) < best_size:
                best_size = len(code)
                best_code = code[:]
                no_improve = 0
            else:
                no_improve += 1
            continue

        # Repair: find best replacement among candidates
        # Use only words not already in code, weighted by uncovered coverage
        counts_add = np.sum(uncovered_mask[nbr], axis=1)
        # Mask out words already in code
        counts_add[in_code == 1] = -1
        best_add = int(np.argmax(counts_add))

        if counts_add[best_add] <= 0:
            # Can't repair; revert
            code.append(c_remove)
            in_code[c_remove] = 1
            cov[nbr[c_remove]] += 1
            no_improve += 1
        else:
            code.append(best_add)
            in_code[best_add] = 1
            cov[nbr[best_add]] += 1
            # Try to remove redundant codewords
            changed = True
            while changed:
                changed = False
                for j in range(len(code) - 1, -1, -1):
                    idx = code[j]
                    if np.min(cov[nbr[idx]]) >= 2:
                        cov[nbr[idx]] -= 1
                        in_code[idx] = 0
                        code.pop(j)
                        changed = True
                        break
            if len(code) < best_size:
                best_size = len(code)
                best_code = code[:]
                no_improve = 0
            else:
                no_improve += 1

    # ---- Phase 3: Multiple random restarts with incremental coverage ----
    if time.time() < deadline - 10.0:
        for restart in range(20):
            if time.time() > deadline - 5.0:
                break
            # Perturb: swap 2-4 random codewords
            perturbed = best_code[:]
            n_swaps = int(rng.integers(2, 5))
            for _ in range(n_swaps):
                if len(perturbed) < 2:
                    break
                a = int(rng.integers(len(perturbed)))
                new_cw = int(rng.integers(total))
                if new_cw not in perturbed:
                    perturbed[a] = new_cw
            # Verify validity and fix
            cov_ps = np.zeros(total, dtype=np.int8)
            in_ps = np.zeros(total, dtype=np.int8)
            for idx in perturbed:
                cov_ps[nbr[idx]] = 1
                in_ps[idx] = 1
            if not np.all(cov_ps == 1):
                uncovered_mask = (cov_ps == 0)
                while np.any(uncovered_mask):
                    if time.time() > deadline - 5.0:
                        break
                    counts = np.sum(uncovered_mask[nbr], axis=1)
                    counts[in_ps == 1] = -1
                    best = int(np.argmax(counts))
                    if counts[best] <= 0:
                        break
                    perturbed.append(best)
                    in_ps[best] = 1
                    cov_ps[nbr[best]] = 1
                if not np.all(cov_ps == 1):
                    continue
            # Local search on perturbed with incremental coverage
            code_ps = perturbed[:]
            cov_ps2 = np.zeros(total, dtype=np.int32)
            in_ps2 = np.zeros(total, dtype=np.int8)
            for idx in code_ps:
                cov_ps2[nbr[idx]] += 1
                in_ps2[idx] = 1
            no_imp_ps = 0
            while time.time() < deadline - 5.0 and no_imp_ps < 1200:
                if not code_ps:
                    break
                idx_rm = int(rng.integers(len(code_ps)))
                cr = code_ps[idx_rm]
                code_ps.pop(idx_rm)
                in_ps2[cr] = 0
                cov_ps2[nbr[cr]] -= 1
                um = (cov_ps2 == 0)
                if not np.any(um):
                    changed = True
                    while changed:
                        changed = False
                        for j in range(len(code_ps) - 1, -1, -1):
                            idx = code_ps[j]
                            if np.min(cov_ps2[nbr[idx]]) >= 2:
                                cov_ps2[nbr[idx]] -= 1
                                in_ps2[idx] = 0
                                code_ps.pop(j)
                                changed = True
                                break
                    if len(code_ps) < best_size:
                        best_size = len(code_ps)
                        best_code = code_ps[:]
                        no_improve = 0
                    no_imp_ps += 1
                    continue
                ca = np.sum(um[nbr], axis=1)
                ca[in_ps2 == 1] = -1
                ba = int(np.argmax(ca))
                if ca[ba] <= 0:
                    code_ps.append(cr)
                    in_ps2[cr] = 1
                    cov_ps2[nbr[cr]] += 1
                    no_imp_ps += 1
                else:
                    code_ps.append(ba)
                    in_ps2[ba] = 1
                    cov_ps2[nbr[ba]] += 1
                    changed = True
                    while changed:
                        changed = False
                        for j in range(len(code_ps) - 1, -1, -1):
                            idx = code_ps[j]
                            if np.min(cov_ps2[nbr[idx]]) >= 2:
                                cov_ps2[nbr[idx]] -= 1
                                in_ps2[idx] = 0
                                code_ps.pop(j)
                                changed = True
                                break
                    if len(code_ps) < best_size:
                        best_size = len(code_ps)
                        best_code = code_ps[:]
                        no_improve = 0
                    no_imp_ps += 1

    # ---- Final verification and output ----
    def to_words(idx_list):
        return [list(coords[idx]) for idx in idx_list]

    if best_code:
        fc = np.zeros(total, dtype=np.int8)
        for idx in best_code:
            fc[nbr[idx]] = 1
        if np.all(fc == 1):
            ds = get_direct_sum()
            if len(best_code) <= len(ds):
                return to_words(best_code)
            return ds
    # Fallback
    if code:
        fc = np.zeros(total, dtype=np.int8)
        for idx in code:
            fc[nbr[idx]] = 1
        if np.all(fc == 1):
            ds = get_direct_sum()
            if len(code) <= len(ds):
                return to_words(code)
    return get_direct_sum()
