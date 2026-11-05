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

    # ---- Phase 2: Local search (remove & repair) ----
    rng = np.random.default_rng(42)
    no_improve = 0
    max_no_improve = 2000

    while time.time() < deadline - 3.0 and no_improve < max_no_improve:
        if not code:
            break

        cov = np.zeros(total, dtype=np.int32)
        for idx in code:
            cov[nbr[idx]] += 1

        i = int(rng.integers(len(code)))
        c_remove = code[i]
        code.pop(i)
        cov[nbr[c_remove]] -= 1

        uncovered_mask = (cov == 0)

        if not np.any(uncovered_mask):
            code = try_reduce(code)
            if len(code) < best_size:
                best_size = len(code)
                best_code = code[:]
                no_improve = 0
            else:
                no_improve += 1
            continue

        # Repair: find best replacement
        counts_add = np.sum(uncovered_mask[nbr], axis=1)
        best_add = int(np.argmax(counts_add))

        current_set = set(code)
        if best_add not in current_set:
            code.append(best_add)
            cov[nbr[best_add]] += 1
            code = try_reduce(code)
            if len(code) < best_size:
                best_size = len(code)
                best_code = code[:]
                no_improve = 0
            else:
                no_improve += 1
        else:
            # best_add already in code; revert
            code.append(c_remove)
            cov[nbr[c_remove]] += 1
            no_improve += 1

    # ---- Phase 3: Multiple random restarts of local search ----
    if time.time() < deadline - 10.0:
        for restart in range(10):
            if time.time() > deadline - 5.0:
                break
            # Perturb: swap 2-4 random codewords
            perturbed = best_code[:]
            for _ in range(rng.integers(2, 5)):
                if len(perturbed) < 2:
                    break
                a = int(rng.integers(len(perturbed)))
                b = int(rng.integers(len(perturbed)))
                if a == b:
                    continue
                # Replace perturbed[a] with a random codeword
                new_cw = int(rng.integers(total))
                if new_cw not in perturbed:
                    perturbed[a] = new_cw
            # Verify validity
            cov = np.zeros(total, dtype=np.int8)
            for idx in perturbed:
                cov[nbr[idx]] = 1
            if not np.all(cov == 1):
                # Invalid perturbation; try to fix by greedy fill
                uncovered_mask = (cov == 0)
                while np.any(uncovered_mask):
                    if time.time() > deadline - 5.0:
                        break
                    counts = np.sum(uncovered_mask[nbr], axis=1)
                    best = int(np.argmax(counts))
                    if counts[best] == 0:
                        break
                    if best not in perturbed:
                        perturbed.append(best)
                        cov[nbr[best]] = 1
                if not np.all(cov == 1):
                    continue
            # Local search on perturbed
            code_ps = perturbed[:]
            no_imp_ps = 0
            while time.time() < deadline - 5.0 and no_imp_ps < 500:
                if not code_ps:
                    break
                cov_ps = np.zeros(total, dtype=np.int32)
                for idx in code_ps:
                    cov_ps[nbr[idx]] += 1
                idx_rm = int(rng.integers(len(code_ps)))
                cr = code_ps[idx_rm]
                code_ps.pop(idx_rm)
                cov_ps[nbr[cr]] -= 1
                um = (cov_ps == 0)
                if not np.any(um):
                    code_ps = try_reduce(code_ps)
                    if len(code_ps) < best_size:
                        best_size = len(code_ps)
                        best_code = code_ps[:]
                        no_improve = 0
                    no_imp_ps += 1
                    continue
                ca = np.sum(um[nbr], axis=1)
                ba = int(np.argmax(ca))
                cs = set(code_ps)
                if ba not in cs:
                    code_ps.append(ba)
                    cov_ps[nbr[ba]] += 1
                    code_ps = try_reduce(code_ps)
                    if len(code_ps) < best_size:
                        best_size = len(code_ps)
                        best_code = code_ps[:]
                        no_improve = 0
                    no_imp_ps += 1
                else:
                    code_ps.append(cr)
                    cov_ps[nbr[cr]] += 1
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
