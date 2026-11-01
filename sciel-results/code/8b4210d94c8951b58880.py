import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    if q == 3 and n == 6 and radius == 1:
        return solve_3_6_1()
    return _generic_greedy(q, n, radius)

def _generic_greedy(q, n, radius):
    all_words = list(itertools.product(range(q), repeat=n))
    N = len(all_words)
    if N > 100000:
        return [list(w) for w in all_words]
    idx = {w: i for i, w in enumerate(all_words)}
    spheres = []
    for i, w in enumerate(all_words):
        sp = [i]
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                nw = list(w)
                nw[pos] = val
                sp.append(idx[tuple(nw)])
        spheres.append(sp)
    covered = [False] * N
    cw = []
    for _ in range(N):
        if all(covered):
            break
        best_i, best_g = -1, -1
        for c in range(N):
            g = sum(1 for p in spheres[c] if not covered[p])
            if g > best_g:
                best_g = g
                best_i = c
        if best_i < 0:
            break
        cw.append(best_i)
        for p in spheres[best_i]:
            if not covered[p]:
                covered[p] = True
    return [list(all_words[c]) for c in cw]

def solve_3_6_1():
    t0 = time.time()
    HARD_LIMIT = t0 + 140.0
    q, n, r = 3, 6, 1
    N = q ** n
    all_words = list(itertools.product(range(q), repeat=n))
    idx = {w: i for i, w in enumerate(all_words)}

    spheres = []
    for i, w in enumerate(all_words):
        sp = [i]
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                nw = list(w)
                nw[pos] = val
                sp.append(idx[tuple(nw)])
        spheres.append(sp)

    sphere_arr = np.array(spheres, dtype=np.int32)
    n_spheres, sph_size = sphere_arr.shape

    def greedy_cover(seed):
        rng = np.random.default_rng(seed)
        covered = np.zeros(N, dtype=np.int32)
        remaining = N
        cw = []
        max_iter = min(200, N)
        for it in range(max_iter):
            if time.time() >= HARD_LIMIT - 5:
                break
            if remaining <= 0:
                break
            gain = np.zeros(n_spheres, dtype=np.int32)
            for k in range(sph_size):
                col = sphere_arr[:, k]
                mask = covered[col] == 0
                np.add.at(gain, col[mask], 1)
            maxg = int(gain.max())
            if maxg <= 0:
                break
            cands = np.where(gain == maxg)[0]
            chosen = int(rng.choice(cands))
            cw.append(chosen)
            newpts = sphere_arr[chosen]
            for p in newpts:
                if covered[p] == 0:
                    covered[p] = 1
                    remaining -= 1
        return cw, covered, remaining

    def is_minimal(cw_indices, covered_vec):
        for i in range(len(cw_indices)):
            c = cw_indices[i]
            # remove c's sphere from covered
            tmp = covered_vec.copy()
            tmp[sphere_arr[c]] = 0
            # but we need to re-add spheres of all other codewords
            # Actually we need to check if all points are still covered by the rest
            # Rebuild coverage without c
            cov2 = np.zeros(N, dtype=np.int32)
            for j in range(len(cw_indices)):
                if j != i:
                    cov2[sphere_arr[cw_indices[j]]] = 1
            if cov2.sum() < N:
                # c is needed, keep it
                continue
            # c is redundant, remove it
            cw_indices = cw_indices[:i] + cw_indices[i+1:]
            covered_vec = cov2
            return is_minimal(cw_indices, covered_vec)
        return cw_indices

    def local_search_reduce(cw_indices, seed):
        rng = random.Random(seed)
        cw = list(cw_indices)
        improved = True
        while improved:
            if time.time() >= HARD_LIMIT - 3:
                break
            improved = False
            rng.shuffle(cw)
            for i in range(len(cw)):
                if time.time() >= HARD_LIMIT - 3:
                    break
                # Try removing cw[i]
                test_cw = [c for j, c in enumerate(cw) if j != i]
                cov = np.zeros(N, dtype=np.int32)
                for c in test_cw:
                    cov[sphere_arr[c]] = 1
                if cov.sum() == N:
                    cw = test_cw
                    improved = True
                    break
        return cw

    def swap_improve(cw_indices, seed):
        """Try to replace one codeword with another that covers more."""
        rng = random.Random(seed)
        cw = list(cw_indices)
        improved = True
        while improved:
            if time.time() >= HARD_LIMIT - 3:
                break
            improved = False
            rng.shuffle(cw)
            for i in range(len(cw)):
                if time.time() >= HARD_LIMIT - 3:
                    break
                current = cw[i]
                # Find points only covered by current
                cov_without = np.zeros(N, dtype=np.int32)
                for j, c in enumerate(cw):
                    if j != i:
                        cov_without[sphere_arr[c]] = 1
                only_by_current = (cov_without == 0)
                # We need a replacement that covers all points in only_by_current
                # and that is not already in cw
                target_set = set(np.where(only_by_current)[0].tolist())
                if not target_set:
                    continue
                best_repl = None
                best_extra = -1
                for c in range(n_spheres):
                    if c in cw:
                        continue
                    sp = set(sphere_arr[c].tolist())
                    if not target_set.issubset(sp):
                        continue
                    extra = len(sp - set(np.where(cov_without > 0)[0].tolist()))
                    if extra > best_extra:
                        best_extra = extra
                        best_repl = c
                if best_repl is not None:
                    cw[i] = best_repl
                    improved = True
                    break
        return cw

    best_cw = None
    best_size = N + 1

    seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    for s in seeds:
        if time.time() >= HARD_LIMIT - 20:
            break
        cw, covered, remaining = greedy_cover(s)
        if remaining > 0:
            # Fallback: add remaining uncovered points as codewords
            unc = np.where(covered == 0)[0]
            for u in unc:
                cw.append(u)
                covered[sphere_arr[u]] = 1
        cw = list(dict.fromkeys(cw))  # deduplicate preserving order

        # Local search to remove redundancies
        cw = local_search_reduce(cw, s)

        if len(cw) < best_size:
            best_size = len(cw)
            best_cw = cw

    # Final validation
    if best_cw is None:
        # Fallback: return all words
        return [list(w) for w in all_words]

    cov = np.zeros(N, dtype=np.int32)
    for c in best_cw:
        cov[sphere_arr[c]] = 1
    if cov.sum() != N:
        # Add missing
        unc = np.where(cov == 0)[0]
        for u in unc:
            best_cw.append(u)
            cov[sphere_arr[u]] = 1

    return [list(all_words[c]) for c in best_cw]
