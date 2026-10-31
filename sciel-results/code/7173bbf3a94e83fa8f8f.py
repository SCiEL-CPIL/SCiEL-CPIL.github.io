import numpy as np
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110

    D = 11
    supports = list(itertools.combinations(range(D), 4))
    n_supports = len(supports)
    sign_options = list(itertools.product([1, -1], repeat=4))
    n_signs = len(sign_options)
    n_vectors = n_supports * n_signs

    # Precompute restriction tables: for each support, for each 3-subset of local positions,
    # map pattern (0-7) to list of sign indices having that pattern
    si_restriction = [None] * n_supports
    for i in range(n_supports):
        d = {}
        for subset in itertools.combinations(range(4), 3):
            subset = tuple(subset)
            p_dict = {p: [] for p in range(8)}
            for sj in range(n_signs):
                sign = sign_options[sj]
                p = 0
                for idx, pos in enumerate(subset):
                    if sign[pos] == 1:
                        p |= (1 << idx)
                p_dict[p].append(sj)
            d[subset] = p_dict
        si_restriction[i] = d

    # Precompute support conflicts: pairs of supports intersecting in exactly 3 coordinates
    support_conflicts = [[] for _ in range(n_supports)]
    for i in range(n_supports):
        Si = set(supports[i])
        for j in range(i + 1, n_supports):
            Sj = set(supports[j])
            inter = Si & Sj
            if len(inter) == 3:
                inter_sorted = tuple(sorted(inter))
                pos_i = tuple(supports[i].index(k) for k in inter_sorted)
                pos_j = tuple(supports[j].index(k) for k in inter_sorted)
                support_conflicts[i].append((j, pos_i, pos_j))
                support_conflicts[j].append((i, pos_j, pos_i))

    # Precompute pattern cache
    subset_list = list(itertools.combinations(range(4), 3))
    subset_to_idx = {tuple(s): i for i, s in enumerate(subset_list)}

    pattern_cache = [None] * n_supports
    for i in range(n_supports):
        pc = [None] * n_signs
        for sj in range(n_signs):
            sign = sign_options[sj]
            pats = []
            for subset in subset_list:
                p = 0
                for idx, pos in enumerate(subset):
                    if sign[pos] == 1:
                        p |= (1 << idx)
                pats.append(p)
            pc[sj] = tuple(pats)
        pattern_cache[i] = pc

    # Greedy with random restarts
    best_vec_arr = None
    best_count = 0

    for trial in range(300):
        if time.time() > deadline - 5:
            break

        all_vecs = np.random.permutation(n_vectors)
        selected_mask = [0] * n_supports
        selected_count = 0

        for vidx in all_vecs:
            si = vidx // n_signs
            sj = vidx % n_signs

            conflict = False
            for (sj2, pos_i, pos_j) in support_conflicts[si]:
                idx_i = subset_to_idx[pos_i]
                idx_j = subset_to_idx[pos_j]
                p = pattern_cache[si][sj][idx_i]

                conflicting_sjs = si_restriction[sj2][pos_j][p]
                mask2 = selected_mask[sj2]
                for sj3 in conflicting_sjs:
                    if mask2 & (1 << sj3):
                        conflict = True
                        break
                if conflict:
                    break

            if not conflict:
                selected_mask[si] |= (1 << sj)
                selected_count += 1

        if selected_count > best_count:
            best_count = selected_count
            vectors = []
            for si in range(n_supports):
                mask = selected_mask[si]
                for sj in range(n_signs):
                    if mask & (1 << sj):
                        v = [0] * D
                        for k in range(4):
                            v[supports[si][k]] = sign_options[sj][k]
                        vectors.append(v)
            best_vec_arr = np.array(vectors, dtype=np.int64)

    if best_vec_arr is None:
        return np.array([], dtype=np.int64).reshape(0, D)

    # Verify
    norms = np.linalg.norm(best_vec_arr, axis=1)
    max_norm = np.max(norms)
    if len(best_vec_arr) > 1:
        dists = np.linalg.norm(best_vec_arr[:, np.newaxis] - best_vec_arr[np.newaxis, :], axis=2)
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
        if min_dist < max_norm - 1e-9:
            return np.array([], dtype=np.int64).reshape(0, D)

    return best_vec_arr
