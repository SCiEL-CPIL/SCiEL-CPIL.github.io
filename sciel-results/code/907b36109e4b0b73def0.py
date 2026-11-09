import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140

    D = 11
    vectors = []

    # E8 type 1: Two +/-2 in first 8 dims
    for i, j in itertools.combinations(range(8), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i] = 2 * si
                v[j] = 2 * sj
                vectors.append(tuple(v))

    if time.time() > deadline:
        return np.array(vectors, dtype=np.int64)

    # E8 type 2: All +/-1 in first 8 dims, even minuses
    for mask in range(256):
        minuses = 0
        v = [0] * D
        for i in range(8):
            if (mask >> i) & 1:
                v[i] = -1
                minuses += 1
            else:
                v[i] = 1
        if minuses % 2 == 0:
            vectors.append(tuple(v))

    if time.time() > deadline:
        return np.array(vectors, dtype=np.int64)

    # D3 type: Two +/-2 in last 3 dims
    for i, j in itertools.combinations(range(3), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[8 + i] = 2 * si
                v[8 + j] = 2 * sj
                vectors.append(tuple(v))

    if time.time() > deadline:
        return np.array(vectors, dtype=np.int64)

    vec_arr = np.array(vectors, dtype=np.int64)
    max_norm_sq = 8

    existing_set = set(vectors)
    candidates = []

    # One +/-2, one +/-1 (norm^2=5)
    for i in range(D):
        for j in range(D):
            if i == j:
                continue
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i] = 2 * si
                    v[j] = sj
                    vt = tuple(v)
                    if vt not in existing_set:
                        candidates.append(vt)

    if time.time() > deadline - 15:
        return np.array(vectors, dtype=np.int64)

    # Two +/-1, one +/-2 (norm^2=6)
    for i in range(D):
        for j in range(D):
            for k in range(D):
                if len({i, j, k}) < 3:
                    continue
                for si in (1, -1):
                    for sj in (1, -1):
                        for sk in (1, -1):
                            v = [0] * D
                            v[i] = 2 * si
                            v[j] = sj
                            v[k] = sk
                            vt = tuple(v)
                            if vt not in existing_set:
                                candidates.append(vt)

    if time.time() > deadline - 8:
        return np.array(vectors, dtype=np.int64)

    # Three +/-1, one +/-2 (norm^2=7)
    for i in range(D):
        for j in range(D):
            for k in range(D):
                for l in range(D):
                    if len({i, j, k, l}) < 4:
                        continue
                    for si in (1, -1):
                        for sj in (1, -1):
                            for sk in (1, -1):
                                for sl in (1, -1):
                                    v = [0] * D
                                    v[i] = 2 * si
                                    v[j] = sj
                                    v[k] = sk
                                    v[l] = sl
                                    vt = tuple(v)
                                    if vt not in existing_set:
                                        candidates.append(vt)

    if time.time() > deadline - 5:
        return np.array(vectors, dtype=np.int64)

    # Vectorized greedy addition
    cand_arr = np.array(candidates, dtype=np.int64)
    n_cand = len(cand_arr)
    batch_size = 1000

    for start in range(0, n_cand, batch_size):
        if time.time() > deadline - 3:
            break
        end = min(start + batch_size, n_cand)
        batch = cand_arr[start:end]

        batch_norms = np.sum(batch**2, axis=1)
        vec_norms = np.sum(vec_arr**2, axis=1)
        dot_products = batch @ vec_arr.T
        dists_sq = batch_norms[:, None] + vec_norms[None, :] - 2 * dot_products

        min_dists = np.min(dists_sq, axis=1)
        valid_mask = min_dists >= max_norm_sq
        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) > 0:
            new_vecs = batch[valid_indices]
            vectors.extend([tuple(v) for v in new_vecs])
            vec_arr = np.vstack([vec_arr, new_vecs])

    return np.array(vectors, dtype=np.int64)
