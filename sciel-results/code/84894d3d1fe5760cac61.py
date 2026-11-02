"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import itertools
import numpy as np
import time


def construct():
    start_time = time.time()
    DEADLINE = 140.0  # seconds, safety margin before 150s kill

    def check_time():
        return time.time() < DEADLINE

    # Generate D11 roots: (±1, ±1, 0, ..., 0)
    d_roots = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = si, sj
                d_roots.append(v)

    vectors = list(d_roots)
    max_norm_sq = 2

    # Try to add B11 roots: ±e_i
    for i in range(D):
        if not check_time():
            break
        for s in (1, -1):
            if not check_time():
                break
            cand = [0] * D
            cand[i] = s
            cand_arr = np.array(cand)
            cand_norm_sq = 1
            ok = True
            for v in vectors:
                v_arr = np.array(v)
                diff = cand_arr - v_arr
                dist_sq = int(np.dot(diff, diff))
                if dist_sq < max_norm_sq:
                    ok = False
                    break
            if ok:
                vectors.append(cand)

    # Try to add weight-4 vectors (norm^2 = 4)
    for comb in itertools.combinations(range(D), 4):
        if not check_time():
            break
        for signs in itertools.product([1, -1], repeat=4):
            if not check_time():
                break
            cand = [0] * D
            for idx, s in zip(comb, signs):
                cand[idx] = s
            cand_arr = np.array(cand)
            cand_norm_sq = 4
            ok = True
            for v in vectors:
                v_arr = np.array(v)
                diff = cand_arr - v_arr
                dist_sq = int(np.dot(diff, diff))
                if dist_sq < max(max_norm_sq, cand_norm_sq):
                    ok = False
                    break
            if ok:
                vectors.append(cand)
                max_norm_sq = max(max_norm_sq, cand_norm_sq)

    return np.array(vectors, dtype=np.int64)

# EVOLVE-BLOCK-END
