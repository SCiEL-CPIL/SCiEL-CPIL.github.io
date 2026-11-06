import numpy as np
import time

N = 64

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 280

    # Baseline: 1D Sidon-like set
    xs, used = [], set()
    x = 0
    while x < N:
        diffs = {abs(x - y) for y in xs}
        if not (diffs & used):
            used |= diffs
            xs.append(x)
        x += 1
    best_set = [(x, 0) for x in xs]

    all_points = np.array([(x, y) for x in range(N) for y in range(N)], dtype=np.int32)

    def is_valid_addition(p, pts):
        """pts is a numpy array of shape (k, 2)"""
        k = len(pts)
        if k < 2:
            return True
        p64 = p.astype(np.int64)
        pts64 = pts.astype(np.int64)
        diffs = pts64 - p64
        d_p = np.sum(diffs ** 2, axis=1)
        i_idx, j_idx = np.triu_indices(k, k=1)
        d_p_i = d_p[i_idx]
        d_p_j = d_p[j_idx]
        d_ij = np.sum((pts64[i_idx] - pts64[j_idx]) ** 2, axis=1)
        if np.any(d_p_i == d_p_j) or np.any(d_p_i == d_ij) or np.any(d_p_j == d_ij):
            return False
        return True

    num_restarts = 50
    for restart in range(num_restarts):
        if time.time() > deadline:
            break

        perm = np.random.permutation(len(all_points))
        current_list = []
        current_arr = np.empty((0, 2), dtype=np.int32)

        for idx in perm:
            if time.time() > deadline:
                break
            p = all_points[idx]
            if is_valid_addition(p, current_arr):
                current_list.append((int(p[0]), int(p[1])))
                if len(current_arr) == 0:
                    current_arr = p.reshape(1, 2).copy()
                else:
                    current_arr = np.vstack([current_arr, p])

        if len(current_list) > len(best_set):
            best_set = current_list[:]

    return best_set
# EVOLVE-BLOCK-END
