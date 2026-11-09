# EVOLVE-BLOCK-START
import numpy as np
import time
import heapq
import random


def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 (q = 49, p = 7).
    Uses priority-queue-based greedy removal to maximize |E| where N = F_49^2 \\ E
    is a valid Nikodym set.
    """
    p = 7
    q = p * p  # 49
    total_points = q * q  # 2401

    deadline = time.time() + 570.0

    k_const = 3
    idx_to_elem = [(i // p, i % p) for i in range(q)]

    mul_table = np.zeros((q, q), dtype=np.int32)
    for i in range(q):
        ai, bi = idx_to_elem[i]
        for j in range(q):
            aj, bj = idx_to_elem[j]
            aa = (ai * aj + bi * bj * k_const) % p
            bb = (ai * bj + bi * aj) % p
            mul_table[i, j] = aa * p + bb

    add_table = np.zeros((q, q), dtype=np.int32)
    for i in range(q):
        ai, bi = idx_to_elem[i]
        for j in range(q):
            aj, bj = idx_to_elem[j]
            aa = (ai + aj) % p
            bb = (bi + bj) % p
            add_table[i, j] = aa * p + bb

    directions = [(0, 1)]
    for m in range(q):
        directions.append((1, m))
    num_dirs = len(directions)

    zx_arr = np.array([z // q for z in range(total_points)], dtype=np.int32)
    zy_arr = np.array([z % q for z in range(total_points)], dtype=np.int32)
    t_vals = np.arange(1, q, dtype=np.int32)

    line_other = np.zeros((total_points, num_dirs, q - 1), dtype=np.int32)
    for v_idx, (dx, dy) in enumerate(directions):
        tdx = mul_table[t_vals, dx]
        tdy = mul_table[t_vals, dy]
        nx = add_table[zx_arr[:, None], tdx[None, :]]
        ny = add_table[zy_arr[:, None], tdy[None, :]]
        line_other[:, v_idx, :] = nx * q + ny

    best_E = np.zeros(total_points, dtype=bool)
    best_size = 0
    num_restarts = 30

    for restart in range(num_restarts):
        if time.time() > deadline:
            break

        rng = random.Random(12345 + restart)
        in_E = np.zeros(total_points, dtype=bool)
        cnt = np.zeros((total_points, num_dirs), dtype=np.int32)
        cover = np.full(total_points, num_dirs, dtype=np.int32)

        heap = [(0, rng.random(), z) for z in range(total_points)]
        heapq.heapify(heap)
        size = 0
        removed = np.zeros(total_points, dtype=bool)

        while heap and time.time() < deadline:
            cost, _, z = heapq.heappop(heap)

            if removed[z]:
                continue

            pts_all = line_other[z]
            pts_flat = pts_all.reshape(-1)
            v_flat = np.repeat(np.arange(num_dirs), q - 1)
            mask = (cover[pts_flat] == 1) & (cnt[pts_flat, v_flat] == 0)
            if np.any(mask):
                continue

            if cover[z] == 0:
                continue

            in_E[z] = True
            removed[z] = True
            size += 1

            for v in range(num_dirs):
                pts = pts_all[v]
                old_cnt = cnt[pts, v]
                cnt[pts, v] += 1
                dec_mask = (old_cnt == 0)
                if np.any(dec_mask):
                    cover[pts[dec_mask]] -= 1

            unique_affected = np.unique(pts_flat)
            unique_affected = unique_affected[~removed[unique_affected]]
            for x in unique_affected:
                x_pts = line_other[x]
                x_pts_flat = x_pts.reshape(-1)
                x_v_flat = np.repeat(np.arange(num_dirs), q - 1)
                x_mask = (cover[x_pts_flat] == 1) & (cnt[x_pts_flat, x_v_flat] == 0)
                new_cost = int(np.sum(x_mask))
                heapq.heappush(heap, (new_cost, rng.random(), int(x)))

        if size > best_size:
            best_size = size
            best_E = in_E.copy()

    points_list = []
    for pidx in range(total_points):
        if not best_E[pidx]:
            xi = pidx // q
            yi = pidx % q
            xa, xb = idx_to_elem[xi]
            ya, yb = idx_to_elem[yi]
            points_list.append([[xa, xb], [ya, yb]])

    if not points_list:
        for pidx in range(total_points):
            xi = pidx // q
            yi = pidx % q
            xa, xb = idx_to_elem[xi]
            ya, yb = idx_to_elem[yi]
            points_list.append([[xa, xb], [ya, yb]])

    return np.array(points_list, dtype=np.int64)


# EVOLVE-BLOCK-END
