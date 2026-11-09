import numpy as np
import time
import random


def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p  # 49
    k = 3  # alpha^2 = 3
    n_points = q * q  # 2401

    # Directions: (1, m) for m in F_q (49), and (0, 1) (1) => 50 total
    # Each direction: (vx_a, vx_b, vy_a, vy_b)
    directions = []
    for ma in range(p):
        for mb in range(p):
            directions.append((1, 0, ma, mb))
    directions.append((0, 0, 0, 1))
    n_dirs = len(directions)  # 50

    # Field element: index = a * p + b, where element = a + b*alpha
    def fe_index(a, b):
        return a * p + b

    def fe_coords(idx):
        return (idx // p) % p, idx % p

    # Point: index = xa * 343 + xb * 49 + ya * 7 + yb
    def pt_index(xa, xb, ya, yb):
        return xa * 343 + xb * 49 + ya * 7 + yb

    def pt_coords(idx):
        xa = (idx // 343) % p
        xb = (idx // 49) % p
        ya = (idx // 7) % p
        yb = idx % p
        return xa, xb, ya, yb

    # Precompute field multiplication: fmul[a, b] = (c0, c1)
    fmul = np.zeros((q, q, 2), dtype=np.int8)
    for a in range(q):
        a0, a1 = fe_coords(a)
        for b in range(q):
            b0, b1 = fe_coords(b)
            c0 = (a0 * b0 + k * a1 * b1) % p
            c1 = (a0 * b1 + a1 * b0) % p
            fmul[a, b, 0] = c0
            fmul[a, b, 1] = c1

    # Precompute line members
    line_members = np.zeros((n_points, n_dirs, q - 1), dtype=np.int32)

    for i in range(n_points):
        xa, xb, ya, yb = pt_coords(i)
        for d in range(n_dirs):
            vx_a, vx_b, vy_a, vy_b = directions[d]
            vx_idx = fe_index(vx_a, vx_b)
            vy_idx = fe_index(vy_a, vy_b)
            for t_idx in range(1, q):
                # t * vx
                tvx_a = fmul[t_idx, vx_idx, 0]
                tvx_b = fmul[t_idx, vx_idx, 1]
                # t * vy
                tvy_a = fmul[t_idx, vy_idx, 0]
                tvy_b = fmul[t_idx, vy_idx, 1]
                nxa = (xa + tvx_a) % p
                nxb = (xb + tvx_b) % p
                nya = (ya + tvy_a) % p
                nyb = (yb + tvy_b) % p
                j = pt_index(nxa, nxb, nya, nyb)
                line_members[i, d, t_idx - 1] = j

    # Initial construction
    y_special = set()
    for u in range(p):
        for v in range(p):
            y_special.add((u * u % p, v))

    S = np.zeros(n_points, dtype=bool)
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    in_unital = (2 * xa + ya * ya - 3 * yb * yb) % p == 0
                    in_y_special = (ya, yb) in y_special
                    if in_unital and not in_y_special:
                        idx = pt_index(xa, xb, ya, yb)
                        S[idx] = True

    is_valid = np.ones((n_points, n_dirs), dtype=bool)
    valid_dirs = np.zeros(n_points, dtype=np.int32)

    for x in range(n_points):
        for d in range(n_dirs):
            if np.any(S[line_members[x, d, :]]):
                is_valid[x, d] = False
            else:
                valid_dirs[x] += 1

    start_time = time.time()
    deadline = start_time + 580

    def try_add(p):
        if S[p]:
            return False
        for d in range(n_dirs):
            members = line_members[p, d, :]
            mask = is_valid[members, d]
            if np.any(mask):
                xv = members[mask]
                if np.any(valid_dirs[xv] == 1):
                    return False
        S[p] = True
        for d in range(n_dirs):
            members = line_members[p, d, :]
            mask = is_valid[members, d]
            if np.any(mask):
                xv = members[mask]
                is_valid[xv, d] = False
                valid_dirs[xv] -= 1
        return True

    # Pass 1: forward
    for p in range(n_points):
        if time.time() > deadline:
            break
        try_add(p)

    # Pass 2: backward
    for p in range(n_points - 1, -1, -1):
        if time.time() > deadline:
            break
        try_add(p)

    # Pass 3: random
    if time.time() < deadline - 5:
        order = list(range(n_points))
        random.shuffle(order)
        for p in order:
            if time.time() > deadline:
                break
            try_add(p)

    # Pass 4: another random
    if time.time() < deadline - 3:
        order = list(range(n_points))
        random.shuffle(order)
        for p in order:
            if time.time() > deadline:
                break
            try_add(p)

    N_indices = np.where(~S)[0]
    result = np.zeros((len(N_indices), 2, 2), dtype=np.int64)
    for idx, i in enumerate(N_indices):
        xa, xb, ya, yb = pt_coords(i)
        result[idx, 0, 0] = xa
        result[idx, 0, 1] = xb
        result[idx, 1, 0] = ya
        result[idx, 1, 1] = yb

    return result
