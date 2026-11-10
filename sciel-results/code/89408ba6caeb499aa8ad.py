# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    p = 7
    q = 49
    alpha_sq = 3

    # Precompute field operations as lookup tables
    # Element representation: elem = a * 7 + b, representing a + b*alpha
    def elem_add(e1, e2):
        a1, b1 = divmod(e1, p)
        a2, b2 = divmod(e2, p)
        return ((a1 + a2) % p) * p + ((b1 + b2) % p)

    def elem_mul(e1, e2):
        a1, b1 = divmod(e1, p)
        a2, b2 = divmod(e2, p)
        ra = (a1 * a2 + b1 * b2 * alpha_sq) % p
        rb = (a1 * b2 + a2 * b1) % p
        return ra * p + rb

    add_table = np.zeros((q, q), dtype=np.int32)
    mul_table = np.zeros((q, q), dtype=np.int32)
    for i in range(q):
        for j in range(q):
            add_table[i, j] = elem_add(i, j)
            mul_table[i, j] = elem_mul(i, j)

    # Directions: 0..48 are slopes m (direction (1, m)), 49 is vertical (0, 1)
    # Precompute for each point idx and direction d, the 48 other points on the line
    line_other = np.zeros((q * q, 50, 48), dtype=np.int32)
    for x in range(q):
        for y in range(q):
            idx = x * q + y
            for d in range(49):  # slope m = d
                for t_idx, t in enumerate(range(1, q)):
                    a_x, b_x = divmod(x, p)
                    a_t, b_t = divmod(t, p)
                    ox = ((a_x + a_t) % p) * p + ((b_x + b_t) % p)
                    td = mul_table[t, d]
                    a_td, b_td = divmod(td, p)
                    a_y, b_y = divmod(y, p)
                    oy = ((a_y + a_td) % p) * p + ((b_y + b_td) % p)
                    line_other[idx, d, t_idx] = ox * q + oy
            for t_idx, t in enumerate(range(1, q)):
                a_t, b_t = divmod(t, p)
                a_y, b_y = divmod(y, p)
                oy = ((a_y + a_t) % p) * p + ((b_y + b_t) % p)
                line_other[idx, 49, t_idx] = x * q + oy

    # Precompute: for each point p and direction d, the 48 points idx such that
    # p is on the punctured line of idx in direction d
    affected = np.zeros((q * q, 50, 48), dtype=np.int32)
    for px in range(q):
        for py in range(q):
            p_idx = px * q + py
            a_px, b_px = divmod(px, p)
            a_py, b_py = divmod(py, p)
            for d in range(49):
                for t_idx, t in enumerate(range(1, q)):
                    a_t, b_t = divmod(t, p)
                    ix = ((a_px - a_t) % p) * p + ((b_px - b_t) % p)
                    td = mul_table[t, d]
                    a_td, b_td = divmod(td, p)
                    iy = ((a_py - a_td) % p) * p + ((b_py - b_td) % p)
                    affected[p_idx, d, t_idx] = ix * q + iy
            for t_idx, t in enumerate(range(1, q)):
                a_t, b_t = divmod(t, p)
                ix = px
                iy = ((a_py - a_t) % p) * p + ((b_py - b_t) % p)
                affected[p_idx, 49, t_idx] = ix * q + iy

    # Build base Nikodym set
    y_special = set()
    for u in range(p):
        for v in range(p):
            y_special.add(((u * u) % p, v))

    N = np.ones(q * q, dtype=bool)
    for xa in range(p):
        for xb in range(p):
            x = xa * p + xb
            for ya in range(p):
                for yb in range(p):
                    y = ya * p + yb
                    in_unital = (2 * xa + ya * ya - yb * yb * alpha_sq) % p == 0
                    in_y_special = (ya, yb) in y_special
                    if in_unital and not in_y_special:
                        N[x * q + y] = False

    # Initialize valid_count
    valid = np.zeros((q * q, 50), dtype=bool)
    valid_count = np.zeros(q * q, dtype=np.int32)
    for idx in range(q * q):
        for d in range(50):
            others = line_other[idx, d, :]
            if np.all(N[others]):
                valid[idx, d] = True
                valid_count[idx] += 1

    assert np.all(valid_count > 0), "Base set is not valid"

    start_time = time.time()
    deadline = start_time + 620  # safety margin

    # Multi-pass randomized local search
    best_N = N.copy()
    best_comp = q * q - N.sum()

    for trial in range(20):
        if time.time() > deadline - 60:
            break
        N = best_N.copy()
        valid = np.zeros((q * q, 50), dtype=bool)
        valid_count = np.zeros(q * q, dtype=np.int32)
        for idx in range(q * q):
            for d in range(50):
                others = line_other[idx, d, :]
                if np.all(N[others]):
                    valid[idx, d] = True
                    valid_count[idx] += 1
        
        # Count how many points have valid_count == 0
        num_invalid = np.sum(valid_count == 0)

        # Shuffle order of points to try
        order = list(range(q * q))
        random.shuffle(order)

        # Multi-pass greedy removal
        changed = True
        while changed and time.time() < deadline - 30:
            changed = False
            for p_idx in order:
                if time.time() > deadline - 30:
                    break
                if not N[p_idx]:
                    continue

                # Try removing p_idx
                changes = []
                for d in range(50):
                    for idx in affected[p_idx, d, :]:
                        if valid[idx, d]:
                            changes.append((idx, d))
                            valid[idx, d] = False
                            old_count = valid_count[idx]
                            valid_count[idx] -= 1
                            if old_count == 1 and valid_count[idx] == 0:
                                num_invalid += 1

                if num_invalid > 0:
                    # Rollback
                    for idx, d in changes:
                        valid[idx, d] = True
                        old_count = valid_count[idx]
                        valid_count[idx] += 1
                        if old_count == 0 and valid_count[idx] == 1:
                            num_invalid -= 1
                else:
                    N[p_idx] = False
                    changed = True
                    comp = q * q - N.sum()
                    if comp > best_comp:
                        best_comp = comp
                        best_N = N.copy()

    N = best_N

    # Build result
    points = []
    for x in range(q):
        for y in range(q):
            if N[x * q + y]:
                xa, xb = divmod(x, p)
                ya, yb = divmod(y, p)
                points.append([[xa, xb], [ya, yb]])

    return np.array(points, dtype=np.int64)

# EVOLVE-BLOCK-END
