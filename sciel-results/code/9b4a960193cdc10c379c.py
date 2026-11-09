# EVOLVE-BLOCK-START
import numpy as np
import time


def nikodym_set() -> np.ndarray:
    """
    Construct a small Nikodym set in F_49^2 using a greedy augmentation of
    the Hermitian unital-based construction.

    The Nikodym condition: for every point x in F_49^2, there exists a direction
    v in {(1,m) : m in F_49} ∪ {(0,1)} such that the punctured line
    {x + t*v : t in F_49, t != 0} is contained in N.

    Equivalently, for x in S (the complement of N), there must exist a direction
    v such that x is the ONLY point of S on the line through x in direction v.
    """
    start_time = time.time()
    deadline = start_time + 600.0
    safety_margin = 30.0

    p = 7
    q = p * p  # 49
    total_points = q * q  # 2401
    n_dirs = q + 1  # 50 directions

    # Field F_49 = F_7[alpha], alpha^2 = k (smallest quadratic non-residue mod 7)
    k = None
    for c in range(2, p):
        if pow(c, (p - 1) // 2, p) == p - 1:
            k = c
            break

    # Field element operations
    def add_el(e1, e2):
        return ((e1[0] + e2[0]) % p, (e1[1] + e2[1]) % p)

    def mul_el(e1, e2):
        a = (e1[0] * e2[0] + e1[1] * e2[1] * k) % p
        b = (e1[0] * e2[1] + e1[1] * e2[0]) % p
        return (a, b)

    def neg_el(e):
        return ((-e[0]) % p, (-e[1]) % p)

    def trace(e):
        return (2 * e[0]) % p

    def norm(e):
        return (e[0] * e[0] - e[1] * e[1] * k) % p

    # Field elements indexed 0..48: (a, b) with index = a*p + b
    field_elements = [(a, b) for a in range(p) for b in range(p)]
    elem_to_idx = {(a, b): a * p + b for a in range(p) for b in range(p)}

    # Precompute tables
    add_table = np.zeros((q, q), dtype=np.int32)
    mul_table = np.zeros((q, q), dtype=np.int32)
    for i in range(q):
        for j in range(q):
            add_table[i, j] = elem_to_idx[add_el(field_elements[i], field_elements[j])]
            mul_table[i, j] = elem_to_idx[mul_el(field_elements[i], field_elements[j])]

    neg_elem = np.zeros(q, dtype=np.int32)
    for i in range(q):
        neg_elem[i] = elem_to_idx[neg_el(field_elements[i])]

    trace_arr = np.array([trace(field_elements[i]) for i in range(q)], dtype=np.int32)
    norm_arr = np.array([norm(field_elements[i]) for i in range(q)], dtype=np.int32)

    # Point encoding: pt_idx = x_idx * q + y_idx
    x_indices = np.arange(total_points) // q
    y_indices = np.arange(total_points) % q

    # Line keys: for direction d, line_keys[d, pt] identifies the line
    # For d in 0..q-1: direction v=(1, m) where m = field_element[d]
    #   key = y - m*x = y + (-m)*x
    # For d = q: vertical direction v=(0,1), key = x

    line_keys = np.zeros((n_dirs, total_points), dtype=np.int32)
    for m in range(q):
        neg_m = neg_elem[m]
        tmp = mul_table[neg_m, x_indices]
        line_keys[m, :] = add_table[tmp, y_indices]
    line_keys[q, :] = x_indices

    # Build line_points: for each (d, L), list of points on that line
    line_points = {}
    for d in range(n_dirs):
        for L in range(q):
            pts = np.where(line_keys[d, :] == L)[0]
            line_points[(d, L)] = pts.tolist()

    # Initial S: points in Hermitian unital U with y_a not a square mod 7
    squares_mod_7 = set((u * u) % p for u in range(p))

    S = np.zeros(total_points, dtype=bool)
    for x_idx in range(q):
        for y_idx in range(q):
            pt = x_idx * q + y_idx
            if (trace_arr[x_idx] + norm_arr[y_idx]) % p == 0:
                ya = y_idx // p
                if ya not in squares_mod_7:
                    S[pt] = True

    # s_count[d, L] = number of S-points on line L in direction d
    s_count = np.zeros((n_dirs, q), dtype=np.int32)
    for d in range(n_dirs):
        for pt in range(total_points):
            if S[pt]:
                s_count[d, line_keys[d, pt]] += 1

    # counts[d, pt] = s_count[d, line_keys[d, pt]]
    counts = np.zeros((n_dirs, total_points), dtype=np.int32)
    for d in range(n_dirs):
        counts[d, :] = s_count[d, line_keys[d, :]]

    # freedom[pt] = number of directions where counts[d, pt] == 0
    freedom = np.sum(counts == 0, axis=0)

    # Greedy: repeatedly add points with freedom > 0
    max_passes = 50
    for _ in range(max_passes):
        if time.time() > deadline - safety_margin:
            break
        added_any = False
        for pt in range(total_points):
            if time.time() > deadline - safety_margin:
                break
            if S[pt] or freedom[pt] <= 0:
                continue
            # Add pt to S
            S[pt] = True
            added_any = True
            for d in range(n_dirs):
                L = line_keys[d, pt]
                old_count = s_count[d, L]
                s_count[d, L] = old_count + 1
                pts_on_line = line_points[(d, L)]
                if old_count == 0:
                    for pt2 in pts_on_line:
                        freedom[pt2] -= 1
                for pt2 in pts_on_line:
                    counts[d, pt2] = old_count + 1
        if not added_any:
            break

    # Build N = complement of S
    points = []
    for pt in range(total_points):
        if not S[pt]:
            x_idx = pt // q
            y_idx = pt % q
            xa = x_idx // p
            xb = x_idx % p
            ya = y_idx // p
            yb = y_idx % p
            points.append([[xa, xb], [ya, yb]])

    return np.array(points, dtype=np.int64)


# EVOLVE-BLOCK-END
