# EVOLVE-BLOCK-START
import numpy as np
import time


def nikodym_set() -> np.ndarray:
    """
    Construct a small Nikodym set in F_49^2 (F_49 = F_7[alpha], alpha^2 = 3).
    Uses a baseline unital-based construction plus a fast greedy expansion of the
    excluded set (complement of N) using line-count bookkeeping.
    """
    start_time = time.time()
    deadline = start_time + 590  # safety margin before 630s kill

    p = 7
    q = p * p  # 49
    k = 3  # alpha^2 = 3

    # ---- Field arithmetic for F_49 = F_7[alpha] ----
    def mul_f49(p1, p2):
        a1, b1 = p1
        a2, b2 = p2
        return ((a1 * a2 + k * b1 * b2) % p, (a1 * b2 + a2 * b1) % p)

    def add_f49(p1, p2):
        return ((p1[0] + p2[0]) % p, (p1[1] + p2[1]) % p)

    def sub_f49(p1, p2):
        return ((p1[0] - p2[0]) % p, (p1[1] - p2[1]) % p)

    # Enumerate all field elements
    # elem[i] = (a, b) for i in 0..48, where i = a*7 + b
    elem = [(i // p, i % p) for i in range(q)]

    # Precompute multiplication and addition tables
    mul_table = np.zeros((q, q), dtype=np.int64)
    add_table = np.zeros((q, q), dtype=np.int64)
    for i in range(q):
        for j in range(q):
            mul_table[i, j] = mul_f49(elem[i], elem[j])[0] * p + mul_f49(elem[i], elem[j])[1]
            add_table[i, j] = add_f49(elem[i], elem[j])[0] * p + add_f49(elem[i], elem[j])[1]

    # Re-compute cleanly to avoid repeated calls
    mul_table = np.zeros((q, q), dtype=np.int64)
    add_table = np.zeros((q, q), dtype=np.int64)
    for i in range(q):
        for j in range(q):
            ei = elem[i]
            ej = elem[j]
            mp = mul_f49(ei, ej)
            ap = add_f49(ei, ej)
            mul_table[i, j] = mp[0] * p + mp[1]
            add_table[i, j] = ap[0] * p + ap[1]

    # ---- All points in F_49^2 ----
    # Point index: pt = x_idx * q + y_idx, where x_idx, y_idx in 0..48
    n_points = q * q  # 2401

    def pt_to_xy(pt):
        return pt // q, pt % q

    def xy_to_pt(x_idx, y_idx):
        return x_idx * q + y_idx

    # ---- Build all lines ----
    # 50 directions: 49 slopes m in F_49, plus vertical
    # For each direction, each line has exactly 49 points.
    # We'll build:
    #   all_lines: list of sets (or lists) of point indices
    #   point_to_lines: for each point, list of line indices it belongs to

    all_lines = []  # list of lists of point indices
    point_to_lines = [[] for _ in range(n_points)]

    # Slope lines: for each slope m_idx in 0..48
    for m_idx in range(q):
        # For each point, compute intercept c = y - m*x
        # Group points by intercept
        intercept_to_pts = {}
        for x_idx in range(q):
            for y_idx in range(q):
                # c = y - m*x
                mx = mul_table[m_idx, x_idx]
                c = add_table[y_idx, q - 1 - mx]  # y + (-mx); need negation
                # Actually, c = y - m*x. Let's compute -m*x properly.
                # -m*x = neg(m*x)
                # In F_7, neg(a,b) = (7-a, 7-b) mod 7
                mx_elem = (mx // p, mx % p)
                neg_mx = ((-mx_elem[0]) % p, (-mx_elem[1]) % p)
                neg_mx_idx = neg_mx[0] * p + neg_mx[1]
                c = add_table[y_idx, neg_mx_idx]
                if c not in intercept_to_pts:
                    intercept_to_pts[c] = []
                intercept_to_pts[c].append(xy_to_pt(x_idx, y_idx))

        for c, pts in intercept_to_pts.items():
            line_idx = len(all_lines)
            all_lines.append(pts)
            for pt in pts:
                point_to_lines[pt].append(line_idx)

    # Vertical lines: for each x_idx
    for x_idx in range(q):
        pts = [xy_to_pt(x_idx, y_idx) for y_idx in range(q)]
        line_idx = len(all_lines)
        all_lines.append(pts)
        for pt in pts:
            point_to_lines[pt].append(line_idx)

    n_lines = len(all_lines)  # should be 50 * 49 = 2450

    # ---- Baseline construction: unital-based initial excluded set ----
    # U: {(x, y) : Trace(x) + Norm(y) = 0}
    # Trace((a,b)) = 2a mod 7
    # Norm((a,b)) = a^2 - 3*b^2 mod 7
    # Y: y-coordinate in special set. Use quadratic residues for y_a.
    # Squares mod 7: 0,1,2,4
    squares_mod_7 = {0, 1, 2, 4}

    in_C = [False] * n_points  # in_C[i] = True means point i is EXCLUDED (not in N)

    for x_idx in range(q):
        for y_idx in range(q):
            pt = xy_to_pt(x_idx, y_idx)
            xa, xb = elem[x_idx]
            ya, yb = elem[y_idx]
            # Trace(x) = 2*xa mod 7
            trace_x = (2 * xa) % p
            # Norm(y) = ya^2 - 3*yb^2 mod 7
            norm_y = (ya * ya - k * yb * yb) % p
            if (trace_x + norm_y) % p == 0:
                # Check if y is in Y: ya is a square mod 7
                if ya not in squares_mod_7:
                    in_C[pt] = True

    # ---- Initialize line counts ----
    # For each line, count how many points are in C (excluded)
    # n_free[pt] = number of lines through pt that have 0 points in C
    # n_iso[pt] = number of lines through pt that have exactly 1 point in C (and that point is pt)
    # Actually, we need:
    #   For a point x NOT in C, it can be added to C if:
    #     For every line L through x:
    #       If L has 0 points in C: after adding x, L has 1 point in C (x).
    #         This is OK for x, but for other points y in L, they lose a "free" line.
    #       If L has 1 point in C (say y0): after adding x, L has 2 points in C.
    #         This is OK, but y0 loses an "isolated" line.
    #       If L has >= 2 points in C: no change in validity for others.
    #
    # The condition for x to be addable:
    #   For every line L through x with 0 C-points: for all y in L \ {x}, n_free[y] >= 2
    #   For every line L through x with exactly 1 C-point y0: n_iso[y0] >= 2
    #
    # Let's maintain:
    #   line_c_count[L] = number of points in C on line L
    #   n_free[pt] = number of lines through pt with line_c_count == 0
    #   n_iso[pt] = number of lines through pt with line_c_count == 1 (and pt is that point)

    line_c_count = np.zeros(n_lines, dtype=np.int64)
    n_free = np.zeros(n_points, dtype=np.int64)
    n_iso = np.zeros(n_points, dtype=np.int64)

    # Initialize
    for L in range(n_lines):
        c_pts = [pt for pt in all_lines[L] if in_C[pt]]
        line_c_count[L] = len(c_pts)
        if len(c_pts) == 0:
            for pt in all_lines[L]:
                n_free[pt] += 1
        elif len(c_pts) == 1:
            y0 = c_pts[0]
            n_iso[y0] += 1

    # ---- Greedy expansion ----
    # Try to add points to C (exclude them from N)
    # Process in order, with multiple passes until no improvement or deadline

    improved = True
    pass_count = 0
    while improved and time.time() < deadline - 5:
        improved = False
        pass_count += 1
        for x_pt in range(n_points):
            if in_C[x_pt]:
                continue
            if time.time() > deadline - 5:
                break

            # Check if x_pt can be added
            # Need: for every line L through x_pt:
            #   if line_c_count[L] == 0: all y in L\{x_pt} have n_free[y] >= 2
            #   if line_c_count[L] == 1: the single C-point y0 has n_iso[y0] >= 2
            valid = True
            updates = []  # (line_idx, type, extra)

            for L in point_to_lines[x_pt]:
                cnt = line_c_count[L]
                if cnt == 0:
                    # Check all other points on this line
                    for y_pt in all_lines[L]:
                        if y_pt == x_pt:
                            continue
                        if n_free[y_pt] < 2:
                            valid = False
                            break
                    if not valid:
                        break
                    updates.append((L, 'empty'))
                elif cnt == 1:
                    # Find the single C-point on this line
                    y0 = -1
                    for y_pt in all_lines[L]:
                        if in_C[y_pt]:
                            y0 = y_pt
                            break
                    if y0 < 0:
                        continue  # shouldn't happen
                    if n_iso[y0] < 2:
                        valid = False
                        break
                    updates.append((L, 'single', y0))
                # else: cnt >= 2, no constraint

            if valid:
                in_C[x_pt] = True
                improved = True
                # Apply updates
                for upd in updates:
                    L = upd[0]
                    if upd[1] == 'empty':
                        line_c_count[L] = 1
                        for y_pt in all_lines[L]:
                            if y_pt == x_pt:
                                continue
                            n_free[y_pt] -= 1
                    elif upd[1] == 'single':
                        y0 = upd[2]
                        line_c_count[L] = 2
                        n_iso[y0] -= 1

    # ---- Build result: N = complement of C ----
    points = []
    for pt in range(n_points):
        if not in_C[pt]:
            x_idx, y_idx = pt_to_xy(pt)
            xa, xb = elem[x_idx]
            ya, yb = elem[y_idx]
            points.append([[xa, xb], [ya, yb]])

    if not points:
        # Fallback: return all points
        for xa in range(p):
            for xb in range(p):
                for ya in range(p):
                    for yb in range(p):
                        points.append([[xa, xb], [ya, yb]])

    return np.array(points, dtype=np.int64)


# EVOLVE-BLOCK-END
