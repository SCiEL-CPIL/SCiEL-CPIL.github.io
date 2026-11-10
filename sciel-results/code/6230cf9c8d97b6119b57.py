# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p
    Q2 = q * q  # 2401
    k = 3  # alpha^2 = 3 in F_7

    # Flat index: idx = xa*343 + xb*49 + ya*7 + yb
    def get_idx(xa, xb, ya, yb):
        return xa * 343 + xb * 49 + ya * 7 + yb

    def get_coords(idx):
        yb = idx % 7
        ya = (idx // 7) % 7
        xb = (idx // 49) % 7
        xa = (idx // 343) % 7
        return xa, xb, ya, yb

    # Field ops
    def mul(u, v):
        ua, ub = u
        va, vb = v
        return ((ua * va + ub * vb * k) % p, (ua * vb + ub * va) % p)

    def add(u, v):
        return ((u[0] + v[0]) % p, (u[1] + v[1]) % p)

    # Precompute all lines
    lines = []
    line_membership = [[] for _ in range(Q2)]

    # Vertical lines: x constant
    for xa in range(p):
        for xb in range(p):
            pts = np.array([get_idx(xa, xb, ya, yb) for ya in range(p) for yb in range(p)], dtype=np.int32)
            li = len(lines)
            lines.append(pts)
            for pt in pts:
                line_membership[pt].append(li)

    # Sloped lines: y = y0 + m*(x - x0)
    for ma in range(p):
        for mb in range(p):
            m = (ma, mb)
            for ba in range(p):
                for bb in range(p):
                    b = (ba, bb)
                    pts = []
                    for xa in range(p):
                        for xb in range(p):
                            x = (xa, xb)
                            mx = mul(m, x)
                            y = add(mx, b)
                            pts.append(get_idx(xa, xb, y[0], y[1]))
                    pts = np.array(pts, dtype=np.int32)
                    li = len(lines)
                    lines.append(pts)
                    for pt in pts:
                        line_membership[pt].append(li)

    num_lines = len(lines)

    # Convert line_membership to numpy arrays for speed
    line_membership_arr = [np.array(lm, dtype=np.int32) for lm in line_membership]

    # Baseline S
    squares = {0, 1, 2, 4}
    S = set()
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    if (2 * xa + ya * ya - yb * yb * k) % p == 0:
                        if ya not in squares:
                            S.add(get_idx(xa, xb, ya, yb))

    # State arrays
    in_S = np.zeros(Q2, dtype=bool)
    for x in S:
        in_S[x] = True

    line_count = np.zeros(num_lines, dtype=np.int32)
    for x in S:
        for li in line_membership_arr[x]:
            line_count[li] += 1

    witness_count = np.zeros(Q2, dtype=np.int32)
    zero_count = np.zeros(Q2, dtype=np.int32)

    for li in range(num_lines):
        lc = line_count[li]
        if lc == 1:
            for x in lines[li]:
                if in_S[x]:
                    witness_count[x] += 1
        elif lc == 0:
            for z in lines[li]:
                if not in_S[z]:
                    zero_count[z] += 1

    # Precompute s_point_on_line for lines with count 1
    s_point_on_line = np.full(num_lines, -1, dtype=np.int32)
    for li in range(num_lines):
        if line_count[li] == 1:
            for x in lines[li]:
                if in_S[x]:
                    s_point_on_line[li] = x
                    break

    deadline = time.time() + 580

    # Local search: try to add points to S (removing from N)
    # We want to maximize |S|, so we add points to S.

    candidates = list(range(Q2))
    random.shuffle(candidates)

    additions = 0
    best_s_size = len(S)
    best_in_S = in_S.copy()
    best_S = set(S)

    # Multiple passes with different random orderings
    passes = 0
    while time.time() < deadline:
        progress = False
        random.shuffle(candidates)
        for y in candidates:
            if time.time() > deadline:
                break
            if in_S[y]:
                continue

            # Check 1: y can be isolated
            has_zero_line = False
            for li in line_membership_arr[y]:
                if line_count[li] == 0:
                    has_zero_line = True
                    break
            if not has_zero_line:
                continue

            # Check 2: S points on count-1 lines remain isolated
            ok = True
            for li in line_membership_arr[y]:
                if line_count[li] == 1:
                    x = s_point_on_line[li]
                    if witness_count[x] <= 1:
                        ok = False
                        break
            if not ok:
                continue

            # Check 3: N points on count-0 lines remain covered
            for li in line_membership_arr[y]:
                if line_count[li] == 0:
                    for z in lines[li]:
                        if z != y and not in_S[z]:
                            if zero_count[z] <= 1:
                                ok = False
                                break
                    if not ok:
                        break
            if not ok:
                continue

            # Add y to S
            in_S[y] = True
            S.add(y)
            additions += 1
            progress = True

            # Update line_count and witness/zero counts
            for li in line_membership_arr[y]:
                old_c = line_count[li]
                line_count[li] = old_c + 1

                if old_c == 0:
                    # l becomes count 1. y is the S point.
                    s_point_on_line[li] = y
                    # y gets a witness
                    witness_count[y] += 1
                    # N points on l lose a zero line
                    for z in lines[li]:
                        if z != y and not in_S[z]:
                            zero_count[z] -= 1
                elif old_c == 1:
                    # l becomes count 2. s_point_on_line no longer valid.
                    x = s_point_on_line[li]
                    witness_count[x] -= 1
                    s_point_on_line[li] = -1
                # else: no change to witness/zero structures

            # Track best
            if len(S) > best_s_size:
                best_s_size = len(S)
                best_in_S = in_S.copy()
                best_S = set(S)

        if not progress:
            passes += 1
            if passes >= 5:
                break
        else:
            passes = 0

    # Use best state found
    in_S = best_in_S
    S = best_S

    # Build N = complement of S
    N_points = []
    for i in range(Q2):
        if not in_S[i]:
            xa, xb, ya, yb = get_coords(i)
            N_points.append([[xa, xb], [ya, yb]])

    return np.array(N_points, dtype=np.int64)

# EVOLVE-BLOCK-END
