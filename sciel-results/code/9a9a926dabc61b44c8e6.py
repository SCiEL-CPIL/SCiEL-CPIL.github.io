# EVOLVE-BLOCK-START
import numpy as np
import time

def nikodym_set() -> np.ndarray:
    """
    Nikodym set in F_49^2.  Minimise |N| (maximise the complement C).

    Structural idea (from the strongest inspiration):  start from the Hermitian-
    unital baseline complement C0, then GREEDILY expand C with efficient line-
    count bookkeeping (n_free / n_iso), and finally run LOCAL-SEARCH refinement
    (add / remove / swap moves) to escape greedy local optima.

    Validity is pointwise: a point x is covered iff some line L through x has
    all of L\{x} contained in N (i.e. not in C).  Hence a point in C is "free"
    (its own line is satisfied by another point of C) iff some line through it
    holds >= 2 points of C.
    """
    start_time = time.time()
    deadline = start_time + 590          # hard safety margin before 630 s kill

    p = 7
    q = p * p
    k = 3                                # alpha^2 = 3  (3 is a QR mod 7)

    # ---- F_49 arithmetic --------------------------------------------------
    def mul_f49(p1, p2):
        a1, b1 = p1
        a2, b2 = p2
        return ((a1 * a2 + k * b1 * b2) % p, (a1 * b2 + a2 * b1) % p)

    # ---- all points -------------------------------------------------------
    all_points = [(xa, xb, ya, yb)
                  for xa in range(p) for xb in range(p)
                  for ya in range(p) for yb in range(p)]
    n_points = len(all_points)           # 2401
    point_idx = {pt: i for i, pt in enumerate(all_points)}

    # ---- precompute every line (vertical + every slope m=(ma,mb) != 0) ----
    all_lines = []
    vert = {}
    for xa in range(p):
        for xb in range(p):
            pts = [i for i, (a, b, c, d) in enumerate(all_points)
                   if a == xa and b == xb]
            vert[(xa, xb)] = pts
            all_lines.append(pts)
    for ma in range(p):
        for mb in range(p):
            if ma == 0 and mb == 0:
                continue
            d = {}
            for i, (xa, xb, ya, yb) in enumerate(all_points):
                mx = mul_f49((ma, mb), (xa, xb))
                c = ((ya - mx[0]) % p, (yb - mx[1]) % p)
                d.setdefault(c, []).append(i)
            for pts in d.values():
                all_lines.append(pts)

    # ---- point -> lines, and line -> points -------------------------------
    point_to_lines = [[] for _ in range(n_points)]
    for li, line in enumerate(all_lines):
        for pt in line:
            point_to_lines[pt].append(li)

    # ---- baseline complement: unital U with y_a a non-square --------------
    y_special = set((u * u % p, v) for u in range(p) for v in range(p))
    in_C = [False] * n_points
    for i, (xa, xb, ya, yb) in enumerate(all_points):
        if (2 * xa + ya * ya - yb * yb * k) % p == 0:
            if (ya, yb) not in y_special:
                in_C[i] = True

    # ---- line-count bookkeeping ------------------------------------------
    # n_free[pt]: # lines through pt with 0 points of C  (=> pt is covered if not in C)
    # n_iso[pt]:  # lines through pt with exactly 1 point of C (that pt itself)
    n_free = [0] * n_points
    n_iso = [0] * n_points
    for line in all_lines:
        c_pts = [pt for pt in line if in_C[pt]]
        if len(c_pts) == 0:
            for pt in line:
                n_free[pt] += 1
        elif len(c_pts) == 1:
            n_iso[c_pts[0]] += 1

    def line_status(line):
        cnt = 0
        the_pt = -1
        for pt in line:
            if in_C[pt]:
                cnt += 1
                the_pt = pt
                if cnt >= 2:
                    return 2, -1
        return cnt, the_pt

    def add(pt):
        if in_C[pt]:
            return
        for li in point_to_lines[pt]:
            line = all_lines[li]
            s, the_pt = line_status(line)
            if s == 0:
                for p2 in line:
                    n_free[p2] -= 1
                n_iso[pt] += 1
            elif s == 1:
                n_iso[the_pt] -= 1
        in_C[pt] = True

    def remove(pt):
        if not in_C[pt]:
            return
        in_C[pt] = False
        for li in point_to_lines[pt]:
            line = all_lines[li]
            s, the_pt = line_status(line)
            if s == 0:
                n_iso[pt] -= 1
                for p2 in line:
                    n_free[p2] += 1
            elif s == 1:
                n_free[pt] += 1
                n_iso[the_pt] += 1

    # ---- greedy expansion -------------------------------------------------
    improved = True
    while improved:
        improved = False
        for x in range(n_points):
            if in_C[x] or n_free[x] == 0:
                continue
            add(x)
            improved = True
        if time.time() > deadline - 60:
            break

    # ---- local-search refinement (add / remove / swap) --------------------
    # A point is "free" iff some line through it holds >= 2 points of C.
    def is_free(pt):
        for li in point_to_lines[pt]:
            s, _ = line_status(all_lines[li])
            if s >= 2:
                return True
        return False

    def any_uncovered_in_C():
        for pt in range(n_points):
            if in_C[pt] and not is_free(pt):
                return pt
        return -1

    def uncovered_nonC():
        lst = []
        for pt in range(n_points):
            if not in_C[pt] and n_free[pt] == 0:
                lst.append(pt)
        return lst

    best_size = sum(in_C)
    best_snapshot = in_C[:]
    ls_iter = 0
    while time.time() < deadline - 40:
        ls_iter += 1
        moved = False

        # (1) add any non-C point that is currently uncovered
        for pt in uncovered_nonC():
            if in_C[pt]:
                continue
            add(pt)
            moved = True

        # (2) remove any C point that is free (keeps validity, shrinks N)
        for pt in range(n_points):
            if in_C[pt] and is_free(pt):
                remove(pt)
                moved = True

        # (3) swap: take a free C point and hand its line to an uncovered non-C point
        if time.time() < deadline - 45:
            unC = uncovered_nonC()
            for u in unC:
                if in_C[u]:
                    continue
                for li in point_to_lines[u]:
                    s, the_pt = line_status(all_lines[li])
                    if s == 1 and the_pt != -1 and in_C[the_pt] and is_free(the_pt):
                        remove(the_pt)
                        add(u)
                        moved = True
                        break

        if sum(in_C) < best_size:
            best_size = sum(in_C)
            best_snapshot = in_C[:]

        if not moved:
            break

    # ---- restore best valid configuration ---------------------------------
    if best_snapshot is not None and sum(best_snapshot) < sum(in_C):
        in_C = best_snapshot

    # ---- final full validity re-check (defensive) -------------------------
    def full_valid():
        for x in range(n_points):
            ok = False
            for li in point_to_lines[x]:
                good = True
                for y in all_lines[li]:
                    if y != x and in_C[y]:
                        good = False
                        break
                if good:
                    ok = True
                    break
            if not ok:
                return False
        return True

    if not full_valid():
        # fall back to the guaranteed-valid baseline complement
        in_C = [False] * n_points
        for i, (xa, xb, ya, yb) in enumerate(all_points):
            if (2 * xa + ya * ya - yb * yb * k) % p == 0:
                if (ya, yb) not in y_special:
                    in_C[i] = True

    # ---- build N = complement of C ----------------------------------------
    pts = [all_points[i] for i in range(n_points) if not in_C[i]]
    result = np.array([[[pt[0], pt[1]], [pt[2], pt[3]]] for pt in pts],
                      dtype=np.int64)
    return result

# EVOLVE-BLOCK-END
