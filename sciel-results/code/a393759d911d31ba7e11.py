import numpy as np
import time
from scipy.optimize import linprog

N = 26


def construct():
    start_time = time.time()
    deadline = start_time + 118.0

    def elapsed():
        return time.time() - start_time

    # ---------- validation ----------
    def validate(x, y, r, tol=1e-8):
        if np.any(r < -tol):
            return False
        if np.any(x - r < -tol) or np.any(y - r < -tol):
            return False
        if np.any(x + r > 1 + tol) or np.any(y + r > 1 + tol):
            return False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(x[i] - x[j], y[i] - y[j])
                if d < r[i] + r[j] - tol:
                    return False
        return True

    # ---------- smooth overlap penalty (for position relaxation) ----------
    def _pair_grad(x, y, r, lam=50.0, s2=1e-4):
        """Return (penalty, gx, gy) for pair-wise overlap with smooth sqrt."""
        gx = np.zeros(N)
        gy = np.zeros(N)
        pen = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d2 = dx * dx + dy * dy
                d = np.sqrt(d2 + s2)
                ov = r[i] + r[j] - d
                if ov > 0.0:
                    g = lam * ov / d
                    gx[i] += g * dx
                    gy[i] += g * dy
                    gx[j] -= g * dx
                    gy[j] -= g * dy
                    pen += 0.5 * lam * ov * ov
        return pen, gx, gy

    def _wall_grad(x, y, r, lam=50.0):
        """Return (penalty, gx, gy) for wall violations (smooth hinge)."""
        gx = np.zeros(N)
        gy = np.zeros(N)
        pen = 0.0
        for i in range(N):
            vl = x[i] - r[i]
            if vl < 0:
                gx[i] += -lam * vl
                pen += 0.5 * lam * vl * vl
            vr = (x[i] + r[i]) - 1.0
            if vr > 0:
                gx[i] -= -lam * vr  # push left: -lam*vr -> gx[i] -= -lam*vr? careful
                # we want to decrease x when vr>0: gradient of 0.5*lam*vr^2 wrt x is lam*vr, so gx[i] += lam*vr, and we subtract in update
                gx[i] += lam * vr
                pen += 0.5 * lam * vr * vr
            vt = y[i] - r[i]
            if vt < 0:
                gy[i] += -lam * vt
                pen += 0.5 * lam * vt * vt
            vb = (y[i] + r[i]) - 1.0
            if vb > 0:
                gy[i] += lam * vb
                pen += 0.5 * lam * vb * vb
        return pen, gx, gy

    # ---------- position relaxation (gradient descent on smooth penalty) ----------
    def relax_positions(x, y, r, max_iter=2000, base_step=0.01, tol=1e-7):
        x = x.astype(float).copy()
        y = y.astype(float).copy()
        step = base_step
        prev_pen = None
        for _ in range(max_iter):
            if elapsed() > deadline:
                break
            pen, px, py = _pair_grad(x, y, r)
            wp, wx, wy = _wall_grad(x, y, r)
            gx = px + wx
            gy = py + wy
            pen_total = pen + wp
            # adaptive step
            if prev_pen is not None:
                if pen_total > prev_pen * 1.001 + 1e-10:
                    step *= 0.5
                else:
                    step = min(base_step, step * 1.02)
            prev_pen = pen_total
            x -= step * gx
            y -= step * gy
            # project to a safe box
            x = np.clip(x, 0.001, 0.999)
            y = np.clip(y, 0.001, 0.999)
            if pen_total < tol:
                break
        return x, y, pen_total

    # ---------- Newton solve for max uniform radius (given positions) ----------
    def newton_uniform_radius(x, y, max_iter=60):
        """Solve for max R such that circle i at (x_i,y_i) radius R is feasible."""
        # wall constraint: R <= min_i dist_to_wall(i)
        wall = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
        D = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :])
        np.fill_diagonal(D, np.inf)
        R = min(0.01, float(np.min(wall)) * 0.99)
        for _ in range(max_iter):
            if elapsed() > deadline:
                break
            active = (D - 2.0 * R) < 1e-9
            A_rows = []
            b_vec = []
            for i in range(N):
                for j in range(i + 1, N):
                    if active[i, j]:
                        A_rows.append(D[i, j])
                        b_vec.append(2.0 * R)
            # minimize -R subject to A @ R <= b  (all vars = R, single var)
            if A_rows:
                A_rows = np.array(A_rows)
                # constraint: R <= b_k / A_k  for each; max R = min_k b_k/A_k
                R_new = float(np.min(np.array(b_vec) / A_rows))
                R_new = min(R_new, float(np.min(wall)))
                R_new = max(0.0, R_new)
                if abs(R_new - R) < 1e-12:
                    R = R_new
                    break
                R = R_new
            else:
                R = float(np.min(wall))
                break
        return R

    # ---------- Newton solve for max non-uniform radii (active contact graph) ----------
    def newton_radii(x, y, r0, max_iter=80):
        """Maximize sum(r) with r initialized near r0. Uses active constraints (pairs touching
        within tolerance) + walls. Solves LP-like via iterative projection."""
        r = r0.astype(float).copy()
        # Build candidate constraint set: all pairs where distance - r_i - r_j < tol
        tol_gap = 1e-7
        D = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :])
        np.fill_diagonal(D, np.inf)
        wall = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
        for it in range(max_iter):
            if elapsed() > deadline:
                break
            # Active pair constraints: D_ij - r_i - r_j <= tol_gap
            active_pairs = []
            for i in range(N):
                for j in range(i + 1, N):
                    if D[i, j] - r[i] - r[j] <= tol_gap:
                        active_pairs.append((i, j))
            # Active wall constraints: r_i <= wall_i (always active if r_i near wall_i)
            # Solve: maximize sum(r_i) s.t. r_i + r_j <= D_ij for active pairs, 0<=r_i<=wall_i
            # Use scipy linprog (HiGHS) if available, else simple iteration
            try:
                # variables r_0..r_{N-1}
                c = -np.ones(N)  # minimize -sum
                A_ub = []
                b_ub = []
                for (i, j) in active_pairs:
                    row = np.zeros(N)
                    row[i] = 1.0
                    row[j] = 1.0
                    A_ub.append(row)
                    b_ub.append(D[i, j])
                A_ub = np.array(A_ub) if A_ub else np.zeros((0, N))
                b_ub = np.array(b_ub) if b_ub else np.zeros(0)
                res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(0.0, w) for w in wall],
                              method='highs', options={'presolve': True})
                if res.success and res.x is not None:
                    r_new = res.x
                    if np.max(np.abs(r_new - r)) < 1e-11:
                        r = r_new
                        break
                    r = r_new
                else:
                    break
            except Exception:
                # Fallback: simple iterative projection
                changed = False
                for i in range(N):
                    r[i] = min(r[i], wall[i])
                for (i, j) in active_pairs:
                    s = r[i] + r[j]
                    if s > D[i, j]:
                        # split excess
                        ex = s - D[i, j]
                        r[i] -= 0.5 * ex
                        r[j] -= 0.5 * ex
                        r[i] = max(0.0, min(r[i], wall[i]))
                        r[j] = max(0.0, min(r[j], wall[j]))
                        changed = True
                if not changed:
                    break
        return r

    # ---------- full pipeline for one radius vector ----------
    def solve_for_radii(x0, y0, r_target):
        """Given target radii, relax positions so overlaps vanish, then return positions."""
        x, y, pen = relax_positions(x0, y0, r_target, max_iter=1500, base_step=0.008)
        # final projection to guarantee wall feasibility
        for i in range(N):
            r = r_target[i]
            x[i] = np.clip(x[i], r, 1.0 - r)
            y[i] = np.clip(y[i], r, 1.0 - r)
        return x, y

    # ---------- initial configuration generators ----------
    def gen_grid5():
        pts = []
        for i in range(5):
            for j in range(5):
                pts.append((0.1 + 0.2 * i, 0.1 + 0.2 * j))
        pts.append((0.5, 0.5))
        x = np.array([p[0] for p in pts])
        y = np.array([p[1] for p in pts])
        return x, y

    def gen_hex():
        pts = []
        for row in range(6):
            for col in range(6):
                xx = 0.1 + col * 0.16 + (0.08 if row % 2 else 0.0)
                yy = 0.1 + row * 0.15
                if xx < 0.95 and yy < 0.95:
                    pts.append((xx, yy))
        pts = pts[:N]
        x = np.array([p[0] for p in pts])
        y = np.array([p[1] for p in pts])
        return x, y

    def gen_ring():
        pts = [(0.5, 0.5)]
        for k in range(1, 14):
            a = 2 * np.pi * k / 13
            pts.append((0.5 + 0.32 * np.cos(a), 0.5 + 0.32 * np.sin(a)))
        for k in range(14, 27):
            a = 2 * np.pi * (k - 13) / 13
            pts.append((0.5 + 0.16 * np.cos(a), 0.5 + 0.16 * np.sin(a)))
        x = np.array([p[0] for p in pts])
        y = np.array([p[1] for p in pts])
        return x, y

    def gen_random(seed):
        rng = np.random.RandomState(seed)
        x = rng.uniform(0.12, 0.88, N)
        y = rng.uniform(0.12, 0.88, N)
        return x, y

    def gen_two_clusters(seed):
        rng = np.random.RandomState(seed)
        x1 = rng.uniform(0.1, 0.42, 13)
        y1 = rng.uniform(0.1, 0.9, 13)
        x2 = rng.uniform(0.58, 0.9, 13)
        y2 = rng.uniform(0.1, 0.9, 13)
        return np.concatenate([x1, x2]), np.concatenate([y1, y2])

    def gen_corner4(seed):
        rng = np.random.RandomState(seed)
        corners = [(0.2, 0.2), (0.8, 0.2), (0.2, 0.8), (0.8, 0.8)]
        x = np.array([])
        y = np.array([])
        per = 6  # 4*6=24
        for (cx, cy) in corners:
            for k in range(per):
                a = 2 * np.pi * k / per + rng.uniform(-0.2, 0.2)
                rr = rng.uniform(0.08, 0.22)
                px = cx + rr * np.cos(a)
                py = cy + rr * np.sin(a)
                px = np.clip(px, 0.08, 0.92)
                py = np.clip(py, 0.08, 0.92)
                x = np.append(x, px)
                y = np.append(y, py)
        # 2 center
        x = np.append(x, [0.5, 0.5])
        y = np.append(y, [0.5, 0.5])
        return x, y

    def gen_strict_hex(seed):
        """Dense hexagonal lattice, clipped to unit square, first N points."""
        rng = np.random.RandomState(seed)
        a = rng.uniform(0.13, 0.17)  # lattice spacing
        dx = a
        dy = a * np.sqrt(3) / 2
        pts = []
        row = 0
        while len(pts) < N + 20:
            yv = 0.08 + row * dy
            if yv > 0.92:
                break
            offset = dx / 2 if row % 2 else 0.0
            col = 0
            while True:
                xv = 0.08 + col * dx + offset
                if xv > 0.92:
                    break
                pts.append((xv, yv))
                col += 1
            row += 1
        pts = pts[:N]
        x = np.array([p[0] for p in pts])
        y = np.array([p[1] for p in pts])
        return x, y

    def all_generators():
        gens = []
        gens.append(gen_grid5)
        gens.append(gen_hex)
        gens.append(gen_ring)
        gens.append(lambda: gen_random(1))
        gens.append(lambda: gen_random(7))
        gens.append(lambda: gen_random(42))
        gens.append(lambda: gen_two_clusters(3))
        gens.append(lambda: gen_two_clusters(11))
        gens.append(lambda: gen_corner4(5))
        gens.append(lambda: gen_corner4(17))
        gens.append(lambda: gen_strict_hex(2))
        gens.append(lambda: gen_strict_hex(9))
        return gens

    # ---------- main annealing loop over starts ----------
    best_config = None
    best_score = 0.0

    gens = all_generators()

    # For each start, run a radius-growth anneal
    for gidx, g in enumerate(gens):
        if elapsed() > deadline - 8:
            break
        try:
            x0, y0 = g()
        except Exception:
            continue
        if len(x0) != N or len(y0) != N:
            continue

        # initial small radii
        r = np.full(N, 0.015)

        # Annealing: slowly grow radii, relaxing positions after each growth
        # Use a schedule: start at r0, end at some large target, with cooling
        n_steps = 400
        r_min = 0.015
        r_max = 0.20  # target uniform radius to attempt
        # We grow each circle's radius proportionally to a global scale
        rng = np.random.RandomState(1000 + gidx)

        cur_x = x0.copy()
        cur_y = y0.copy()
        cur_r = r.copy()
        cur_score = 0.0

        for s in range(n_steps):
            if elapsed() > deadline - 5:
                break
            t = s / max(1, n_steps - 1)
            # target uniform radius at this step
            R_target = r_min + (r_max - r_min) * t
            # set all radii to R_target (uniform growth)
            new_r = np.full(N, R_target)
            # relax positions
            nx, ny = relax_positions(cur_x, cur_y, new_r,
                                     max_iter=300, base_step=0.006)
            # project to walls
            for i in range(N):
                rr = new_r[i]
                nx[i] = np.clip(nx[i], rr, 1.0 - rr)
                ny[i] = np.clip(ny[i], rr, 1.0 - rr)
            # check feasibility
            if validate(nx, ny, new_r, tol=1e-7):
                cur_x, cur_y, cur_r = nx, ny, new_r
                cur_score = float(np.sum(cur_r))
            else:
                # infeasible: try to shrink a bit and retry (annealing rejection)
                # reduce target
                R_target = r_min + (r_max - r_min) * (s / max(1, n_steps - 1)) * 0.9
                new_r = np.full(N, R_target)
                nx, ny = relax_positions(cur_x, cur_y, new_r,
                                         max_iter=200, base_step=0.005)
                for i in range(N):
                    rr = new_r[i]
                    nx[i] = np.clip(nx[i], rr, 1.0 - rr)
                    ny[i] = np.clip(ny[i], rr, 1.0 - rr)
                if validate(nx, ny, new_r, tol=1e-7):
                    cur_x, cur_y, cur_r = nx, ny, new_r
                    cur_score = float(np.sum(cur_r))
                # else keep previous

        # After growth, try to maximize radii non-uniformly via Newton on the
        # final contact structure
        try:
            r_final = newton_radii(cur_x, cur_y, cur_r, max_iter=60)
            # re-relax positions to fix any drift
            cur_x, cur_y = relax_positions(cur_x, cur_y, r_final,
                                           max_iter=400, base_step=0.005)
            for i in range(N):
                rr = r_final[i]
                cur_x[i] = np.clip(cur_x[i], rr, 1.0 - rr)
                cur_y[i] = np.clip(cur_y[i], rr, 1.0 - rr)
            if validate(cur_x, cur_y, r_final, tol=1e-8):
                sc = float(np.sum(r_final))
                if sc > best_score:
                    best_score = sc
                    best_config = np.column_stack([cur_x, cur_y, r_final])
        except Exception:
            pass

        # Also record the uniform-growth best if better
        if validate(cur_x, cur_y, cur_r, tol=1e-8):
            sc = float(np.sum(cur_r))
            if sc > best_score:
                best_score = sc
                best_config = np.column_stack([cur_x, cur_y, cur_r])

    # ---------- final local refinement: per-circle radius increase with Newton ----------
    if best_config is not None and elapsed() < deadline - 3:
        x_cur = best_config[:, 0].copy()
        y_cur = best_config[:, 1].copy()
        r_cur = best_config[:, 2].copy()
        # Try to push each circle's radius up one at a time, relaxing neighbors
        for round_idx in range(3):
            if elapsed() > deadline - 2:
                break
            improved = False
            for i in range(N):
                if elapsed() > deadline - 2:
                    break
                # try increasing r_i slightly
                for _ in range(30):
                    r_try = r_cur.copy()
                    r_try[i] += 0.004
                    # relax all positions
                    nx, ny = relax_positions(x_cur, y_cur, r_try,
                                             max_iter=300, base_step=0.005)
                    for k in range(N):
                        rr = r_try[k]
                        nx[k] = np.clip(nx[k], rr, 1.0 - rr)
                        ny[k] = np.clip(ny[k], rr, 1.0 - rr)
                    if validate(nx, ny, r_try, tol=1e-8):
                        x_cur, y_cur, r_cur = nx, ny, r_try
                        improved = True
                        break
                    else:
                        break
            if not improved:
                break
        if validate(x_cur, y_cur, r_cur, tol=1e-8):
            sc = float(np.sum(r_cur))
            if sc > best_score:
                best_score = sc
                best_config = np.column_stack([x_cur, y_cur, r_cur])

    # ---------- fallback ----------
    if best_config is None or best_score < 0.1:
        # simple grid fallback
        side = 6
        s = 1.0 / side
        cells = [((i + 0.5) * s, (j + 0.5) * s, 0.4 * s)
                 for i in range(side) for j in range(side)]
        best_config = np.array(cells[:N])
        best_score = float(np.sum(best_config[:, 2]))

    # ---------- final safety shrink ----------
    x_f = best_config[:, 0].copy()
    y_f = best_config[:, 1].copy()
    r_f = best_config[:, 2].copy()
    if not validate(x_f, y_f, r_f, tol=1e-8):
        for factor in [0.999, 0.99, 0.98, 0.95, 0.9, 0.8, 0.7, 0.5]:
            r_test = r_f * factor
            if validate(x_f, y_f, r_test, tol=1e-8):
                r_f = r_test
                best_config = np.column_stack([x_f, y_f, r_f])
                break
        else:
            side = 6
            s = 1.0 / side
            cells = [((i + 0.5) * s, (j + 0.5) * s, 0.4 * s)
                     for i in range(side) for j in range(side)]
            best_config = np.array(cells[:N])

    return best_config
