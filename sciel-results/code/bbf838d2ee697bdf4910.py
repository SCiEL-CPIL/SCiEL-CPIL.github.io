"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 132.0  # Safety margin before 150s limit

    n = 8
    iu = np.triu_indices(n, k=1)
    num_pairs = len(iu[0])
    theta = 2 * np.pi * np.arange(n) / n

    def log_score(pts):
        """log of score Delta/n^n; Delta = product over ordered pairs of distances."""
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs ** 2, axis=2))
        d_ij = dists[iu]
        d_ij_safe = np.maximum(d_ij, 1e-15)
        log_delta = 2.0 * np.sum(np.log(d_ij_safe))
        return log_delta - n * np.log(n)

    def diameter(pts):
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs ** 2, axis=2))
        return np.max(dists)

    # --- Two concentric squares parameterization ---
    def two_squares_points(r1, r2, phi):
        """
        Points 0-3: inner square, radius r1, angles 0, pi/2, pi, 3pi/2
        Points 4-7: outer square, radius r2, angles phi, phi+pi/2, phi+pi, phi+3pi/2
        """
        a1 = np.array([0.0, np.pi/2, np.pi, 3*np.pi/2])
        a2 = a1 + phi
        p1 = np.column_stack([r1 * np.cos(a1), r1 * np.sin(a1)])
        p2 = np.column_stack([r2 * np.cos(a2), r2 * np.sin(a2)])
        return np.vstack([p1, p2])

    def two_squares_log_score(r1, r2, phi):
        """Compute log score for two concentric squares, scaled to diameter 2."""
        pts = two_squares_points(r1, r2, phi)
        d = diameter(pts)
        if d <= 1e-12:
            return -np.inf
        pts_scaled = pts * (2.0 / d)
        return log_score(pts_scaled)

    def two_squares_optimize(r1_init, r2_init, phi_init):
        """
        Optimize (r1, r2, phi) with penalty for diameter > 2.
        Returns best (r1, r2, phi) and score.
        """
        if time.time() > deadline - 10:
            return None, -np.inf

        def sq_obj(params):
            r1, r2, phi = params
            if r1 < 0.01 or r2 < 0.01:
                return 1e10
            pts = two_squares_points(r1, r2, phi)
            d = diameter(pts)
            if d < 1e-12:
                return 1e10
            pts_s = pts * (2.0 / d)
            ls = log_score(pts_s)
            # Penalty if any distance exceeds 2 after scaling (shouldn't happen, but guard)
            diffs = pts_s[:, np.newaxis, :] - pts_s[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diffs ** 2, axis=2))
            max_d = np.max(dists)
            penalty = 0.0
            if max_d > 2.001:
                penalty = 1000.0 * (max_d - 2.001) ** 2
            return -ls + penalty

        def sq_grad(params):
            """Numerical gradient for the 3D parameter space."""
            eps = 1e-7
            grad = np.zeros(3)
            for k in range(3):
                p_plus = params.copy()
                p_minus = params.copy()
                p_plus[k] += eps
                p_minus[k] -= eps
                grad[k] = (sq_obj(p_plus) - sq_obj(p_minus)) / (2 * eps)
            return grad

        try:
            res = minimize(
                sq_obj,
                np.array([r1_init, r2_init, phi_init]),
                jac=sq_grad,
                method='L-BFGS-B',
                bounds=[(0.01, 5.0), (0.01, 5.0), (-np.pi, np.pi)],
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 5000},
            )
            r1_opt, r2_opt, phi_opt = res.x
            # Ensure valid
            if r1_opt < 0.01 or r2_opt < 0.01:
                return None, -np.inf
            score = two_squares_log_score(r1_opt, r2_opt, phi_opt)
            return (r1_opt, r2_opt, phi_opt), score
        except Exception:
            return None, -np.inf

    def generate_initial_points():
        inits = []
        octagon = np.column_stack([np.cos(theta), np.sin(theta)])
        inits.append(octagon.copy())
        rng = np.random.RandomState(42)
        for scale in [0.05, 0.1, 0.2, 0.3, 0.5, 0.8]:
            inits.append((octagon + rng.randn(n, 2) * scale).copy())
        for angle_offset in [0, np.pi / 4, np.pi / 3, np.pi/6]:
            c1 = np.column_stack([np.cos(theta[:4]), np.sin(theta[:4])]) * 0.7
            c2 = np.column_stack([np.cos(theta[4:]), np.sin(theta[4:])]) * 0.7 + np.array([0.5, 0.0])
            inits.append(np.vstack([c1, c2]).copy())
        for _ in range(8):
            angles = rng.uniform(0, 2 * np.pi, n)
            radii = rng.uniform(0.3, 1.0, n)
            inits.append(np.column_stack([radii * np.cos(angles), radii * np.sin(angles)]).copy())
        inits.append(np.array([
            [1, 1], [1, -1], [-1, -1], [-1, 1],
            [0, 1], [1, 0], [0, -1], [-1, 0]
        ], dtype=float).copy())
        # Two concentric squares
        r1, r2 = 0.6, 1.0
        sq1 = np.column_stack([r1*np.cos(theta[:4]), r1*np.sin(theta[:4])])
        sq2 = np.column_stack([r2*np.cos(theta[4:]+np.pi/8), r2*np.sin(theta[4:]+np.pi/8)])
        inits.append(np.vstack([sq1, sq2]).copy())
        # Alternating radii
        radii_alt = np.array([1.0, 0.6, 1.0, 0.6, 1.0, 0.6, 1.0, 0.6])
        inits.append(np.column_stack([radii_alt*np.cos(theta), radii_alt*np.sin(theta)]).copy())
        return inits

    def scaled_log_score(pts):
        d = diameter(pts)
        if d <= 1e-12:
            return -np.inf
        return log_score(pts * (2.0 / d))

    def obj_and_grad(x, W, T):
        """
        Objective = -log_score + W * max(0, soft_diam - 2)
        soft_diam = (1/T) * logsumexp(T * (d_ij - 2))  ~ smooth max of (d_ij - 2)
        Analytical gradient.
        """
        pts = x.reshape(n, 2)
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs ** 2, axis=2))
        d_ij = dists[iu]
        d_ij_safe = np.maximum(d_ij, 1e-15)
        log_delta = 2.0 * np.sum(np.log(d_ij_safe))
        log_s = log_delta - n * np.log(n)

        # Gradient of log_score
        grad_log = np.zeros_like(pts)
        for idx in range(num_pairs):
            i, j = iu[0][idx], iu[1][idx]
            d = d_ij[idx]
            if d < 1e-15:
                continue
            diff = pts[i] - pts[j]
            grad_log[i] += 2.0 * diff / (d * d)
            grad_log[j] -= 2.0 * diff / (d * d)

        # Soft diameter (log domain) and its gradient
        a = T * (d_ij - 2.0)
        m = np.max(a)
        e = np.exp(a - m)
        s = np.sum(e)
        smax = (m + np.log(s)) / T
        w = e / s

        val = -log_s + W * max(smax, 0.0)
        grad = -grad_log
        if smax > 0.0:
            for idx in range(num_pairs):
                i, j = iu[0][idx], iu[1][idx]
                d = d_ij[idx]
                if d < 1e-15:
                    continue
                diff = pts[i] - pts[j]
                coef = W * w[idx] / d
                grad[i] += coef * diff
                grad[j] -= coef * diff
        return val, grad

    def run_lbfgs(x0, W, T, maxiter, maxfun):
        if time.time() > deadline - 8:
            return None
        try:
            res = minimize(
                lambda v: obj_and_grad(v, W, T)[0],
                x0,
                jac=lambda v: obj_and_grad(v, W, T)[1].flatten(),
                method='L-BFGS-B',
                options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-12, 'maxfun': maxfun},
            )
            return res.x.reshape(n, 2)
        except Exception:
            return None

    # --- Stage 1: Grid search over two-concentric-squares parameter space ---
    best_score = -np.inf
    best_points = None
    best_sq_params = None

    # Coarse grid
    r1_vals = np.linspace(0.1, 1.5, 12)
    r2_vals = np.linspace(0.3, 2.0, 15)
    phi_vals = np.linspace(0, np.pi, 9)  # By symmetry, phi in [0, pi] suffices

    for r1 in r1_vals:
        if time.time() > deadline - 60:
            break
        for r2 in r2_vals:
            for phi in phi_vals:
                ls = two_squares_log_score(r1, r2, phi)
                if ls > best_score:
                    best_score = ls
                    best_sq_params = (r1, r2, phi)
                    pts = two_squares_points(r1, r2, phi)
                    d = diameter(pts)
                    best_points = (pts * (2.0 / d)).copy()

    # Fine grid around best
    if best_sq_params is not None:
        r1_c, r2_c, phi_c = best_sq_params
        r1_fine = np.linspace(max(0.05, r1_c - 0.15), r1_c + 0.15, 15)
        r2_fine = np.linspace(max(0.05, r2_c - 0.15), r2_c + 0.15, 15)
        phi_fine = np.linspace(max(0, phi_c - 0.3), min(np.pi, phi_c + 0.3), 15)
        for r1 in r1_fine:
            if time.time() > deadline - 45:
                break
            for r2 in r2_fine:
                for phi in phi_fine:
                    ls = two_squares_log_score(r1, r2, phi)
                    if ls > best_score:
                        best_score = ls
                        best_sq_params = (r1, r2, phi)
                        pts = two_squares_points(r1, r2, phi)
                        d = diameter(pts)
                        best_points = (pts * (2.0 / d)).copy()

    # --- Stage 2: Local refinement in 3D parameter space from multiple starts ---
    if best_sq_params is not None and time.time() < deadline - 40:
        r1_c, r2_c, phi_c = best_sq_params
        # Multiple starting points for local optimization
        sq_starts = [
            (r1_c, r2_c, phi_c),
            (r1_c * 0.8, r2_c, phi_c),
            (r1_c * 1.2, r2_c, phi_c),
            (r1_c, r2_c * 0.8, phi_c),
            (r1_c, r2_c * 1.2, phi_c),
            (r1_c, r2_c, (phi_c + 0.2) % np.pi),
            (r1_c, r2_c, (phi_c - 0.2) % np.pi),
            (0.5, 1.0, np.pi/8),
            (0.7, 1.2, 0.3),
            (0.4, 1.5, 0.5),
        ]
        for s in sq_starts:
            if time.time() > deadline - 30:
                break
            params, score = two_squares_optimize(s[0], s[1], s[2])
            if params is not None and score > best_score:
                best_score = score
                best_sq_params = params
                pts = two_squares_points(*params)
                d = diameter(pts)
                best_points = (pts * (2.0 / d)).copy()

    # --- Stage 3: Evaluate diverse non-square starts ---
    for init_pts in generate_initial_points():
        if time.time() > deadline - 30:
            break
        ls = scaled_log_score(init_pts)
        if ls > best_score:
            best_score = ls
            d = diameter(init_pts)
            best_points = (init_pts * (2.0 / d)).copy()

    # --- Stage 4: Constrained optimization from best starts in full 16D space ---
    if best_points is not None and time.time() < deadline - 30:
        scored = []
        for init_pts in generate_initial_points()[:10]:
            ls = scaled_log_score(init_pts)
            if ls > -np.inf:
                d = diameter(init_pts)
                scored.append((ls, init_pts * (2.0 / d)))
        scored.sort(key=lambda t: -t[0])

        penalty_weights = [20.0, 50.0, 100.0, 200.0]
        temps = [100.0, 200.0]
        for _, sp in scored[:6]:
            for pw in penalty_weights[:3]:
                for smt in temps:
                    if time.time() > deadline - 20:
                        break
                    opt = run_lbfgs(sp.flatten(), pw, smt, 300, 4000)
                    if opt is None:
                        continue
                    d = diameter(opt)
                    if d <= 1e-10:
                        continue
                    opt = opt * (2.0 / d)
                    ls = log_score(opt)
                    if ls > best_score:
                        best_score = ls
                        best_points = opt.copy()

    # --- Stage 5: Local-refinement restarts around the best point ---
    if best_points is not None and time.time() < deadline - 20:
        rng2 = np.random.RandomState(7)
        for _ in range(6):
            if time.time() > deadline - 12:
                break
            pert = best_points + rng2.randn(n, 2) * 0.05
            for pw in [50.0, 100.0]:
                for smt in [100.0, 200.0]:
                    if time.time() > deadline - 10:
                        break
                    opt = run_lbfgs(pert.flatten(), pw, smt, 400, 6000)
                    if opt is None:
                        continue
                    d = diameter(opt)
                    if d <= 1e-10:
                        continue
                    opt = opt * (2.0 / d)
                    ls = log_score(opt)
                    if ls > best_score:
                        best_score = ls
                        best_points = opt.copy()

    # --- Stage 6: Final polish ---
    if best_points is not None and time.time() < deadline - 12:
        for pw in [100.0, 200.0, 500.0]:
            for smt in [200.0, 400.0]:
                if time.time() > deadline - 8:
                    break
                opt = run_lbfgs(best_points.flatten(), pw, smt, 600, 8000)
                if opt is None:
                    continue
                d = diameter(opt)
                if d <= 1e-10:
                    continue
                opt = opt * (2.0 / d)
                ls = log_score(opt)
                if ls > best_score:
                    best_score = ls
                    best_points = opt.copy()

    # --- Stage 7: Nelder-Mead polish ---
    if best_points is not None and time.time() < deadline - 10:
        x0 = best_points.flatten()
        try:
            res = minimize(
                lambda v: -log_score(v.reshape(n, 2)),
                x0,
                method='Nelder-Mead',
                options={'maxiter': 4000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True},
            )
            opt = res.x.reshape(n, 2)
            d = diameter(opt)
            if d > 1e-10:
                opt = opt * (2.0 / d)
                ls = log_score(opt)
                if ls > best_score:
                    best_score = ls
                    best_points = opt.copy()
        except Exception:
            pass

    # --- Finalize ---
    if best_points is None:
        best_points = np.column_stack([np.cos(theta), np.sin(theta)])

    d = diameter(best_points)
    if d > 1e-10:
        best_points = best_points * (2.0 / d)

    # Ensure distinctness
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(best_points[i] - best_points[j]) < 1e-10:
                best_points[j] += np.array([1e-8, 1e-8])

    max_coord = np.max(np.abs(best_points))
    if max_coord > 1e5:
        best_points = best_points * (1e5 / max_coord)
        d = diameter(best_points)
        if d > 1e-10:
            best_points = best_points * (2.0 / d)

    return [[float(p[0]), float(p[1])] for p in best_points]
# EVOLVE-BLOCK-END
