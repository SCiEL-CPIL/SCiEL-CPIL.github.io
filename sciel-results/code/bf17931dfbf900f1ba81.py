import time
import math
import numpy as np
from scipy.optimize import minimize

# EVOLVE-BLOCK-START
def construct():
    n = 8
    start_time = time.time()
    deadline = start_time + 145.0

    iu = np.triu_indices(n, k=1)

    def compute_score(pts):
        pts = np.asarray(pts, dtype=float)
        if not np.all(np.isfinite(pts)):
            return 0.0
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        d = dists[iu]
        if np.min(d) < 1e-12:
            return 0.0
        D = np.max(d)
        if D < 1e-12:
            return 0.0
        scaled_d = d * (2.0 / D)
        prod_unordered = np.prod(scaled_d)
        Delta = prod_unordered ** 2
        return Delta / (n ** n)

    def neg_logscore(pts):
        """Return negative log-score; lower is better."""
        pts = np.asarray(pts, dtype=float)
        if not np.all(np.isfinite(pts)):
            return 1e10
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        d = dists[iu]
        if np.min(d) < 1e-10:
            return 1e10
        D = np.max(d)
        if D < 1e-10:
            return 1e10
        log_score = 2.0 * (np.sum(np.log(d)) + 28.0 * np.log(2.0) - 28.0 * np.log(D) - 8.0 * np.log(8.0))
        return -log_score

    # ---- Conjugate-symmetric representation: 4 complex numbers -> 8 points ----
    # Points: z0, conj(z0), z1, conj(z1), z2, conj(z2), z3, conj(z3)
    # x = [a0, b0, a1, b1, a2, b2, a3, b3]
    def build_conj_symmetric(x):
        a0, b0, a1, b1, a2, b2, a3, b3 = x
        pts = np.array([
            [a0, b0], [a0, -b0],
            [a1, b1], [a1, -b1],
            [a2, b2], [a2, -b2],
            [a3, b3], [a3, -b3]
        ])
        return pts

    def neg_logscore_cs(x):
        pts = build_conj_symmetric(x)
        return neg_logscore(pts)

    # ---- Two-square family for initial good starting points ----
    def build_two_squares(t, phi):
        angles_outer = np.array([0.0, np.pi/2, np.pi, 3.0*np.pi/2])
        angles_inner = angles_outer + phi
        pts_outer = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
        pts_inner = t * np.column_stack([np.cos(angles_inner), np.sin(angles_inner)])
        return np.vstack([pts_outer, pts_inner])

    def neg_logscore_ts(x):
        t, phi = x
        if t <= 1e-6 or t > 5.0:
            return 1e10
        pts = build_two_squares(t, phi)
        return neg_logscore(pts)

    # ---- Regular octagon baseline ----
    angles_reg = np.linspace(0, 2 * np.pi, n, endpoint=False)
    reg_pts = np.column_stack([np.cos(angles_reg), np.sin(angles_reg)])
    s_reg = compute_score(reg_pts)

    best_score = s_reg
    best_pts = reg_pts.copy()

    # ---- Phase 1: Quick two-square optimization ----
    rng = np.random.RandomState(42)

    t_grid = np.linspace(0.2, 1.5, 15)
    phi_grid = np.linspace(0.0, np.pi/2, 10)
    coarse_best = -np.inf
    coarse_t = 0.5
    coarse_phi = np.pi/4

    for t0 in t_grid:
        for phi0 in phi_grid:
            pts = build_two_squares(t0, phi0)
            s = compute_score(pts)
            if s > coarse_best:
                coarse_best = s
                coarse_t = t0
                coarse_phi = phi0

    if coarse_best > best_score:
        best_score = coarse_best
        best_pts = build_two_squares(coarse_t, coarse_phi)

    try:
        res = minimize(neg_logscore_ts, np.array([coarse_t, coarse_phi]), method='Nelder-Mead',
                       options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-15})
        if np.all(np.isfinite(res.x)):
            t2, phi2 = float(res.x[0]), float(res.x[1])
            if t2 > 1e-6:
                cand = build_two_squares(t2, phi2 % (np.pi/2))
                s2 = compute_score(cand)
                if s2 > best_score:
                    best_score = s2
                    best_pts = cand
    except Exception:
        pass

    # ---- Extract conjugate-symmetric params from a point set ----
    def extract_cs_params(pts):
        centroid = np.mean(pts, axis=0)
        pts_c = pts - centroid
        y_order = np.argsort(pts_c[:, 1])[::-1]
        upper = pts_c[y_order[:4]]
        params = np.array([
            upper[0, 0], upper[0, 1],
            upper[1, 0], upper[1, 1],
            upper[2, 0], upper[2, 1],
            upper[3, 0], upper[3, 1]
        ])
        return params

    cs_params = extract_cs_params(best_pts)

    # ---- Phase 2: Conjugate-symmetric 8D optimization (main improvement) ----
    n_cs_starts = 12
    for i in range(n_cs_starts):
        if time.time() > deadline - 35:
            break
        if i == 0:
            x0 = cs_params.copy()
        else:
            x0 = cs_params + 0.1 * rng.randn(8)
        try:
            res = minimize(neg_logscore_cs, x0, method='L-BFGS-B',
                           bounds=[(-10, 10)] * 8,
                           options={'maxiter': 3000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 50})
            if np.all(np.isfinite(res.x)):
                cand = build_conj_symmetric(res.x)
                s2 = compute_score(cand)
                if s2 > best_score:
                    best_score = s2
                    best_pts = cand
        except Exception:
            pass

    # ---- Phase 3: Diverse 8D starts ----
    if time.time() < deadline - 25:
        reg_cs = np.array([1.0, 0.0,
                           1.0/math.sqrt(2), 1.0/math.sqrt(2),
                           0.0, 1.0,
                           -1.0/math.sqrt(2), 1.0/math.sqrt(2)])

        starts_diverse = [reg_cs.copy()]
        for pert in [0.05, 0.1, 0.2, 0.3]:
            starts_diverse.append(reg_cs + pert * rng.randn(8))

        for t_try in [0.3, 0.5, 0.7, 1.0]:
            for phi_try in [0.0, np.pi/8, np.pi/4, 3*np.pi/8]:
                pts = build_two_squares(t_try, phi_try)
                starts_diverse.append(extract_cs_params(pts))

        for x0 in starts_diverse:
            if time.time() > deadline - 20:
                break
            try:
                res = minimize(neg_logscore_cs, x0, method='L-BFGS-B',
                               bounds=[(-10, 10)] * 8,
                               options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 40})
                if np.all(np.isfinite(res.x)):
                    cand = build_conj_symmetric(res.x)
                    s2 = compute_score(cand)
                    if s2 > best_score:
                        best_score = s2
                        best_pts = cand
            except Exception:
                pass

    # ---- Phase 4: Full 16D polish from best candidate ----
    if time.time() < deadline - 12:
        def objective_16d(x):
            pts = x.reshape(n, 2)
            return neg_logscore(pts)

        try:
            res = minimize(objective_16d, best_pts.flatten(), method='L-BFGS-B',
                           bounds=[(-10, 10)] * 16,
                           options={'maxiter': 3000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 50})
            if np.all(np.isfinite(res.x)):
                refined = res.x.reshape(n, 2)
                s_ref = compute_score(refined)
                if s_ref > best_score:
                    best_score = s_ref
                    best_pts = refined
        except Exception:
            pass

    # ---- Phase 5: Final 8D polish ----
    if time.time() < deadline - 6:
        cs_final = extract_cs_params(best_pts)
        try:
            res = minimize(neg_logscore_cs, cs_final, method='L-BFGS-B',
                           bounds=[(-10, 10)] * 8,
                           options={'maxiter': 3000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 50})
            if np.all(np.isfinite(res.x)):
                cand = build_conj_symmetric(res.x)
                s2 = compute_score(cand)
                if s2 > best_score:
                    best_score = s2
                    best_pts = cand
        except Exception:
            pass

    # ---- Phase 6: 16D from perturbed best ----
    if time.time() < deadline - 3:
        rng2 = np.random.RandomState(999)
        for trial in range(3):
            if time.time() > deadline - 2:
                break
            perturbed = best_pts + 0.01 * rng2.randn(n, 2)
            def objective_16d(x):
                pts = x.reshape(n, 2)
                return neg_logscore(pts)
            try:
                res = minimize(objective_16d, perturbed.flatten(), method='L-BFGS-B',
                               bounds=[(-10, 10)] * 16,
                               options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 40})
                if np.all(np.isfinite(res.x)):
                    refined = res.x.reshape(n, 2)
                    s_ref = compute_score(refined)
                    if s_ref > best_score:
                        best_score = s_ref
                        best_pts = refined
            except Exception:
                pass

    # Final validity check
    best_pts = np.asarray(best_pts, dtype=float)
    diffs = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    d = dists[iu]
    if np.min(d) < 1e-6 or not np.all(np.isfinite(best_pts)):
        best_pts = reg_pts

    max_coord = np.max(np.abs(best_pts))
    if max_coord > 1e6:
        best_pts = best_pts / max_coord * 1e5

    return best_pts.tolist()

# EVOLVE-BLOCK-END
