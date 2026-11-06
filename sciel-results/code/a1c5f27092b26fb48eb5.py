import time
import math
import numpy as np
from scipy.optimize import minimize

# EVOLVE-BLOCK-START
def construct():
    n = 8
    start_time = time.time()
    deadline = start_time + 140.0

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

    # ---- Structural family 1: Two concentric squares with independent rotations ----
    # Outer square at radius 1, angles: 0, pi/2, pi, 3pi/2
    # Inner square at radius t, angles: phi, phi+pi/2, phi+pi, phi+3pi/2
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

    # ---- Structural family 2: Two concentric squares, but outer also rotated ----
    # Outer at radius 1, rotation alpha; inner at radius t, rotation beta
    def build_two_squares2(t, alpha, beta):
        angles_outer = np.array([0.0, np.pi/2, np.pi, 3.0*np.pi/2]) + alpha
        angles_inner = np.array([0.0, np.pi/2, np.pi, 3.0*np.pi/2]) + beta
        pts_outer = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
        pts_inner = t * np.column_stack([np.cos(angles_inner), np.sin(angles_inner)])
        return np.vstack([pts_outer, pts_inner])

    def neg_logscore_ts2(x):
        t, alpha, beta = x
        if t <= 1e-6 or t > 5.0:
            return 1e10
        pts = build_two_squares2(t, alpha, beta)
        return neg_logscore(pts)

    # ---- Structural family 3: Antiprism-like: 4 outer at angles 0,pi/2,pi,3pi/2;
    #      4 inner at radius t, angles offset by pi/4 from outer ----
    # This is actually the same as family 1 with phi=pi/4, but let's generalize.
    # More interesting: 4 outer points NOT at 90-degree spacing.
    # Let outer angles be: a, b, c, d (summing to 2pi), inner at radius t at angles a+pi/4, etc.
    # Too many parameters. Instead: regular octagon perturbed by alternating radii.
    # 8 points at angles 0, pi/4, ..., 7pi/4 with radii r1, r2, ..., r8
    # where r_odd = 1 (outer) and r_even = t (inner), or vice versa.
    # This is actually the same as two squares with phi=0 or phi=pi/4.
    
    # ---- Structural family 4: "Alternating" - 4 points on circle radius 1,
    #      4 points on circle radius t, but the inner 4 are at angles 
    #      0, pi/2, pi, 3pi/2 (aligned with outer) rather than rotated.
    # This is phi=0 in family 1.
    
    # ---- New structural idea: 3-3-2 or 4-2-2 splits ----
    # Let's try: 4 points forming a square (radius 1), 2 points on x-axis at ±s,
    # 2 points on y-axis at ±s. But that's 8 points total with structure.
    # Actually: 4 outer at angles 0, pi/2, pi, 3pi/2 (radius 1)
    #           2 inner at angles 0, pi (radius t1)
    #           2 inner at angles pi/2, 3pi/2 (radius t2)
    # This allows different inner radii for horizontal vs vertical pairs.
    def build_4_2_2(t1, t2):
        angles_outer = np.array([0.0, np.pi/2, np.pi, 3.0*np.pi/2])
        pts_outer = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
        # 2 points on x-axis at radius t1
        pts_h = t1 * np.array([[1.0, 0.0], [-1.0, 0.0]])
        # 2 points on y-axis at radius t2
        pts_v = t2 * np.array([[0.0, 1.0], [0.0, -1.0]])
        return np.vstack([pts_outer, pts_h, pts_v])

    def neg_logscore_4_2_2(x):
        t1, t2 = x
        if t1 <= 1e-6 or t2 <= 1e-6 or t1 > 5.0 or t2 > 5.0:
            return 1e10
        pts = build_4_2_2(t1, t2)
        return neg_logscore(pts)

    # ---- Structural family 5: Two squares, but one is a "diamond" (rotated 45 deg)
    #      and the other is axis-aligned, with independent radii and a relative rotation.
    #      This is already covered by family 2 with alpha=0, beta=pi/4.
    
    # ---- General 16D optimization from multiple structured starts ----
    def objective_16d(x):
        pts = x.reshape(n, 2)
        return neg_logscore(pts)

    # ---- Regular octagon baseline ----
    angles_reg = np.linspace(0, 2 * np.pi, n, endpoint=False)
    reg_pts = np.column_stack([np.cos(angles_reg), np.sin(angles_reg)])
    s_reg = compute_score(reg_pts)

    best_score = s_reg
    best_pts = reg_pts.copy()

    # ---- Phase 1: Optimize two-square family (t, phi) with dense multi-start ----
    rng = np.random.RandomState(42)
    
    # Grid search for good starting region
    t_grid = np.linspace(0.1, 2.0, 30)
    phi_grid = np.linspace(0.0, np.pi/2, 20)
    
    coarse_best = -np.inf
    coarse_t = 0.5
    coarse_phi = np.pi/4
    
    for t0 in t_grid:
        if time.time() > deadline - 90:
            break
        for phi0 in phi_grid:
            if time.time() > deadline - 90:
                break
            pts = build_two_squares(t0, phi0)
            s = compute_score(pts)
            if s > coarse_best:
                coarse_best = s
                coarse_t = t0
                coarse_phi = phi0

    if coarse_best > best_score:
        best_score = coarse_best
        best_pts = build_two_squares(coarse_t, coarse_phi)

    # Multi-start Nelder-Mead on two-square family
    n_starts = 50
    for i in range(n_starts):
        if time.time() > deadline - 50:
            break
        if i == 0:
            x0 = np.array([coarse_t, coarse_phi])
        else:
            t0 = coarse_t + rng.uniform(-0.4, 0.4)
            phi0 = (coarse_phi + rng.uniform(-0.4, 0.4)) % (np.pi/2)
            x0 = np.array([max(0.05, t0), phi0])
        try:
            res = minimize(neg_logscore_ts, x0, method='Nelder-Mead',
                           options={'maxiter': 8000, 'xatol': 1e-14, 'fatol': 1e-15})
            if np.all(np.isfinite(res.x)):
                t2, phi2 = float(res.x[0]), float(res.x[1])
                if t2 > 1e-6:
                    phi2 = phi2 % (np.pi/2)
                    cand = build_two_squares(t2, phi2)
                    s2 = compute_score(cand)
                    if s2 > best_score:
                        best_score = s2
                        best_pts = cand
        except Exception:
            pass

    # ---- Phase 2: Optimize two-square with independent rotations (t, alpha, beta) ----
    if time.time() < deadline - 40:
        # The relative rotation is what matters, so alpha=0 w.l.o.g.
        # But let's try a few starts with alpha != 0 just in case
        for i in range(20):
            if time.time() > deadline - 40:
                break
            if i == 0:
                x0 = np.array([coarse_t, 0.0, coarse_phi])
            else:
                t0 = coarse_t + rng.uniform(-0.3, 0.3)
                a0 = rng.uniform(-0.2, 0.2)
                b0 = (coarse_phi + rng.uniform(-0.3, 0.3)) % (np.pi/2)
                x0 = np.array([max(0.05, t0), a0, b0])
            try:
                res = minimize(neg_logscore_ts2, x0, method='Nelder-Mead',
                               options={'maxiter': 5000, 'xatol': 1e-13, 'fatol': 1e-14})
                if np.all(np.isfinite(res.x)):
                    t2, a2, b2 = float(res.x[0]), float(res.x[1]), float(res.x[2])
                    if t2 > 1e-6:
                        cand = build_two_squares2(t2, a2, b2)
                        s2 = compute_score(cand)
                        if s2 > best_score:
                            best_score = s2
                            best_pts = cand
            except Exception:
                pass

    # ---- Phase 3: 4-2-2 structure ----
    if time.time() < deadline - 30:
        # Try some starts
        starts_422 = [
            [0.5, 0.5],
            [0.7, 0.5],
            [0.5, 0.7],
            [0.6, 0.6],
            [0.4, 0.6],
            [0.6, 0.4],
        ]
        for x0 in starts_422:
            if time.time() > deadline - 30:
                break
            try:
                res = minimize(neg_logscore_4_2_2, np.array(x0), method='Nelder-Mead',
                               options={'maxiter': 5000, 'xatol': 1e-13, 'fatol': 1e-14})
                if np.all(np.isfinite(res.x)):
                    t1, t2 = float(res.x[0]), float(res.x[1])
                    if t1 > 1e-6 and t2 > 1e-6:
                        cand = build_4_2_2(t1, t2)
                        s2 = compute_score(cand)
                        if s2 > best_score:
                            best_score = s2
                            best_pts = cand
            except Exception:
                pass

    # ---- Phase 4: Full 16D polish from best structured candidate ----
    if time.time() < deadline - 15:
        bounds = [(-10, 10)] * 16
        try:
            res = minimize(objective_16d, best_pts.flatten(), method='L-BFGS-B',
                           bounds=bounds,
                           options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 50})
            if np.all(np.isfinite(res.x)):
                refined = res.x.reshape(n, 2)
                s_ref = compute_score(refined)
                if s_ref > best_score:
                    best_score = s_ref
                    best_pts = refined
        except Exception:
            pass

    # ---- Phase 5: More 16D polish with perturbations ----
    if time.time() < deadline - 8:
        rng2 = np.random.RandomState(777)
        for trial in range(5):
            if time.time() > deadline - 5:
                break
            scale = 0.02 * (trial + 1)
            perturbed = best_pts + scale * rng2.randn(n, 2)
            bounds = [(-10, 10)] * 16
            try:
                res = minimize(objective_16d, perturbed.flatten(), method='L-BFGS-B',
                               bounds=bounds,
                               options={'maxiter': 3000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 40})
                if np.all(np.isfinite(res.x)):
                    refined = res.x.reshape(n, 2)
                    s_ref = compute_score(refined)
                    if s_ref > best_score:
                        best_score = s_ref
                        best_pts = refined
            except Exception:
                pass

    # ---- Phase 6: One more round of structured search with the best found as seed ----
    if time.time() < deadline - 5:
        # Extract structure from best_pts: fit to two squares
        # Sort points by radius
        radii = np.linalg.norm(best_pts, axis=1)
        # Try to identify 4 outer and 4 inner
        # For now, just do a few more 16D optimizations from slightly perturbed versions
        rng3 = np.random.RandomState(555)
        for trial in range(3):
            if time.time() > deadline - 3:
                break
            perturbed = best_pts + 0.01 * rng3.randn(n, 2)
            bounds = [(-10, 10)] * 16
            try:
                res = minimize(objective_16d, perturbed.flatten(), method='L-BFGS-B',
                               bounds=bounds,
                               options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 30})
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
