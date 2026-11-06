import time
import math
import numpy as np
from scipy.optimize import minimize

# EVOLVE-BLOCK-START
def construct():
    n = 8
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    def compute_score(pts):
        """Compute the actual score for a set of points."""
        pts = np.asarray(pts, dtype=float)
        if not np.all(np.isfinite(pts)):
            return 0.0
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        iu = np.triu_indices(n, k=1)
        d = dists[iu]
        if np.min(d) < 1e-12:
            return 0.0
        D = np.max(d)
        if D < 1e-12:
            return 0.0
        # Scale to diameter 2
        scaled_d = d * (2.0 / D)
        prod_unordered = np.prod(scaled_d)
        Delta = prod_unordered ** 2  # Product over ordered pairs
        return Delta / (n ** n)

    def objective(x):
        """Objective for optimization: minimize negative log score."""
        pts = x.reshape(n, 2)
        if not np.all(np.isfinite(pts)):
            return 1e10
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        iu = np.triu_indices(n, k=1)
        d = dists[iu]
        if np.min(d) < 1e-10:
            return 1e10
        D = np.max(d)
        if D < 1e-10:
            return 1e10
        log_score = 2.0 * (np.sum(np.log(d)) + 28.0 * np.log(2.0) - 28.0 * np.log(D) - 8.0 * np.log(8.0))
        return -log_score

    def neg_logscore_tp(x):
        """Objective for two-square family: (t, phi) -> negative log score."""
        t, phi = x
        if t <= 1e-6 or t > 5.0:
            return 1e10
        angles_outer = np.array([0.0, np.pi/2, np.pi, 3.0*np.pi/2])
        angles_inner = angles_outer + phi
        pts_outer = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
        pts_inner = t * np.column_stack([np.cos(angles_inner), np.sin(angles_inner)])
        pts = np.vstack([pts_outer, pts_inner])
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        iu = np.triu_indices(n, k=1)
        d = dists[iu]
        if np.min(d) < 1e-10:
            return 1e10
        D = np.max(d)
        if D < 1e-10:
            return 1e10
        log_score = 2.0 * (np.sum(np.log(d)) + 28.0 * np.log(2.0) - 28.0 * np.log(D) - 8.0 * np.log(8.0))
        return -log_score

    def build_two_squares_phi(t, phi):
        angles_outer = np.array([0.0, np.pi/2, np.pi, 3.0*np.pi/2])
        angles_inner = angles_outer + phi
        pts_outer = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
        pts_inner = t * np.column_stack([np.cos(angles_inner), np.sin(angles_inner)])
        return np.vstack([pts_outer, pts_inner])

    # ---- Regular octagon baseline ----
    angles_reg = np.linspace(0, 2 * np.pi, n, endpoint=False)
    reg_pts = np.column_stack([np.cos(angles_reg), np.sin(angles_reg)])
    s_reg = compute_score(reg_pts)

    best_score = s_reg
    best_pts = reg_pts.copy()

    # ---- Structural family 1: two concentric squares with variable (t, phi) ----
    # Multi-start over t and phi
    t_starts = np.linspace(0.1, 2.0, 15)
    phi_starts = np.linspace(0, np.pi/2, 11)  # Symmetry: phi in [0, pi/2]

    for t0 in t_starts:
        if time.time() > deadline - 30:
            break
        for phi0 in phi_starts:
            if time.time() > deadline - 30:
                break
            x0 = np.array([t0, phi0])
            try:
                res = minimize(neg_logscore_tp, x0, method='Nelder-Mead',
                               options={'maxiter': 3000, 'xatol': 1e-13, 'fatol': 1e-14})
                if np.all(np.isfinite(res.x)):
                    t2, phi2 = float(res.x[0]), float(res.x[1])
                    if t2 > 1e-6:
                        cand = build_two_squares_phi(t2, phi2)
                        s2 = compute_score(cand)
                        if s2 > best_score:
                            best_score = s2
                            best_pts = cand
            except Exception:
                pass

    # ---- Structural family 2: three concentric regular polygons (2+2+4 or 2+3+3 etc.) ----
    # Try: 2 points at radius r1 angle 0,pi; 2 points at radius r2 angle pi/3, 4pi/3; 4 points at radius r3 angles pi/4, 3pi/4, 5pi/4, 7pi/4
    # Actually let's try a more general approach: 8 points on 2 concentric circles with variable angles
    # Family: 4 points on outer circle (radius 1), 4 points on inner circle (radius t), with variable angular offsets

    def build_four_four(t, phi1, phi2):
        """4 points at radius 1 with angles phi1, phi1+pi/2, phi1+pi, phi1+3pi/2
           4 points at radius t with angles phi2, phi2+pi/2, phi2+pi, phi2+3pi/2"""
        a1 = np.array([phi1, phi1 + np.pi/2, phi1 + np.pi, phi1 + 3.0*np.pi/2])
        a2 = np.array([phi2, phi2 + np.pi/2, phi2 + np.pi, phi2 + 3.0*np.pi/2])
        pts1 = np.column_stack([np.cos(a1), np.sin(a1)])
        pts2 = t * np.column_stack([np.cos(a2), np.sin(a2)])
        return np.vstack([pts1, pts2])

    def neg_logscore_44(x):
        t, phi1, phi2 = x
        if t <= 1e-6 or t > 5.0:
            return 1e10
        pts = build_four_four(t, phi1, phi2)
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        iu = np.triu_indices(n, k=1)
        d = dists[iu]
        if np.min(d) < 1e-10:
            return 1e10
        D = np.max(d)
        if D < 1e-10:
            return 1e10
        log_score = 2.0 * (np.sum(np.log(d)) + 28.0 * np.log(2.0) - 28.0 * np.log(D) - 8.0 * np.log(8.0))
        return -log_score

    # Multi-start for 4+4 family
    for t0 in np.linspace(0.2, 1.8, 8):
        if time.time() > deadline - 20:
            break
        for phi1_0 in np.linspace(0, np.pi/4, 4):
            if time.time() > deadline - 20:
                break
            for phi2_0 in np.linspace(0, np.pi/4, 4):
                if time.time() > deadline - 20:
                    break
                x0 = np.array([t0, phi1_0, phi2_0])
                try:
                    res = minimize(neg_logscore_44, x0, method='Nelder-Mead',
                                   options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-13})
                    if np.all(np.isfinite(res.x)):
                        t2, p1, p2 = float(res.x[0]), float(res.x[1]), float(res.x[2])
                        if t2 > 1e-6:
                            cand = build_four_four(t2, p1, p2)
                            s2 = compute_score(cand)
                            if s2 > best_score:
                                best_score = s2
                                best_pts = cand
                except Exception:
                    pass

    # ---- Structural family 3: 3+3+2 (two triangles + one pair) ----
    def build_332(r1, r2, phi1, phi2, phi3):
        """3 points at radius r1, 3 points at radius r2, 2 points at radius 1"""
        a1 = np.array([phi1, phi1 + 2*np.pi/3, phi1 + 4*np.pi/3])
        a2 = np.array([phi2, phi2 + 2*np.pi/3, phi2 + 4*np.pi/3])
        a3 = np.array([phi3, phi3 + np.pi])
        pts1 = r1 * np.column_stack([np.cos(a1), np.sin(a1)])
        pts2 = r2 * np.column_stack([np.cos(a2), np.sin(a2)])
        pts3 = np.column_stack([np.cos(a3), np.sin(a3)])
        return np.vstack([pts1, pts2, pts3])

    def neg_logscore_332(x):
        r1, r2, phi1, phi2, phi3 = x
        if r1 <= 1e-6 or r2 <= 1e-6 or r1 > 5.0 or r2 > 5.0:
            return 1e10
        pts = build_332(r1, r2, phi1, phi2, phi3)
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        iu = np.triu_indices(n, k=1)
        d = dists[iu]
        if np.min(d) < 1e-10:
            return 1e10
        D = np.max(d)
        if D < 1e-10:
            return 1e10
        log_score = 2.0 * (np.sum(np.log(d)) + 28.0 * np.log(2.0) - 28.0 * np.log(D) - 8.0 * np.log(8.0))
        return -log_score

    for r1_0 in np.linspace(0.3, 1.5, 5):
        if time.time() > deadline - 15:
            break
        for r2_0 in np.linspace(0.3, 1.5, 5):
            if time.time() > deadline - 15:
                break
            x0 = np.array([r1_0, r2_0, 0.0, np.pi/3, np.pi/6])
            try:
                res = minimize(neg_logscore_332, x0, method='Nelder-Mead',
                               options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-13})
                if np.all(np.isfinite(res.x)):
                    r1, r2, p1, p2, p3 = [float(v) for v in res.x]
                    if r1 > 1e-6 and r2 > 1e-6:
                        cand = build_332(r1, r2, p1, p2, p3)
                        s2 = compute_score(cand)
                        if s2 > best_score:
                            best_score = s2
                            best_pts = cand
            except Exception:
                pass

    # ---- Full 16-dim local polish from best structured candidate ----
    if time.time() < deadline - 10:
        bounds = [(-5, 5)] * 16
        try:
            res = minimize(objective, best_pts.flatten(), method='L-BFGS-B',
                           bounds=bounds,
                           options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12})
            if np.all(np.isfinite(res.x)):
                refined = res.x.reshape(n, 2)
                s_ref = compute_score(refined)
                if s_ref > best_score:
                    best_score = s_ref
                    best_pts = refined
        except Exception:
            pass

    # ---- Second polish pass with different initial perturbation ----
    if time.time() < deadline - 5:
        rng = np.random.RandomState(42)
        for trial in range(5):
            if time.time() > deadline - 5:
                break
            perturbed = best_pts + 0.05 * rng.randn(n, 2)
            bounds = [(-5, 5)] * 16
            try:
                res = minimize(objective, perturbed.flatten(), method='L-BFGS-B',
                               bounds=bounds,
                               options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12})
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
    iu = np.triu_indices(n, k=1)
    d = dists[iu]
    if np.min(d) < 1e-6 or not np.all(np.isfinite(best_pts)):
        best_pts = reg_pts

    # Ensure coordinate bounds
    max_coord = np.max(np.abs(best_pts))
    if max_coord > 1e6:
        best_pts = best_pts / max_coord * 1e5

    return best_pts.tolist()

# EVOLVE-BLOCK-END
