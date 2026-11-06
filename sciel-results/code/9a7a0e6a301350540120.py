import time
import math
import numpy as np
from scipy.optimize import minimize

# EVOLVE-BLOCK-START
def construct():
    n = 8
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    iu = np.triu_indices(n, k=1)
    
    def compute_score(pts):
        """Compute the actual score for a set of points."""
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
        d = dists[iu]
        if np.min(d) < 1e-10:
            return 1e10
        D = np.max(d)
        if D < 1e-10:
            return 1e10
        log_score = 2.0 * (np.sum(np.log(d)) + 28.0 * np.log(2.0) - 28.0 * np.log(D) - 8.0 * np.log(8.0))
        return -log_score

    def build_two_squares(t, phi):
        """Build two concentric squares: outer radius 1, inner radius t, rotation phi."""
        angles_outer = np.array([0.0, np.pi/2, np.pi, 3.0*np.pi/2])
        angles_inner = angles_outer + phi
        pts_outer = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
        pts_inner = t * np.column_stack([np.cos(angles_inner), np.sin(angles_inner)])
        return np.vstack([pts_outer, pts_inner])

    def neg_logscore_tp(x):
        """Objective for two-square family: (t, phi) -> negative log score."""
        t, phi = x
        if t <= 1e-6 or t > 5.0:
            return 1e10
        pts = build_two_squares(t, phi)
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

    # ---- Regular octagon baseline ----
    angles_reg = np.linspace(0, 2 * np.pi, n, endpoint=False)
    reg_pts = np.column_stack([np.cos(angles_reg), np.sin(angles_reg)])
    s_reg = compute_score(reg_pts)

    best_score = s_reg
    best_pts = reg_pts.copy()

    # ---- Phase 1: Densely optimize two-square family (t, phi) ----
    # The known good region is t in [0.3, 1.2], phi near pi/4
    # Use a grid of starting points and Nelder-Mead refinement
    
    # First, do a coarse grid to find the best region
    t_grid = np.linspace(0.2, 1.5, 25)
    phi_grid = np.linspace(0.0, np.pi/2, 15)
    
    coarse_best_score = -np.inf
    coarse_best_t = 0.6
    coarse_best_phi = np.pi/4
    
    for t0 in t_grid:
        if time.time() > deadline - 80:
            break
        for phi0 in phi_grid:
            if time.time() > deadline - 80:
                break
            pts = build_two_squares(t0, phi0)
            s = compute_score(pts)
            if s > coarse_best_score:
                coarse_best_score = s
                coarse_best_t = t0
                coarse_best_phi = phi0
    
    if coarse_best_score > best_score:
        best_score = coarse_best_score
        best_pts = build_two_squares(coarse_best_t, coarse_best_phi)
    
    # Now do fine multi-start optimization around the best region
    # Use multiple starting points in a neighborhood of the best found
    rng = np.random.RandomState(123)
    
    # Create a set of starting points clustered around the best region
    n_starts = 40
    t_starts = []
    phi_starts = []
    
    # Add the best point
    t_starts.append(coarse_best_t)
    phi_starts.append(coarse_best_phi)
    
    # Add points in a local neighborhood
    for _ in range(n_starts - 1):
        t0 = coarse_best_t + rng.uniform(-0.3, 0.3)
        phi0 = coarse_best_phi + rng.uniform(-0.3, 0.3)
        # Wrap phi to [0, pi/2] due to symmetry
        phi0 = phi0 % (np.pi/2)
        t_starts.append(max(0.1, t0))
        phi_starts.append(phi0)
    
    # Also add some points in the broader known-good region
    for _ in range(10):
        t0 = rng.uniform(0.3, 1.2)
        phi0 = rng.uniform(0.0, np.pi/2)
        t_starts.append(t0)
        phi_starts.append(phi0)
    
    # Run Nelder-Mead from each starting point
    for i in range(len(t_starts)):
        if time.time() > deadline - 40:
            break
        x0 = np.array([t_starts[i], phi_starts[i]])
        try:
            res = minimize(neg_logscore_tp, x0, method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-15})
            if np.all(np.isfinite(res.x)):
                t2, phi2 = float(res.x[0]), float(res.x[1])
                if t2 > 1e-6:
                    # Normalize phi to [0, pi/2]
                    phi2 = phi2 % (np.pi/2)
                    cand = build_two_squares(t2, phi2)
                    s2 = compute_score(cand)
                    if s2 > best_score:
                        best_score = s2
                        best_pts = cand
        except Exception:
            pass

    # ---- Phase 2: Try a few alternative structural families briefly ----
    # Family: 4 points on outer circle, 4 on inner with different angular structure
    # This is less likely to beat the two-square family, but worth a quick check
    
    def build_asymmetric(t, phi1, phi2, phi3, phi4):
        """4 outer points at angles phi1-phi4, 4 inner points at radius t with same angles."""
        angles = np.array([phi1, phi2, phi3, phi4])
        pts_outer = np.column_stack([np.cos(angles), np.sin(angles)])
        pts_inner = t * np.column_stack([np.cos(angles), np.sin(angles)])
        return np.vstack([pts_outer, pts_inner])
    
    def neg_logscore_asym(x):
        t, p1, p2, p3, p4 = x
        if t <= 1e-6 or t > 5.0:
            return 1e10
        pts = build_asymmetric(t, p1, p2, p3, p4)
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
    
    # Quick multi-start for asymmetric family
    if time.time() < deadline - 20:
        asym_starts = [
            [0.6, 0.0, np.pi/2, np.pi, 3*np.pi/2],  # symmetric
            [0.5, 0.1, np.pi/2+0.1, np.pi-0.1, 3*np.pi/2-0.1],
            [0.7, 0.05, np.pi/2-0.05, np.pi+0.05, 3*np.pi/2-0.05],
        ]
        for x0 in asym_starts:
            if time.time() > deadline - 15:
                break
            try:
                res = minimize(neg_logscore_asym, np.array(x0), method='Nelder-Mead',
                               options={'maxiter': 2000, 'xatol': 1e-13, 'fatol': 1e-14})
                if np.all(np.isfinite(res.x)):
                    t2, p1, p2, p3, p4 = [float(v) for v in res.x]
                    if t2 > 1e-6:
                        cand = build_asymmetric(t2, p1, p2, p3, p4)
                        s2 = compute_score(cand)
                        if s2 > best_score:
                            best_score = s2
                            best_pts = cand
            except Exception:
                pass

    # ---- Phase 3: Full 16D local polish from best structured candidate ----
    if time.time() < deadline - 10:
        bounds = [(-5, 5)] * 16
        try:
            res = minimize(objective, best_pts.flatten(), method='L-BFGS-B',
                           bounds=bounds,
                           options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-13})
            if np.all(np.isfinite(res.x)):
                refined = res.x.reshape(n, 2)
                s_ref = compute_score(refined)
                if s_ref > best_score:
                    best_score = s_ref
                    best_pts = refined
        except Exception:
            pass

    # ---- Phase 4: Second polish with perturbation ----
    if time.time() < deadline - 5:
        rng2 = np.random.RandomState(999)
        for trial in range(3):
            if time.time() > deadline - 3:
                break
            perturbed = best_pts + 0.03 * rng2.randn(n, 2)
            bounds = [(-5, 5)] * 16
            try:
                res = minimize(objective, perturbed.flatten(), method='L-BFGS-B',
                               bounds=bounds,
                               options={'maxiter': 1500, 'ftol': 1e-15, 'gtol': 1e-12})
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

    # Ensure coordinate bounds
    max_coord = np.max(np.abs(best_pts))
    if max_coord > 1e6:
        best_pts = best_pts / max_coord * 1e5

    return best_pts.tolist()

# EVOLVE-BLOCK-END
