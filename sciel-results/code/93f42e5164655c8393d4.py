import numpy as np
from scipy.optimize import minimize
import time
import math

def construct():
    n = 8
    start_time = time.time()
    deadline = 140  # 10s safety margin before 150s kill

    def regular_octagon():
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return np.column_stack([np.cos(angles), np.sin(angles)])

    def score(points):
        diffs = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        D = np.max(dists)
        if D == 0:
            return 0.0
        scaled_dists = dists * (2.0 / D)
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        prod = np.prod(scaled_dists[mask])
        return prod / (n ** n)

    best_score = score(regular_octagon())
    best_points = regular_octagon()

    def build_pts(x):
        pts = np.zeros((n, 2))
        pts[0] = [-1.0, 0.0]
        pts[1] = [1.0, 0.0]
        pts[2:] = x.reshape(n-2, 2)
        return pts

    def compute_dists(pts):
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        return np.linalg.norm(diffs, axis=2)

    def log_prod_obj(x):
        """Smooth objective: negative log product of all pairwise distances."""
        pts = build_pts(x)
        dists = compute_dists(pts)
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        d = dists[mask]
        if np.min(d) < 1e-8:
            return 1e10
        return -np.sum(np.log(d))

    def smooth_penalty_obj(x, penalty_weight=1e4):
        """Objective with smooth soft-max penalty for diameter constraint."""
        pts = build_pts(x)
        dists = compute_dists(pts)
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        d = dists[mask]
        
        if np.min(d) < 1e-8:
            return 1e10
        
        log_sum = np.sum(np.log(d))
        
        # Soft-max approximation of max(d) using log-sum-exp
        # max(d) ≈ (1/β) * log(sum(exp(β*d)))
        beta = 50.0
        d_scaled = d * beta
        d_max_scaled = np.max(d_scaled)
        log_sum_exp = d_max_scaled + np.log(np.sum(np.exp(d_scaled - d_max_scaled)))
        soft_max = log_sum_exp / beta
        
        # Penalty for exceeding diameter 2
        if soft_max > 2.0:
            penalty = penalty_weight * (soft_max - 2.0) ** 2
        else:
            penalty = 0.0
        
        return -log_sum + penalty

    bounds = [(-1.0, 1.0), (-1.8, 1.8)] * (n-2)

    # Initial guess from regular octagon
    oct_pts = regular_octagon()
    # p0=(-1,0)=idx4, p1=(1,0)=idx0, remaining: 1,2,3,5,6,7
    init_x = np.array([
        oct_pts[1][0], oct_pts[1][1],
        oct_pts[2][0], oct_pts[2][1],
        oct_pts[3][0], oct_pts[3][1],
        oct_pts[5][0], oct_pts[5][1],
        oct_pts[6][0], oct_pts[6][1],
        oct_pts[7][0], oct_pts[7][1]
    ])

    def run_slsqp_penalty(x0, maxiter, ftol):
        """SLSQP with smooth penalty instead of explicit constraints."""
        try:
            res = minimize(
                smooth_penalty_obj,
                x0,
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': maxiter, 'ftol': ftol}
            )
            pts = build_pts(res.x)
            dists = compute_dists(pts)
            D = np.max(dists)
            mask = np.triu(np.ones((n, n), dtype=bool), k=1)
            if D <= 2.0 + 1e-5 and np.min(dists[mask]) > 1e-5:
                return pts
            return None
        except Exception:
            return None

    def run_nelder_mead(x0, maxiter, tol):
        """Nelder-Mead with penalty for constraint violations."""
        def penalized_obj(x):
            pts = build_pts(x)
            dists = compute_dists(pts)
            mask = np.triu(np.ones((n, n), dtype=bool), k=1)
            d = dists[mask]
            
            max_dist = np.max(d)
            if max_dist > 2.0:
                penalty = 1e6 * (max_dist - 2.0) ** 2
            else:
                penalty = 0.0
            
            min_dist = np.min(d)
            if min_dist < 1e-5:
                penalty += 1e6 * (1e-5 - min_dist) ** 2
            
            log_sum = np.sum(np.log(np.maximum(d, 1e-10)))
            return -log_sum + penalty
        
        try:
            res = minimize(
                penalized_obj,
                x0,
                method='Nelder-Mead',
                options={'maxiter': maxiter, 'xatol': tol, 'fatol': tol, 'adaptive': True}
            )
            pts = build_pts(res.x)
            dists = compute_dists(pts)
            D = np.max(dists)
            mask = np.triu(np.ones((n, n), dtype=bool), k=1)
            if D <= 2.0 + 1e-4 and np.min(dists[mask]) > 1e-5:
                return pts
            return None
        except Exception:
            return None

    # Run 1: From Octagon with SLSQP (smooth penalty)
    if time.time() - start_time < deadline:
        pts = run_slsqp_penalty(init_x, 800, 1e-12)
        if pts is not None:
            cand_score = score(pts)
            if cand_score > best_score:
                best_score = cand_score
                best_points = pts

    # Run 2: Nelder-Mead polish from SLSQP result
    if time.time() - start_time < deadline - 5:
        x0 = best_points[2:].flatten()
        pts = run_nelder_mead(x0, 8000, 1e-12)
        if pts is not None:
            cand_score = score(pts)
            if cand_score > best_score:
                best_score = cand_score
                best_points = pts

    # Run 3..N: Random restarts
    np.random.seed(42)
    max_restarts = 20
    for k in range(max_restarts):
        if time.time() - start_time >= deadline - 30:
            break
        
        pts_init = np.zeros((n, 2))
        pts_init[0] = [-1.0, 0.0]
        pts_init[1] = [1.0, 0.0]
        for i in range(2, n):
            x = np.random.uniform(-1.0, 1.0)
            max_y_sq = 3.0 - x*x - 2.0*abs(x)
            if max_y_sq < 0:
                max_y_sq = 0.0
            max_y = np.sqrt(max_y_sq)
            y = np.random.uniform(-max_y, max_y)
            pts_init[i] = [x, y]
        
        x0 = pts_init[2:].flatten()
        
        # SLSQP with smooth penalty
        pts = run_slsqp_penalty(x0, 500, 1e-11)
        if pts is not None:
            cand_score = score(pts)
            if cand_score > best_score:
                best_score = cand_score
                best_points = pts
                x0 = pts[2:].flatten()
        
        # Nelder-Mead polish
        if time.time() - start_time < deadline - 15:
            pts = run_nelder_mead(x0, 5000, 1e-11)
            if pts is not None:
                cand_score = score(pts)
                if cand_score > best_score:
                    best_score = cand_score
                    best_points = pts

    # Final high-precision Nelder-Mead polish
    if time.time() - start_time < deadline - 10:
        x_best = best_points[2:].flatten()
        pts = run_nelder_mead(x_best, 15000, 1e-14)
        if pts is not None:
            cand_score = score(pts)
            if cand_score > best_score:
                best_score = cand_score
                best_points = pts

    # Coordinate descent with binary search refinement
    if time.time() - start_time < deadline - 5:
        pts_cd = best_points.copy()
        for sweep in range(5):
            if time.time() - start_time >= deadline - 3:
                break
            improved = False
            for i in range(n):
                for dim in range(2):
                    base = pts_cd[i, dim]
                    best_val = base
                    best_sc = score(pts_cd)
                    
                    # Try a range of step sizes with binary search refinement
                    for step in [0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]:
                        for direction in [-1, 1]:
                            candidate = pts_cd.copy()
                            candidate[i, dim] = base + direction * step
                            dists = compute_dists(candidate)
                            D = np.max(dists)
                            mask = np.triu(np.ones((n, n), dtype=bool), k=1)
                            if D > 2.0 + 1e-8 or np.min(dists[mask]) < 1e-8:
                                continue
                            sc = score(candidate)
                            if sc > best_sc:
                                best_sc = sc
                                best_val = candidate[i, dim]
                        # Binary search refinement if improvement found
                        if best_val != base:
                            lo = min(base, best_val)
                            hi = max(base, best_val)
                            for _ in range(20):
                                mid = (lo + hi) / 2.0
                                candidate = pts_cd.copy()
                                candidate[i, dim] = mid
                                dists = compute_dists(candidate)
                                D = np.max(dists)
                                mask = np.triu(np.ones((n, n), dtype=bool), k=1)
                                if D > 2.0 + 1e-8 or np.min(dists[mask]) < 1e-8:
                                    # Try the other side
                                    if mid > base:
                                        hi = mid
                                    else:
                                        lo = mid
                                    continue
                                sc = score(candidate)
                                if sc > best_sc:
                                    best_sc = sc
                                    best_val = mid
                                    if mid > base:
                                        lo = mid
                                    else:
                                        hi = mid
                                else:
                                    if mid > base:
                                        hi = mid
                                    else:
                                        lo = mid
                    
                    if abs(best_val - pts_cd[i, dim]) > 1e-12:
                        pts_cd[i, dim] = best_val
                        improved = True
            if not improved:
                break
        cand_score = score(pts_cd)
        if cand_score > best_score:
            best_score = cand_score
            best_points = pts_cd

    # One more NM polish if time allows
    if time.time() - start_time < deadline - 2:
        x_best = best_points[2:].flatten()
        pts = run_nelder_mead(x_best, 10000, 1e-15)
        if pts is not None:
            cand_score = score(pts)
            if cand_score > best_score:
                best_score = cand_score
                best_points = pts

    return best_points.tolist()
