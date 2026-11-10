import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin for 150s limit

    n = 8
    
    # Precompute pair indices for 28 unordered pairs
    pair_indices = np.array([(i, j) for i in range(n) for j in range(i + 1, n)])
    idx_i = pair_indices[:, 0]
    idx_j = pair_indices[:, 1]
    
    def calculate_score(points):
        """
        points: 8x2 array
        Returns: log of the scaled product of distances.
        Score = Delta / 8^8, where Delta is product of ordered pair distances.
        Delta = (prod_{i<j} d_ij)^2.
        log(Score) = 2 * sum_{i<j} log(d_ij) - 8 * log(8).
        Scaling: d_scaled = d * (2 / D).
        log(d_scaled) = log(d) + log(2) - log(D).
        Sum_{i<j} log(d_scaled) = Sum log(d) + 28 * log(2) - 28 * log(D).
        log(Score) = 2 * [Sum log(d) + 28 * log(2) - 28 * log(D)] - 8 * log(8).
        """
        diff = points[idx_i, :] - points[idx_j, :]
        dists = np.sqrt(np.sum(diff**2, axis=1))
        if np.any(dists < 1e-12):
            return -np.inf
        D = np.max(dists)
        if D < 1e-12:
            return -np.inf
        sum_log_d = np.sum(np.log(dists))
        log_scaled_sum = sum_log_d + 28.0 * np.log(2.0) - 28.0 * np.log(D)
        log_score = 2.0 * log_scaled_sum - 8.0 * np.log(8.0)
        return log_score

    def objective_16(x):
        """Objective for 16D optimization (8 points x 2 coords)."""
        if time.time() > deadline:
            return 1e10
        pts = x.reshape(8, 2)
        s = calculate_score(pts)
        if np.isinf(s) or np.isnan(s):
            return 1e10
        return -s

    def build_two_squares(R1, R2, theta):
        """
        Build 8 points from two squares.
        Square 1: radius R1, angles 0, pi/2, pi, 3pi/2.
        Square 2: radius R2, angles theta, theta+pi/2, theta+pi, theta+3pi/2.
        Centered at origin.
        """
        angles = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
        p1 = np.column_stack([R1 * np.cos(angles), R1 * np.sin(angles)])
        p2 = np.column_stack([R2 * np.cos(angles + theta), R2 * np.sin(angles + theta)])
        return np.vstack([p1, p2])

    def objective_3d(params):
        """Objective for 3D optimization (R1, R2, theta)."""
        if time.time() > deadline - 10.0:
            return 1e10
        R1, R2, theta = params
        if R1 < 0.05 or R2 < 0.05:
            return 1e10
        pts = build_two_squares(R1, R2, theta)
        s = calculate_score(pts)
        if np.isinf(s) or np.isnan(s):
            return 1e10
        return -s

    best_score = -np.inf
    best_points = None

    # 1. Regular Octagon Baseline
    angles_oct = np.linspace(0, 2*np.pi, 8, endpoint=False)
    pts_oct = np.column_stack([np.cos(angles_oct), np.sin(angles_oct)])
    s_oct = calculate_score(pts_oct)
    if s_oct > best_score:
        best_score = s_oct
        best_points = pts_oct.copy()

    # 2. Global Search on 3D Two-Square Manifold
    # This is faster and more robust than 5D or 16D global search.
    # The optimal solution is likely to have high symmetry or be close to it.
    bounds_3d = [
        (0.1, 3.0),   # R1
        (0.1, 3.0),   # R2
        (0.0, np.pi/4)# theta (symmetry allows 0 to pi/4)
    ]
    
    candidates = []
    
    try:
        res_de = differential_evolution(
            objective_3d,
            bounds_3d,
            maxiter=200,
            popsize=30,
            tol=1e-10,
            seed=42,
            init='sobol',
            workers=1,
            polish=False
        )
        if res_de.fun < 1e9:
            x_opt = res_de.x
            pts_cand = build_two_squares(*x_opt)
            s_cand = calculate_score(pts_cand)
            candidates.append((s_cand, pts_cand))
            if s_cand > best_score:
                best_score = s_cand
                best_points = pts_cand.copy()
    except Exception:
        pass

    # 3. Local Refinement on 16D Space
    # Take the best candidates from the global search and the baseline,
    # and refine them in the full 16D space. This allows breaking symmetry
    # if it leads to a higher score.
    
    # Add octagon to candidates list for refinement
    candidates.append((s_oct, pts_oct.copy()))
    
    # Sort candidates by score to prioritize refinement of the best ones
    candidates.sort(key=lambda t: -t[0])
    
    # Refine top candidates
    for s_init, pts_init in candidates:
        if time.time() > deadline - 15.0:
            break
        
        x_init = pts_init.flatten()
        
        # Step 3a: Powell Method (good for initial convergence)
        try:
            res_powell = minimize(
                objective_16,
                x_init,
                method='Powell',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            x_best = res_powell.x
            s_powell = -res_powell.fun
            if np.isfinite(s_powell) and s_powell > -1e9:
                if s_powell > best_score:
                    best_score = s_powell
                    best_points = x_best.reshape(8, 2).copy()
        except Exception:
            x_best = x_init
            s_powell = s_init

        # Step 3b: Nelder-Mead Polish (high precision)
        if time.time() < deadline - 5.0:
            try:
                res_nm = minimize(
                    objective_16,
                    x_best,
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
                )
                x_nm = res_nm.x
                s_nm = -res_nm.fun
                if np.isfinite(s_nm) and s_nm > best_score:
                    best_score = s_nm
                    best_points = x_nm.reshape(8, 2).copy()
            except Exception:
                pass

    # 4. Final Safety Polish
    if best_points is not None and time.time() < deadline - 2.0:
        try:
            res_final = minimize(
                objective_16,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
            )
            s_final = -res_final.fun
            if np.isfinite(s_final) and s_final > best_score:
                best_score = s_final
                best_points = res_final.x.reshape(8, 2).copy()
        except Exception:
            pass

    if best_points is None:
        best_points = pts_oct

    # Verify distinctness
    diffs = best_points[:, np.newaxis, :] - best_points[:, :, np.newaxis]
    d = np.sqrt(np.sum(diffs**2, axis=2))
    np.fill_diagonal(d, 1.0)
    if np.min(d) < 1e-6:
        # Fallback to octagon if points collapsed
        best_points = pts_oct

    # Scale to diameter 2
    D = np.max(d)
    if D > 0:
        best_points = best_points * (2.0 / D)
        
    return [[float(p[0]), float(p[1])] for p in best_points]
