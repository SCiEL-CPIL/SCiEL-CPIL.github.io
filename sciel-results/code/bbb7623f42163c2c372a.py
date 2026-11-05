import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    n = 8
    N_pairs = n * (n - 1)  # 56 ordered pairs

    def exact_log_score(points):
        """
        Computes log of score = prod(d_ij_ordered) / n^n after scaling to diameter 2.
        """
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        mask = ~np.eye(n, dtype=bool)
        off_diag_dists_sq = dists_sq[mask]
        
        if np.any(off_diag_dists_sq < 1e-24):
            return -np.inf
            
        off_diag_dists = np.sqrt(off_diag_dists_sq)
        D = np.max(off_diag_dists)
        if D < 1e-12:
            return -np.inf
            
        scaled_dists = off_diag_dists * (2.0 / D)
        scaled_dists = np.maximum(scaled_dists, 1e-15)
        
        log_prod = np.sum(np.log(scaled_dists))
        return log_prod - n * np.log(n)

    def points_from_symmetry(r1, r2, theta1, theta2):
        """
        Construct 8 points from two squares.
        Square 1: radius r1, rotation theta1
        Square 2: radius r2, rotation theta2
        """
        a1 = 2 * np.pi * np.arange(4) / 4 + theta1
        a2 = 2 * np.pi * np.arange(4) / 4 + theta2
        
        pts1 = np.column_stack([r1 * np.cos(a1), r1 * np.sin(a1)])
        pts2 = np.column_stack([r2 * np.cos(a2), r2 * np.sin(a2)])
        
        return np.vstack([pts1, pts2])

    def neg_log_score_sym(params):
        """
        Objective for minimization: -log_score.
        params: [r1, r2, theta1, theta2]
        """
        r1, r2, theta1, theta2 = params
        if r1 < 1e-6 or r2 < 1e-6:
            return 1e10
        pts = points_from_symmetry(r1, r2, theta1, theta2)
        return -exact_log_score(pts)

    # 1. Initialize with regular octagon (r1=r2=1, theta1=0, theta2=pi/4)
    best_score = -np.inf
    best_params = np.array([1.0, 1.0, 0.0, np.pi/4.0])
    best_points = points_from_symmetry(*best_params)
    best_score = exact_log_score(best_points)

    # 2. Grid Search over symmetry parameters
    # Radii ratio and rotation angle are the key degrees of freedom.
    # We can fix r1=1 due to scaling invariance (diameter scaling).
    # Let r2 = r1 * ratio.
    # theta1 = 0 (fix rotation of first square).
    # theta2 = rotation relative to first square.
    
    # Search space:
    # ratio: 0.5 to 2.0 (symmetric, so 0.5 to 1.0 is sufficient if we swap, but let's be safe)
    # theta2: 0 to pi/2 (symmetry of square: pi/2 period)
    
    # Coarse grid
    ratios = np.linspace(0.5, 1.5, 20)
    thetas = np.linspace(0.0, np.pi/2, 20)
    
    for r in ratios:
        for t in thetas:
            if time.time() > deadline - 60:
                break
            params = np.array([1.0, r, 0.0, t])
            s = exact_log_score(points_from_symmetry(*params))
            if s > best_score:
                best_score = s
                best_params = params
                best_points = points_from_symmetry(*params)
                
    # 3. Local Optimization (Polish)
    # Use L-BFGS-B to refine the best candidate from the grid.
    # We can also try polishing a few of the top candidates from the grid.
    
    # Let's collect top candidates from grid to ensure we don't miss a local opt.
    # Actually, just polishing the single best is usually enough if the grid is fine.
    # But let's do a few random restarts within the symmetry manifold as well.
    
    candidates_to_polish = [best_params.copy()]
    
    # Add some random perturbations around the best
    np.random.seed(42)
    for _ in range(5):
        pert = np.random.normal(0, 0.1, 4)
        p = best_params + pert
        # Ensure radii are positive
        if p[0] > 0.1 and p[1] > 0.1:
            candidates_to_polish.append(p)

    # Also try the regular octagon explicitly
    candidates_to_polish.append(np.array([1.0, 1.0, 0.0, np.pi/4.0]))
    
    for p_init in candidates_to_polish:
        if time.time() > deadline - 10:
            break
        try:
            res = minimize(
                neg_log_score_sym,
                p_init,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if res.fun < np.inf:
                candidate_params = res.x
                # Ensure radii are positive
                if candidate_params[0] > 1e-6 and candidate_params[1] > 1e-6:
                    candidate_points = points_from_symmetry(*candidate_params)
                    s = exact_log_score(candidate_points)
                    if s > best_score:
                        best_score = s
                        best_params = candidate_params
                        best_points = candidate_points
        except Exception:
            pass

    # 4. If symmetry assumption is too restrictive, try a global search in 16D?
    # The prompt suggests the symmetry manifold is likely.
    # However, to be safe, let's check if a quick DE on the full space improves things.
    # But DE is slow. Given the budget, if the symmetry search found something > 1.0, we trust it.
    # If it's still 1.0, maybe we should try a bit more.
    
    # Let's check the score.
    # If best_score is close to 0 (log scale), it's ~1.0.
    # log(1) = 0.
    
    # If we are stuck at 1.0, we might need to break symmetry.
    # But the directive says "Make a STRUCTURAL change... Model the 8 points as two concentric squares".
    # So we stick to this model.
    
    # Final validation
    diff = best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    min_d = np.sqrt(np.min(dists_sq))
    
    # Fallback to regular octagon if points collapsed or score is bad
    reg_oct = points_from_symmetry(1.0, 1.0, 0.0, np.pi/4.0)
    reg_score = exact_log_score(reg_oct)
    
    if min_d < 1e-8 or best_score < reg_score - 1e-6:
        best_points = reg_oct

    max_coord = np.max(np.abs(best_points))
    if max_coord > 1e6:
        best_points = best_points / max_coord * 1e5

    return [[float(p[0]), float(p[1])] for p in best_points]

if __name__ == "__main__":
    pts = construct()
    if pts:
        print("Constructed 8 points.")
        print(pts[0])
