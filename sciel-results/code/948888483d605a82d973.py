import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13
_DEADLINE_MARGIN = 10.0  # seconds before hard kill

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    time_limit = 150.0 - _DEADLINE_MARGIN  # Leave 10s margin for safety

    # Precompute indices for all triples
    triples = list(combinations(range(N), 3))
    n_triples = len(triples)

    i_idx = np.array([t[0] for t in triples])
    j_idx = np.array([t[1] for t in triples])
    k_idx = np.array([t[2] for t in triples])

    def compute_score(points):
        """Compute min triangle area / convex hull area for given points."""
        try:
            hull = ConvexHull(points)
            area_hull = hull.volume
        except Exception:
            return 0.0, 0.0

        if area_hull < 1e-9:
            return 0.0, 0.0

        x = points[:, 0]
        y = points[:, 1]
        v1x = x[j_idx] - x[i_idx]
        v1y = y[j_idx] - y[i_idx]
        v2x = x[k_idx] - x[i_idx]
        v2y = y[k_idx] - y[i_idx]
        areas = 0.5 * np.abs(v1x * v2y - v1y * v2x)
        min_area = np.min(areas)
        score = min_area / area_hull
        return score, area_hull

    def get_worst_triangles(points, n_worst=5):
        """Return indices of the n_worst smallest triangles."""
        x = points[:, 0]
        y = points[:, 1]
        v1x = x[j_idx] - x[i_idx]
        v1y = y[j_idx] - y[i_idx]
        v2x = x[k_idx] - x[i_idx]
        v2y = y[k_idx] - y[i_idx]
        areas = 0.5 * np.abs(v1x * v2y - v1y * v2x)
        worst_indices = np.argsort(areas)[:n_worst]
        return worst_indices, areas[worst_indices]

    def _soft_min_score(points, beta=50.0):
        """Smooth surrogate using log-sum-exp for min triangle area."""
        try:
            hull = ConvexHull(points)
            area_hull = hull.volume
        except Exception:
            return 1e6

        if area_hull < 1e-9:
            return 1e6

        x = points[:, 0]
        y = points[:, 1]
        v1x = x[j_idx] - x[i_idx]
        v1y = y[j_idx] - y[i_idx]
        v2x = x[k_idx] - x[i_idx]
        v2y = y[k_idx] - y[i_idx]
        areas = 0.5 * np.abs(v1x * v2y - v1y * v2x)

        neg_beta_areas = -beta * areas
        max_val = np.max(neg_beta_areas)
        if max_val < -1e6:
            return 1e6
        log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_areas - max_val)))
        soft_min_area = -log_sum_exp / beta
        score = soft_min_area / area_hull
        return -score

    def _epigraph_objective(params):
        """
        Epigraph formulation: maximize t subject to all triangle areas >= t * hull_area.
        params = [x1, y1, ..., x13, y13, t]
        We minimize -t with constraints that each triangle area >= t * hull_area.
        For simplicity in L-BFGS-B, we use a penalty approach.
        """
        points = params[:2*N].reshape(N, 2)
        t = params[2*N]
        
        try:
            hull = ConvexHull(points)
            area_hull = hull.volume
        except Exception:
            return 1e6

        if area_hull < 1e-9:
            return 1e6

        x = points[:, 0]
        y = points[:, 1]
        v1x = x[j_idx] - x[i_idx]
        v1y = y[j_idx] - y[i_idx]
        v2x = x[k_idx] - x[i_idx]
        v2y = y[k_idx] - y[i_idx]
        areas = 0.5 * np.abs(v1x * v2y - v1y * v2x)
        
        # Penalty for violating areas >= t * area_hull
        violations = np.maximum(0, t * area_hull - areas)
        penalty = np.sum(violations ** 2)
        
        # Objective: minimize -t + penalty
        return -t + 100.0 * penalty

    def _refine_worst_triangles(points, max_iters=20, n_worst=10):
        """
        Refine points by optimizing only the coordinates involved in the worst triangles.
        """
        current = points.copy()
        current_score, _ = compute_score(current)
        
        for it in range(max_iters):
            if time.time() > start_time + time_limit - 5:
                break
            
            worst_idx, worst_areas = get_worst_triangles(current, n_worst=n_worst)
            if len(worst_idx) == 0:
                break
            
            # Get unique point indices involved in these worst triangles
            involved_points = set()
            for wi in worst_idx:
                tri = triples[wi]
                involved_points.update([tri[0], tri[1], tri[2]])
            
            if len(involved_points) < 3:
                break
                
            involved_indices = np.array(sorted(involved_points))
            
            # Extract current coordinates of involved points
            x0 = current[involved_indices].flatten()
            
            # Define local objective: minimize negative min area of the specific worst triangles
            # We only care about the triangles in 'worst_idx' for the local step
            # To be safe, we can also penalize if we make *other* triangles significantly worse,
            # but for a local step, focusing on the bottleneck is standard.
            
            def local_obj(x_flat):
                # Reconstruct full points
                p = current.copy()
                p[involved_indices] = x_flat.reshape(-1, 2)
                
                # Compute areas of the specific worst triangles
                # We need to map the global triple indices to the local calculation
                # Since we are only changing specific points, we can just recompute the areas for those triples
                areas_local = np.zeros(len(worst_idx))
                for k, wi in enumerate(worst_idx):
                    i, j, kk = triples[wi]
                    v1 = p[j] - p[i]
                    v2 = p[kk] - p[i]
                    areas_local[k] = 0.5 * np.abs(v1[0]*v2[1] - v1[1]*v2[0])
                
                # We want to maximize the minimum of these areas
                # Use a soft-min to keep it smooth
                beta = 50.0
                neg_beta_areas = -beta * areas_local
                max_val = np.max(neg_beta_areas)
                if max_val < -1e6:
                    return 1e6
                log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_areas - max_val)))
                soft_min = -log_sum_exp / beta
                return -soft_min
            
            try:
                res = opt.minimize(
                    local_obj,
                    x0,
                    method='Nelder-Mead',
                    options={'maxiter': 100, 'adaptive': True, 'xatol': 1e-10, 'fatol': 1e-12}
                )
                new_coords = res.x.reshape(-1, 2)
                new_points = current.copy()
                new_points[involved_indices] = new_coords
                
                new_score, _ = compute_score(new_points)
                if new_score > current_score:
                    current = new_points
                    current_score = new_score
                else:
                    # If no improvement, break early to save time? Or continue with smaller steps?
                    # Nelder-Mead might have converged. Let's break if improvement is tiny.
                    if current_score - new_score < 1e-10:
                         break
            except Exception:
                break
                
        return current, current_score

    def _hard_score_neg(points_flat):
        """Negative hard score for final evaluation."""
        points = points_flat.reshape(-1, 2)
        score, _ = compute_score(points)
        return -score

    def _soft_objective(points_flat, beta=50.0):
        """Soft-min objective for gradient-based optimization."""
        points = points_flat.reshape(-1, 2)
        return _soft_min_score(points, beta)

    def _make_symmetric_points(n_hull, n_inner, r_hull, r_inner, rot_hull, rot_inner):
        """
        Generate points with two concentric regular polygons.
        n_hull + n_inner = 13
        """
        if n_inner == 0:
            angles_hull = np.linspace(rot_hull, rot_hull + 2*np.pi, n_hull, endpoint=False)
            xs = r_hull * np.cos(angles_hull)
            ys = r_hull * np.sin(angles_hull)
            return np.column_stack([xs, ys])
        
        angles_hull = np.linspace(rot_hull, rot_hull + 2*np.pi, n_hull, endpoint=False)
        angles_inner = np.linspace(rot_inner, rot_inner + 2*np.pi, n_inner, endpoint=False)
        
        xs = np.concatenate([r_hull * np.cos(angles_hull), r_inner * np.cos(angles_inner)])
        ys = np.concatenate([r_hull * np.sin(angles_hull), r_inner * np.sin(angles_inner)])
        return np.column_stack([xs, ys])

    def _symmetric_objective(params, n_hull, n_inner, beta=50.0):
        """
        Objective for symmetric parameterization.
        params = [r_hull, r_inner, rot_hull, rot_inner]
        """
        r_hull, r_inner, rot_hull, rot_inner = params
        if r_hull < 0.1 or r_inner < 0.0 or r_hull < r_inner:
            return 1e6
        points = _make_symmetric_points(n_hull, n_inner, r_hull, r_inner, rot_hull, rot_inner)
        return _soft_min_score(points, beta)

    def _symmetric_hard_objective(params, n_hull, n_inner):
        """Hard score for symmetric parameterization."""
        r_hull, r_inner, rot_hull, rot_inner = params
        if r_hull < 0.1 or r_inner < 0.0 or r_hull < r_inner:
            return 1e6
        points = _make_symmetric_points(n_hull, n_inner, r_hull, r_inner, rot_hull, rot_inner)
        score, _ = compute_score(points)
        return -score

    best_points = None
    best_score = -float('inf')

    # Strategy 1: Optimize symmetric configurations (k-gon + m-gon)
    # Focus on the most promising splits
    splits = [(12, 1), (11, 2), (10, 3), (9, 4), (8, 5), (7, 6)]

    for n_hull, n_inner in splits:
        if time.time() > start_time + time_limit - 30:
            break

        # Initial guess
        r_hull_init = 1.0
        r_inner_init = 0.4 if n_inner > 0 else 0.0
        rot_hull_init = 0.0
        rot_inner_init = np.pi / n_inner if n_inner > 1 else 0.0

        # Phase 1: Coarse optimization with soft-min
        if time.time() > start_time + time_limit - 30:
            break

        x0 = np.array([r_hull_init, r_inner_init, rot_hull_init, rot_inner_init])
        
        for beta in [20.0, 50.0, 100.0]:
            if time.time() > start_time + time_limit - 30:
                break
            try:
                result = opt.minimize(
                    lambda x, b=beta, nh=n_hull, ni=n_inner: _symmetric_objective(x, nh, ni, b),
                    x0,
                    method='L-BFGS-B',
                    bounds=[(0.3, 2.0), (0.0, 1.0), (-np.pi, np.pi), (-np.pi, np.pi)],
                    options={'maxiter': 60, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                x0 = result.x
                points = _make_symmetric_points(n_hull, n_inner, *x0)
                s, _ = compute_score(points)
                if s > best_score:
                    best_score = s
                    best_points = points.copy()
            except Exception:
                pass

        # Phase 2: Fine-tune with hard score using Nelder-Mead
        if time.time() > start_time + time_limit - 30:
            break

        try:
            res = opt.minimize(
                lambda x, nh=n_hull, ni=n_inner: _symmetric_hard_objective(x, nh, ni),
                x0,
                method='Nelder-Mead',
                options={'maxiter': 100, 'adaptive': True, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            x0 = res.x
            points = _make_symmetric_points(n_hull, n_inner, *x0)
            s, _ = compute_score(points)
            if s > best_score:
                best_score = s
                best_points = points.copy()
        except Exception:
            pass

    # Strategy 2: Full free-space optimization from best symmetric result
    if best_points is not None and time.time() < start_time + time_limit - 25:
        current = best_points.copy()
        
        # Multi-beta annealing in free space
        for beta in [30.0, 80.0, 150.0]:
            if time.time() > start_time + time_limit - 25:
                break
            try:
                result = opt.minimize(
                    lambda x, b=beta: _soft_objective(x, b),
                    current.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                optimized = result.x.reshape(-1, 2)
                opt_score, _ = compute_score(optimized)
                if opt_score > best_score:
                    best_score = opt_score
                    best_points = optimized.copy()
                current = optimized
            except Exception:
                pass
        
        # Strategy 2b: Targeted worst-triangle optimization
        if best_points is not None and time.time() < start_time + time_limit - 15:
            targeted, targeted_score = _refine_worst_triangles(best_points, max_iters=25, n_worst=10)
            if targeted_score > best_score:
                best_score = targeted_score
                best_points = targeted.copy()

    # Strategy 3: Random restarts in free space (reduced)
    if time.time() < start_time + time_limit - 25:
        np.random.seed(42)
        n_restarts = 3
        for _ in range(n_restarts):
            if time.time() > start_time + time_limit - 25:
                break
            
            # Generate random configuration: mix of hull and interior points
            n_hull = np.random.randint(8, 13)
            n_inner = N - n_hull
            r_hull = np.random.uniform(0.9, 1.2)
            r_inner = np.random.uniform(0.2, 0.5) if n_inner > 0 else 0.0
            rot_hull = np.random.uniform(0, 2*np.pi)
            rot_inner = np.random.uniform(0, 2*np.pi) if n_inner > 1 else 0.0
            
            config = _make_symmetric_points(n_hull, n_inner, r_hull, r_inner, rot_hull, rot_inner)
            # Add small noise
            config = config + np.random.normal(0, 0.05, config.shape)

            s, _ = compute_score(config)
            if s > best_score:
                best_score = s
                best_points = config.copy()

            # Quick optimization
            if time.time() > start_time + time_limit - 25:
                break

            current = config.copy()
            for beta in [50.0, 100.0]:
                if time.time() > start_time + time_limit - 25:
                    break
                try:
                    result = opt.minimize(
                        lambda x, b=beta: _soft_objective(x, b),
                        current.flatten(),
                        method='L-BFGS-B',
                        options={'maxiter': 30, 'ftol': 1e-12, 'gtol': 1e-10}
                    )
                    optimized = result.x.reshape(-1, 2)
                    opt_score, _ = compute_score(optimized)
                    if opt_score > best_score:
                        best_score = opt_score
                        best_points = optimized.copy()
                    current = optimized
                except Exception:
                    pass

    # Final high-precision polish - this is where the big gains come from
    if best_points is not None and time.time() < start_time + time_limit - 10:
        # Stage 1: High-beta soft-min polish
        for beta in [200.0, 400.0, 600.0]:
            if time.time() > start_time + time_limit - 10:
                break
            try:
                result = opt.minimize(
                    lambda x, b=beta: _soft_objective(x, b),
                    best_points.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 60, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                polished = result.x.reshape(-1, 2)
                pol_score, _ = compute_score(polished)
                if pol_score > best_score:
                    best_score = pol_score
                    best_points = polished.copy()
            except Exception:
                pass

        # Stage 2: Targeted worst-triangle refinement (multiple passes)
        for _ in range(3):
            if best_points is not None and time.time() < start_time + time_limit - 8:
                targeted, targeted_score = _refine_worst_triangles(best_points, max_iters=30, n_worst=12)
                if targeted_score > best_score:
                    best_score = targeted_score
                    best_points = targeted.copy()
            else:
                break

        # Stage 3: Final Nelder-Mead polish with very tight tolerances
        if time.time() < start_time + time_limit - 5:
            try:
                res = opt.minimize(
                    _hard_score_neg,
                    best_points.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'adaptive': True, 'xatol': 1e-12, 'fatol': 1e-14}
                )
                final_points = res.x.reshape(-1, 2)
                final_score, _ = compute_score(final_points)
                if final_score > best_score:
                    best_score = final_score
                    best_points = final_points.copy()
            except Exception:
                pass

        # Stage 4: One more round of targeted refinement after Nelder-Mead
        if best_points is not None and time.time() < start_time + time_limit - 3:
            targeted, targeted_score = _refine_worst_triangles(best_points, max_iters=40, n_worst=15)
            if targeted_score > best_score:
                best_score = targeted_score
                best_points = targeted.copy()

    if best_points is None:
        # Fallback: regular 13-gon
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])

    return best_points

# EVOLVE-BLOCK-END
