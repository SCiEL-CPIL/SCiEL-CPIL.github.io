import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    time_limit = 145.0  # Leave 5s margin for safety

    # Precompute indices for all triples
    triples = list(combinations(range(N), 3))
    n_triples = len(triples)

    i_idx = np.array([t[0] for t in triples])
    j_idx = np.array([t[1] for t in triples])
    k_idx = np.array([t[2] for t in triples])

    def _triangle_areas(points):
        """Compute all triangle areas for given points."""
        x = points[:, 0]
        y = points[:, 1]
        v1x = x[j_idx] - x[i_idx]
        v1y = y[j_idx] - y[i_idx]
        v2x = x[k_idx] - x[i_idx]
        v2y = y[k_idx] - y[i_idx]
        return 0.5 * np.abs(v1x * v2y - v1y * v2x)

    def _min_triangle_area(points):
        """Compute minimum triangle area."""
        return np.min(_triangle_areas(points))

    def _hull_area(points):
        """Compute convex hull area."""
        try:
            hull = ConvexHull(points)
            return hull.volume
        except Exception:
            return 0.0

    def _score(points):
        """Compute min triangle area / convex hull area."""
        area_hull = _hull_area(points)
        if area_hull < 1e-9:
            return 0.0
        min_area = _min_triangle_area(points)
        return min_area / area_hull

    def _soft_min(points, beta=50.0):
        """Smooth surrogate using log-sum-exp for min triangle area."""
        area_hull = _hull_area(points)
        if area_hull < 1e-9:
            return 1e6

        areas = _triangle_areas(points)
        neg_beta_areas = -beta * areas
        max_val = np.max(neg_beta_areas)
        if max_val < -1e6:
            return 1e6
        log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_areas - max_val)))
        soft_min_area = -log_sum_exp / beta
        return -soft_min_area / area_hull

    def _neg_soft_score(points_flat, beta=50.0):
        """Negative soft score for optimization."""
        points = points_flat.reshape(-1, 2)
        return _soft_min(points, beta)

    def _neg_score(points_flat):
        """Negative hard score for final evaluation."""
        points = points_flat.reshape(-1, 2)
        return -_score(points)

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
        return _soft_min(points, beta)

    def _symmetric_hard_objective(params, n_hull, n_inner):
        """Hard score for symmetric parameterization."""
        r_hull, r_inner, rot_hull, rot_inner = params
        if r_hull < 0.1 or r_inner < 0.0 or r_hull < r_inner:
            return 1e6
        points = _make_symmetric_points(n_hull, n_inner, r_hull, r_inner, rot_hull, rot_inner)
        return -_score(points)

    def _refine_soft_min(points, beta=50.0, maxiter=100):
        """Refine points using L-BFGS-B with soft-min objective."""
        try:
            result = opt.minimize(
                lambda x, b=beta: _neg_soft_score(x, b),
                points.flatten(),
                method='L-BFGS-B',
                options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            return result.x.reshape(-1, 2)
        except Exception:
            return points

    def _refine_nelder_mead(points, maxiter=200):
        """Refine points using Nelder-Mead with hard score."""
        try:
            result = opt.minimize(
                _neg_score,
                points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': maxiter, 'adaptive': True, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            return result.x.reshape(-1, 2)
        except Exception:
            return points

    def _targeted_worst_triangle_opt(points, max_iter=50):
        """
        Targeted optimization focusing on the worst triangle.
        For each iteration, find the worst triangle and optimize to increase its area.
        """
        current = points.copy()
        
        for iteration in range(max_iter):
            if time.time() > start_time + time_limit - 5:
                break
                
            # Find the worst triangle
            areas = _triangle_areas(current)
            worst_idx = np.argmin(areas)
            worst_area = areas[worst_idx]
            
            if worst_area < 1e-10:
                break
            
            # Get the three points of the worst triangle
            i, j, k = triples[worst_idx]
            
            # Compute the area as a function of point positions
            # Area = 0.5 * |cross(P_j - P_i, P_k - P_i)|
            # We want to maximize this area
            
            def worst_triangle_area(params):
                pts = params.reshape(-1, 2)
                pi = pts[i]
                pj = pts[j]
                pk = pts[k]
                v1 = pj - pi
                v2 = pk - pi
                return 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
            
            def neg_worst_triangle_area(params):
                return -worst_triangle_area(params)
            
            # Optimize to increase the worst triangle area
            try:
                result = opt.minimize(
                    neg_worst_triangle_area,
                    current.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 20, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                new_points = result.x.reshape(-1, 2)
                
                # Check if this improves the overall score
                old_score = _score(current)
                new_score = _score(new_points)
                
                if new_score > old_score:
                    current = new_points
                else:
                    # Try a smaller step
                    step_size = 0.1
                    for _ in range(5):
                        new_points = current + step_size * (new_points - current)
                        new_score = _score(new_points)
                        if new_score > old_score:
                            current = new_points
                            break
                        step_size *= 0.5
            except Exception:
                break
        
        return current

    def _symmetric_optimize(n_hull, n_inner):
        """Optimize symmetric configuration."""
        # Initial guess
        r_hull_init = 1.0
        r_inner_init = 0.4 if n_inner > 0 else 0.0
        rot_hull_init = 0.0
        rot_inner_init = np.pi / n_inner if n_inner > 1 else 0.0

        x0 = np.array([r_hull_init, r_inner_init, rot_hull_init, rot_inner_init])
        best_local_points = None
        best_local_score = -float('inf')
        
        # Phase 1: Coarse optimization with soft-min
        for beta in [20.0, 50.0, 100.0]:
            if time.time() > start_time + time_limit - 30:
                break
            try:
                result = opt.minimize(
                    lambda x, b=beta: _symmetric_objective(x, n_hull, n_inner, b),
                    x0,
                    method='L-BFGS-B',
                    bounds=[(0.3, 2.0), (0.0, 1.0), (-np.pi, np.pi), (-np.pi, np.pi)],
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                x0 = result.x
                points = _make_symmetric_points(n_hull, n_inner, *x0)
                s = _score(points)
                if s > best_local_score:
                    best_local_score = s
                    best_local_points = points.copy()
            except Exception:
                pass

        # Phase 2: Fine-tune with hard score using Nelder-Mead
        if time.time() > start_time + time_limit - 30:
            return best_local_points, best_local_score

        try:
            res = opt.minimize(
                lambda x: _symmetric_hard_objective(x, n_hull, n_inner),
                x0,
                method='Nelder-Mead',
                options={'maxiter': 200, 'adaptive': True, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            x0 = res.x
            points = _make_symmetric_points(n_hull, n_inner, *x0)
            s = _score(points)
            if s > best_local_score:
                best_local_score = s
                best_local_points = points.copy()
        except Exception:
            pass

        return best_local_points, best_local_score

    # === NEW: Barycentric parameterization with normalized hull ===
    
    def _hull_area_from_vertices(hull_pts):
        """Compute area of convex polygon from ordered vertices using shoelace formula."""
        n = len(hull_pts)
        if n < 3:
            return 0.0
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += hull_pts[i, 0] * hull_pts[j, 1] - hull_pts[j, 0] * hull_pts[i, 1]
        return abs(area) * 0.5

    def _barycentric_to_point(bary_coords, hull_pts):
        """Convert barycentric coordinates relative to hull vertices to Cartesian point."""
        # bary_coords: (n_inner, n_hull)
        # hull_pts: (n_hull, 2)
        # Return: (n_inner, 2)
        return bary_coords @ hull_pts

    def _params_to_points(params, n_hull):
        """
        Convert parameter vector to point array.
        params layout: [hull_x1, hull_y1, ..., hull_x_n, hull_y_n, bary_inner1_1, bary_inner1_2, ..., bary_inner_m_n]
        where m = N - n_hull
        """
        n_inner = N - n_hull
        hull_flat = params[:2*n_hull]
        hull_pts = hull_flat.reshape(n_hull, 2)
        
        # Normalize hull to unit area
        hull_area = _hull_area_from_vertices(hull_pts)
        if hull_area < 1e-12:
            # Degenerate, return regular polygon
            angles = np.linspace(0, 2*np.pi, n_hull, endpoint=False)
            hull_pts = np.column_stack([np.cos(angles), np.sin(angles)])
            scale = 1.0 / np.sqrt(_hull_area_from_vertices(hull_pts))
            hull_pts = hull_pts * scale
        else:
            scale = 1.0 / np.sqrt(hull_area)
            hull_pts = hull_pts * scale
        
        if n_inner == 0:
            return hull_pts
        
        bary_flat = params[2*n_hull:]
        bary_coords = bary_flat.reshape(n_inner, n_hull)
        
        # Ensure barycentric coords are valid (positive, sum to 1)
        # Use softmax-like transformation to ensure positivity
        # Actually, let's just clip and normalize
        bary_coords = np.maximum(bary_coords, 1e-6)
        bary_coords = bary_coords / bary_coords.sum(axis=1, keepdims=True)
        
        inner_pts = _barycentric_to_point(bary_coords, hull_pts)
        
        return np.vstack([hull_pts, inner_pts])

    def _soft_min_normalized(params, n_hull, beta=50.0):
        """
        Soft-min objective with normalized hull area.
        Since hull area is 1 by construction, score = min_triangle_area.
        """
        points = _params_to_points(params, n_hull)
        areas = _triangle_areas(points)
        neg_beta_areas = -beta * areas
        max_val = np.max(neg_beta_areas)
        if max_val < -1e6:
            return 1e6
        log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_areas - max_val)))
        soft_min_area = -log_sum_exp / beta
        return -soft_min_area  # Maximize min area (hull area = 1)

    def _hard_score_normalized(params, n_hull):
        """Hard score with normalized hull area."""
        points = _params_to_points(params, n_hull)
        return -_min_triangle_area(points)  # Hull area = 1

    def _init_barycentric_params(n_hull, seed=42):
        """Initialize barycentric parameters for a given hull size."""
        np.random.seed(seed)
        n_inner = N - n_hull
        
        # Initialize hull as regular polygon
        angles = np.linspace(0, 2*np.pi, n_hull, endpoint=False)
        hull_pts = np.column_stack([np.cos(angles), np.sin(angles)])
        
        # Normalize to unit area
        hull_area = _hull_area_from_vertices(hull_pts)
        scale = 1.0 / np.sqrt(hull_area)
        hull_pts = hull_pts * scale
        
        params = list(hull_pts.flatten())
        
        if n_inner > 0:
            # Initialize interior points with random barycentric coords
            for i in range(n_inner):
                bary = np.random.dirichlet(np.ones(n_hull))
                params.extend(bary.tolist())
        
        return np.array(params)

    def _optimize_barycentric(n_hull, max_time=30.0):
        """Optimize using barycentric parameterization."""
        deadline = start_time + max_time
        x0 = _init_barycentric_params(n_hull)
        
        best_params = x0.copy()
        best_score = _score(_params_to_points(x0, n_hull))
        
        # Multi-beta annealing
        for beta in [30.0, 80.0, 150.0, 300.0]:
            if time.time() > deadline:
                break
            try:
                result = opt.minimize(
                    lambda x, b=beta: _soft_min_normalized(x, n_hull, b),
                    best_params,
                    method='L-BFGS-B',
                    options={'maxiter': 150, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                new_params = result.x
                new_score = _score(_params_to_points(new_params, n_hull))
                if new_score > best_score:
                    best_score = new_score
                    best_params = new_params
            except Exception:
                pass
        
        # Nelder-Mead polish
        if time.time() < deadline:
            try:
                result = opt.minimize(
                    lambda x: _hard_score_normalized(x, n_hull),
                    best_params,
                    method='Nelder-Mead',
                    options={'maxiter': 300, 'adaptive': True, 'xatol': 1e-12, 'fatol': 1e-14}
                )
                new_params = result.x
                new_score = _score(_params_to_points(new_params, n_hull))
                if new_score > best_score:
                    best_score = new_score
                    best_params = new_params
            except Exception:
                pass
        
        return _params_to_points(best_params, n_hull), best_score

    def _generate_initializations():
        """Generate diverse initial configurations."""
        inits = []
        
        # Regular 13-gon
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        inits.append(np.column_stack([np.cos(angles), np.sin(angles)]))
        
        # Two-ring configurations
        for n_hull in [10, 11, 12]:
            n_inner = N - n_hull
            r_hull = 1.0
            r_inner = 0.3 + 0.1 * (13 - n_hull)
            rot_hull = 0.0
            rot_inner = np.pi / n_inner if n_inner > 1 else 0.0
            inits.append(_make_symmetric_points(n_hull, n_inner, r_hull, r_inner, rot_hull, rot_inner))
        
        # Random configurations
        np.random.seed(42)
        for _ in range(3):
            angles = np.sort(np.random.uniform(0, 2*np.pi, N))
            radii = np.random.uniform(0.5, 1.0, N)
            inits.append(np.column_stack([radii * np.cos(angles), radii * np.sin(angles)]))
        
        return inits

    best_points = None
    best_score = -float('inf')

    # Strategy 1: Optimize symmetric configurations (k-gon + m-gon)
    splits = [(13, 0), (12, 1), (11, 2), (10, 3), (9, 4), (8, 5), (7, 6)]

    for n_hull, n_inner in splits:
        if time.time() > start_time + time_limit - 40:
            break

        local_points, local_score = _symmetric_optimize(n_hull, n_inner)
        
        if local_points is not None and local_score > best_score:
            best_score = local_score
            best_points = local_points.copy()

    # Strategy 2: Barycentric parameterization optimization (NEW)
    for n_hull in [7, 8, 9, 10, 11, 12, 13]:
        if time.time() > start_time + time_limit - 25:
            break
        
        local_points, local_score = _optimize_barycentric(n_hull, max_time=15.0)
        
        if local_points is not None and local_score > best_score:
            best_score = local_score
            best_points = local_points.copy()

    # Strategy 3: Targeted worst-triangle optimization from best result
    if best_points is not None and time.time() < start_time + time_limit - 15:
        targeted_points = _targeted_worst_triangle_opt(best_points, max_iter=25)
        targeted_score = _score(targeted_points)
        
        if targeted_score > best_score:
            best_score = targeted_score
            best_points = targeted_points.copy()

    # Strategy 4: Full free-space optimization from best result
    if best_points is not None and time.time() < start_time + time_limit - 10:
        current = best_points.copy()
        
        # Multi-beta annealing in free space
        for beta in [50.0, 120.0, 250.0]:
            if time.time() > start_time + time_limit - 10:
                break
            current = _refine_soft_min(current, beta, maxiter=60)
            opt_score = _score(current)
            if opt_score > best_score:
                best_score = opt_score
                best_points = current.copy()

    # Strategy 5: Final high-precision polish
    if best_points is not None and time.time() < start_time + time_limit - 5:
        # High-precision soft-min polish
        current = _refine_soft_min(best_points, beta=300.0, maxiter=80)
        pol_score = _score(current)
        if pol_score > best_score:
            best_score = pol_score
            best_points = current.copy()

        # Nelder-Mead final polish
        if time.time() < start_time + time_limit - 2:
            final_points = _refine_nelder_mead(best_points, maxiter=150)
            final_score = _score(final_points)
            if final_score > best_score:
                best_score = final_score
                best_points = final_points.copy()

    if best_points is None:
        # Fallback: regular 13-gon
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])

    return best_points

# EVOLVE-BLOCK-END
