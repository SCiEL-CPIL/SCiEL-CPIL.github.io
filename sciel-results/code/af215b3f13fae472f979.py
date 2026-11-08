import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13

def construct():
    start_time = time.time()
    time_limit = 148.0  # Safety margin for 150s limit

    # Precompute triangle indices
    tri_indices = list(combinations(range(N), 3))
    tri_i = np.array([t[0] for t in tri_indices], dtype=np.int32)
    tri_j = np.array([t[1] for t in tri_indices], dtype=np.int32)
    tri_k = np.array([t[2] for t in tri_indices], dtype=np.int32)
    n_tris = len(tri_i)

    def compute_score(points):
        """Compute min triangle area / convex hull area."""
        try:
            pts = points - points.mean(axis=0)
            hull = ConvexHull(pts)
            area_hull = hull.volume
        except Exception:
            return 0.0
        if area_hull < 1e-9:
            return 0.0
        
        x = pts[:, 0]
        y = pts[:, 1]
        v1x = x[tri_j] - x[tri_i]
        v1y = y[tri_j] - y[tri_i]
        v2x = x[tri_k] - x[tri_i]
        v2y = y[tri_k] - y[tri_i]
        cross = np.abs(v1x * v2y - v1y * v2x)
        areas = 0.5 * cross
        min_area = np.min(areas)
        return min_area / area_hull

    def _normalize_points(points):
        """Center and scale points to unit convex hull area."""
        pts = points - points.mean(axis=0)
        try:
            hull = ConvexHull(pts)
            area = hull.volume
            if area < 1e-9:
                return pts
            pts = pts * np.sqrt(1.0 / area)
        except Exception:
            pass
        return pts

    def _soft_objective(points_flat, beta=50.0):
        """
        Objective function for optimization.
        Returns negative soft-min triangle area (normalized to unit hull area).
        """
        pts = points_flat.reshape(-1, 2)
        pts = _normalize_points(pts)
        
        x = pts[:, 0]
        y = pts[:, 1]
        v1x = x[tri_j] - x[tri_i]
        v1y = y[tri_j] - y[tri_i]
        v2x = x[tri_k] - x[tri_i]
        v2y = y[tri_k] - y[tri_i]
        cross = np.abs(v1x * v2y - v1y * v2x)
        areas = 0.5 * cross
        
        # Soft min via log-sum-exp
        neg_beta_areas = -beta * areas
        max_val = np.max(neg_beta_areas)
        if max_val < -1e6:
            return 1e6
        log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_areas - max_val)))
        soft_min_area = -log_sum_exp / beta
        
        # Since hull area is 1, score is just min_area
        return -soft_min_area

    def _hard_objective(points_flat):
        """
        Objective function for final polish.
        Returns negative min triangle area (normalized to unit hull area).
        """
        pts = points_flat.reshape(-1, 2)
        pts = _normalize_points(pts)
        score = compute_score(pts)
        if score == 0.0:
            return 1e6
        return -score

    def _fourier_deform(params):
        """
        Generate 13 points as a deformation of a regular 13-gon.
        params: array of shape (2*K+2,) where K is the number of Fourier modes.
        params[0] = radius, params[1] = angle offset
        params[2:] = Fourier coefficients (amplitude, phase) for each mode
        
        We use modes 1..K. The radius at angle theta is:
        r(theta) = R * (1 + sum_{k=1}^{K} a_k * cos(k*theta + phi_k))
        
        To preserve convexity, we need |a_k| small enough.
        """
        R = params[0]
        angle_offset = params[1]
        
        # Number of Fourier modes
        K = (len(params) - 2) // 2
        
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False) + angle_offset
        
        radii = np.ones(N)
        for k in range(1, K + 1):
            a_k = params[2 + 2*(k-1)]
            phi_k = params[2 + 2*(k-1) + 1]
            radii += a_k * np.cos(k * angles + phi_k)
        
        # Ensure radii are positive (convexity constraint)
        radii = np.maximum(radii, 0.1)
        
        xs = R * radii * np.cos(angles)
        ys = R * radii * np.sin(angles)
        
        return np.column_stack([xs, ys])

    def _fourier_objective(params):
        """
        Objective for Fourier parameterization.
        """
        pts = _fourier_deform(params)
        return _soft_objective(pts.flatten(), beta=100.0)

    def _fourier_hard_objective(params):
        """
        Hard objective for Fourier parameterization.
        """
        pts = _fourier_deform(params)
        score = compute_score(pts)
        if score == 0.0:
            return 1e6
        return -score

    def _coordinate_descent(points, max_iter=200):
        """
        Coordinate descent refinement with adaptive step sizes.
        Moves one point at a time to improve the score.
        """
        pts = points.copy()
        best_score = compute_score(pts)
        step = 0.05
        
        for _ in range(max_iter):
            if time.time() > start_time + time_limit - 2:
                break
                
            improved = False
            for i in range(N):
                original = pts[i].copy()
                # Try moving point i in 8 directions
                for dx in [-step, 0, step]:
                    for dy in [-step, 0, step]:
                        if dx == 0 and dy == 0:
                            continue
                        pts[i] = original + np.array([dx, dy])
                        new_score = compute_score(pts)
                        if new_score > best_score:
                            best_score = new_score
                            improved = True
                        else:
                            pts[i] = original
                # If no improvement, try smaller step
                if not improved:
                    for dx in [-step/2, 0, step/2]:
                        for dy in [-step/2, 0, step/2]:
                            if dx == 0 and dy == 0:
                                continue
                            pts[i] = original + np.array([dx, dy])
                            new_score = compute_score(pts)
                            if new_score > best_score:
                                best_score = new_score
                                improved = True
                            else:
                                pts[i] = original
                    if not improved:
                        step /= 2
                        if step < 1e-7:
                            break
            if step < 1e-7:
                break
        
        return pts

    def _regular_polygon(n, radius=1.0, center=(0.0, 0.0)):
        if n == 1:
            return np.array([center])
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        xs = center[0] + radius * np.cos(angles)
        ys = center[1] + radius * np.sin(angles)
        return np.column_stack([xs, ys])

    best_score = -float('inf')
    best_points = None

    # Strategy 1: Fourier parameterization with differential evolution
    # This reduces the search space and preserves convexity by construction
    
    # Fourier modes: use modes 1 through 5 (10 params) + radius + angle = 12 params
    K_FOURIER = 5
    n_fourier_params = 2 + 2 * K_FOURIER
    
    # Bounds for Fourier parameters
    # Radius: [0.5, 2.0]
    # Angle: [0, 2*pi]
    # Amplitudes: [-0.3, 0.3] (small to preserve convexity)
    # Phases: [0, 2*pi]
    
    bounds_fourier = [(0.5, 2.0), (0, 2*np.pi)]
    for _ in range(K_FOURIER):
        bounds_fourier.append((-0.3, 0.3))
        bounds_fourier.append((0, 2*np.pi))
    
    # Try differential evolution on Fourier parameters
    if time.time() < start_time + time_limit - 60:
        rng = np.random.RandomState(42)
        try:
            de_res = opt.differential_evolution(
                _fourier_objective,
                bounds_fourier,
                seed=42,
                maxiter=50,
                tol=1e-8,
                popsize=15,
                mutation=(0.3, 0.9),
                recombination=0.8,
                polish=True,
                init='latinhypercube'
            )
            if de_res.success or de_res.fun < 1e5:
                best_fourier_params = de_res.x
                best_fourier_pts = _fourier_deform(best_fourier_params)
                best_fourier_score = compute_score(best_fourier_pts)
                
                if best_fourier_score > best_score:
                    best_score = best_fourier_score
                    best_points = best_fourier_pts.copy()
                    
                # Refine with L-BFGS on Fourier params
                if time.time() < start_time + time_limit - 30:
                    try:
                        res = opt.minimize(
                            _fourier_objective,
                            best_fourier_params,
                            method='L-BFGS-B',
                            bounds=bounds_fourier,
                            options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-12}
                        )
                        refined_params = res.x
                        refined_pts = _fourier_deform(refined_params)
                        refined_score = compute_score(refined_pts)
                        
                        if refined_score > best_score:
                            best_score = refined_score
                            best_points = refined_pts.copy()
                    except Exception:
                        pass
        except Exception:
            pass

    # Strategy 2: Direct optimization from symmetric initializations
    # Generate diverse symmetric initial configurations
    initializations = []
    
    # 1. Regular 13-gon
    initializations.append(_regular_polygon(13, 1.0))
    
    # 2. 12 outer + 1 center
    initializations.append(np.vstack([_regular_polygon(12, 1.0), np.array([[0.0, 0.0]])]))
    
    # 3. 11 outer + 2 inner (dipole)
    for r in [0.3, 0.5, 0.7]:
        outer = _regular_polygon(11, 1.0)
        inner = np.array([[-r, 0.0], [r, 0.0]])
        initializations.append(np.vstack([outer, inner]))
        
    # 4. 10 outer + 3 inner (triangle)
    for r in [0.3, 0.5, 0.7]:
        outer = _regular_polygon(10, 1.0)
        inner = _regular_polygon(3, r)
        initializations.append(np.vstack([outer, inner]))
        
    # 5. 9 outer + 4 inner (square)
    for r in [0.3, 0.5, 0.7]:
        outer = _regular_polygon(9, 1.0)
        inner = _regular_polygon(4, r)
        initializations.append(np.vstack([outer, inner]))
        
    # 6. 8 outer + 5 inner (pentagon)
    for r in [0.3, 0.5, 0.7]:
        outer = _regular_polygon(8, 1.0)
        inner = _regular_polygon(5, r)
        initializations.append(np.vstack([outer, inner]))
        
    # 7. 7 outer + 6 inner (hexagon)
    for r in [0.3, 0.5, 0.7]:
        outer = _regular_polygon(7, 1.0)
        inner = _regular_polygon(6, r)
        initializations.append(np.vstack([outer, inner]))
        
    # 8. 6 outer + 1 center + 6 inner (hexagon)
    for r in [0.3, 0.5, 0.7]:
        outer = _regular_polygon(6, 1.0)
        center = np.array([[0.0, 0.0]])
        inner = _regular_polygon(6, r)
        initializations.append(np.vstack([outer, center, inner]))
        
    # Add some random perturbations to top symmetric configs to break symmetry if beneficial
    rng = np.random.RandomState(42)
    for i in range(0, len(initializations), 3):
        perturbed = initializations[i].copy()
        perturbed += rng.randn(*perturbed.shape) * 0.1
        initializations.append(perturbed)

    # Optimize each initialization
    for init in initializations:
        if time.time() > start_time + time_limit - 15:
            break
            
        current = init.copy()
        
        # Step 1: L-BFGS-B with soft objective (annealing beta)
        betas = [20.0, 50.0, 100.0, 200.0, 500.0]
        
        for beta in betas:
            if time.time() > start_time + time_limit - 8:
                break
            try:
                res = opt.minimize(
                    lambda x, b=beta: _soft_objective(x, b),
                    current.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 80, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                current = res.x.reshape(-1, 2)
            except Exception:
                pass
        
        # Step 2: Nelder-Mead with hard objective (robust local search)
        if time.time() > start_time + time_limit - 4:
            break
        try:
            res = opt.minimize(
                _hard_objective,
                current.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
            )
            current = res.x.reshape(-1, 2)
        except Exception:
            pass
        
        # Step 3: Coordinate descent refinement
        if time.time() > start_time + time_limit - 2:
            break
        try:
            current = _coordinate_descent(current, max_iter=100)
        except Exception:
            pass
        
        # Step 4: Final Nelder-Mead polish
        if time.time() > start_time + time_limit - 1:
            break
        try:
            res = opt.minimize(
                _hard_objective,
                current.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 100, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
            )
            candidate = res.x.reshape(-1, 2)
            score = compute_score(candidate)
            
            if score > best_score:
                best_score = score
                best_points = candidate.copy()
        except Exception:
            pass

    # Final check: if no points were found (shouldn't happen), fallback
    if best_points is None:
        best_points = initializations[0]

    return best_points
