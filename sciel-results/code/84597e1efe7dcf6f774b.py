import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
import warnings

N = 13

def _triangle_areas_vectorized(points):
    """Compute all triangle areas for N points using vectorized cross products."""
    n = points.shape[0]
    # Get all unique triples
    idx = np.array([i for i in range(n) for j in range(i+1, n) for k in range(j+1, n)])
    # Reshape into (m, 3)
    idx = idx.reshape(-1, 3)
    p1 = points[idx[:, 0]]
    p2 = points[idx[:, 1]]
    p3 = points[idx[:, 2]]
    # Cross product in 2D
    areas = 0.5 * np.abs((p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) -
                         (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0]))
    return areas

def _hull_area(points):
    """Compute convex hull area using scipy or manual shoelace after sorting by angle."""
    try:
        from scipy.spatial import ConvexHull
        if np.ptp(points, axis=0).max() < 1e-12:
            return 0.0
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is area
    except Exception:
        # Fallback: sort by angle around centroid and use shoelace
        centroid = np.mean(points, axis=0)
        angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
        sorted_idx = np.argsort(angles)
        sorted_points = points[sorted_idx]
        x = sorted_points[:, 0]
        y = sorted_points[:, 1]
        x_next = np.roll(x, -1)
        y_next = np.roll(y, -1)
        area = 0.5 * np.abs(np.sum(x * y_next - x_next * y))
        return area

def _score(points):
    """Compute the score: min triangle area / convex hull area."""
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    areas = _triangle_areas_vectorized(points)
    min_area = np.min(areas)
    return min_area / hull_area

def _neg_score(points):
    """Negative score for minimization."""
    return -_score(points)

def _polar_to_cartesian(params):
    """Convert polar parameters (r, theta for each point) to cartesian."""
    n = len(params) // 2
    r = params[:n]
    theta = params[n:]
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack([x, y])

def _cartesian_to_polar(points):
    """Convert cartesian points to polar parameters."""
    x = points[:, 0]
    y = points[:, 1]
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return np.concatenate([r, theta])

def _polar_objective(params):
    """Objective function in polar coordinates."""
    points = _polar_to_cartesian(params)
    return _neg_score(points)

def _smooth_objective(points, beta=100.0):
    """Smooth approximation of min using log-sum-exp."""
    areas = _triangle_areas_vectorized(points)
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    # Soft min: -log(sum(exp(-beta * areas))) / beta
    # But we want min(areas) / hull_area, so we minimize -min(areas)/hull_area
    # = max(-areas/hull_area)
    # Soft max of -areas/hull_area = -log(sum(exp(-beta * (-areas/hull_area)))) / beta
    # = log(sum(exp(beta * areas/hull_area))) / beta
    # Wait, let's think again.
    # We want to minimize f = min(areas) / hull_area
    # This is equivalent to maximizing -f = -min(areas) / hull_area = max(-areas / hull_area)
    # Soft max of values v_i = log(sum(exp(beta * v_i))) / beta
    # Here v_i = -areas_i / hull_area
    # So soft max = log(sum(exp(beta * (-areas_i / hull_area)))) / beta
    # = -log(sum(exp(-beta * areas_i / hull_area))) / beta
    # So we minimize soft_max = -log(sum(exp(-beta * areas_i / hull_area))) / beta
    scaled = areas / hull_area
    max_scaled = np.max(scaled)
    exp_terms = np.exp(-beta * (scaled - max_scaled))
    log_sum = np.log(np.sum(exp_terms)) + beta * max_scaled
    soft_max = -log_sum / beta
    return soft_max

def _de_objective(params):
    """Differential evolution objective in polar coordinates."""
    points = _polar_to_cartesian(params)
    return _neg_score(points)

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    np.random.seed(42)
    
    # Initial guess: Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    best_points = points.copy()
    best_score = _score(points)
    
    # Try multiple initial configurations
    configs = [
        "regular",  # Regular polygon
        "two_rings",  # 7 inner + 6 outer
        "three_rings",  # 5+4+4
    ]
    
    for config_name in configs:
        if time.time() > deadline - 30:
            break
            
        if config_name == "regular":
            pts = np.column_stack([np.cos(angles), np.sin(angles)])
        elif config_name == "two_rings":
            # 7 points on outer ring, 6 on inner ring
            outer_angles = np.linspace(0, 2 * np.pi, 7, endpoint=False)
            inner_angles = np.linspace(0, 2 * np.pi, 6, endpoint=False) + np.pi/6
            outer_pts = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
            inner_pts = 0.5 * np.column_stack([np.cos(inner_angles), np.sin(inner_angles)])
            pts = np.vstack([outer_pts, inner_pts])
        elif config_name == "three_rings":
            # 5+4+4
            a1 = np.linspace(0, 2*np.pi, 5, endpoint=False)
            a2 = np.linspace(0, 2*np.pi, 4, endpoint=False) + np.pi/4
            a3 = np.linspace(0, 2*np.pi, 4, endpoint=False) + np.pi/4
            p1 = np.column_stack([np.cos(a1), np.sin(a1)])
            p2 = 0.6 * np.column_stack([np.cos(a2), np.sin(a2)])
            p3 = 0.3 * np.column_stack([np.cos(a3), np.sin(a3)])
            pts = np.vstack([p1, p2, p3])
        
        score = _score(pts)
        if score > best_score:
            best_score = score
            best_points = pts.copy()
    
    # Differential Evolution in polar coordinates
    if time.time() < deadline - 20:
        try:
            # Bounds: r in [0, 2], theta in [0, 2*pi]
            bounds = [(0, 2)] * N + [(0, 2 * np.pi)] * N
            
            # Convert best points to polar for initial population
            polar_best = _cartesian_to_polar(best_points)
            
            res = differential_evolution(
                _de_objective,
                bounds,
                seed=42,
                maxiter=50,
                popsize=15,
                tol=1e-6,
                init='sobol',
                updating='deferred',
                polish=False,
                workers=1,
                x0=polar_best,
                mutation=(0.3, 0.9),
                recombination=0.7
            )
            
            if res.fun < -best_score:
                new_points = _polar_to_cartesian(res.x)
                new_score = _score(new_points)
                if new_score > best_score:
                    best_score = new_score
                    best_points = new_points.copy()
        except Exception as e:
            pass
    
    # Local refinement with Nelder-Mead
    if time.time() < deadline - 10:
        try:
            res = minimize(
                _neg_score,
                best_points,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            if res.fun < -best_score:
                new_score = _score(res.x)
                if new_score > best_score:
                    best_score = new_score
                    best_points = res.x.copy()
        except Exception:
            pass
    
    # Coordinate descent polish
    if time.time() < deadline - 5:
        try:
            current_points = best_points.copy()
            current_score = _score(current_points)
            
            step_size = 0.01
            for _ in range(50):
                if time.time() > deadline - 5:
                    break
                improved = False
                for i in range(N):
                    for coord in range(2):
                        for direction in [-1, 1]:
                            trial = current_points.copy()
                            trial[i, coord] += direction * step_size
                            trial_score = _score(trial)
                            if trial_score > current_score:
                                current_points = trial
                                current_score = trial_score
                                improved = True
                if not improved:
                    step_size *= 0.5
                    if step_size < 1e-6:
                        break
                if current_score > best_score:
                    best_score = current_score
                    best_points = current_points.copy()
        except Exception:
            pass
    
    # Final check
    final_score = _score(best_points)
    if final_score > best_score:
        best_score = final_score
    
    return best_points

# EVOLVE-BLOCK-START
# The construct function is defined above.
# EVOLVE-BLOCK-END
