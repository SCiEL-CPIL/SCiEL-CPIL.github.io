# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
from itertools import combinations

def _triangle_area_vec(points, i, j, k):
    """Compute area of triangle formed by points[i], points[j], points[k]."""
    p1, p2, p3 = points[i], points[j], points[k]
    return 0.5 * abs(p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1]))

def _min_triangle_area(points):
    """Find minimum area among all C(13,3)=286 triangles."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                a = 0.5 * abs(points[i,0]*(points[j,1]-points[k,1]) + 
                              points[j,0]*(points[k,1]-points[i,1]) + 
                              points[k,0]*(points[i,1]-points[j,1]))
                if a < min_area:
                    min_area = a
    return min_area

def _hull_area(points):
    """Calculate convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume = area
    except Exception:
        return float('inf')

def _ratio(points):
    """Compute A_min / A_hull. Higher is better."""
    a_min = _min_triangle_area(points)
    a_hull = _hull_area(points)
    if a_hull <= 0 or not np.isfinite(a_hull) or not np.isfinite(a_min):
        return 0.0
    if a_min <= 0:
        return 0.0
    return a_min / a_hull

def _negative_ratio(params):
    """Objective for minimization (we minimize negative ratio)."""
    points = params.reshape(13, 2)
    if not np.all(np.isfinite(points)):
        return 1e10
    r = _ratio(points)
    return -r

def _make_initial_points(regular_hull_n=7, interior_n=6, hull_radius=1.0, interior_radius=0.5):
    """Create initial configuration: regular polygon hull + interior hexagonal pattern."""
    points = []
    # Hull points on regular heptagon
    for i in range(regular_hull_n):
        angle = 2 * np.pi * i / regular_hull_n
        points.append([hull_radius * np.cos(angle), hull_radius * np.sin(angle)])
    # Interior points: regular hexagon
    for i in range(interior_n):
        angle = 2 * np.pi * i / interior_n + np.pi / interior_n
        points.append([interior_radius * np.cos(angle), interior_radius * np.sin(angle)])
    return np.array(points)

def _make_initial_points_v2():
    """Alternative initial: 6-gon hull + 7 interior points (1 center + 6 around)."""
    points = []
    # Hull: regular hexagon
    for i in range(6):
        angle = 2 * np.pi * i / 6
        points.append([1.0 * np.cos(angle), 1.0 * np.sin(angle)])
    # Interior: center + 6 points at radius 0.5
    points.append([0.0, 0.0])
    for i in range(6):
        angle = 2 * np.pi * i / 6 + np.pi / 6
        points.append([0.5 * np.cos(angle), 0.5 * np.sin(angle)])
    return np.array(points)

def _make_initial_points_v3():
    """Another initial: 5-gon hull + 8 interior points."""
    points = []
    # Hull: regular pentagon
    for i in range(5):
        angle = 2 * np.pi * i / 5
        points.append([1.0 * np.cos(angle), 1.0 * np.sin(angle)])
    # Interior: 2 rings
    # Ring 1: 4 points at radius 0.4
    for i in range(4):
        angle = 2 * np.pi * i / 4 + np.pi / 8
        points.append([0.4 * np.cos(angle), 0.4 * np.sin(angle)])
    # Ring 2: 4 points at radius 0.7
    for i in range(4):
        angle = 2 * np.pi * i / 4 + np.pi / 4
        points.append([0.7 * np.cos(angle), 0.7 * np.sin(angle)])
    return np.array(points)

def _local_optimize(points, max_iters=200, timeout=None, start_time=None):
    """Use Nelder-Mead to locally optimize the ratio."""
    if start_time is None:
        start_time = time.time()
    
    params = points.flatten()
    best_points = points.copy()
    best_ratio = _ratio(points)
    
    try:
        result = minimize(
            _negative_ratio, 
            params, 
            method='Nelder-Mead',
            options={
                'maxiter': max_iters,
                'xatol': 1e-6,
                'fatol': 1e-8,
                'adaptive': True
            }
        )
        if result.x is not None and np.all(np.isfinite(result.x)):
            new_points = result.x.reshape(13, 2)
            new_ratio = _ratio(new_points)
            if new_ratio > best_ratio:
                best_points = new_points
                best_ratio = new_ratio
    except Exception:
        pass
    
    if timeout is not None:
        elapsed = time.time() - start_time
        if elapsed > timeout:
            return best_points
    
    return best_points

def _simulated_anneal(points, iterations=500, timeout=None, start_time=None):
    """Simple simulated annealing to escape local minima."""
    if start_time is None:
        start_time = time.time()
    
    best_points = points.copy()
    best_ratio = _ratio(points)
    current_points = points.copy()
    current_ratio = best_ratio
    
    T = 0.01  # Initial temperature
    cooling = 0.999
    
    for it in range(iterations):
        if timeout is not None:
            elapsed = time.time() - start_time
            if elapsed > timeout:
                break
        
        # Perturb one point
        idx = np.random.randint(0, 13)
        new_points = current_points.copy()
        new_points[idx] += np.random.normal(0, T, 2)
        
        new_ratio = _ratio(new_points)
        
        if new_ratio > current_ratio or np.random.random() < np.exp((new_ratio - current_ratio) / max(T, 1e-10)):
            current_points = new_points
            current_ratio = new_ratio
            if current_ratio > best_ratio:
                best_points = current_points.copy()
                best_ratio = current_ratio
        
        T *= cooling
    
    return best_points

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize A_min / A_hull.
    Uses structured initialization + local optimization + simulated annealing.
    """
    start_time = time.time()
    deadline = start_time + 600  # Safety margin before 630s limit
    
    np.random.seed(42)
    
    # Generate multiple initial configurations
    initial_configs = [
        _make_initial_points(regular_hull_n=7, interior_n=6, hull_radius=1.0, interior_radius=0.5),
        _make_initial_points_v2(),
        _make_initial_points_v3(),
    ]
    
    # Add some randomized perturbations of the structured configs
    for base in initial_configs[:]:
        for _ in range(3):
            perturbed = base + np.random.normal(0, 0.05, (13, 2))
            initial_configs.append(perturbed)
    
    best_points = None
    best_ratio = 0.0
    
    for config in initial_configs:
        elapsed = time.time() - start_time
        if elapsed > deadline - 30:
            break
        
        # Local optimization
        optimized = _local_optimize(config, max_iters=300, timeout=20, start_time=start_time)
        ratio = _ratio(optimized)
        
        if ratio > best_ratio:
            best_ratio = ratio
            best_points = optimized.copy()
        
        # Simulated annealing from the optimized point
        if time.time() - start_time < deadline - 30:
            sa_result = _simulated_anneal(optimized, iterations=200, timeout=15, start_time=start_time)
            sa_ratio = _ratio(sa_result)
            if sa_ratio > best_ratio:
                best_ratio = sa_ratio
                best_points = sa_result.copy()
    
    # Final local polish on the best
    if best_points is not None and time.time() - start_time < deadline - 10:
        final = _local_optimize(best_points, max_iters=500, timeout=20, start_time=start_time)
        final_ratio = _ratio(final)
        if final_ratio > best_ratio:
            best_points = final
            best_ratio = final_ratio
    
    if best_points is None:
        # Fallback: return a valid configuration
        best_points = _make_initial_points()
    
    # Ensure all points are finite
    best_points = np.nan_to_num(best_points, nan=0.0, posinf=0.0, neginf=0.0)
    
    return best_points

# EVOLVE-BLOCK-END
