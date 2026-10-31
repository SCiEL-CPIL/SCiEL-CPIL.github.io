"""Heilbronn triangle: place 11 points in unit-area equilateral triangle to maximize min triangle area."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 11
TIME_LIMIT = 110  # seconds, leave safety margin before 120s kill

# Vertices of the equilateral triangle
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2.0])


def bary_to_cart(bary):
    """Convert barycentric coordinates (b0, b1, b2) to Cartesian."""
    return bary[0] * V0 + bary[1] * V1 + bary[2] * V2


def cart_to_bary(pt):
    """Convert Cartesian point to barycentric coordinates relative to V0, V1, V2."""
    # Solve: pt = b0*V0 + b1*V1 + b2*V2, b0+b1+b2=1
    # Using V0=(0,0), V1=(1,0), V2=(0.5, sqrt(3)/2)
    x, y = pt[0], pt[1]
    b2 = y / (np.sqrt(3) / 2.0)
    b1 = x - 0.5 * b2
    b0 = 1.0 - b1 - b2
    return np.array([b0, b1, b2])


def in_triangle(pt, eps=1e-9):
    """Check if point is inside or on the boundary of the triangle."""
    bary = cart_to_bary(pt)
    return all(b >= -eps and b <= 1.0 + eps for b in bary)


def all_triangle_areas(points):
    """Compute areas of all triangles formed by triples of points.
    Returns array of areas (absolute values)."""
    n = len(points)
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
                areas.append(area)
    return np.array(areas)


def min_triangle_area(points):
    """Return the minimum triangle area among all triples."""
    areas = all_triangle_areas(points)
    return np.min(areas)


def objective(params):
    """Objective function to minimize: negative of min triangle area plus penalties for out-of-bounds."""
    n = len(params) // 2
    points = params.reshape(n, 2)
    
    # Penalty for points outside the triangle
    penalty = 0.0
    for i in range(n):
        bary = cart_to_bary(points[i])
        for b in bary:
            if b < 0:
                penalty += 100.0 * b * b
            elif b > 1:
                penalty += 100.0 * (b - 1) * (b - 1)
    
    mta = min_triangle_area(points)
    return -mta + penalty


def random_initial_points(rng):
    """Generate random initial points inside the triangle using barycentric coordinates."""
    points = []
    for _ in range(N):
        u = rng.random()
        v = rng.random()
        while u + v > 1.0:
            u = rng.random()
            v = rng.random()
        bary = np.array([1.0 - u - v, u, v])
        points.append(bary_to_cart(bary))
    return np.array(points)


def structured_initial_points():
    """Create a structured initial configuration: 3x3 grid-like + 2 extra points."""
    # Use barycentric grid points
    # A 3x3 grid in barycentric space gives 9 points, plus 2 more
    grid_points = []
    for i in range(3):
        for j in range(3):
            b0 = (2 - i - j) / 2.0
            b1 = i / 2.0
            b2 = j / 2.0
            # Clip to valid range
            b0 = max(0.0, min(1.0, b0))
            b1 = max(0.0, min(1.0, b1))
            b2 = max(0.0, min(1.0, b2))
            total = b0 + b1 + b2
            bary = np.array([b0 / total, b1 / total, b2 / total])
            grid_points.append(bary_to_cart(bary))
    
    # Add 2 extra points near center
    center = (V0 + V1 + V2) / 3.0
    extra1 = center + np.array([0.05, 0.03])
    extra2 = center + np.array([-0.04, 0.05])
    
    points = grid_points[:9] + [extra1, extra2]
    return np.array(points)


def optimize_from_initial(points, rng, time_budget):
    """Run optimization from initial points, returning best found."""
    start_time = time.time()
    best_points = points.copy()
    best_score = min_triangle_area(points)
    
    x0 = points.flatten()
    
    # Use Nelder-Mead for robustness
    for restart in range(20):
        if time.time() - start_time > time_budget:
            break
        
        # Perturb initial point for restarts after the first
        if restart == 0:
            x_start = x0.copy()
        else:
            noise_scale = 0.02 * (1.0 + 0.1 * restart)
            x_start = x0 + rng.gauss(0, noise_scale, size=len(x0))
            # Ensure points stay roughly in triangle
            pts = x_start.reshape(N, 2)
            for i in range(N):
                if not in_triangle(pts[i], eps=0.05):
                    # Pull back toward centroid
                    centroid = (V0 + V1 + V2) / 3.0
                    pts[i] = 0.7 * pts[i] + 0.3 * centroid
            x_start = pts.flatten()
        
        try:
            result = minimize(
                objective,
                x_start,
                method='Nelder-Mead',
                options={
                    'maxiter': 500,
                    'xatol': 1e-5,
                    'fatol': 1e-7,
                    'adaptive': True,
                }
            )
            
            if result.success or result.fun < objective(x_start):
                opt_points = result.x.reshape(N, 2)
                # Check validity
                valid = True
                for i in range(N):
                    if not in_triangle(opt_points[i], eps=1e-6):
                        valid = False
                        break
                if valid:
                    score = min_triangle_area(opt_points)
                    if score > best_score:
                        best_score = score
                        best_points = opt_points.copy()
        except Exception:
            continue
    
    return best_points, best_score


def construct():
    start_time = time.time()
    rng = np.random.RandomState(42)
    
    best_points = None
    best_score = -1.0
    
    # Try multiple initial configurations
    initial_configs = []
    
    # Structured grid
    initial_configs.append(structured_initial_points())
    
    # Random configurations
    for _ in range(3):
        initial_configs.append(random_initial_points(rng))
    
    # One more: slightly perturbed grid
    grid = structured_initial_points()
    perturbed = grid + rng.normal(0, 0.03, size=grid.shape)
    # Clip to triangle
    for i in range(N):
        if not in_triangle(perturbed[i], eps=0.05):
            centroid = (V0 + V1 + V2) / 3.0
            perturbed[i] = 0.5 * perturbed[i] + 0.5 * centroid
    initial_configs.append(perturbed)
    
    # Allocate time budget
    time_per_config = (TIME_LIMIT - 10) / len(initial_configs)
    
    for idx, init_pts in enumerate(initial_configs):
        remaining = TIME_LIMIT - 15 - (time.time() - start_time)
        if remaining < 5:
            break
        
        time_budget = min(time_per_config, remaining)
        points, score = optimize_from_initial(init_pts, rng, time_budget)
        
        if score > best_score:
            best_score = score
            best_points = points
    
    # Final local polish on best
    if best_points is not None:
        remaining = TIME_LIMIT - 10 - (time.time() - start_time)
        if remaining > 10:
            try:
                x0 = best_points.flatten()
                result = minimize(
                    objective,
                    x0,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 1000,
                        'xatol': 1e-6,
                        'fatol': 1e-8,
                        'adaptive': True,
                    }
                )
                opt_points = result.x.reshape(N, 2)
                valid = all(in_triangle(opt_points[i], eps=1e-6) for i in range(N))
                if valid:
                    score = min_triangle_area(opt_points)
                    if score > best_score:
                        best_score = score
                        best_points = opt_points
            except Exception:
                pass
    
    if best_points is None:
        # Fallback: return structured grid
        best_points = structured_initial_points()
    
    return best_points


# Quick self-test
if __name__ == "__main__":
    pts = construct()
    print(f"Number of points: {len(pts)}")
    print(f"Min triangle area: {min_triangle_area(pts)}")
    print(f"All in triangle: {all(in_triangle(pts[i]) for i in range(N))}")
