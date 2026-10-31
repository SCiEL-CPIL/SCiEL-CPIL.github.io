"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from itertools import combinations

N = 11
# Unit equilateral triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2.0])
# Area of unit equilateral triangle
TRI_AREA = 0.5 * np.sqrt(3) / 2.0  # = sqrt(3)/4


def _point_in_triangle(p):
    """Check if point p is inside or on the boundary of the unit equilateral triangle."""
    # Using barycentric coordinates
    x, y = p
    # Check if y is in range
    if y < -1e-10 or y > np.sqrt(3) / 2.0 + 1e-10:
        return False
    # Left edge: from (0,0) to (0.5, sqrt(3)/2) -> line y = sqrt(3)*x
    # Right edge: from (1,0) to (0.5, sqrt(3)/2) -> line y = -sqrt(3)*(x-1)
    if y > np.sqrt(3) * x + 1e-10:
        return False
    if y > -np.sqrt(3) * (x - 1.0) + 1e-10:
        return False
    return True


def _project_to_triangle(p):
    """Project point p to the closest point inside the triangle (simple clamping approach)."""
    x, y = p
    # Clamp y
    y = np.clip(y, 0.0, np.sqrt(3) / 2.0)
    # Clamp x based on y
    # Left boundary: x >= y / sqrt(3)
    # Right boundary: x <= 1 - y / sqrt(3)
    x_min = y / np.sqrt(3)
    x_max = 1.0 - y / np.sqrt(3)
    x = np.clip(x, x_min, x_max)
    return np.array([x, y])


def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by 3 points."""
    n = len(points)
    min_area = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                # Area = 0.5 * |cross(p2-p1, p3-p1)|
                cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
                area = 0.5 * abs(cross)
                if area < min_area:
                    min_area = area
    return min_area


def _objective(points):
    """Return the minimum triangle area (we want to maximize this)."""
    return _min_triangle_area(points)


def _nelder_mead(points, max_iter=500, tol=1e-10):
    """
    Nelder-Mead simplex optimization to maximize min triangle area.
    points: array of shape (N, 2)
    """
    n_params = N * 2
    # Initial simplex
    simplex = []
    for i in range(n_params + 1):
        if i == 0:
            simplex.append(points.flatten().copy())
        else:
            step = 0.01 * np.ones(n_params)
            step[i - 1] += 0.05
            simplex.append(points.flatten() + step)
    
    # Evaluate initial simplex
    values = []
    for v in simplex:
        pts = v.reshape(N, 2)
        # Project to triangle
        pts = np.array([_project_to_triangle(p) for p in pts])
        val = _min_triangle_area(pts)
        values.append(val)
    
    # Nelder-Mead iterations
    for _ in range(max_iter):
        # Sort by value (ascending, since we want to maximize)
        order = np.argsort(values)
        simplex = [simplex[i] for i in order]
        values = [values[i] for i in order]
        
        # Check convergence
        if abs(values[-1] - values[0]) < tol:
            break
        
        # Centroid of all but worst
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        alpha = 1.0
        x_ref = centroid + alpha * (centroid - simplex[-1])
        pts_ref = np.array([_project_to_triangle(p) for p in x_ref.reshape(N, 2)])
        val_ref = _min_triangle_area(pts_ref)
        
        if values[0] <= val_ref < values[-2]:
            simplex[-1] = x_ref
            values[-1] = val_ref
        elif val_ref >= values[-2]:
            # Expansion
            gamma = 2.0
            x_exp = centroid + gamma * (x_ref - centroid)
            pts_exp = np.array([_project_to_triangle(p) for p in x_exp.reshape(N, 2)])
            val_exp = _min_triangle_area(pts_exp)
            if val_exp > val_ref:
                simplex[-1] = x_exp
                values[-1] = val_exp
            else:
                simplex[-1] = x_ref
                values[-1] = val_ref
        else:
            # Contraction
            rho = 0.5
            x_con = centroid + rho * (simplex[-1] - centroid)
            pts_con = np.array([_project_to_triangle(p) for p in x_con.reshape(N, 2)])
            val_con = _min_triangle_area(pts_con)
            if val_con > values[-1]:
                simplex[-1] = x_con
                values[-1] = val_con
            else:
                # Shrink
                for i in range(1, n_params + 1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    pts_shrink = np.array([_project_to_triangle(p) for p in simplex[i].reshape(N, 2)])
                    values[i] = _min_triangle_area(pts_shrink)
    
    # Return best point
    best_idx = np.argmax(values)
    best_pts = np.array([_project_to_triangle(p) for p in simplex[best_idx].reshape(N, 2)])
    return best_pts


def _random_points(rng):
    """Generate random points inside the triangle."""
    points = []
    for _ in range(N):
        while True:
            u = rng.random()
            v = rng.random()
            if u + v < 1.0:
                p = u * V1 + v * V2
                points.append(p)
                break
    return np.array(points)


def _lattice_points():
    """Generate points on a hexagonal/triangular lattice inside the triangle."""
    # Try a few lattice configurations
    points = []
    # 11 points: try a rough grid
    # Use barycentric coordinates
    # For an equilateral triangle, we can use a triangular lattice
    # Let's try to place points at regular intervals
    # One approach: 4 rows
    # Row 0: 1 point
    # Row 1: 2 points
    # Row 2: 3 points
    # Row 3: 3 points
    # Row 4: 2 points
    # Total: 11
    # Actually, let's use a simpler approach: generate a grid in barycentric coords
    
    # Barycentric coordinates (a, b, c) with a+b+c=1, a,b,c >= 0
    # Point = a*V0 + b*V1 + c*V2
    
    # Let's try a few manual configurations
    configs = []
    
    # Configuration 1: Roughly uniform
    # 4 points on each side, 3 in interior
    # Let's just use a simple grid
    for a in np.linspace(0.1, 0.8, 4):
        for b in np.linspace(0.1, 0.8 - a, 3):
            c = 1 - a - b
            if c >= 0:
                p = a * V0 + b * V1 + c * V2
                if _point_in_triangle(p):
                    configs.append(p)
    
    # If we don't have exactly 11, take the first 11
    if len(configs) >= N:
        points = configs[:N]
    else:
        # Pad with random
        rng = random.Random(42)
        while len(configs) < N:
            u = rng.random()
            v = rng.random()
            if u + v < 1.0:
                p = u * V1 + v * V2
                configs.append(p)
        points = configs
    
    return np.array(points)


def construct():
    """Construct 11 points inside the unit equilateral triangle to maximize the minimum triangle area."""
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    rng = random.Random(123)
    
    # Initial candidates
    candidates = []
    
    # 1. Random points
    for seed in range(50):
        rng2 = random.Random(seed)
        pts = _random_points(rng2)
        candidates.append(pts)
    
    # 2. Lattice-based
    try:
        lattice = _lattice_points()
        candidates.append(lattice)
    except:
        pass
    
    # 3. Perturbed lattice
    for seed in range(20):
        rng3 = random.Random(1000 + seed)
        try:
            base = _lattice_points()
            noise = 0.02 * rng3.gauss(0, 1, size=(N, 2))
            pts = base + noise
            pts = np.array([_project_to_triangle(p) for p in pts])
            candidates.append(pts)
        except:
            pass
    
    # Optimize each candidate with Nelder-Mead
    best_points = None
    best_value = -np.inf
    
    for i, pts in enumerate(candidates):
        if time.time() > deadline:
            break
        
        # Quick evaluation
        val = _min_triangle_area(pts)
        if val > best_value:
            best_value = val
            best_points = pts.copy()
        
        # Only optimize promising candidates
        if val > 0.005 or i < 10:  # Optimize top candidates
            if time.time() > deadline:
                break
            optimized = _nelder_mead(pts, max_iter=300)
            opt_val = _min_triangle_area(optimized)
            if opt_val > best_value:
                best_value = opt_val
                best_points = optimized
    
    # If we have time, do more random restarts
    while time.time() < deadline - 10:
        rng4 = random.Random(rng.randint(0, 10**6))
        pts = _random_points(rng4)
        val = _min_triangle_area(pts)
        if val > best_value * 0.8:  # Only optimize if reasonably good
            optimized = _nelder_mead(pts, max_iter=200)
            opt_val = _min_triangle_area(optimized)
            if opt_val > best_value:
                best_value = opt_val
                best_points = optimized
    
    if best_points is None:
        # Fallback
        best_points = _random_points(random.Random(999))
    
    return best_points
