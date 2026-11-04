"""Improved program for Heilbronn triangles using simulated annealing."""
import numpy as np
import itertools
import time
from scipy.optimize import minimize

N = 11
SQRT3 = np.sqrt(3)

def _triangle_area(p1, p2, p3):
    """Compute area of triangle formed by three 2D points."""
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]))

def _min_triangle_area(points):
    """Compute the minimum area among all triangles formed by N points."""
    min_area = float('inf')
    for i, j, k in itertools.combinations(range(len(points)), 3):
        a = _triangle_area(points[i], points[j], points[k])
        if a < min_area:
            min_area = a
    return min_area

def _inside_triangle(pt):
    """Check if point is inside or on the boundary of the unit equilateral triangle."""
    x, y = pt
    # Triangle vertices: (0,0), (1,0), (0.5, sqrt(3)/2)
    if y < -1e-9:
        return False
    if y > SQRT3 * x + 1e-9:
        return False
    if y > -SQRT3 * (x - 1) + 1e-9:
        return False
    return True

def _project_to_triangle(pt):
    """Project a point onto the triangle if it's outside."""
    x, y = pt
    # Check if inside
    if _inside_triangle(pt):
        return pt
    
    # Find closest point on the triangle edges
    # Edge 1: from (0,0) to (1,0)
    # Edge 2: from (1,0) to (0.5, sqrt(3)/2)
    # Edge 3: from (0.5, sqrt(3)/2) to (0,0)
    
    best_dist = float('inf')
    best_pt = pt
    
    # For each edge, project and check
    edges = [
        ((0, 0), (1, 0)),
        ((1, 0), (0.5, SQRT3/2)),
        ((0.5, SQRT3/2), (0, 0))
    ]
    
    for a, b in edges:
        a = np.array(a, dtype=float)
        b = np.array(b, dtype=float)
        ab = b - a
        t = np.dot(pt - a, ab) / np.dot(ab, ab)
        t = np.clip(t, 0, 1)
        proj = a + t * ab
        d = np.linalg.norm(pt - proj)
        if d < best_dist:
            best_dist = d
            best_pt = proj
    
    return best_pt

def _random_point_in_triangle(rng):
    """Generate a random point uniformly inside the triangle using barycentric coordinates."""
    u = rng.random()
    v = rng.random()
    if u + v > 1:
        u = 1 - u
        v = 1 - v
    # Barycentric: (1-u-v)*A + u*B + v*C where A=(0,0), B=(1,0), C=(0.5, sqrt(3)/2)
    x = u * 1 + v * 0.5
    y = v * SQRT3 / 2
    return np.array([x, y])

def _initial_configuration(rng):
    """Generate an initial configuration of N points."""
    points = []
    # Place some points at strategic locations
    # Vertices and midpoints as starting points
    vertices = [
        np.array([0.1, 0.05]),
        np.array([0.9, 0.05]),
        np.array([0.5, SQRT3/2 - 0.05]),
    ]
    for v in vertices:
        points.append(v.copy())
    
    # Add remaining points randomly
    while len(points) < N:
        p = _random_point_in_triangle(rng)
        points.append(p)
    
    return np.array(points)

def _simulated_annealing(points, rng, max_iters, deadline):
    """Run simulated annealing to improve the configuration."""
    current = points.copy()
    current_score = _min_triangle_area(current)
    best = current.copy()
    best_score = current_score
    
    T0 = 0.1  # Initial temperature
    T_end = 0.001  # Final temperature
    cooling_rate = (T_end / T0) ** (1.0 / max_iters)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Generate a move
        move_type = rng.random()
        candidate = current.copy()
        
        if move_type < 0.7:
            # Gaussian jitter on a random point
            idx = rng.randint(0, N - 1)
            sigma = T0 * 0.1 * (T0 ** 0.5)
            candidate[idx] += rng.gauss(0, sigma)
            candidate[idx] = _project_to_triangle(candidate[idx])
        elif move_type < 0.9:
            # Swap two points
            i, j = rng.sample(range(N), 2)
            candidate[i], candidate[j] = candidate[j].copy(), candidate[i].copy()
            # Add small jitter after swap
            candidate[i] += rng.gauss(0, 0.01, size=2)
            candidate[j] += rng.gauss(0, 0.01, size=2)
            candidate[i] = _project_to_triangle(candidate[i])
            candidate[j] = _project_to_triangle(candidate[j])
        else:
            # Move two points
            i, j = rng.sample(range(N), 2)
            sigma = T0 * 0.1
            candidate[i] += rng.gauss(0, sigma, size=2)
            candidate[j] += rng.gauss(0, sigma, size=2)
            candidate[i] = _project_to_triangle(candidate[i])
            candidate[j] = _project_to_triangle(candidate[j])
        
        candidate_score = _min_triangle_area(candidate)
        
        # Acceptance criterion
        if candidate_score >= current_score or rng.random() < np.exp((candidate_score - current_score) / T0):
            current = candidate
            current_score = candidate_score
            if current_score > best_score:
                best = current.copy()
                best_score = current_score
        
        # Cool down
        T0 *= cooling_rate
    
    return best, best_score

def _local_polish(points, deadline):
    """Perform local optimization using Nelder-Mead."""
    def objective(x):
        pts = x.reshape(-1, 2)
        # Penalize points outside triangle
        penalty = 0
        for i in range(len(pts)):
            if not _inside_triangle(pts[i]):
                penalty += 100 * np.linalg.norm(pts[i] - _project_to_triangle(pts[i]))
        min_a = _min_triangle_area(pts)
        return -min_a + penalty
    
    x0 = points.flatten()
    try:
        result = minimize(objective, x0, method='Nelder-Mead', 
                         options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10})
        if time.time() < deadline:
            best_pts = result.x.reshape(-1, 2)
            # Project any points outside
            for i in range(len(best_pts)):
                if not _inside_triangle(best_pts[i]):
                    best_pts[i] = _project_to_triangle(best_pts[i])
            return best_pts
    except:
        pass
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 120  # Safety margin before 150s limit
    
    rng = np.random.RandomState(42)
    
    best_points = None
    best_score = -float('inf')
    
    # Multi-start with simulated annealing
    n_starts = 8
    sa_iters_per_start = 3000
    
    for start in range(n_starts):
        if time.time() > deadline - 30:  # Leave time for polish
            break
        
        # Generate initial configuration
        if start == 0:
            # Start with a structured configuration
            points = []
            # Place points in a grid-like pattern
            for i in range(4):
                for j in range(3):
                    if i + j < N:
                        x = 0.15 + i * 0.2
                        y = 0.1 + j * 0.25
                        if _inside_triangle([x, y]):
                            points.append(np.array([x, y]))
            while len(points) < N:
                points.append(_random_point_in_triangle(rng))
            points = np.array(points)
        else:
            points = _initial_configuration(rng)
        
        # Run simulated annealing
        improved_points, score = _simulated_annealing(points, rng, sa_iters_per_start, deadline)
        
        if score > best_score:
            best_score = score
            best_points = improved_points
    
    # Final local polish
    if best_points is not None and time.time() < deadline - 10:
        polished = _local_polish(best_points, deadline)
        polished_score = _min_triangle_area(polished)
        if polished_score > best_score:
            best_points = polished
            best_score = polished_score
    
    # Ensure all points are inside the triangle
    for i in range(N):
        if not _inside_triangle(best_points[i]):
            best_points[i] = _project_to_triangle(best_points[i])
    
    return best_points
