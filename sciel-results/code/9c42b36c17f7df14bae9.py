"""
Heilbronn triangles: Place 11 points to maximize the minimum triangle area.
Strategy: Combined approach - smart initialization, simulated annealing with
critical triplet push, and final Nelder-Mead polish.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 11
S3H = np.sqrt(3) / 2
DEADLINE = 140.0  # Safety margin before 150s limit

def _min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by three points."""
    n = len(pts)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = pts[i], pts[j], pts[k]
                # Area using cross product
                area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - 
                                 (p2[1] - p1[1]) * (p3[0] - p1[0]))
                min_area = min(min_area, area)
    return min_area

def _vectorized_min_areas(pts):
    """Vectorized computation of all triangle areas, return the minimum."""
    n = len(pts)
    # Generate all unique triplets
    triplets = np.array([(i, j, k) for i in range(n) 
                         for j in range(i+1, n) 
                         for k in range(j+1, n)])
    if len(triplets) == 0:
        return 0.0
    p1 = pts[triplets[:, 0]]
    p2 = pts[triplets[:, 1]]
    p3 = pts[triplets[:, 2]]
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _check_inside(pts, tol=1e-9):
    """Check if all points are inside or on the boundary of the triangle."""
    x = pts[:, 0]
    y = pts[:, 1]
    return (np.all(y >= -tol) and 
            np.all(y <= S3H * x + tol) and 
            np.all(y <= -S3H * x + S3H + tol))

def _project_to_triangle(pts):
    """Project points to be inside the triangle (simple clamping)."""
    x = pts[:, 0].copy()
    y = pts[:, 1].copy()
    # Ensure y >= 0
    y = np.maximum(y, 0.0)
    # Ensure y <= S3H * x
    y = np.minimum(y, S3H * x)
    # Ensure y <= -S3H * x + S3H
    y = np.minimum(y, -S3H * x + S3H)
    return np.column_stack((x, y))

def _random_point_in_triangle():
    """Generate a random point uniformly inside the triangle."""
    r1 = np.random.random()
    r2 = np.random.random()
    if r1 + r2 > 1:
        r1 = 1 - r1
        r2 = 1 - r2
    x = r1 + 0.5 * r2
    y = S3H * r2
    return np.array([x, y])

def _smart_init(n_points=N):
    """Generate a smart initial configuration."""
    pts = []
    # Add vertices with small perturbations
    v1 = np.array([0.0, 0.0])
    v2 = np.array([1.0, 0.0])
    v3 = np.array([0.5, S3H])
    pts.append(v1 + 0.01 * np.random.randn(2))
    pts.append(v2 + 0.01 * np.random.randn(2))
    pts.append(v3 + 0.01 * np.random.randn(2))
    
    # Add points along edges
    for i in range(2):
        t = (i + 1) / 3.0
        pts.append(t * v1 + (1-t) * v2 + 0.005 * np.random.randn(2))
        pts.append(t * v2 + (1-t) * v3 + 0.005 * np.random.randn(2))
        pts.append(t * v3 + (1-t) * v1 + 0.005 * np.random.randn(2))
    
    # Add center and random points
    center = (v1 + v2 + v3) / 3.0
    pts.append(center + 0.01 * np.random.randn(2))
    
    while len(pts) < n_points:
        pts.append(_random_point_in_triangle())
    
    pts = np.array(pts[:n_points])
    return _project_to_triangle(pts)

def _single_point_step(pts, step_size):
    """Move one random point by a Gaussian step."""
    idx = np.random.randint(len(pts))
    delta = step_size * np.random.randn(2)
    new_pts = pts.copy()
    new_pts[idx] += delta
    return new_pts

def _critical_triplet_push(pts, step_size):
    """
    Find the critical triplet (minimizing area) and push points to increase area.
    """
    n = len(pts)
    # Find all triplet areas
    areas = np.zeros((n, n, n))
    idx = np.indices((n, n, n))
    mask = (idx[0] < idx[1]) & (idx[1] < idx[2])
    triplets = np.argwhere(mask)
    
    p1 = pts[triplets[:, 0]]
    p2 = pts[triplets[:, 1]]
    p3 = pts[triplets[:, 2]]
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas_flat = 0.5 * np.abs(cross)
    
    min_idx = np.argmin(areas_flat)
    i, j, k = triplets[min_idx]
    
    new_pts = pts.copy()
    # Push points apart to increase area
    # Move each point slightly away from the centroid of the triplet
    centroid = (pts[i] + pts[j] + pts[k]) / 3.0
    for idx in [i, j, k]:
        direction = pts[idx] - centroid
        norm = np.linalg.norm(direction)
        if norm > 1e-10:
            direction = direction / norm
        else:
            direction = np.random.randn(2)
            direction = direction / np.linalg.norm(direction)
        new_pts[idx] += step_size * direction
    
    return _project_to_triangle(new_pts), _min_triangle_area(new_pts)

def construct():
    start_time = time.time()
    np.random.seed(42)
    
    # Initial configuration
    best_pts = _smart_init()
    best_score = _min_triangle_area(best_pts)
    
    # Simulated Annealing
    T = 0.05
    T_min = 0.0001
    alpha = 0.98
    step_size = 0.05
    
    it = 0
    while time.time() < DEADLINE - 20:
        new_pts = _single_point_step(best_pts, step_size)
        new_pts = _project_to_triangle(new_pts)
        new_score = _min_triangle_area(new_pts)
        
        delta = new_score - best_score
        
        if delta > 0:
            best_pts = new_pts
            best_score = new_score
        else:
            if T > T_min:
                p = np.exp(delta / T)
                if np.random.random() < p:
                    best_pts = new_pts
                    best_score = new_score
        
        T *= alpha
        step_size *= 0.995
        it += 1
        
        # Periodic critical triplet push
        if it % 50 == 0:
            pushed_pts, pushed_score = _critical_triplet_push(best_pts, step_size=0.02)
            if pushed_score > best_score:
                best_pts = pushed_pts
                best_score = pushed_score
        
        # Also try smaller push
        if it % 50 == 25:
            pushed_pts2, pushed_score2 = _critical_triplet_push(best_pts, step_size=0.005)
            if pushed_score2 > best_score:
                best_pts = pushed_pts2
                best_score = pushed_score2
    
    # Final polish with Nelder-Mead
    if time.time() < DEADLINE - 5:
        def objective(x):
            pts = _project_to_triangle(x.reshape(N, 2))
            return -_min_triangle_area(pts)
        
        x0 = best_pts.flatten()
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8})
            res_pts = _project_to_triangle(res.x.reshape(N, 2))
            res_score = _min_triangle_area(res_pts)
            if res_score > best_score:
                best_pts = res_pts
                best_score = res_score
        except Exception:
            pass
    
    # Ensure final points are inside
    best_pts = _project_to_triangle(best_pts)
    
    return best_pts

if __name__ == "__main__":
    pts = construct()
    print("Points:", pts)
    print("Min area:", _min_triangle_area(pts))
