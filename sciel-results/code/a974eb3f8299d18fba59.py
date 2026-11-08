"""Heilbronn triangles: maximize min triangle area for 11 points in unit-area equilateral triangle."""
import numpy as np
import time
import random

N = 11
H = np.sqrt(3) / 2.0
AREA_EQUI = np.sqrt(3) / 4.0

# Pre-compute indices for all combinations of 3 points
combos = []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            combos.append((i, j, k))
combos = np.array(combos)
idx_i = combos[:, 0]
idx_j = combos[:, 1]
idx_k = combos[:, 2]

def _min_triangle_area_vec(points):
    """
    Vectorized computation of the minimum triangle area.
    points: shape (N, 2)
    """
    pi = points[idx_i]  # shape (165, 2)
    pj = points[idx_j]
    pk = points[idx_k]
    
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def _is_inside_triangle(points):
    """
    Check if all points are inside the unit equilateral triangle.
    Vertices: (0,0), (1,0), (0.5, sqrt(3)/2)
    """
    y = points[:, 1]
    x = points[:, 0]
    
    cond1 = y >= -1e-9
    cond2 = y <= np.sqrt(3) * x + 1e-9
    cond3 = y <= np.sqrt(3) * (1 - x) + 1e-9
    
    return np.all(cond1) and np.all(cond2) and np.all(cond3)

def _clip_to_triangle(points):
    """
    Clip points to be inside the triangle.
    """
    x = points[:, 0]
    y = points[:, 1]
    
    # y >= 0
    y = np.maximum(y, 0)
    
    # y <= sqrt(3) * x  =>  x >= y / sqrt(3)
    x_min = y / np.sqrt(3)
    
    # y <= sqrt(3) * (1 - x)  =>  x <= 1 - y / sqrt(3)
    x_max = 1 - y / np.sqrt(3)
    
    # If x_min > x_max, the point is outside. We need to project it.
    # For simplicity, if x < x_min, set x = x_min. If x > x_max, set x = x_max.
    # But this might still be outside if y is too large.
    # A more robust way is to check if the point is inside after clipping.
    # If not, project to the closest point on the boundary or center.
    
    x = np.maximum(x, x_min)
    x = np.minimum(x, x_max)
    
    # Re-check validity. If still invalid, move to centroid.
    centroid = np.array([0.5, H / 3.0])
    for i in range(N):
        if not _is_inside_triangle(points[i:i+1]):
            points[i] = centroid
    
    return points

def _generate_structured_initial():
    """
    Generate a structured initial configuration.
    We use a grid-like structure or known good patterns.
    For n=11, a common pattern is 3 points on each side (including corners) and 2 in the center?
    Or 4-4-3 distribution?
    Let's try a few structured patterns and pick the best.
    """
    patterns = []
    
    # Pattern 1: 3x3 grid + 2 center? No, 11 points.
    # Let's try to place points in rows.
    # Row 0 (bottom): 4 points
    # Row 1 (middle): 4 points
    # Row 2 (top): 3 points
    
    # Barycentric coordinates for grid points.
    # Let's use a function to generate grid points.
    
    def make_grid(n_rows, counts):
        pts = []
        total = sum(counts)
        for r in range(n_rows):
            for c in range(counts[r]):
                # y is proportional to r
                # x is spread across the width at that y
                # y = r * (H / (n_rows - 1)) ? No, spacing should be uniform in area?
                # Let's just use linear y spacing.
                y = (r + 0.5) * H / n_rows  # Center of row
                # Width at y is 1 - y/H = 1 - (r+0.5)/n_rows
                w = 1 - y / H
                # x is spread from y/sqrt(3) to 1 - y/sqrt(3)
                x_min = y / np.sqrt(3)
                x_max = 1 - y / np.sqrt(3)
                if counts[r] == 1:
                    x = (x_min + x_max) / 2
                    pts.append([x, y])
                else:
                    for k in range(counts[r]):
                        x = x_min + (x_max - x_min) * (k + 0.5) / counts[r]
                        pts.append([x, y])
        return np.array(pts)
    
    # Try different row distributions
    dists = [
        [4, 4, 3],
        [3, 4, 4],
        [5, 3, 3],
        [3, 5, 3],
        [2, 4, 5],
        [4, 3, 4],
    ]
    
    for d in dists:
        if len(d) == 3:
            pts = make_grid(3, d)
            if len(pts) == N:
                patterns.append(pts)
        
    # Also try a random structured set
    # And the centroid + random
    rng = np.random.RandomState(42)
    for _ in range(5):
        pts = np.zeros((N, 2))
        for j in range(N):
            while True:
                u = rng.random()
                w = rng.random()
                if u + w <= 1.0:
                    pts[j] = [u + 0.5 * w, w * H]
                    break
        patterns.append(pts)
        
    best_pts = None
    best_score = -np.inf
    for p in patterns:
        if _is_inside_triangle(p):
            s = _min_triangle_area_vec(p)
            if s > best_score:
                best_score = s
                best_pts = p.copy()
                
    return best_pts, best_score

def _nelder_mead(points, max_iter=1000, tol=1e-8):
    """
    Nelder-Mead optimization.
    points: shape (N, 2)
    """
    # Flatten points to 1D vector of length 2N
    x0 = points.flatten()
    n_dims = 2 * N
    
    # Initial simplex
    simplex = np.zeros((n_dims + 1, n_dims))
    simplex[0] = x0
    for i in range(n_dims):
        simplex[i + 1] = x0.copy()
        simplex[i + 1][i] += 0.01
    
    def objective(x):
        pts = x.reshape(N, 2)
        if not _is_inside_triangle(pts):
            return -np.inf
        return -_min_triangle_area_vec(pts)
    
    # Nelder-Mead parameters
    alpha = 0.5
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    
    f_simplex = np.array([objective(simplex[i]) for i in range(n_dims + 1)])
    
    for it in range(max_iter):
        # Sort simplex
        order = np.argsort(f_simplex)
        simplex = simplex[order]
        f_simplex = f_simplex[order]
        
        # Check convergence
        if np.std(f_simplex) < tol:
            break
        
        # Centroid of all but worst
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        xr = centroid + alpha * (centroid - simplex[-1])
        fr = objective(xr)
        
        if fr < f_simplex[0]:
            # Expansion
            xe = centroid + gamma * (centroid - simplex[-1])
            fe = objective(xe)
            if fe < fr:
                simplex[-1] = xe
                f_simplex[-1] = fe
            else:
                simplex[-1] = xr
                f_simplex[-1] = fr
        elif fr < f_simplex[-2]:
            simplex[-1] = xr
            f_simplex[-1] = fr
        else:
            # Contraction
            xc = centroid + rho * (simplex[-1] - centroid)
            fc = objective(xc)
            if fc < f_simplex[-1]:
                simplex[-1] = xc
                f_simplex[-1] = fc
            else:
                # Shrink
                for i in range(1, n_dims + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    f_simplex[i] = objective(simplex[i])
                    
    best_idx = np.argmin(f_simplex)
    best_x = simplex[best_idx]
    best_pts = best_x.reshape(N, 2)
    best_score = -f_simplex[best_idx]
    
    return best_pts, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # 1. Structured Initialization
    init_pts, init_score = _generate_structured_initial()
    
    best_pts = init_pts
    best_score = init_score
    
    # 2. Nelder-Mead Local Optimization
    if best_pts is not None:
        opt_pts, opt_score = _nelder_mead(best_pts, max_iter=500)
        if opt_score > best_score:
            best_pts = opt_pts
            best_score = opt_score
    
    # 3. Random Restarts with Nelder-Mead
    rng = np.random.RandomState(42)
    num_restarts = 20
    
    for r in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
            
        # Generate a new random starting point
        # Mix of random and slightly perturbed best
        if r % 2 == 0:
            # Random
            pts = np.zeros((N, 2))
            for j in range(N):
                while True:
                    u = rng.random()
                    w = rng.random()
                    if u + w <= 1.0:
                        pts[j] = [u + 0.5 * w, w * H]
                        break
        else:
            # Perturbed best
            pts = best_pts.copy()
            for j in range(N):
                pts[j] += rng.normal(0, 0.05, size=2)
            pts = _clip_to_triangle(pts)
            
        if not _is_inside_triangle(pts):
            continue
            
        # Local optimization
        opt_pts, opt_score = _nelder_mead(pts, max_iter=200)
        if opt_score > best_score:
            best_pts = opt_pts
            best_score = opt_score
            
    # Final check and return
    if best_pts is None:
        # Fallback
        pts = np.zeros((N, 2))
        for j in range(N):
            while True:
                u = rng.random()
                w = rng.random()
                if u + w <= 1.0:
                    pts[j] = [u + 0.5 * w, w * H]
                    break
        return pts
        
    if not _is_inside_triangle(best_pts):
        best_pts = _clip_to_triangle(best_pts)
        
    return best_pts
