"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations
from scipy.optimize import minimize

def _bary_to_cart(bary):
    """Convert barycentric coordinates (n,3) to Cartesian (n,2).
    Triangle vertices: A=(0,0), B=(1,0), C=(0.5, sqrt(3)/2)
    bary[i] = (lambda_A, lambda_B, lambda_C), sum to 1, all >= 0
    """
    bary = np.clip(bary, 0, None)
    sums = bary.sum(axis=1, keepdims=True)
    sums = np.where(sums < 1e-12, 1.0, sums)
    bary = bary / sums
    x = bary[:, 1] + 0.5 * bary[:, 2]
    y = (3 ** 0.5) / 2 * bary[:, 2]
    return np.column_stack([x, y])

def _min_triangle_area(points):
    """Compute the minimum triangle area among all C(n,3) triples."""
    n = len(points)
    if n < 3:
        return 0.0
    min_area = np.inf
    for i, j, k in combinations(range(n), 3):
        d1 = points[j] - points[i]
        d2 = points[k] - points[i]
        cross = d1[0] * d2[1] - d1[1] * d2[0]
        area = 0.5 * abs(cross)
        if area < min_area:
            min_area = area
    return float(min_area)

def _is_inside_triangle(p, eps=1e-9):
    """Check if point p=(x,y) is inside or on the boundary of the equilateral triangle."""
    x, y = p
    if y < -eps:
        return False
    if y > np.sqrt(3)/2 + eps:
        return False
    if y > np.sqrt(3) * x + eps:
        return False
    if y > np.sqrt(3) * (1 - x) + eps:
        return False
    return True

def _penalty(points):
    """Penalty for points outside the triangle."""
    pen = 0.0
    for p in points:
        if not _is_inside_triangle(p):
            x, y = p
            d_ab = max(0, -y)
            d_ac = max(0, np.sqrt(3)*x - y) / 2
            d_bc = max(0, np.sqrt(3)*(1-x) - y) / 2
            pen += (d_ab + d_ac + d_bc) ** 2
    return pen

def _lattice_init():
    """Generate a symmetric lattice-based initial configuration."""
    # Use a 3x3 triangular lattice pattern, selecting 11 points
    # Barycentric coordinates for a triangular grid
    # Level 0: vertices (3 points)
    # Level 1: edge midpoints (3 points)
    # Level 2: interior points (5 points from a 3x3 grid minus edges)
    
    points_bary = []
    
    # Vertices
    points_bary.append([1.0, 0.0, 0.0])
    points_bary.append([0.0, 1.0, 0.0])
    points_bary.append([0.0, 0.0, 1.0])
    
    # Edge midpoints
    points_bary.append([0.5, 0.5, 0.0])
    points_bary.append([0.5, 0.0, 0.5])
    points_bary.append([0.0, 0.5, 0.5])
    
    # Interior points - use a 3x3 grid in barycentric space
    # For a 3x3 grid, coordinates are at positions (i/3, j/3, k/3) where i+j+k=3
    # But we need to select 5 more points from the interior
    
    # Interior points at 1/3 and 2/3 fractions
    # Center
    points_bary.append([1/3, 1/3, 1/3])
    
    # Four points forming a smaller triangle around the center
    # These are at the centroids of the 4 sub-triangles of the 2x2 subdivision
    # Actually, let's use points at (2/3, 1/6, 1/6) and its rotations
    points_bary.append([2/3, 1/6, 1/6])
    points_bary.append([1/6, 2/3, 1/6])
    points_bary.append([1/6, 1/6, 2/3])
    
    # One more interior point - let's use a point on one of the medians
    points_bary.append([1/2, 1/4, 1/4])
    
    # We have 3 + 3 + 5 = 11 points
    bary = np.array(points_bary)
    return _bary_to_cart(bary)

def _random_valid_point(rng):
    """Generate a random point inside the triangle using barycentric coords."""
    while True:
        u = rng.uniform(0, 1)
        v = rng.uniform(0, 1)
        if u + v <= 1:
            bary = np.array([1 - u - v, u, v])
            return _bary_to_cart(bary.reshape(1, 3))[0]

def _simulated_annealing(initial_points, max_iter=50000, t0=0.05, t_end=0.0001, timeout=None):
    """Simulated annealing to maximize minimum triangle area."""
    start_time = time.time()
    rng = np.random.RandomState(42)
    
    current_points = initial_points.copy()
    current_score = _min_triangle_area(current_points)
    
    best_points = current_points.copy()
    best_score = current_score
    
    # Precompute all triangle indices
    n = len(current_points)
    tri_indices = list(combinations(range(n), 3))
    
    for it in range(max_iter):
        if timeout is not None and time.time() - start_time > timeout:
            break
        
        # Temperature schedule
        t = t0 * (t_end / t0) ** (it / max_iter)
        
        # Pick a random point to move
        idx = rng.randint(n)
        old_point = current_points[idx].copy()
        
        # Generate a new position - random perturbation
        # Use a step size that decreases with temperature
        step_size = 0.05 * (t / t0) + 0.001
        new_point = old_point + rng.normal(0, step_size, size=2)
        
        # Check if new point is inside the triangle
        if not _is_inside_triangle(new_point):
            # Reflect or reject
            # Try to project back
            if rng.random() < 0.5:
                continue
            else:
                # Move towards centroid
                centroid = np.array([0.5, np.sqrt(3)/6])
                new_point = 0.5 * old_point + 0.5 * centroid
                if not _is_inside_triangle(new_point):
                    continue
        
        # Compute new score
        current_points[idx] = new_point
        new_score = _min_triangle_area(current_points)
        
        # Accept or reject
        if new_score >= current_score:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_points = current_points.copy()
        else:
            # Accept with probability exp((new_score - current_score) / t)
            delta = new_score - current_score
            if delta < -10 * t:
                # Too bad, reject
                current_points[idx] = old_point
            elif rng.random() < np.exp(delta / t):
                current_score = new_score
            else:
                current_points[idx] = old_point
    
    return best_points, best_score

def _nelder_mead_refine(points, max_iter=2000, timeout=None):
    """Refine points using Nelder-Mead in Cartesian space."""
    start_time = time.time()
    
    def obj(flat):
        if timeout is not None and time.time() - start_time > timeout:
            return 1e6
        pts = flat.reshape(N, 2)
        pen = _penalty(pts)
        if pen > 1e-8:
            return 1e6 + pen
        return -_min_triangle_area(pts)
    
    flat = points.flatten()
    try:
        res = minimize(obj, flat, method='Nelder-Mead', 
                       options={'maxiter': max_iter, 'xatol': 1e-10, 'fatol': 1e-12})
        pts = res.x.reshape(N, 2)
        if _penalty(pts) < 1e-6:
            return pts, -res.fun
        else:
            return points, _min_triangle_area(points)
    except Exception:
        return points, _min_triangle_area(points)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    best_points = None
    best_score = -np.inf
    
    # Phase 1: Start with lattice initialization
    lattice_pts = _lattice_init()
    lattice_score = _min_triangle_area(lattice_pts)
    if lattice_score > best_score:
        best_score = lattice_score
        best_points = lattice_pts.copy()
    
    # Phase 2: Simulated Annealing from lattice
    elapsed = time.time() - start_time
    if elapsed < deadline - 20:
        remaining = deadline - time.time() - 20
        sa_points, sa_score = _simulated_annealing(lattice_pts, max_iter=30000, timeout=remaining)
        if sa_score > best_score:
            best_score = sa_score
            best_points = sa_points.copy()
    
    # Phase 3: Nelder-Mead refinement from SA result
    elapsed = time.time() - start_time
    if elapsed < deadline - 10:
        remaining = deadline - time.time() - 10
        if remaining > 5:
            refined, score = _nelder_mead_refine(best_points, max_iter=3000, timeout=remaining)
            if score > best_score:
                best_score = score
                best_points = refined
    
    # Phase 4: Additional SA from random starts if time allows
    elapsed = time.time() - start_time
    if elapsed < deadline - 15:
        rng = np.random.RandomState(123)
        for trial in range(3):
            if time.time() > deadline - 15:
                break
            remaining = deadline - time.time() - 15
            if remaining < 10:
                break
            
            # Random initial configuration
            rand_pts = np.array([_random_valid_point(rng) for _ in range(N)])
            rand_score = _min_triangle_area(rand_pts)
            
            # SA from random start
            sa_pts, sa_sc = _simulated_annealing(rand_pts, max_iter=10000, timeout=remaining)
            
            # NM refinement
            if time.time() < deadline - 5:
                nm_pts, nm_sc = _nelder_mead_refine(sa_pts, max_iter=1000, timeout=deadline - time.time() - 5)
                if nm_sc > best_score:
                    best_score = nm_sc
                    best_points = nm_pts.copy()
    
    # Final refinement
    elapsed = time.time() - start_time
    if best_points is not None and elapsed < deadline - 5:
        remaining = deadline - time.time() - 5
        if remaining > 3:
            refined, score = _nelder_mead_refine(best_points, max_iter=2000, timeout=remaining)
            if score > best_score:
                best_score = score
                best_points = refined
    
    if best_points is None:
        best_points = _lattice_init()
    
    return best_points
# EVOLVE-BLOCK-END
