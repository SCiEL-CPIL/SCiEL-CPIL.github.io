import numpy as np
from scipy.optimize import minimize
import time
from itertools import combinations

N = 11
TIME_LIMIT = 145.0

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
H = np.sqrt(3) / 2
Centroid = np.array([0.5, H / 3.0])

# Pre-compute all triangle indices for area calculation
tri_indices = np.array(list(combinations(range(N), 3)))

def get_areas(points):
    """Vectorized calculation of areas for all 165 triangles."""
    p_i = points[tri_indices[:, 0]]
    p_j = points[tri_indices[:, 1]]
    p_k = points[tri_indices[:, 2]]
    
    # Cross product in 2D
    cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - \
            (p_k[:, 0] - p_i[:, 0]) * (p_j[:, 1] - p_i[:, 1])
    areas = 0.5 * np.abs(cross)
    return areas

def min_area(points):
    return np.min(get_areas(points))

def clamp_to_triangle(points, eps=1e-6):
    """
    Project points onto the convex hull of the triangle (0,0), (1,0), (0.5, H).
    Uses barycentric coordinates for stable clipping.
    """
    x = points[:, 0]
    y = points[:, 1]
    
    # Barycentric coordinates
    # v = 2*y / H
    v = 2.0 * y / H
    # u = 1 - v - x
    u = 1.0 - v - x
    # w = 1 - u - v = x
    
    # Clip to [0, 1]
    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)
    
    # If u+v > 1, scale down
    sums = u + v
    invalid = sums > (1 - eps)
    if np.any(invalid):
        scale = (1 - eps) / np.where(sums > 0, sums, 1.0)
        u[invalid] *= scale[invalid]
        v[invalid] *= scale[invalid]
        
    # Convert back to Cartesian
    x_new = v + 0.5 * (1 - u - v)
    y_new = (1 - u - v) * H
    
    return np.column_stack([x_new, y_new])

def param_to_points(params):
    """
    Convert parameter vector to point coordinates.
    Strategy: 
    1. Define a base symmetric configuration (e.g., hexagonal lattice subset or radial).
    2. params control deviations from this base to allow symmetry breaking.
    
    To maximize robustness, we use a "Radial + Angular" parameterization relative to the centroid.
    However, simple radial is hard to constrain to the triangle boundary.
    
    Alternative: 
    Use 22 Cartesian coordinates but optimize them using Nelder-Mead with a penalty for going outside.
    But the directive asks for a DIFFERENT strategy.
    
    Let's try: **Active Set / Boundary Fitting**.
    The optimal solution for Heilbronn triangles often has points on the boundary.
    We can parameterize points as:
    - 3 points fixed at vertices? No, usually interior or mixed.
    - Let's use a **Polar-like** parameterization relative to the centroid, 
      but mapped through the barycentric space to ensure validity.
      
    Actually, a very effective structural change is to use **Sobol Sampling + Local Nelder-Mead** 
    on the raw Cartesian coordinates, but with a **Restart Policy** based on 
    "Delaunay Edge Flipping" or simply multiple diverse seeds.
    
    Wait, the current program uses L-BFGS-B on soft-min. 
    A materially different strategy: **Nelder-Mead on Hard Min with Multi-Start**.
    Nelder-Mead is derivative-free and handles non-smooth objectives (hard min) better 
    than gradient-based methods if initialized well. The "structural" change is 
    abandoning the smooth surrogate (soft-min) entirely and using a global-ish 
    multi-start local search on the true objective.
    """
    pts = params.reshape(N, 2)
    return clamp_to_triangle(pts)

def objective(params):
    pts = param_to_points(params)
    return -min_area(pts)

def nelder_mead_optimize(initial_pts, max_iter=2000):
    """
    Run Nelder-Mead on the hard minimum area objective.
    """
    initial_params = initial_pts.flatten()
    
    try:
        result = minimize(
            objective,
            initial_params,
            method='Nelder-Mead',
            options={
                'maxiter': max_iter,
                'xatol': 1e-8,
                'fatol': 1e-10,
                'adaptive': True
            }
        )
        return result.x.reshape(N, 2)
    except Exception:
        return initial_pts

def generate_hexagonal_lattice():
    """
    Generate a high-quality symmetric starting point.
    For N=11, a hexagonal packing is a good heuristic.
    We place points in a hex grid and select the best 11 that fit in the triangle.
    """
    best_pts = None
    best_val = -np.inf
    
    # Try different scales and offsets
    for scale in [0.2, 0.25, 0.3, 0.35, 0.4]:
        # Generate grid points in barycentric coords
        pts = []
        # Hex grid in 2D Cartesian is easier, then filter
        x_min, x_max = -0.2, 1.2
        y_min, y_max = -0.2, H + 0.2
        
        dx = scale
        dy = scale * np.sqrt(3) / 2
        
        x = np.arange(x_min, x_max, dx)
        y = np.arange(y_min, y_max, dy)
        
        # Create meshgrid
        XX, YY = np.meshgrid(x, y)
        
        # Offset every other row for hex packing
        offset = np.where(np.arange(len(y)) % 2 == 1, dx / 2, 0)
        XX = XX + offset[:, np.newaxis]
        
        candidates = np.column_stack([XX.ravel(), YY.ravel()])
        
        # Filter to triangle
        # Barycentric check
        v = 2.0 * candidates[:, 1] / H
        u = 1.0 - v - candidates[:, 0]
        w = 1.0 - u - v
        
        mask = (u >= -1e-6) & (v >= -1e-6) & (w >= -1e-6)
        valid_pts = candidates[mask]
        
        if len(valid_pts) < N:
            continue
            
        # If more than N, we need to select N. 
        # Simple heuristic: keep the ones closest to the centroid? 
        # Or just take the first N? 
        # Better: Randomly sample subsets if too many? 
        # For simplicity, if we have exactly N or close, use them.
        # If we have many, we can pick a subset that maximizes min area via a quick greedy.
        
        if len(valid_pts) == N:
            val = min_area(valid_pts)
            if val > best_val:
                best_val = val
                best_pts = valid_pts
        else:
            # Select N points from valid_pts to maximize min area
            # Greedy: Start with centroid, add point that maximizes min distance to existing?
            # Or just pick a random subset of size N from valid_pts
            rng = np.random.default_rng(42)
            # Try a few random subsets
            for _ in range(10):
                idx = rng.choice(len(valid_pts), size=N, replace=False)
                subset = valid_pts[idx]
                val = min_area(subset)
                if val > best_val:
                    best_val = val
                    best_pts = subset
                    
    if best_pts is None:
        # Fallback: Regular grid
        best_pts = np.zeros((N, 2))
        for i in range(N):
            best_pts[i] = [0.5, H/3.0] + 0.1 * np.random.randn(2)
            
    return clamp_to_triangle(best_pts)

def generate_random_points(seed):
    rng = np.random.default_rng(seed)
    pts = np.zeros((N, 2))
    for i in range(N):
        # Random point in triangle
        u = rng.random()
        v = rng.random() * (1 - u)
        x = v + 0.5 * (1 - u - v)
        y = (1 - u - v) * H
        pts[i] = [x, y]
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_pts = None
    best_val = -np.inf
    
    def update_best(pts):
        nonlocal best_pts, best_val
        pts = clamp_to_triangle(pts)
        val = min_area(pts)
        if val > best_val:
            best_val = val
            best_pts = pts.copy()
            return True
        return False

    # 1. Start with a strong heuristic: Hexagonal Lattice
    hex_pts = generate_hexagonal_lattice()
    update_best(hex_pts)
    
    # 2. Multi-start Nelder-Mead on Hard Min
    # We will run Nelder-Mead from several different starting points.
    # Starting points:
    # - Hex lattice
    # - Random points (several seeds)
    # - Perturbed hex lattice
    
    starts = [hex_pts]
    
    # Add random starts
    for seed in range(10):
        if time.time() > deadline - 10:
            break
        starts.append(generate_random_points(seed))
        
    # Add perturbed hex starts
    for i in range(5):
        if time.time() > deadline - 10:
            break
        rng = np.random.default_rng(100 + i)
        perturbed = hex_pts + rng.normal(0, 0.05, hex_pts.shape)
        starts.append(perturbed)
        
    # Optimize each start
    for i, start_pts in enumerate(starts):
        if time.time() > deadline - 5:
            break
            
        # Allocate time per optimization
        # Nelder-Mead can be slow. Limit iterations based on time.
        # ~10s per run is reasonable.
        max_iter = 2000
        
        optimized_pts = nelder_mead_optimize(start_pts, max_iter=max_iter)
        update_best(optimized_pts)
        
        # If we improved significantly, try a second polish
        if time.time() > deadline - 5:
            break
        
    # 3. Final polish on the best found
    if best_pts is not None and time.time() < deadline - 2:
        final_pts = nelder_mead_optimize(best_pts, max_iter=500)
        update_best(final_pts)
        
    if best_pts is None:
        # Absolute fallback
        best_pts = generate_random_points(0)
        
    return clamp_to_triangle(best_pts)

if __name__ == "__main__":
    pts = construct()
    print(pts)
