import numpy as np
from scipy.spatial import Voronoi
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time
from itertools import combinations

N = 11
TIME_LIMIT = 145.0

# Triangle vertices in Cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
H = np.sqrt(3) / 2

# Pre-compute triangle indices
tri_indices = list(combinations(range(N), 3))
M = len(tri_indices)
tri_idx = np.array(tri_indices)  # (M, 3)

def bary_to_cart(bary):
    """Convert barycentric coords (u, v) to Cartesian (x, y)."""
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = v + 0.5 * w
    y = w * H
    return np.column_stack([x, y])

def cart_to_bary(cart):
    """Convert Cartesian (x, y) to barycentric (u, v)."""
    x = cart[:, 0]
    y = cart[:, 1]
    w = y / H
    v = x - 0.5 * w
    u = 1.0 - v - w
    return np.column_stack([u, v])

def calc_min_area_cartesian(cart):
    """Calculate the minimum area of triangles formed by N points in Cartesian coords."""
    x = cart[:, 0]
    y = cart[:, 1]
    
    xi = x[tri_idx[:, 0]]
    yi = y[tri_idx[:, 0]]
    xj = x[tri_idx[:, 1]]
    yj = y[tri_idx[:, 1]]
    xk = x[tri_idx[:, 2]]
    yk = y[tri_idx[:, 2]]
    
    det = (xj - xi) * (yk - yi) - (yj - yi) * (xk - xi)
    areas = 0.5 * np.abs(det)
    return np.min(areas)

def get_areas_cartesian(cart):
    """Calculate all triangle areas."""
    x = cart[:, 0]
    y = cart[:, 1]
    
    xi = x[tri_idx[:, 0]]
    yi = y[tri_idx[:, 0]]
    xj = x[tri_idx[:, 1]]
    yj = y[tri_idx[:, 1]]
    xk = x[tri_idx[:, 2]]
    yk = y[tri_idx[:, 2]]
    
    det = (xj - xi) * (yk - yi) - (yj - yi) * (xk - xi)
    areas = 0.5 * np.abs(det)
    return areas

def clamp_cart(cart, eps=1e-9):
    """Clamp Cartesian points to be inside the triangle."""
    cart = cart.copy()
    x = cart[:, 0]
    y = cart[:, 1]
    
    y = np.clip(y, eps, H - eps)
    
    x_min = y / (2 * H)
    x_max = 1 - y / (2 * H)
    
    x = np.clip(x, x_min + eps, x_max - eps)
    
    return np.column_stack([x, y])

def compute_voronoi_centroids(points):
    """
    Compute the centroids of Voronoi cells for points, clipped to the triangle.
    Returns an array of new points (centroids).
    """
    # We need to clip the Voronoi diagram to the triangle.
    # A robust way is to add the triangle vertices as "fake" points with very low weight
    # or simply compute the Voronoi and then intersect each cell with the triangle.
    # However, scipy's Voronoi doesn't handle clipping directly.
    
    # Alternative approach:
    # 1. Compute Voronoi of the points.
    # 2. For each point, find the vertices of its Voronoi cell.
    # 3. Intersect this polygon with the triangle polygon.
    # 4. Compute the centroid of the intersection.
    
    # This is computationally expensive and prone to edge cases.
    
    # A simpler, more robust "Voronoi-like" heuristic:
    # Iteratively move each point towards the centroid of its nearest neighbors' exclusion zone?
    # No, let's stick to the actual Voronoi centroid but handle clipping carefully.
    
    # Actually, a common trick for "Voronoi in a convex polygon" is to reflect the points
    # across the boundaries and compute the Voronoi of the reflected set, then take the part inside.
    # But that's complex.
    
    # Let's try a different geometric interpretation:
    # The "Voronoi centroid" for a point p_i is the center of mass of the region in the triangle
    # that is closer to p_i than to any other point.
    
    # We can approximate this by:
    # 1. Creating a fine grid inside the triangle.
    # 2. Assigning each grid point to its nearest data point.
    # 3. Computing the centroid of the assigned grid points for each data point.
    
    # This is a discrete approximation of the Voronoi centroid. It is robust and easy to implement.
    
    # Grid resolution trade-off:
    # Too low -> poor approximation.
    # Too high -> slow.
    # Let's use a 100x100 grid or similar.
    
    # Generate grid in barycentric coordinates for easy clipping
    n_grid = 100
    u_vals = np.linspace(0, 1, n_grid)
    v_vals = np.linspace(0, 1, n_grid)
    
    # Create meshgrid
    U, V = np.meshgrid(u_vals, v_vals)
    
    # Filter points inside the triangle (u >= 0, v >= 0, u+v <= 1)
    mask = (U >= 0) & (V >= 0) & (U + V <= 1)
    
    # Get valid grid points
    u_valid = U[mask]
    v_valid = V[mask]
    
    # Convert to Cartesian
    w_valid = 1 - u_valid - v_valid
    x_grid = v_valid + 0.5 * w_valid
    y_grid = w_valid * H
    
    grid_points = np.column_stack([x_grid, y_grid])
    
    # For each data point, find the grid points closest to it
    # This is equivalent to computing the Voronoi cell in the discrete grid
    # and taking the centroid.
    
    new_points = np.zeros_like(points)
    
    for i in range(N):
        # Distance from grid points to point i
        dists = np.linalg.norm(grid_points - points[i], axis=1)
        
        # Assign grid point to i if it is closer to i than to any other point
        # We can do this by finding the minimum distance to any point
        all_dists = np.linalg.norm(grid_points[:, np.newaxis, :] - points[np.newaxis, :, :], axis=2)
        min_dists = np.min(all_dists, axis=1)
        closest_indices = np.argmin(all_dists, axis=1)
        
        # Grid points belonging to cell i
        cell_mask = (closest_indices == i)
        
        if np.any(cell_mask):
            # Centroid of the cell
            cell_points = grid_points[cell_mask]
            new_points[i] = np.mean(cell_points, axis=0)
        else:
            # Fallback: keep original point
            new_points[i] = points[i]
            
    return new_points

def voronoi_refine(points, max_iter=50, deadline=None):
    """
    Refine points by iteratively moving them to their Voronoi cell centroids.
    This minimizes the variance of cell areas, promoting a uniform distribution.
    """
    current_points = points.copy()
    
    for i in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
            
        new_points = compute_voronoi_centroids(current_points)
        
        # Clamp to triangle
        new_points = clamp_cart(new_points)
        
        # Check for convergence
        if np.linalg.norm(new_points - current_points) < 1e-6:
            break
            
        current_points = new_points
        
    return current_points

def generate_lattice_init():
    """Generate a lattice-based initialization."""
    bary = []
    candidates = [
        [0, 0], [0, 1], [1, 0],  # Vertices
        [0, 0.5], [0.5, 0], [0.5, 0.5],  # Midpoints
        [1/3, 1/3],  # Center
        [0, 0.25], [0.25, 0], [0.25, 0.25], [0.5, 0.25]  # Additional
    ]
    
    selected = candidates[:11]
    bary = np.array(selected)
    
    # Add small perturbation
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 0.005, bary.shape)
    bary += noise
    bary = np.clip(bary, 1e-9, 1-1e-9)
    # Ensure u+v <= 1
    sums = bary[:, 0] + bary[:, 1]
    invalid = sums > 1 - 1e-9
    if np.any(invalid):
        scale = (1 - 1e-9) / sums[invalid]
        bary[invalid, 0] *= scale
        bary[invalid, 1] *= scale
        
    return bary_to_cart(bary)

def generate_random_init(seed):
    rng = np.random.default_rng(seed)
    u = rng.random(N)
    v = rng.random(N) * (1 - u)
    bary = np.column_stack([u, v])
    return bary_to_cart(bary)

def refine_local(cart, deadline):
    """
    Multi-stage local refinement:
    1. Voronoi Centroid Method (Geometric Fairness).
    2. L-BFGS-B with soft objective (Polish).
    """
    current_cart = cart.copy()
    current_min = calc_min_area_cartesian(current_cart)
    
    # Stage 1: Voronoi Refinement
    if time.time() < deadline - 10:
        voronoi_cart = voronoi_refine(current_cart, max_iter=20, deadline=deadline - 10)
        voronoi_min = calc_min_area_cartesian(voronoi_cart)
        
        if voronoi_min > current_min:
            current_cart = voronoi_cart
            current_min = voronoi_min
            
    # Stage 2: L-BFGS-B with soft objective
    if time.time() < deadline - 5:
        def soft_obj(x):
            c = x.reshape(N, 2)
            c = clamp_cart(c)
            areas = get_areas_cartesian(c)
            # Minimize negative log-sum-exp of areas (approximates max-min)
            alpha = 50.0
            neg_alpha_areas = -alpha * areas
            max_neg_alpha = np.max(neg_alpha_areas)
            lse = max_neg_alpha + np.log(np.sum(np.exp(neg_alpha_areas - max_neg_alpha)))
            return -lse / alpha
        
        try:
            res = minimize(
                soft_obj,
                current_cart.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-10}
            )
            
            new_cart = clamp_cart(res.x.reshape(N, 2))
            new_min = calc_min_area_cartesian(new_cart)
            
            if new_min > current_min:
                current_cart = new_cart
                current_min = new_min
        except Exception:
            pass
    
    return current_cart, current_min

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT - 5  # Safety margin
    
    best_cart = None
    best_val = -np.inf
    
    def update_best(cart):
        nonlocal best_cart, best_val
        if cart is None:
            return
        cart = clamp_cart(cart)
        val = calc_min_area_cartesian(cart)
        if val > best_val:
            best_val = val
            best_cart = cart.copy()
            
    # Phase 1: Generate diverse initializations
    inits = []
    
    # Lattice init
    inits.append(generate_lattice_init())
    
    # Random inits
    for i in range(5):
        inits.append(generate_random_init(100 + i))
        
    # Phase 2: Refine each initialization
    for init_cart in inits:
        if time.time() > deadline - 15:
            break
            
        # Quick evaluation
        init_val = calc_min_area_cartesian(init_cart)
        
        # Refine
        refined_cart, refined_val = refine_local(init_cart, deadline)
        
        update_best(refined_cart)
        
        # Also try a few more random perturbations from the refined solution
        if time.time() < deadline - 10:
            rng = np.random.default_rng()
            for _ in range(3):
                perturbed = refined_cart + rng.normal(0, 0.01, refined_cart.shape)
                perturbed = clamp_cart(perturbed)
                pert_val = calc_min_area_cartesian(perturbed)
                
                if pert_val > best_val:
                    # Refine this perturbation too
                    refined2, val2 = refine_local(perturbed, deadline)
                    update_best(refined2)
                    
    # Fallback
    if best_cart is None:
        best_cart = generate_random_init(0)
        
    return best_cart
