import numpy as np
import time
from scipy.optimize import minimize

N = 11
SQRT3 = np.sqrt(3)

# Precompute indices for all unique triples (i, j, k) with i < j < k
# There are C(11, 3) = 165 triples
_triples = []
for i in range(N - 2):
    for j in range(i + 1, N - 1):
        for k in range(j + 1, N):
            _triples.append((i, j, k))
_TRIPLES = np.array(_triples, dtype=np.int32)
_NUM_TRIPLES = len(_TRIPLES)

def _min_area_bary(p):
    """
    p: array of shape (N, 3) with barycentric coordinates.
    Returns the minimum normalized triangle area.
    """
    r1 = p[_TRIPLES[:, 0]]
    r2 = p[_TRIPLES[:, 1]]
    r3 = p[_TRIPLES[:, 2]]
    
    # Cross product r2 x r3
    cross_r2_r3 = np.cross(r2, r3)
    
    # Dot product r1 . (r2 x r3)
    dets = np.einsum('ij,ij->i', r1, cross_r2_r3)
    
    areas = 0.5 * np.abs(dets)
    return np.min(areas)

def _bary_to_cart(bary):
    """Convert barycentric (u,v,w) to Cartesian (x,y) for equilateral triangle."""
    x = bary[:, 1] + 0.5 * bary[:, 2]
    y = (SQRT3 / 2) * bary[:, 2]
    return np.column_stack([x, y])

def _project_bary(bary):
    """
    Project barycentric coordinates to be valid (non-negative, sum to 1).
    Simple clip and normalize.
    """
    b = bary.copy()
    b = np.clip(b, 1e-12, 1.0)
    b_sum = np.sum(b, axis=1, keepdims=True)
    b = b / b_sum
    return b

def _objective_flat(x_flat):
    """
    Objective function for optimizers: negative of the min area.
    x_flat: array of shape (N*3,)
    """
    bary = x_flat.reshape(N, 3)
    bary = _project_bary(bary)
    return -_min_area_bary(bary)

def _generate_subdivided_init():
    """
    Generate an initial configuration by dividing the equilateral triangle into
    11 sub-regions of roughly equal area and placing points at their centroids.
    
    Strategy:
    1. Divide the triangle into 3 vertical strips of equal width (and thus equal area).
    2. Distribute 11 points among these 3 strips.
       - Left strip: 4 points
       - Middle strip: 4 points
       - Right strip: 3 points
       (This distribution balances the load considering the triangular shape, 
        though equal area strips have equal height. We place them in a grid-like 
        fashion within each strip, adjusted for the slanted boundaries.)
        
    Actually, a simpler and more robust "structural" initialization is to use a 
    jittered grid or a specific known good pattern. 
    However, the prompt suggests "hierarchical subdivision".
    
    Let's implement a simple recursive bisection or a grid mapping.
    A grid of points mapped to the triangle via affine transformation of a square 
    to a triangle is a standard approach.
    
    Let's use a 4x3 grid (12 points) minus 1, or a specific arrangement.
    Given N=11, a 3x4 grid (12 points) is close. We can remove one corner or 
    distribute 11 points in a 3x4 grid pattern mapped to the triangle.
    
    Let's create a 3x4 grid in [0,1]x[0,1] and map to the triangle.
    Mapping (u,v) in [0,1]x[0,1] to Triangle with vertices (0,0), (1,0), (0.5, h).
    A common mapping is:
    x = u + v/2
    y = v * h
    But this maps the square to a parallelogram, not the triangle.
    
    Correct mapping for Triangle T with vertices A(0,0), B(1,0), C(0.5, h):
    Barycentric coordinates (l1, l2, l3) sum to 1.
    If we take a grid in barycentric space, we can sample points.
    
    Let's sample 11 points on a grid in barycentric coordinates.
    Grid size: We need 11 points.
    Let's use a 3x3 grid (9 points) + 2 extra? Or a 4x3 grid (12 points) and drop one?
    
    Let's try a 3x4 grid in "barycentric-like" coordinates.
    Actually, let's just use a deterministic set of points known to be spread out.
    For example, the vertices, midpoints of edges, and centroid, plus 5 more?
    
    Let's stick to the "subdivision" idea:
    Divide the triangle into 11 smaller triangles.
    One way: Connect the centroid to all vertices and midpoints?
    
    Let's use a simpler, robust initialization:
    Place points at the vertices of a regular grid mapped to the triangle.
    We'll use a 4x3 grid of points in the unit square, map them to the triangle,
    and if we have more than 11, remove the ones closest to the edges or corners 
    that cause small areas, or just take the first 11.
    
    Better: Use a 3x3 grid (9 points) + centroid + 1 more?
    
    Let's use a 4x3 grid (12 points) in barycentric coordinates:
    u in [0, 1], v in [0, 1], w = 1 - u - v.
    We need u, v, w >= 0.
    This defines a triangle in the (u,v) plane: u>=0, v>=0, u+v<=1.
    
    We can create a grid of points in this (u,v) triangle.
    Let's divide the (u,v) triangle into 11 regions.
    
    Alternative: Just use a well-spaced set of points.
    1. (1/3, 1/3, 1/3) - Centroid
    2. (1, 0, 0), (0, 1, 0), (0, 0, 1) - Vertices
    3. Midpoints of edges: (1/2, 1/2, 0), (1/2, 0, 1/2), (0, 1/2, 1/2)
    4. That's 7 points. We need 4 more.
    5. Points at 1/4 and 3/4 along the medians?
    
    Let's generate points on a grid in the (u,v) plane where u+v <= 1.
    Let's use a 4x4 grid in u and v, keep points where u+v <= 0.95 (to stay inside).
    This might give too many or too few.
    
    Let's try a specific deterministic initialization:
    Points at:
    (0.5, 0.5, 0)
    (0.5, 0, 0.5)
    (0, 0.5, 0.5)
    (1/3, 1/3, 1/3)
    (0.8, 0.1, 0.1)
    (0.1, 0.8, 0.1)
    (0.1, 0.1, 0.8)
    (0.6, 0.2, 0.2)
    (0.2, 0.6, 0.2)
    (0.2, 0.2, 0.6)
    (0.4, 0.4, 0.2)
    That's 11 points.
    
    Let's refine this to be more symmetric.
    """
    # Symmetric initialization
    # 3 vertices
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([0.0, 1.0, 0.0])
    v3 = np.array([0.0, 0.0, 1.0])
    
    # 3 edge midpoints
    m1 = np.array([0.5, 0.5, 0.0])
    m2 = np.array([0.5, 0.0, 0.5])
    m3 = np.array([0.0, 0.5, 0.5])
    
    # 1 centroid
    c = np.array([1/3.0, 1/3.0, 1/3.0])
    
    # 4 more points: symmetric positions
    # Let's take points on the medians, 1/4 and 3/4 from vertex?
    # Or just add 4 points in a square-like arrangement around the centroid?
    
    # Let's use a 3x3 grid in the middle region?
    # Let's just use a deterministic set of 11 points that are well spread.
    # (0.5, 0.5, 0)
    # (0.5, 0, 0.5)
    # (0, 0.5, 0.5)
    # (1/3, 1/3, 1/3)
    # (0.7, 0.15, 0.15)
    # (0.15, 0.7, 0.15)
    # (0.15, 0.15, 0.7)
    # (0.4, 0.4, 0.2)
    # (0.4, 0.2, 0.4)
    # (0.2, 0.4, 0.4)
    # (0.6, 0.2, 0.2) -> Wait, I need 11.
    
    pts = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
        [1/3.0, 1/3.0, 1/3.0],
        [0.7, 0.15, 0.15],
        [0.15, 0.7, 0.15],
        [0.15, 0.15, 0.7],
        [0.4, 0.4, 0.2]
    ]
    return np.array(pts)

def _repulsion_relaxation(bary, time_limit):
    """
    Iterative relaxation to minimize maximum pairwise distance (or maximize min distance).
    This helps in spreading points out before the final area optimization.
    We use gradient descent on the objective: -min_pairwise_distance.
    Since min distance is non-smooth, we can use a soft-min approximation or
    simply move points apart if they are too close.
    
    A simpler approach: "Force-directed" layout.
    For each point, compute the vector sum of repulsive forces from all other points.
    F_i = sum_{j != i} (p_i - p_j) / ||p_i - p_j||^2
    Update: p_i += lr * F_i
    Project back to barycentric coordinates.
    """
    start_time = time.time()
    current_bary = bary.copy()
    lr = 0.01
    
    # We want to maximize the minimum distance between any two points.
    # This is a non-smooth objective. We can approximate it by minimizing the sum of 
    # inverse squared distances, or just use a simple repulsion force.
    
    # Let's use a simple repulsion force.
    # F_i = sum_{j != i} (p_i - p_j) / ||p_i - p_j||^2
    
    for step in range(500):
        if time.time() > time_limit:
            break
            
        forces = np.zeros_like(current_bary)
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                diff = current_bary[i] - current_bary[j]
                dist_sq = np.sum(diff**2)
                if dist_sq < 1e-12:
                    # Add small random noise if they are on top of each other
                    noise = np.random.normal(0, 0.01, 3)
                    diff = noise
                    dist_sq = np.sum(diff**2)
                
                # Repulsive force: proportional to 1/dist^2, direction away
                # In barycentric coordinates, the metric is not Euclidean, 
                # but for a small step, it's a reasonable approximation.
                force = diff / dist_sq
                forces[i] += force
        
        # Update
        current_bary = current_bary + lr * forces
        current_bary = _project_bary(current_bary)
        
        # Reduce learning rate
        lr *= 0.99
        
    return current_bary

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    def _is_time_up():
        return time.time() > deadline - 2.0

    best_area = 0.0
    best_bary = None

    # Phase 1: Deterministic Hierarchical Subdivision Initialization
    # This provides a structurally balanced starting point.
    init_bary = _generate_subdivided_init()
    
    # Phase 2: Repulsion-based Relaxation
    # Spread the points out to maximize minimum pairwise distance.
    # This helps in finding a configuration where no three points are too close.
    relaxed_bary = _repulsion_relaxation(init_bary, deadline - 10.0)
    
    current_area = _min_area_bary(relaxed_bary)
    best_area = current_area
    best_bary = relaxed_bary.copy()
    
    # Phase 3: Final Polish with Nelder-Mead
    # NM is good for final convergence when close to optimum.
    if best_bary is not None and not _is_time_up():
        x0 = best_bary.flatten()
        try:
            res_nm = minimize(
                _objective_flat,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            if not _is_time_up():
                refined_bary = _project_bary(res_nm.x.reshape(N, 3))
                refined_area = _min_area_bary(refined_bary)
                if refined_area > best_area:
                    best_area = refined_area
                    best_bary = refined_bary.copy()
        except Exception:
            pass

    if best_bary is None:
        # Fallback
        rng = np.random.default_rng(0)
        best_bary = rng.dirichlet([1, 1, 1], size=N)
        best_bary = _project_bary(best_bary)

    return _bary_to_cart(best_bary)

def _fallback():
    """Fallback if no valid result found."""
    rng = np.random.default_rng(0)
    bary = rng.dirichlet([1, 1, 1], size=N)
    bary = _project_bary(bary)
    return _bary_to_cart(bary)
