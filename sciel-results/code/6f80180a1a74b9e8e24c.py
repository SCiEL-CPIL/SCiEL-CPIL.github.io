import numpy as np
from scipy.optimize import differential_evolution, minimize
import time
from itertools import combinations

N = 11
TIME_LIMIT = 115.0

# Triangle vertices
P0 = np.array([0.0, 0.0])
P1 = np.array([1.0, 0.0])
P2 = np.array([0.5, np.sqrt(3) / 2])
H = np.sqrt(3) / 2

# Pre-compute triangle indices for vectorization
tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices) # shape (165, 3)

def project_to_triangle(points):
    """
    Project points to the interior or boundary of the equilateral triangle.
    Points are given in Cartesian coordinates.
    """
    # Barycentric coordinates relative to P0, P1, P2
    # P = u*P0 + v*P1 + w*P2, u+v+w=1
    # Using the formula:
    # v = dot(P-P0, P1-P0) / dot(P1-P0, P1-P0) ... wait, easier to use linear algebra
    
    # Let's use the standard barycentric calculation
    # P0=(0,0), P1=(1,0), P2=(0.5, H)
    # x = v + 0.5*w
    # y = H*w
    # w = y/H
    # v = x - 0.5*w
    # u = 1 - v - w
    
    x = points[:, 0]
    y = points[:, 1]
    
    w = y / H
    v = x - 0.5 * w
    u = 1.0 - v - w
    
    # Check if inside: u>=0, v>=0, w>=0
    # If not, project to boundary.
    # Simple projection: clip barycentric coords to [0, 1] and renormalize?
    # No, clipping u,v,w to 0 and renormalizing gives the closest point on the boundary in the barycentric metric,
    # which is the Euclidean closest point on the boundary for a triangle?
    # Actually, yes, for a triangle, the closest point on the boundary to an exterior point
    # is found by clamping the barycentric coordinates to the valid range [0,1] and normalizing
    # IF we are talking about the "barycentric" distance. For Euclidean distance, it's slightly different
    # but clamping and normalizing is a very good approximation and keeps points inside.
    
    # Better:
    # If u < 0, point is outside side P1-P2. Project to line P1-P2.
    # If v < 0, point is outside side P0-P2. Project to line P0-P2.
    # If w < 0, point is outside side P0-P1. Project to line P0-P1.
    # If multiple are negative, it's a vertex.
    
    # Let's just clamp and renormalize. It's fast and robust.
    # u, v, w must be >= 0 and sum to 1.
    
    # If any coordinate is negative, it's outside.
    # We find the minimum coordinate. If min < 0, we set it to 0 and renormalize the others.
    # This effectively projects to the edge opposite the negative coordinate.
    
    min_u = np.min(u)
    min_v = np.min(v)
    min_w = np.min(w)
    
    # If all positive, it's inside.
    # If one is negative, project to opposite edge.
    # If two are negative, project to vertex.
    
    # Vectorized approach:
    # Create a mask for points outside
    outside = (u < 0) | (v < 0) | (w < 0)
    
    if np.any(outside):
        pts_out = points[outside]
        u_out = u[outside]
        v_out = v[outside]
        w_out = w[outside]
        
        # For each outside point, find the minimum barycentric coord
        # Set it to 0, renormalize others to sum to 1.
        min_coords = np.minimum(np.minimum(u_out, v_out), w_out)
        
        # Sum of the other two
        # If u is min, new u=0, new v = v/(v+w), new w = w/(v+w)
        # This is equivalent to:
        # 1. Clip all to 0
        # 2. Normalize
        
        u_clip = np.clip(u_out, 0, None)
        v_clip = np.clip(v_out, 0, None)
        w_clip = np.clip(w_out, 0, None)
        
        sums = u_clip + v_clip + w_clip
        # sums should be 1 for inside points, but for outside points after clipping, sum < 1
        # We renormalize
        u_out = u_clip / sums
        v_out = v_clip / sums
        w_out = w_clip / sums
        
        # Convert back to Cartesian
        x_out = v_out + 0.5 * w_out
        y_out = w_out * H
        
        points[outside] = np.column_stack([x_out, y_out])
        
    return points

def calc_min_area(points):
    """Calculate the minimum area of triangles formed by N points."""
    # points is (N, 2)
    # Use cross product area formula
    # Area = 0.5 * |det(P_j - P_i, P_k - P_i)|
    
    pi = points[tri_idx[:, 0]]
    pj = points[tri_idx[:, 1]]
    pk = points[tri_idx[:, 2]]
    
    v1 = pj - pi
    v2 = pk - pi
    
    # 2D cross product
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def objective(x):
    """
    Objective function for optimization.
    x is a flat array of 2*N coordinates.
    """
    points = x.reshape(N, 2)
    points = project_to_triangle(points)
    area = calc_min_area(points)
    return -area

def generate_perimeter_init(n_points):
    """Generate n_points evenly spaced on the perimeter of the triangle."""
    # Side lengths are all 1.
    # Total perimeter = 3.
    # We can parameterize the perimeter by t in [0, 3).
    # t in [0, 1): P0 to P1
    # t in [1, 2): P1 to P2
    # t in [2, 3): P2 to P0
    
    ts = np.linspace(0, 3, n_points, endpoint=False)
    points = []
    for t in ts:
        if t < 1:
            p = P0 + t * (P1 - P0)
        elif t < 2:
            t_prime = t - 1
            p = P1 + t_prime * (P2 - P1)
        else:
            t_prime = t - 2
            p = P2 + t_prime * (P0 - P2)
        points.append(p)
    return np.array(points)

def generate_center_ring_init(n_ring):
    """1 center point + n_ring points on the perimeter."""
    center = (P0 + P1 + P2) / 3.0
    ring_pts = generate_perimeter_init(n_ring)
    return np.vstack([center, ring_pts])

def refine_local(points_init, time_limit=5.0):
    """
    Refine a set of points using local optimization.
    """
    start_time = time.time()
    x0 = points_init.flatten()
    
    # Use SLSQP with bounds [0,1]x[0,1] for each point? 
    # The triangle is inside [0,1]x[0, sqrt(3)/2].
    # Let's just use bounds [0, 1] for x and [0, H] for y.
    bounds = []
    for i in range(N):
        bounds.append((0, 1))
        bounds.append((0, H))
        
    try:
        res = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-12}
        )
        if res.success or res.fun < -calc_min_area(project_to_triangle(x0.reshape(N, 2))):
            points_init = project_to_triangle(res.x.reshape(N, 2))
    except Exception:
        pass
        
    # Try Nelder-Mead for a bit more precision
    if time.time() - start_time < time_limit:
        try:
            res2 = minimize(
                objective,
                points_init.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            if res2.fun < -calc_min_area(points_init):
                 points_init = project_to_triangle(res2.x.reshape(N, 2))
        except Exception:
            pass
            
    return points_init

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_points = None
    best_score = -np.inf
    
    def update_best(points):
        nonlocal best_points, best_score
        points = project_to_triangle(points)
        score = calc_min_area(points)
        if score > best_score:
            best_score = score
            best_points = points.copy()
            return True
        return False

    # 1. Global Search with Differential Evolution
    # We will run DE on the Cartesian coordinates.
    # Bounds: x in [0, 1], y in [0, H]
    bounds = [(0, 1), (0, H)] * N
    
    # To save time, we run DE with a moderate population
    # N=11, 22 variables.
    
    try:
        # Use a seed for reproducibility
        de_result = differential_evolution(
            objective,
            bounds,
            seed=42,
            maxiter=50,
            popsize=15,
            tol=1e-5,
            polish=False,
            init='latinhypercube'
        )
        
        de_points = project_to_triangle(de_result.x.reshape(N, 2))
        update_best(de_points)
        
        # Refine the DE result
        if time.time() < deadline - 20:
            refined_de = refine_local(de_points, time_limit=10.0)
            update_best(refined_de)
            
    except Exception:
        pass

    # 2. Multi-start Local Optimization
    # Try different structured initializations
    
    inits = []
    
    # Perimeter only
    inits.append(generate_perimeter_init(N))
    
    # Center + Ring
    inits.append(generate_center_ring_init(N-1))
    
    # Random
    rng = np.random.default_rng(123)
    for _ in range(5):
        # Random points in bounding box, then project
        pts = rng.random((N, 2))
        pts[:, 0] *= 1.0
        pts[:, 1] *= H
        inits.append(project_to_triangle(pts))
        
    # Perturbed best
    if best_points is not None:
        for _ in range(5):
            noise = rng.normal(0, 0.05, best_points.shape)
            inits.append(project_to_triangle(best_points + noise))

    for init_pts in inits:
        if time.time() > deadline - 10:
            break
        try:
            refined = refine_local(init_pts, time_limit=5.0)
            update_best(refined)
        except Exception:
            pass

    # 3. Final Polish
    if best_points is not None and time.time() < deadline - 2:
        try:
            # High precision SLSQP
            bounds = [(0, 1), (0, H)] * N
            res_final = minimize(
                objective,
                best_points.flatten(),
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-15}
            )
            if res_final.fun < -calc_min_area(best_points):
                 best_points = project_to_triangle(res_final.x.reshape(N, 2))
        except Exception:
            pass

    return best_points

# EVOLVE-BLOCK-END
