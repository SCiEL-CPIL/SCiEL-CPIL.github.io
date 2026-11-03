"""Pack 12 unit hexagons inside a regular hexagon, minimizing outer side length."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START

def hex_vertices(cx, cy, angle_rad, side=1.0):
    """Return vertices of a regular hexagon centered at (cx,cy) with given rotation in radians."""
    verts = []
    for i in range(6):
        theta = angle_rad + i * math.pi / 3
        verts.append((cx + side * math.cos(theta), cy + side * math.sin(theta)))
    return np.array(verts)

def get_sat_axes(v1, v2):
    """Get SAT axes from two polygons."""
    axes = []
    for poly in [v1, v2]:
        for i in range(len(poly)):
            j = (i + 1) % len(poly)
            edge = poly[j] - poly[i]
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-9:
                axes.append(normal / norm)
    return axes

def min_overlap_depth(v1, v2, axes):
    """Calculate minimum overlap depth using provided axes. Returns 0 if no overlap."""
    min_depth = float('inf')
    for normal in axes:
        proj1 = v1 @ normal
        proj2 = v2 @ normal
        min1, max1 = proj1.min(), proj1.max()
        min2, max2 = proj2.min(), proj2.max()
        
        # Check for gap
        if max1 < min2 or max2 < min1:
            return 0.0
            
        # Calculate overlap depth for this axis
        depth = min(max1, max2) - max(min1, min2)
        if depth < min_depth:
            min_depth = depth
            
    return min_depth

def compute_objective(params):
    """
    Given flat params [x1, y1, a1, ..., x12, y12, a12],
    compute the objective: outer side length + overlap penalty.
    """
    positions = params[:2*N].reshape(N, 2)
    angles = params[2*N:]
    
    # 1. Check Overlaps
    total_overlap = 0.0
    verts_list = [hex_vertices(positions[i,0], positions[i,1], angles[i]) for i in range(N)]
    
    for i in range(N):
        for j in range(i+1, N):
            v1 = verts_list[i]
            v2 = verts_list[j]
            axes = get_sat_axes(v1, v2)
            depth = min_overlap_depth(v1, v2, axes)
            if depth > 1e-9:
                total_overlap += depth * depth # Squared penalty for smoothness

    # 2. Compute Min Outer Side
    # Outer hexagon centered at (0,0), angle 0.
    # Normals at pi/6 + k*pi/3
    max_proj = 0.0
    for k in range(6):
        normal_angle = math.pi/6 + k * math.pi/3
        n = np.array([math.cos(normal_angle), math.sin(normal_angle)])
        for i in range(N):
            proj = verts_list[i] @ n
            max_proj = max(max_proj, np.max(proj))
            
    # S * sqrt(3) / 2 = max_proj => S = max_proj / (sqrt(3)/2)
    side = max_proj / (math.sqrt(3) / 2.0)
    
    # Penalty for overlap
    penalty = 100.0 * total_overlap
    
    return side + penalty

def is_valid(params, tol=1e-5):
    """Check if configuration is valid (no overlaps)."""
    positions = params[:2*N].reshape(N, 2)
    angles = params[2*N:]
    verts_list = [hex_vertices(positions[i,0], positions[i,1], angles[i]) for i in range(N)]
    
    for i in range(N):
        for j in range(i+1, N):
            v1 = verts_list[i]
            v2 = verts_list[j]
            axes = get_sat_axes(v1, v2)
            depth = min_overlap_depth(v1, v2, axes)
            if depth > tol:
                return False
    return True

def get_grid_initial_guess():
    """Generate a valid initial guess using a hexagonal grid cluster."""
    # Basis vectors for hexagonal lattice of touching hexagons
    # Distance between centers is sqrt(3)
    u = np.array([math.sqrt(3), 0])
    v = np.array([math.sqrt(3)/2, 1.5]) # 3/2 = 1.5
    
    points = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            p = i * u + j * v
            points.append(p)
            
    # Sort by distance to origin
    points.sort(key=lambda p: np.dot(p, p))
    
    # Take the first 12
    best_points = points[:12]
    
    # Flatten to params: x1, y1, a1, ..., x12, y12, a12
    # Angles are 0 initially
    params = []
    for p in best_points:
        params.append(p[0])
        params.append(p[1])
        params.append(0.0)
        
    return np.array(params)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin
    
    best_side = float('inf')
    best_params = None
    
    # Initial guesses
    guesses = []
    
    # 1. Grid cluster
    guesses.append(get_grid_initial_guess())
    
    # 2. Perturbed grid
    np.random.seed(42)
    for _ in range(5):
        p = get_grid_initial_guess()
        noise = np.random.randn(2*N) * 0.2
        p[:2*N] += noise
        # Angles stay 0 or small noise
        p[2*N:] += np.random.rand(N) * 0.1
        guesses.append(p)
        
    # 3. Random starts
    for _ in range(5):
        pos = np.random.randn(N, 2) * 2.0
        ang = np.random.rand(N) * (math.pi / 3)
        guesses.append(np.concatenate([pos.flatten(), ang]))
        
    for i, x0 in enumerate(guesses):
        if time.time() > deadline - 5:
            break
            
        # Use Powell method, which is good for non-smooth problems
        try:
            res = minimize(compute_objective, x0, method='Powell', 
                           options={'maxiter': 200, 'xtol': 1e-6, 'ftol': 1e-6})
            
            # Check if the result is valid (no overlaps)
            if is_valid(res.x):
                # Calculate the true side length (without penalty)
                positions = res.x[:2*N].reshape(N, 2)
                angles = res.x[2*N:]
                verts_list = [hex_vertices(positions[i,0], positions[i,1], angles[i]) for i in range(N)]
                
                max_proj = 0.0
                for k in range(6):
                    normal_angle = math.pi/6 + k * math.pi/3
                    n = np.array([math.cos(normal_angle), math.sin(normal_angle)])
                    for v in verts_list:
                        max_proj = max(max_proj, np.max(v @ n))
                
                side = max_proj / (math.sqrt(3) / 2.0)
                
                if side < best_side:
                    best_side = side
                    best_params = res.x.copy()
                    
        except Exception as e:
            pass
            
    # If no valid solution found from optimization, use the grid guess directly
    if best_params is None:
        x0 = get_grid_initial_guess()
        if is_valid(x0):
            positions = x0[:2*N].reshape(N, 2)
            angles = x0[2*N:]
            verts_list = [hex_vertices(positions[i,0], positions[i,1], angles[i]) for i in range(N)]
            
            max_proj = 0.0
            for k in range(6):
                normal_angle = math.pi/6 + k * math.pi/3
                n = np.array([math.cos(normal_angle), math.sin(normal_angle)])
                for v in verts_list:
                    max_proj = max(max_proj, np.max(v @ n))
                    
            best_side = max_proj / (math.sqrt(3) / 2.0)
            best_params = x0
        else:
            # Fallback: should not happen, but return something
            best_side = 10.0
            best_params = get_grid_initial_guess()

    # Extract results
    positions = best_params[:2*N].reshape(N, 2)
    angles = best_params[2*N:]
    
    inner = []
    for i in range(N):
        inner.append([positions[i,0], positions[i,1], math.degrees(angles[i])])
        
    return inner, (0.0, 0.0), best_side, 0.0

# EVOLVE-BLOCK-END
