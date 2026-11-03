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
        # Depth is the amount one needs to move to separate
        # It's min(max1 - min2, max2 - min1) if they overlap
        # Actually, if max1 < min2 is false, they overlap.
        # The depth is min(max1, max2) - max(min1, min2)
        depth = min(max1, max2) - max(min1, min2)
        if depth < min_depth:
            min_depth = depth
            
    return min_depth

def compute_min_outer_side_params(params):
    """
    Given flat params [x1, y1, a1, ..., x12, y12, a12],
    compute the minimum outer side length required to contain all hexagons
    centered at (0,0) with angle 0.
    Also returns overlap penalty if any.
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
            if depth > 1e-6:
                total_overlap += depth

    if total_overlap > 1e-6:
        # Penalize overlaps heavily
        return 1000.0 + total_overlap * 100.0

    # 2. Compute Min Outer Side
    # Outer hexagon centered at (0,0), angle 0.
    # Vertices of outer hexagon with side S:
    # The distance from center to edge (apothem) is S * sqrt(3) / 2.
    # The constraint is that for all vertices of all inner hexagons,
    # the projection on the 6 outward normals must be <= S * sqrt(3) / 2.
    
    max_proj = 0.0
    # Outward normals for outer hexagon (angle 0)
    # Edge midpoints are at angles pi/6, pi/6+pi/3, ...
    # Normal to edge k points outward at angle pi/6 + k*pi/3
    for k in range(6):
        normal_angle = math.pi/6 + k * math.pi/3
        n = np.array([math.cos(normal_angle), math.sin(normal_angle)])
        for i in range(N):
            proj = verts_list[i] @ n
            max_proj = max(max_proj, np.max(proj))
            
    # max_proj = S * sqrt(3) / 2  => S = max_proj / (sqrt(3)/2)
    side = max_proj / (math.sqrt(3) / 2.0)
    
    return side

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_side = float('inf')
    best_params = None
    
    # Initial guesses
    # 1. Grid pattern
    # 2. Random cluster
    # 3. Hexagonal packing (1 center + 6 + 5)
    
    initial_guesses = []
    
    # Guess 1: 4x3 Grid
    pos = []
    ang = []
    idx = 0
    for row in range(4):
        for col in range(3):
            cx = col * 1.5
            cy = row * math.sqrt(3)
            if row % 2 == 1:
                cx += 0.75
            # Center the grid
            pos.append([cx - 1.5, cy - 1.5 * math.sqrt(3)])
            ang.append(0.0)
    p1 = np.array(pos).flatten()
    a1 = np.array(ang)
    initial_guesses.append(np.concatenate([p1, a1]))
    
    # Guess 2: Random
    np.random.seed(42)
    p2 = np.random.randn(N, 2) * 2.0
    a2 = np.random.rand(N) * math.pi
    initial_guesses.append(np.concatenate([p2.flatten(), a2]))
    
    # Guess 3: Hexagonal Cluster (1 center, 6 around, 5 more)
    # This is a valid packing for 7 hexagons. For 12, we can add 5 more.
    # Let's just use a dense random start near 0.
    p3 = np.random.randn(N, 2) * 1.5
    a3 = np.random.rand(N) * math.pi
    initial_guesses.append(np.concatenate([p3.flatten(), a3]))
    
    # Run optimization for each guess
    for i, x0 in enumerate(initial_guesses):
        if time.time() > deadline - 10:
            break
            
        # Nelder-Mead is good for this non-smooth problem
        # We minimize the function that returns side length if valid, else penalty
        res = minimize(compute_min_outer_side_params, x0, method='Nelder-Mead', 
                       options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6})
        
        if res.success or (res.fun < best_side and res.fun < 100.0): # Check if valid (side length < 100)
            # Verify validity strictly
            test_side = compute_min_outer_side_params(res.x)
            if test_side < 100.0: # Valid
                if test_side < best_side:
                    best_side = test_side
                    best_params = res.x.copy()
                    
    # If no valid solution found, return a fallback
    if best_params is None:
        # Fallback to initial grid, even if it has overlaps, we should return something
        # But the task requires disjoint hexagons.
        # If we can't find a valid one, we should try to return the best "valid" one we can construct.
        # Let's force a valid one by scaling up the grid.
        # But the score is the side length. If we return an invalid one, score is 0.
        # We must return a valid one.
        # Let's try a simple valid packing:
        # Place hexagons in a line? No, too long.
        # Place in a circle?
        # Let's just use the first guess and if it's invalid, we are in trouble.
        # However, the grid guess is likely valid if we scale it? No, the grid is fixed.
        # Let's assume the optimizer found something.
        
        # Fallback: Return a configuration that is known to be valid but not optimal.
        # 12 hexagons in a 3x4 grid is valid if the outer side is large enough.
        # We calculate the side for the grid.
        best_params = initial_guesses[0]
        best_side = compute_min_outer_side_params(best_params)
        # If this is still invalid (overlap), we have a problem.
        # The 4x3 grid of unit hexagons:
        # Width: 3 * 1.5 + 0.75 (offset) + 1 (radius) ?
        # Let's check if grid has overlaps.
        # Distance between centers in grid:
        # Horizontal: 1.5. Min distance for hexagons: sqrt(3) ~ 1.732.
        # So 1.5 < 1.732. The grid HAS overlaps.
        # So the grid is an INVALID initial guess for the "no overlap" constraint.
        # The optimizer will push them apart.
        
        # If best_side is still > 100 (penalty), we failed.
        # Let's try one more time with a larger random spread.
        if best_side > 100.0:
            p4 = np.random.randn(N, 2) * 5.0
            a4 = np.random.rand(N) * math.pi
            x0 = np.concatenate([p4.flatten(), a4])
            res = minimize(compute_min_outer_side_params, x0, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6})
            if res.fun < 100.0:
                best_side = res.fun
                best_params = res.x.copy()
                
    # Extract results
    positions = best_params[:2*N].reshape(N, 2)
    angles = best_params[2*N:]
    
    inner = []
    for i in range(N):
        inner.append([positions[i,0], positions[i,1], math.degrees(angles[i])])
        
    return inner, (0.0, 0.0), best_side, 0.0

# EVOLVE-BLOCK-END
