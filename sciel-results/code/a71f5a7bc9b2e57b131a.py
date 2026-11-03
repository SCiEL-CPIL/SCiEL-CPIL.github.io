"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math

N = 12
DEADLINE = 145.0  # Safety margin before 150s kill

# EVOLVE-BLOCK-START
def get_hex_vertices(cx, cy, side=1.0, rot_deg=0.0):
    """
    Return vertices of a regular hexagon centered at (cx, cy) with given side length and rotation.
    Vertices are in counter-clockwise order.
    """
    rot_rad = np.radians(rot_deg)
    angles = np.radians(np.array([0, 60, 120, 180, 240, 300])) + rot_rad
    x = cx + side * np.cos(angles)
    y = cy + side * np.sin(angles)
    return np.column_stack((x, y))

def min_dist_convex_poly(v1, v2):
    """
    Compute the minimum distance between two convex polygons.
    Returns a negative value if they overlap (approximating penetration depth), 
    or positive distance if separated.
    Using Separating Axis Theorem (SAT) for exact overlap check.
    """
    axes = []
    for v in (v1, v2):
        for i in range(len(v)):
            p1 = v[i]
            p2 = v[(i+1)%len(v)]
            # Edge vector
            e = p2 - p1
            # Normal vector
            n = np.array([-e[1], e[0]])
            norm = np.linalg.norm(n)
            if norm > 1e-12:
                n = n / norm
                axes.append(n)
    
    min_penetration = float('inf')
    for n in axes:
        # Project v1
        proj1 = v1 @ n
        min1, max1 = np.min(proj1), np.max(proj1)
        # Project v2
        proj2 = v2 @ n
        min2, max2 = np.min(proj2), np.max(proj2)
        
        # Check for overlap
        if max1 < min2 or max2 < min1:
            # Separated. Distance is the gap.
            # For SAT, if separated on any axis, the polygons are disjoint.
            # The actual distance is not simply the gap on this axis, but for penalty purposes,
            # a large positive value indicates separation. 
            # To be more accurate for optimization, we could return the gap, but 
            # standard SAT just confirms disjointness.
            # Let's return the gap on this separating axis as a proxy for distance.
            gap = min2 - max1 if min2 > max1 else min1 - max2
            return max(gap, 0.1) # Ensure positive and non-zero gradient if possible
            
        # Calculate overlap amount
        # The overlap interval is [max(min1, min2), min(max1, max2)]
        # The depth of penetration is the length of the intersection? 
        # Actually, for SAT, if they overlap on all axes, they are intersecting.
        # The "minimum translation vector" magnitude is the min overlap width.
        overlap = min(max1, max2) - max(min1, min2)
        min_penetration = min(min_penetration, overlap)
        
    if min_penetration < 0:
        return 0.0 # Should not happen if we returned early
    return -min_penetration # Negative indicates overlap/penetration

def calculate_outer_hex_side(points, outer_rot_deg):
    """
    Calculate the minimum side length L of a regular hexagon centered at (0,0)
    with rotation outer_rot_deg that encloses all given points.
    """
    # Normals of the outer hexagon sides
    # Sides are at angles: outer_rot + 90, 150, 210, 270, 330, 30?
    # A regular hexagon centered at origin.
    # Vertices at 0, 60, ...
    # Normals to sides are at 90, 150, 210, 270, 330, 30.
    angles = np.radians(np.array([90, 150, 210, 270, 330, 30]) + outer_rot_deg)
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    
    # Compute projections
    # points shape (M, 2), normals shape (6, 2)
    # proj shape (M, 6)
    proj = points @ normals.T
    max_proj = np.max(np.abs(proj))
    
    apothem = max_proj
    # Apothem a = sqrt(3)/2 * L  => L = 2/sqrt(3) * a
    return (2.0 / np.sqrt(3.0)) * apothem

def objective(params, time_limit):
    """
    params: array of length 2*N + 1
    [x0, y0, ..., x11, y11, outer_rot_deg]
    """
    if time.time() > time_limit:
        return 1e6 # Large penalty if timeout approaching
    
    xys = params[:2*N].reshape(N, 2)
    outer_rot = params[2*N]
    
    # Fix rotations of inner hexes to 0 for simplicity and speed, 
    # as symmetric packings usually have aligned hexes.
    
    vertices_list = []
    for i in range(N):
        cx, cy = xys[i]
        v = get_hex_vertices(cx, cy, 1.0, 0.0)
        vertices_list.append(v)
    
    # Check overlaps and calculate penalty
    penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            dist = min_dist_convex_poly(vertices_list[i], vertices_list[j])
            if dist < 0:
                # Overlap, penalize heavily
                penalty += (-dist) ** 2
    
    # Calculate bounding hexagon
    all_points = np.vstack(vertices_list)
    L = calculate_outer_hex_side(all_points, outer_rot)
    
    # Total objective
    return L + 100.0 * penalty

def is_valid(xys, outer_rot, tol=1e-3):
    """Check if configuration is valid (no overlaps, all inside)."""
    vertices_list = [get_hex_vertices(x, y, 1.0, 0.0) for x, y in xys]
    
    # Check overlaps
    for i in range(N):
        for j in range(i+1, N):
            dist = min_dist_convex_poly(vertices_list[i], vertices_list[j])
            if dist < -tol:
                return False
    
    # Check bounds
    L = calculate_outer_hex_side(np.vstack(vertices_list), outer_rot)
    # If L is calculated, it means they fit in a hexagon of side L centered at 0.
    # But we need to ensure they fit in the *minimal* hexagon?
    # The objective minimizes L. If we return the best L found, it's valid by construction
    # of the bounding box calculation.
    return True

def construct():
    start_time = time.time()
    
    best_L = 40.0
    best_params = None
    
    # Initial configurations
    def make_brick_3x4():
        pts = []
        sqrt3 = np.sqrt(3)
        for r in range(4):
            y = r * 1.5
            for c in range(3):
                x = c * sqrt3
                if r % 2 == 1:
                    x += sqrt3 / 2.0
                pts.append([x, y])
        pts = np.array(pts)
        pts -= np.mean(pts, axis=0)
        return pts

    def make_brick_2x6():
        pts = []
        sqrt3 = np.sqrt(3)
        for r in range(2):
            y = r * 1.5
            for c in range(6):
                x = c * sqrt3
                if r % 2 == 1:
                    x += sqrt3 / 2.0
                pts.append([x, y])
        pts = np.array(pts)
        pts -= np.mean(pts, axis=0)
        return pts

    def make_cluster():
        pts = []
        sqrt3 = np.sqrt(3)
        pts.append([0,0])
        for k in range(6):
            ang = np.radians(60*k)
            pts.append([sqrt3*np.cos(ang), sqrt3*np.sin(ang)])
        for k in range(5):
            ang = np.radians(60*k + 30)
            pts.append([2*sqrt3*np.cos(ang), 2*sqrt3*np.sin(ang)])
        pts = np.array(pts)
        pts -= np.mean(pts, axis=0)
        return pts

    configs = [
        make_brick_3x4(),
        make_brick_2x6(),
        make_cluster()
    ]
    
    # Use scipy minimize if available, otherwise manual Nelder-Mead
    try:
        from scipy.optimize import minimize
        use_scipy = True
    except ImportError:
        use_scipy = False

    def nelder_mead_manual(func, x0, max_iter=100, alpha=1.0, gamma=2.0, rho=0.5, sigma=2.0):
        """Simple manual Nelder-Mead implementation."""
        n = len(x0)
        simplex = [x0.copy()]
        f0 = func(x0)
        simplex.append(f0)
        
        for i in range(n):
            x = x0.copy()
            x[i] += 0.05
            simplex.append(x)
            simplex.append(func(x))
            
        for it in range(max_iter):
            if time.time() > start_time + 140:
                break
                
            # Sort simplex by function value
            idx = np.argsort(simplex[1::2])
            sorted_simplex = []
            for i in idx:
                sorted_simplex.append(simplex[2*i])
                sorted_simplex.append(simplex[2*i+1])
            simplex = sorted_simplex
            
            # Check convergence
            if abs(simplex[1] - simplex[-1]) < 1e-6:
                break
                
            x_worst = simplex[-2]
            f_worst = simplex[-1]
            x_best = simplex[0]
            f_best = simplex[1]
            
            # Centroid of all but worst
            x_centroid = np.mean(simplex[0:-2:2], axis=0)
            
            # Reflection
            x_r = x_centroid + alpha * (x_centroid - x_worst)
            f_r = func(x_r)
            
            if f_r < f_best:
                # Expansion
                x_e = x_centroid + gamma * (x_r - x_centroid)
                f_e = func(x_e)
                if f_e < f_r:
                    simplex[-2] = x_e
                    simplex[-1] = f_e
                else:
                    simplex[-2] = x_r
                    simplex[-1] = f_r
            elif f_r < simplex[-3]: # Second worst
                simplex[-2] = x_r
                simplex[-1] = f_r
            else:
                # Contraction
                x_c = x_centroid + rho * (x_worst - x_centroid)
                f_c = func(x_c)
                if f_c < f_worst:
                    simplex[-2] = x_c
                    simplex[-1] = f_c
                else:
                    # Shrink
                    for i in range(1, n+1):
                        simplex[2*i] = x_best + sigma * (simplex[2*i] - x_best)
                        simplex[2*i+1] = func(simplex[2*i])
                        
        # Return best
        idx = np.argmin(simplex[1::2])
        return simplex[2*idx], simplex[2*idx+1]

    for cfg in configs:
        if time.time() > start_time + 140:
            break
            
        xys = cfg
        init_params = np.concatenate((xys.flatten(), [0.0]))
        
        if use_scipy:
            try:
                res = minimize(objective, init_params, args=(start_time + 140,), method='Nelder-Mead', 
                               options={'maxiter': 300, 'xatol': 1e-4, 'fatol': 1e-4, 'adaptive': True})
                
                L_val = calculate_outer_hex_side(
                    np.vstack([get_hex_vertices(x, y, 1.0, 0.0) for x, y in res.x[:2*N].reshape(N, 2)]), 
                    res.x[2*N]
                )
                # Verify no overlaps
                verts = [get_hex_vertices(x, y, 1.0, 0.0) for x, y in res.x[:2*N].reshape(N, 2)]
                overlap = False
                for i in range(N):
                    for j in range(i+1, N):
                        if min_dist_convex_poly(verts[i], verts[j]) < -1e-3:
                            overlap = True
                
                if not overlap and L_val < best_L:
                    best_L = L_val
                    best_params = res.x
                    
            except Exception:
                pass
        else:
            try:
                res_x, res_f = nelder_mead_manual(objective, init_params, max_iter=100)
                L_val = calculate_outer_hex_side(
                    np.vstack([get_hex_vertices(x, y, 1.0, 0.0) for x, y in res_x[:2*N].reshape(N, 2)]), 
                    res_x[2*N]
                )
                verts = [get_hex_vertices(x, y, 1.0, 0.0) for x, y in res_x[:2*N].reshape(N, 2)]
                overlap = False
                for i in range(N):
                    for j in range(i+1, N):
                        if min_dist_convex_poly(verts[i], verts[j]) < -1e-3:
                            overlap = True
                
                if not overlap and L_val < best_L:
                    best_L = L_val
                    best_params = res_x
            except Exception:
                pass

    # If no valid improvement found, return seed
    if best_params is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    # Extract best result
    xys = best_params[:2*N].reshape(N, 2)
    outer_rot = best_params[2*N]
    
    inner = []
    for i in range(N):
        inner.append([xys[i][0], xys[i][1], 0.0])
        
    return inner, (0.0, 0.0), best_L, outer_rot
# EVOLVE-BLOCK-END
