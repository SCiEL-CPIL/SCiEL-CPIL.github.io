import numpy as np
import time
import warnings
from scipy.optimize import minimize, linprog
from itertools import product

N = 11
DEADLINE = 140.0  # Safety margin before 150s kill

def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx,cy)."""
    angle = np.radians(angle_deg)
    verts = np.empty((6, 2))
    for i in range(6):
        theta = angle + np.pi * i / 3.0
        verts[i, 0] = cx + side * np.cos(theta)
        verts[i, 1] = cy + side * np.sin(theta)
    return verts

def _hex_edge_normals(angle_deg):
    """Return 6 edge normals for a hexagon with given rotation angle (degrees)."""
    angle = np.radians(angle_deg)
    normals = np.empty((6, 2))
    for i in range(6):
        # Normals point outward from center, perpendicular to edges
        # For a hexagon with vertices at angles 0, pi/3, ..., the edge normals are at pi/6, pi/2, ...
        # Actually, let's be precise.
        # Vertices: v_i = (cos(theta_i), sin(theta_i)) where theta_i = angle + i*pi/3
        # Edge i is between v_i and v_{i+1}.
        # Normal to edge i points outward.
        # The angle of the normal is (theta_i + theta_{i+1})/2 = angle + (i + 0.5)*pi/3 = angle + pi/6 + i*pi/3
        mid_angle = angle + np.pi / 6.0 + np.pi * i / 3.0
        normals[i, 0] = np.cos(mid_angle)
        normals[i, 1] = np.sin(mid_angle)
    return normals

def _hex_vertices_batch(centers, angles, side=1.0):
    """Vectorized vertex calculation for multiple hexagons.
    centers: (N, 2)
    angles: (N,)
    Returns: (N, 6, 2)
    """
    N = len(centers)
    angles_rad = np.radians(angles)
    # Base angles: 0, pi/3, ...
    base_angles = np.pi * np.arange(6) / 3.0
    # (N, 1) + (1, 6)
    thetas = angles_rad[:, np.newaxis] + base_angles[np.newaxis, :]
    
    verts = np.empty((N, 6, 2))
    verts[:, :, 0] = centers[:, 0:1] + side * np.cos(thetas)
    verts[:, :, 1] = centers[:, 1:2] + side * np.sin(thetas)
    return verts

def _hex_min_distance(centers_i, angle_i, centers_j, angle_j):
    """Compute minimum distance between two hexagons using SAT.
    Returns: (min_gap, separating_axis_index)
    min_gap > 0 means separated, min_gap < 0 means overlapping.
    """
    v1 = _hex_vertices(centers_i[0], centers_i[1], 1.0, angle_i)
    v2 = _hex_vertices(centers_j[0], centers_j[1], 1.0, angle_j)
    
    n1 = _hex_edge_normals(angle_i)
    n2 = _hex_edge_normals(angle_j)
    
    all_normals = np.vstack([n1, n2])
    
    min_gap = float('inf')
    best_axis = -1
    
    for k in range(12):
        ax = all_normals[k]
        proj1 = v1 @ ax
        proj2 = v2 @ ax
        
        # Gap is min of (max1 - min2) and (max2 - min1)
        gap1 = proj1.max() - proj2.min()
        gap2 = proj2.max() - proj1.min()
        gap = min(gap1, gap2)
        
        if gap < min_gap:
            min_gap = gap
            best_axis = k
    
    return min_gap, best_axis

def _get_valid_fallback():
    """Return a known valid, non-overlapping configuration."""
    # Use a grid layout that is guaranteed to be valid
    # Hexagon width (flat-to-flat) is 2*sqrt(3) ~ 3.464
    # Spacing 4.0 is safe
    centers = []
    angles = [0.0] * N
    x = 0.0
    y = 0.0
    for i in range(N):
        centers.append((x, y))
        x += 4.0
        if i % 2 == 1:
            y += 4.0
            x = -4.0 + (i // 2) * 8.0
    
    # Optimize outer for this fallback
    side, center, angle = _optimize_outer(centers, angles)
    return centers, angles, center, side, angle

def _optimize_outer(centers, inner_angles):
    """Find optimal outer hexagon for fixed inner positions using coarse-to-fine sweep."""
    best_side = float('inf')
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    # Coarse sweep
    for outer_angle in np.linspace(0, 60, 31):
        side, center = _min_enclosing_side_lp(centers, inner_angles, outer_angle)
        if side < best_side:
            best_side = side
            best_center = center
            best_angle = outer_angle
            
    # Fine sweep around best
    if best_side < float('inf'):
        for outer_angle in np.linspace(max(0, best_angle - 3), min(60, best_angle + 3), 13):
            side, center = _min_enclosing_side_lp(centers, inner_angles, outer_angle)
            if side < best_side:
                best_side = side
                best_center = center
                best_angle = outer_angle

    return best_side, best_center, best_angle

def _min_enclosing_side_lp(centers, inner_angles, outer_angle_deg):
    """
    Compute minimum side length of a regular hexagon (fixed orientation)
    enclosing all inner hexagons using Linear Programming.
    """
    sqrt3 = np.sqrt(3)
    # Normals of outer hexagon
    angle = np.radians(outer_angle_deg)
    normals = _hex_edge_normals(outer_angle_deg)
    
    # Get all vertices
    all_points = np.vstack([_hex_vertices(centers[i][0], centers[i][1], 1.0, inner_angles[i]) 
                            for i in range(len(centers))])
    
    # Projections
    projections = all_points @ normals.T
    A = projections.max(axis=0)
    
    # Variables: x = [cx, cy, r]
    # Minimize r
    c_obj = np.array([0.0, 0.0, 1.0])
    
    # Constraints: n_i . c - r <= -A_i
    A_mat = np.zeros((6, 3))
    b_vec = np.zeros(6)
    
    for i in range(6):
        A_mat[i, 0] = normals[i, 0]
        A_mat[i, 1] = normals[i, 1]
        A_mat[i, 2] = -1.0
        b_vec[i] = -A[i]
        

    res = linprog(c_obj, A_ub=A_mat, b_ub=b_vec, method='highs')
    
    if res.status != 0:
        return float('inf'), np.array([0.0, 0.0])
        
    cx, cy, r = res.x
    side = 2 * r / sqrt3
    return side, np.array([cx, cy])

def _hex_overlap(c1, a1, c2, a2, tol=1e-9):
    """Check if two unit hexagons overlap using SAT."""
    min_gap, _ = _hex_min_distance(c1, a1, c2, a2)
    return min_gap < -tol

def _check_validity(centers, inner_angles, outer_center, outer_side, outer_angle_deg, tol=1e-7):
    """Check containment and non-overlap."""
    sqrt3 = np.sqrt(3)
    normals = _hex_edge_normals(outer_angle_deg)
        
    inradius = outer_side * sqrt3 / 2
    
    for i in range(len(centers)):
        verts = _hex_vertices(centers[i][0], centers[i][1], 1.0, inner_angles[i])
        for vx, vy in verts:
            for nx, ny in normals:
                dist = (vx - outer_center[0]) * nx + (vy - outer_center[1]) * ny
                if dist > inradius + tol:
                    return False
    
    for i in range(N):
        for j in range(i + 1, N):
            if _hex_overlap(centers[i], inner_angles[i], centers[j], inner_angles[j]):
                return False
    
    return True

def _smooth_objective(vars, beta_overlap=10.0, beta_side=10.0):
    """
    vars: [x1, y1, th1, ..., x11, y11, th11, phi_outer]
    Returns: Smooth approximation of (OuterSide + OverlapPenalty)
    """
    # Unpack
    centers = np.empty((N, 2))
    angles = np.empty(N)
    for i in range(N):
        centers[i, 0] = vars[3*i]
        centers[i, 1] = vars[3*i+1]
        angles[i] = vars[3*i+2]
        
    phi_outer = vars[3*N]
    
    # 1. Containment Cost (Smooth Max of Projections)
    normals = _hex_edge_normals(phi_outer)
        
    # Vertices of all inner hexagons
    verts = _hex_vertices_batch(centers, angles, 1.0) # (N, 6, 2)
    
    max_proj = np.zeros(6)
    for k in range(6):
        n = normals[k]
        proj = np.einsum('nij,j->ni', verts, n)
        max_proj[k] = np.max(np.abs(proj))
        
    log_sum_exp = np.log(np.sum(np.exp(beta_side * max_proj)))
    smooth_side = (2.0 / np.sqrt(3.0)) * (log_sum_exp / beta_side)
    
    # 2. Overlap Penalty
    penalty = 0.0
    
    for i in range(N):
        for j in range(i + 1, N):
            min_gap, _ = _hex_min_distance(centers[i], angles[i], centers[j], angles[j])
            
            if min_gap > 0:
                p = 0.0
            else:
                x = -beta_overlap * min_gap
                if x > 50:
                    p = -min_gap
                else:
                    p = (1.0 / beta_overlap) * np.log(1.0 + np.exp(x))
            penalty += p * p
            
    return smooth_side + 10.0 * penalty

def _local_optimize(initial_vars, max_iter=1000):
    """Run L-BFGS-B optimization."""
    # Initial value
    f0 = _smooth_objective(initial_vars)
    
    # Bounds: Angles in [0, 360), Positions in [-20, 20]
    bounds = []
    for i in range(N):
        bounds.append((-20, 20)) # x
        bounds.append((-20, 20)) # y
        bounds.append((0, 360)) # angle
    bounds.append((0, 60)) # outer angle
    
    try:
        res = minimize(
            _smooth_objective,
            initial_vars,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-6, 'gtol': 1e-6, 'disp': False}
        )
        return res.x, res.fun
    except Exception as e:
        return initial_vars, f0

def _try_arrangement(centers, angles, deadline):
    """Try to optimize an arrangement and return (side, centers, angles, outer_center, outer_angle) or None."""
    if time.time() > deadline - 5:
        return None
    
    # Convert to vars
    vars_list = []
    for i in range(N):
        vars_list.append(centers[i][0])
        vars_list.append(centers[i][1])
        vars_list.append(angles[i])
    vars_list.append(0.0) # outer angle
    initial_vars = np.array(vars_list)
    
    opt_vars, opt_val = _local_optimize(initial_vars, max_iter=500)
    
    centers_opt = [(opt_vars[3*i], opt_vars[3*i+1]) for i in range(N)]
    angles_opt = [opt_vars[3*i+2] for i in range(N)]
    
    side, center, angle = _optimize_outer(centers_opt, angles_opt)
    
    if np.isfinite(side) and _check_validity(centers_opt, angles_opt, center, side, angle):
        return side, centers_opt, angles_opt, center, angle
    return None

def construct():
    start_time = time.time()
    
    best_side = float('inf')
    best_result = None
    
    def update_best(result):
        nonlocal best_side, best_result
        if result is not None:
            side, centers, angles, center, angle = result
            if side < best_side:
                best_side = side
                best_result = (centers, angles, center, side, angle)
    
    # 1. Hexagonal Lattice (center + ring 1 + 4 from ring 2)
    d = 2 * np.sqrt(3)
    centers = [(0.0, 0.0)]
    angles = [0.0]
    for i in range(6):
        angle = np.pi / 3 * i
        centers.append((d * np.cos(angle), d * np.sin(angle)))
        angles.append(0.0)
        
    d2 = 4 * np.sqrt(3)
    for i in range(4):
        angle = np.pi / 3 * i
        centers.append((d2 * np.cos(angle), d2 * np.sin(angle)))
        angles.append(0.0)
        
    result = _try_arrangement(centers, angles, DEADLINE)
    update_best(result)
    
    # 2. Random Jittered Lattice
    if time.time() < DEADLINE - 15:
        np.random.seed(42)
        jittered_centers = [(c[0] + np.random.normal(0, 0.5), c[1] + np.random.normal(0, 0.5)) for c in centers]
        jittered_angles = [a + np.random.normal(0, 10.0) for a in angles]
        
        result = _try_arrangement(jittered_centers, jittered_angles, DEADLINE)
        update_best(result)

    # 3. Random Circle
    if time.time() < DEADLINE - 15:
        np.random.seed(123)
        random_centers = []
        random_angles = []
        for i in range(N):
            x = np.random.normal(0, 3.0)
            y = np.random.normal(0, 3.0)
            a = np.random.uniform(0, 360)
            random_centers.append((x, y))
            random_angles.append(a)
        
        result = _try_arrangement(random_centers, random_angles, DEADLINE)
        update_best(result)
    
    # 4. Simulated Annealing with Force-Directed Relaxation
    if time.time() < DEADLINE - 20:
        # Start from best found or lattice
        sa_centers = best_result[0] if best_result else centers
        sa_angles = best_result[1] if best_result else angles
        
        current_side = best_side if best_result else float('inf')
        
        # Coarse force-directed relaxation
        for temp in [1.0, 0.5, 0.25, 0.1, 0.05, 0.02]:
            if time.time() > DEADLINE - 10:
                break
            
            for step in range(100):
                if time.time() > DEADLINE - 10:
                    break
                
                # Pick a random hexagon and perturb it
                i = np.random.randint(N)
                dx = np.random.normal(0, temp)
                dy = np.random.normal(0, temp)
                da = np.random.normal(0, temp * 5)
                
                old_x, old_y = sa_centers[i]
                old_a = sa_angles[i]
                
                sa_centers[i] = (old_x + dx, old_y + dy)
                sa_angles[i] = old_a + da
                
                # Quick validity check
                valid = True
                for j in range(N):
                    if i != j and _hex_overlap(sa_centers[i], sa_angles[i], sa_centers[j], sa_angles[j]):
                        valid = False
                        break
                
                if valid:
                    # Compute new side
                    side, _, _ = _optimize_outer(sa_centers, sa_angles)
                    if np.isfinite(side) and side < current_side:
                        current_side = side
                        update_best((side, sa_centers, sa_angles, _optimize_outer(sa_centers, sa_angles)[1], _optimize_outer(sa_centers, sa_angles)[2]))
                    elif np.random.random() > np.exp(-(side - current_side) / temp):
                        # Accept worse solution with probability
                        pass
                    else:
                        # Reject
                        sa_centers[i] = (old_x, old_y)
                        sa_angles[i] = old_a
                else:
                    # Reject if invalid
                    sa_centers[i] = (old_x, old_y)
                    sa_angles[i] = old_a
    
    if best_result is None:
        fb_centers, fb_angles, fb_center, fb_side, fb_angle = _get_valid_fallback()
        best_result = (fb_centers, fb_angles, fb_center, fb_side, fb_angle)
        
    centers, inner_angles, outer_center, outer_side, outer_angle = best_result
    
    inner = [[float(centers[i][0]), float(centers[i][1]), float(inner_angles[i])] for i in range(N)]
    return inner, (float(outer_center[0]), float(outer_center[1])), float(outer_side), float(outer_angle)
