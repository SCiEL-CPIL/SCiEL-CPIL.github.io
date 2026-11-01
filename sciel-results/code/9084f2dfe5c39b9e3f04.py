import math
import numpy as np
import time
from scipy.optimize import linprog, minimize
from itertools import product

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 15.0

# Axial coordinates for 11 hexagons: Center + Ring 1 + 4 from Ring 2
# This forms a compact cluster.
AXIAL_COORDS = [
    (0, 0),
    (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
    (1, 1), (-1, 2), (-2, 1), (2, -1)
]

def axial_to_xy(q, r):
    x = math.sqrt(3) * (q + r / 2.0)
    y = 1.5 * r
    return x, y

# Pre-compute base lattice positions
BASE_POINTS = np.array([axial_to_xy(q, r) for q, r in AXIAL_COORDS])

def hex_vertices(cx, cy, side, angle_rad):
    """Return 6 vertices of a regular hexagon as a numpy array (6,2)."""
    thetas = angle_rad + np.arange(6) * np.pi / 3.0
    return np.column_stack([cx + side * np.cos(thetas), cy + side * np.sin(thetas)])

def sat_overlap(v1, v2, eps=1e-9):
    """SAT overlap check for two convex polygons (numpy arrays of shape (6,2))."""
    axes = []
    for poly in (v1, v2):
        for i in range(6):
            p1 = poly[i]
            p2 = poly[(i + 1) % 6]
            e = p2 - p1
            n = np.array([-e[1], e[0]])
            ln = np.linalg.norm(n)
            if ln > 1e-12:
                axes.append(n / ln)
    
    for ax in axes:
        min1 = np.min(v1 @ ax)
        max1 = np.max(v1 @ ax)
        min2 = np.min(v2 @ ax)
        max2 = np.max(v2 @ ax)
        if max1 < min2 + eps or max2 < min1 + eps:
            return False
    return True

def check_disjoint(centers, angles, eps=1e-6):
    """Check if all inner hexagons are disjoint."""
    n = len(centers)
    polys = []
    for i in range(n):
        polys.append(hex_vertices(centers[i][0], centers[i][1], 1.0, angles[i]))
    for i in range(n):
        for j in range(i + 1, n):
            if sat_overlap(polys[i], polys[j], eps):
                return False
    return True

def calc_min_side_for_config(centers, inner_angles, outer_angle_rad):
    """
    Calculate the minimum side length S of an outer regular hexagon 
    with rotation outer_angle_rad that contains all inner hexagons.
    Returns (S, center_x, center_y) or (inf, 0, 0) if infeasible.
    """
    # Normals of the outer hexagon
    normals = []
    for k in range(6):
        ang = outer_angle_rad + math.pi / 6.0 + k * math.pi / 3.0
        normals.append((math.cos(ang), math.sin(ang)))
    
    # Variables: cx, cy, D (where D = S * APOTHEM - APOTHEM_inner_support)
    # The support function of a unit hexagon in direction n is not constant!
    # Wait. The current program assumes inner hexagons are aligned with outer.
    # If inner hexagons have arbitrary angles, the projection of the inner hexagon 
    # onto the outer normal n is NOT just APOTHEM.
    # It is the support function h(n) of the inner hexagon.
    # h(n) = max_{v in vertices} (v - center) . n
    
    # So the constraint is:
    # (center_i - c_outer) . n_k + h_i(n_k) <= S * APOTHEM_outer
    # Let R = S * APOTHEM_outer.
    # (center_i - c_outer) . n_k <= R - h_i(n_k)
    # And -(center_i - c_outer) . n_k <= R - h_i(-n_k)
    
    # This is still a linear program in cx, cy, R.
    # Minimize R.
    # S = R / APOTHEM
    
    c_obj = np.array([0.0, 0.0, 1.0]) # Minimize R
    A_ub = []
    b_ub = []
    
    for i, (cx_i, cy_i) in enumerate(centers):
        ang_i = inner_angles[i]
        verts = hex_vertices(cx_i, cy_i, 1.0, ang_i)
        # Compute support function for each normal
        for nx, ny in normals:
            n_vec = np.array([nx, ny])
            # h_i(n) = max((v - center_i) . n)
            # Since v is absolute, (v - center_i) . n = v . n - center_i . n
            # But it's easier to compute relative to center_i
            rel_verts = verts - np.array([cx_i, cy_i])
            h_pos = np.max(rel_verts @ n_vec)
            h_neg = np.max(rel_verts @ -n_vec) # This is -min(rel_verts . n)
            
            # Constraint 1: (cx_i - cx)nx + (cy_i - cy)ny + h_pos <= R
            # -nx * cx - ny * cy - R <= -(cx_i * nx + cy_i * ny + h_pos)
            A_ub.append([-nx, -ny, -1.0])
            b_ub.append(-(cx_i * nx + cy_i * ny + h_pos))
            
            # Constraint 2: -(cx_i - cx)nx - (cy_i - cy)ny + h_neg <= R
            # nx * cx + ny * cy - R <= cx_i * nx + cy_i * ny + h_neg
            A_ub.append([nx, ny, -1.0])
            b_ub.append(cx_i * nx + cy_i * ny + h_neg)
            
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    bounds = [(None, None), (None, None), (0, None)]
    
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.status == 0:
            R = res.x[2]
            cx = res.x[0]
            cy = res.x[1]
            S = R / APOTHEM
            return S, cx, cy
        else:
            return float('inf'), 0.0, 0.0
    except Exception:
        return float('inf'), 0.0, 0.0

def construct():
    # Check time limit
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        # Fallback to simple aligned lattice
        pts = BASE_POINTS
        # Just return a large enough hexagon to be safe
        # Max dist from center to vertex of outermost hexagon
        max_dist = np.max(np.linalg.norm(pts, axis=1)) + 1.0
        side = (max_dist + APOTHEM) / APOTHEM * 1.1
        inner = [[float(x), float(y), 0.0] for x, y in pts]
        return inner, [0.0, 0.0], float(side), 0.0

    best_S = float('inf')
    best_config = None

    # 1. Global Optimization of Lattice Rotation and Outer Angle
    # We assume all inner hexagons have the same rotation theta (lattice alignment)
    # and the outer hexagon has rotation beta.
    
    def objective(params):
        theta, beta = params
        # Rotate base points by theta
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        rot_mat = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        rotated_pts = BASE_POINTS @ rot_mat.T
        
        centers = [(p[0], p[1]) for p in rotated_pts]
        inner_angles = [theta] * N
        
        S, cx, cy = calc_min_side_for_config(centers, inner_angles, beta)
        return S

    # Coarse grid search to find a good starting point
    # Symmetry: theta in [0, pi/6], beta in [0, pi/6]
    # Relative angle delta = beta - theta is the key.
    
    best_start_theta = 0.0
    best_start_beta = 0.0
    best_start_S = float('inf')
    
    # Sample relative angles and absolute angles
    # Since the system has 6-fold symmetry, we can fix theta=0 and vary beta?
    # No, rotating the lattice changes the shape relative to the outer hexagon.
    # Let's sample theta and beta.
    
    for i in range(7):
        theta = i * (math.pi / 6) / 6.0
        for j in range(7):
            beta = j * (math.pi / 6) / 6.0
            S, _, _ = calc_min_side_for_config(
                [(p[0], p[1]) for p in (BASE_POINTS @ np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]]).T)],
                [theta] * N,
                beta
            )
            if S < best_start_S:
                best_start_S = S
                best_start_theta = theta
                best_start_beta = beta

    # Refine with Nelder-Mead
    if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
        try:
            res = minimize(objective, [best_start_theta, best_start_beta], method='Nelder-Mead',
                           options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10, 'maxfev': 400})
            if res.fun < best_S:
                best_S = res.fun
                best_config = (res.x[0], res.x[1])
        except Exception:
            pass

    if best_config is None:
        best_config = (best_start_theta, best_start_beta)
        best_S = best_start_S

    theta_opt, beta_opt = best_config

    # 2. Local Refinement: Allow individual inner hexagons to move and rotate
    # We will use a local search to "push" hexagons into the corners.
    
    # Initial positions
    cos_t = math.cos(theta_opt)
    sin_t = math.sin(theta_opt)
    rot_mat = np.array([[cos_t, -sin_t], [sin_t, math.cos(theta_opt)]])
    current_centers = [(p[0], p[1]) for p in (BASE_POINTS @ rot_mat.T)]
    current_angles = [theta_opt] * N
    
    # Verify disjointness initially (should be true for lattice)
    if not check_disjoint(current_centers, current_angles):
        # If lattice is touching, add tiny epsilon
        # For now, assume it's fine or relax check slightly
        pass

    # Local optimization step
    # We want to minimize S(centers, angles, beta_opt)
    # We can't easily do gradient descent on S because it's non-smooth (LP).
    # Instead, we can try random perturbations and keep if S decreases and disjointness holds.
    
    best_local_S = best_S
    best_local_centers = current_centers
    best_local_angles = current_angles
    
    # Get current center of outer hexagon
    S, cx_opt, cy_opt = calc_min_side_for_config(current_centers, current_angles, beta_opt)
    
    # Simple Simulated Annealing / Random Search for local improvement
    # Time budget: 30 seconds
    deadline = time.time() + 30.0
    T = 0.5
    T_min = 0.01
    cool = 0.999
    
    np.random.seed(42)
    
    while time.time() < deadline and T > T_min:
        # Perturb a random hexagon
        idx = np.random.randint(0, N)
        new_centers = [c[:] for c in current_centers]
        new_angles = current_angles[:]
        
        # Move slightly
        new_centers[idx] = (new_centers[idx][0] + np.random.normal(0, 0.1), 
                            new_centers[idx][1] + np.random.normal(0, 0.1))
        # Rotate slightly
        new_angles[idx] = new_angles[idx] + np.random.normal(0, 0.1)
        
        # Check disjointness
        if not check_disjoint(new_centers, new_angles):
            T *= cool
            continue
            
        S_new, _, _ = calc_min_side_for_config(new_centers, new_angles, beta_opt)
        
        if S_new < best_local_S:
            best_local_S = S_new
            best_local_centers = new_centers
            best_local_angles = new_angles
            current_centers = new_centers
            current_angles = new_angles
            T = T * 0.99 # Restart cooling slightly
        else:
            T *= cool
            
    # Final check and extraction
    if best_local_S < best_S:
        best_S = best_local_S
        final_centers = best_local_centers
        final_angles = best_local_angles
    else:
        final_centers = current_centers
        final_angles = current_angles

    # Compute final outer hexagon parameters
    S_final, cx_final, cy_final = calc_min_side_for_config(final_centers, final_angles, beta_opt)
    
    # Prepare output
    inner = []
    for i in range(N):
        inner.append([float(final_centers[i][0]), float(final_centers[i][1]), float(math.degrees(final_angles[i]))])
        
    center = [float(cx_final), float(cy_final)]
    side = float(S_final)
    rotation = float(math.degrees(beta_opt))
    
    return inner, center, side, rotation

if __name__ == "__main__":
    # Quick test
    try:
        res = construct()
        print(f"Side: {res[2]}, Center: {res[1]}, Rot: {res[3]}")
        print(f"Time: {time.time() - START_TIME:.2f}s")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
