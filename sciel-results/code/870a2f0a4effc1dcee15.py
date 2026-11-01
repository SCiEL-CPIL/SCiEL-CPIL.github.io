"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
SQRT3 = math.sqrt(3)

# EVOLVE-BLOCK-START

def hex_vertices(cx, cy, side, angle_rad):
    """Return 6 vertices of a regular hexagon as a numpy array (6,2)."""
    thetas = angle_rad + np.arange(6) * np.pi / 3.0
    return np.column_stack([cx + side * np.cos(thetas), cy + side * np.sin(thetas)])

def compute_required_side_fast(inner, outer_center, outer_angle_deg):
    """
    Compute minimum outer side length to contain all inner hexagons.
    inner: list of (x, y, angle_degrees)
    outer_center: (x, y)
    outer_angle_deg: rotation of outer hexagon
    """
    if not inner:
        return 0.0
    ox, oy = outer_center
    # Normals to the sides of the outer hexagon in its local frame
    # The outer hexagon has sides perpendicular to angles 30, 90, 150, ...
    # In the local frame of the outer hexagon (rotated by -outer_angle), 
    # the supporting lines are at distance 'side' from center.
    # Actually, the distance from center to side in a regular hexagon of side S is S * sqrt(3)/2.
    # So if max projection of any vertex onto the normal is P, then S * sqrt(3)/2 >= P => S >= 2P/sqrt(3).
    
    normals = np.array([[math.cos(math.radians(30 + i * 60)),
                         math.sin(math.radians(30 + i * 60))] for i in range(6)])
    
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))
    # Rotation matrix to transform global coords to outer local coords
    # v_local = R * (v_global - center)
    R = np.array([[cos_a, sin_a], [-sin_a, cos_a]])
    
    max_proj = -np.inf
    
    for (ix, iy, iang) in inner:
        verts = hex_vertices(ix, iy, 1.0, math.radians(iang))
        # Translate to center
        local_pts = (verts - np.array([ox, oy])) @ R.T
        # Project onto normals
        projs = local_pts @ normals.T
        cur_max = projs.max()
        if cur_max > max_proj:
            max_proj = cur_max
            
    if max_proj < 0:
        max_proj = 0.0
        
    return 2.0 * max_proj / SQRT3

def compute_required_side_smooth(inner, outer_center, outer_angle_deg, beta=50.0):
    """
    Smooth surrogate for compute_required_side using log-sum-exp.
    """
    if not inner:
        return 0.0
    ox, oy = outer_center
    normals = np.array([[math.cos(math.radians(30 + i * 60)),
                         math.sin(math.radians(30 + i * 60))] for i in range(6)])
    
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))
    R = np.array([[cos_a, sin_a], [-sin_a, cos_a]])
    
    all_projs = []
    for (ix, iy, iang) in inner:
        verts = hex_vertices(ix, iy, 1.0, math.radians(iang))
        local_pts = (verts - np.array([ox, oy])) @ R.T
        projs = local_pts @ normals.T
        all_projs.append(projs)
        
    P = np.concatenate([p.ravel() for p in all_projs])
    if P.size == 0:
        return 0.0
        
    m = P.max()
    if m < -1e6:
        return 0.0
    # log-sum-exp trick for stability
    smooth_max = m + math.log(np.sum(np.exp(beta * (P - m)))) / beta
    if smooth_max < 0:
        smooth_max = 0.0
        
    return 2.0 * smooth_max / SQRT3

def sat_overlap(v1, v2, eps=1e-9):
    """SAT overlap check for two convex polygons."""
    axes = []
    for poly in (v1, v2):
        for i in range(len(poly)):
            p1 = poly[i]
            p2 = poly[(i + 1) % len(poly)]
            e = (p2[0] - p1[0], p2[1] - p1[1])
            n = (-e[1], e[0])
            ln = math.hypot(n[0], n[1])
            if ln > 1e-12:
                axes.append((n[0] / ln, n[1] / ln))
                
    for ax in axes:
        min1 = min(p[0] * ax[0] + p[1] * ax[1] for p in v1)
        max1 = max(p[0] * ax[0] + p[1] * ax[1] for p in v1)
        min2 = min(p[0] * ax[0] + p[1] * ax[1] for p in v2)
        max2 = max(p[0] * ax[0] + p[1] * ax[1] for p in v2)
        if max1 < min2 + eps or max2 < min1 + eps:
            return False
    return True

def check_disjoint(inner, eps=1e-6):
    """Pairwise SAT check that all inner hexagons are disjoint."""
    n = len(inner)
    polys = []
    for (x, y, a) in inner:
        polys.append(hex_vertices(x, y, 1.0, math.radians(a)))
    for i in range(n):
        for j in range(i + 1, n):
            if sat_overlap(polys[i], polys[j], eps):
                return False
    return True

def overlap_penalty(inner, min_dist=SQRT3 - 0.001):
    """
    Penalty for overlaps. 
    For two regular hexagons of side 1, the minimum distance between centers 
    to be disjoint is sqrt(3) if aligned, or 2.0 if rotated 30 deg. 
    Using sqrt(3) is a safe lower bound for axis-aligned, but we allow rotation.
    A robust penalty is based on the actual minimum distance between boundaries, 
    but that's expensive. We use center distance with a conservative threshold.
    If centers are closer than sqrt(3), they *might* overlap. 
    Actually, max distance for guaranteed overlap is 2.0 (touching at vertices of opposite hexes).
    Min distance for guaranteed separation is sqrt(3) (touching at midpoints of parallel sides).
    If we rotate, the min separation distance can be smaller? No, the circumradius is 1. 
    Two circles of radius 1 need dist > 2. Hexagons are smaller than circles.
    The width of a hexagon is 2. The "radius" to side is sqrt(3)/2.
    If two hexagons are aligned, they touch at dist sqrt(3).
    If one is rotated 30 deg, they touch at dist 1.0 (vertex to vertex? No, vertex to side?).
    Actually, the minimum distance between centers for two unit hexagons to be disjoint 
    is 1.0 (vertex of one touches vertex of other? No, vertex of one touches side of other 
    if rotated appropriately? No, the max width is 2, so min dist is 1 if they point at each other).
    Wait, circumradius is 1. So two hexagons with centers dist < 2 *could* overlap.
    If dist < 1, they definitely overlap (centers inside each other's circumcircles).
    Let's use a penalty that activates when dist < 1.5 to be safe, with a smooth ramp.
    """
    pen = 0.0
    n = len(inner)
    for i in range(n):
        xi, yi, _ = inner[i]
        for j in range(i + 1, n):
            xj, yj, _ = inner[j]
            d = math.hypot(xi - xj, yi - yj)
            # If d < 1.0, they overlap significantly.
            # If d < sqrt(3) ~ 1.732, they might overlap depending on angle.
            # We penalize if d < 1.2 to push them apart strongly.
            if d < 1.2:
                pen += (1.2 - d) ** 2
    return pen

def generate_seed_lattice():
    """
    Generate a standard hexagonal lattice seed.
    For N=11, we take the center and the first ring (6) and 4 from the second ring.
    """
    coords = [(0, 0)]
    # Ring 1
    for i in range(6):
        q, r = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)][i]
        coords.append((q, r))
    # Ring 2 - pick 4 to make 11
    # We want them to be close to the center to minimize bounding box.
    # (2,0), (1,1), (0,2), (-1,2) are good candidates.
    ring2 = [(2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1), (-2, 0), (-1, -1), (0, -2), (1, -2), (2, -1)]
    for q, r in ring2:
        if len(coords) < N:
            coords.append((q, r))
            
    centers = []
    for q, r in coords:
        x = 1.5 * q
        y = SQRT3 * (r + q / 2.0)
        centers.append((x, y))
        
    return [[x, y, 0.0] for x, y in centers]

def generate_seed_compact():
    """
    Generate a compact seed where the outer 4 are pushed inwards slightly 
    to encourage rotation and tighter packing.
    """
    seed = generate_seed_lattice()
    # Apply a slight random perturbation to break symmetry
    np.random.seed(42)
    for i in range(N):
        seed[i][0] += np.random.uniform(-0.1, 0.1)
        seed[i][1] += np.random.uniform(-0.1, 0.1)
        seed[i][2] = np.random.uniform(0, 60)
    return seed

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # 1. Generate multiple seeds
    seeds = [
        generate_seed_lattice(),
        generate_seed_compact()
    ]
    
    # Add a few more randomized seeds
    for k in range(3):
        np.random.seed(100 + k)
        s = generate_seed_lattice()
        for i in range(N):
            s[i][0] += np.random.uniform(-0.5, 0.5)
            s[i][1] += np.random.uniform(-0.5, 0.5)
            s[i][2] = np.random.uniform(0, 60)
        seeds.append(s)
        
    best_inner = None
    best_side = float('inf')
    best_outer_center = (0, 0)
    best_outer_angle = 0.0
    
    for seed in seeds:
        if time.time() > deadline - 20:
            break
            
        # Phase 1: Optimize outer container for this fixed inner layout
        # This helps find the best orientation for the outer hexagon
        
        def obj_outer_smooth(p):
            return compute_required_side_smooth(seed, (p[0], p[1]), p[2], beta=30.0)
            
        def obj_outer_exact(p):
            return compute_required_side_fast(seed, (p[0], p[1]), p[2])
            
        # Initial guess for outer center: centroid of inner
        cx_avg = sum(p[0] for p in seed) / N
        cy_avg = sum(p[1] for p in seed) / N
        
        best_outer_val = float('inf')
        best_outer_params = [cx_avg, cy_avg, 0.0]
        
        # Try a few initial angles
        for init_angle in [0, 15, 30, 45, 60]:
            if time.time() > deadline - 15:
                break
            p0 = [cx_avg, cy_avg, float(init_angle)]
            try:
                # Smooth phase
                r1 = minimize(obj_outer_smooth, p0, method='Nelder-Mead',
                              options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-8, 'maxfev': 200})
                # Exact phase
                r2 = minimize(obj_outer_exact, r1.x, method='Nelder-Mead',
                              options={'maxiter': 300, 'xatol': 1e-9, 'fatol': 1e-11, 'maxfev': 500})
                if r2.fun < best_outer_val:
                    best_outer_val = r2.fun
                    best_outer_params = r2.x
            except Exception:
                pass
        
        current_outer_center = (best_outer_params[0], best_outer_params[1])
        current_outer_angle = best_outer_params[2]
        current_side = best_outer_val
        
        # Phase 2: Joint Optimization
        # We optimize inner positions/angles and outer container simultaneously.
        
        def obj_joint_smooth(params, beta=40.0):
            il = []
            for i in range(N):
                il.append([params[i * 3], params[i * 3 + 1], params[i * 3 + 2]])
            ocx, ocy, oa = params[N * 3], params[N * 3 + 1], params[N * 3 + 2]
            side = compute_required_side_smooth(il, (ocx, ocy), oa, beta=beta)
            pen = overlap_penalty(il)
            return side + 100.0 * pen
            
        def obj_joint_exact(params):
            il = []
            for i in range(N):
                il.append([params[i * 3], params[i * 3 + 1], params[i * 3 + 2]])
            ocx, ocy, oa = params[N * 3], params[N * 3 + 1], params[N * 3 + 2]
            side = compute_required_side_fast(il, (ocx, ocy), oa)
            pen = overlap_penalty(il)
            return side + 1000.0 * pen
            
        p0 = np.zeros(N * 3 + 3)
        for i, (x, y, a) in enumerate(seed):
            p0[i * 3] = x
            p0[i * 3 + 1] = y
            p0[i * 3 + 2] = a
        p0[N * 3] = current_outer_center[0]
        p0[N * 3 + 1] = current_outer_center[1]
        p0[N * 3 + 2] = current_outer_angle
        
        # Use Differential Evolution for global search in the joint space
        # DE is slow, so we limit iterations and use a tight time check
        bounds = []
        for i in range(N):
            bounds.append((-5, 5))
            bounds.append((-5, 5))
            bounds.append((0, 60))
        bounds.append((cx_avg - 2, cx_avg + 2))
        bounds.append((cy_avg - 2, cy_avg + 2))
        bounds.append((0, 60))
        
        try:
            # Small popsize and maxiter to stay within time
            result_de = minimize(obj_joint_smooth, p0, method='differential_evolution',
                                 bounds=bounds, maxiter=20, popsize=10, tol=1e-5,
                                 seed=42, init='latinhypercube', mutation=(0.5, 1.0))
            
            # Polish the DE result with Nelder-Mead on the exact objective
            if result_de.success or result_de.fun < current_side:
                p_polish = result_de.x
                # Nelder-Mead polish
                r_polish = minimize(obj_joint_exact, p_polish, method='Nelder-Mead',
                                    options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'maxfev': 1000})
                
                if r_polish.fun < current_side:
                    current_side = r_polish.fun
                    p_best = r_polish.x
                else:
                    p_best = p_polish
            else:
                p_best = p0
        except Exception:
            p_best = p0
            # If DE fails, do a simple Nelder-Mead from start
            try:
                r_nm = minimize(obj_joint_exact, p0, method='Nelder-Mead',
                                options={'maxiter': 300, 'xatol': 1e-7, 'fatol': 1e-9, 'maxfev': 500})
                if r_nm.fun < current_side:
                    current_side = r_nm.fun
                    p_best = r_nm.x
            except Exception:
                pass
        
        # Extract result
        inner_res = []
        for i in range(N):
            inner_res.append([p_best[i * 3], p_best[i * 3 + 1], p_best[i * 3 + 2]])
        outer_c = (p_best[N * 3], p_best[N * 3 + 1])
        outer_a = p_best[N * 3 + 2]
        
        # Validate disjointness
        if check_disjoint(inner_res, eps=1e-5):
            # Final exact side calculation
            exact_side = compute_required_side_fast(inner_res, outer_c, outer_a)
            if exact_side < best_side:
                best_side = exact_side
                best_inner = inner_res
                best_outer_center = outer_c
                best_outer_angle = outer_a
        else:
            # If invalid, try to repair by projecting overlaps
            # For now, skip if invalid, as DE usually handles penalty well
            # But we can try a local repair: push overlapping hexes apart
            pass

    # If we found a valid solution, do a final polish on the outer container
    if best_inner is not None:
        def obj_final_outer(p):
            return compute_required_side_fast(best_inner, (p[0], p[1]), p[2])
            
        p0 = [best_outer_center[0], best_outer_center[1], best_outer_angle]
        try:
            r_final = minimize(obj_final_outer, p0, method='Nelder-Mead',
                               options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12, 'maxfev': 2000})
            if r_final.fun < best_side:
                best_side = r_final.fun
                best_outer_center = (r_final.x[0], r_final.x[1])
                best_outer_angle = r_final.x[2]
        except Exception:
            pass

    # Fallback if no valid solution found
    if best_inner is None:
        # Return the initial lattice as a valid fallback (it is disjoint)
        best_inner = generate_seed_lattice()
        cx_avg = sum(p[0] for p in best_inner) / N
        cy_avg = sum(p[1] for p in best_inner) / N
        best_outer_center = (cx_avg, cy_avg)
        best_outer_angle = 0.0
        best_side = compute_required_side_fast(best_inner, best_outer_center, best_outer_angle)

    return best_inner, best_outer_center, best_side, best_outer_angle

# EVOLVE-BLOCK-END
