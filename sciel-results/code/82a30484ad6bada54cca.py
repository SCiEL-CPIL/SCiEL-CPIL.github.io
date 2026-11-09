# EVOLVE-BLOCK-START

import numpy as np
from scipy.optimize import differential_evolution, minimize
import time
import math

N = 12

def _hex_verts(cx, cy, s, ang_deg):
    """Return 6 vertices of a regular hexagon centered at (cx, cy) with side s and rotation ang_deg."""
    ang = math.radians(ang_deg)
    verts = []
    for k in range(6):
        theta = ang + k * math.pi / 3.0
        x = cx + s * math.cos(theta)
        y = cy + s * math.sin(theta)
        verts.append((x, y))
    return np.array(verts)

def _point_in_poly(pt, poly):
    """Ray casting algorithm for point in polygon."""
    x, y = pt
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside

def _segments_intersect(p1, p2, p3, p4):
    """Check if segment p1-p2 intersects segment p3-p4."""
    def ccw(A, B, C):
        return (C[1]-A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])
    return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)

def _check_overlap(v1, v2):
    """
    Check if two hexagons (defined by vertices) overlap.
    Returns True if overlap, False otherwise.
    Also returns a penalty value (0 if no overlap, >0 if overlap).
    """
    # Check vertices of v1 in v2
    for pt in v1:
        if _point_in_poly(pt, v2):
            return True, 10.0
    # Check vertices of v2 in v1
    for pt in v2:
        if _point_in_poly(pt, v1):
            return True, 10.0
    # Check edge intersections
    for i in range(6):
        for j in range(6):
            if _segments_intersect(v1[i], v1[(i+1)%6], v2[j], v2[(j+1)%6]):
                return True, 10.0
    return False, 0.0

def _objective(params, deadline):
    """
    Objective function.
    params: (N*3,) -> x, y, angle
    """
    if time.time() > deadline:
        return 1e10
    
    p = params.reshape(N, 3)
    xs = p[:, 0]
    ys = p[:, 1]
    angles = p[:, 2]
    
    # Calculate S required
    normals = np.array([
        [0.0, 1.0],
        [math.sqrt(3)/2, 0.5],
        [math.sqrt(3)/2, -0.5],
        [0.0, -1.0],
        [-math.sqrt(3)/2, -0.5],
        [-math.sqrt(3)/2, 0.5]
    ])
    apothem_unit = math.sqrt(3) / 2.0
    
    max_proj = 0.0
    verts_list = []
    for i in range(N):
        verts = _hex_verts(xs[i], ys[i], 1.0, angles[i])
        verts_list.append(verts)
        projs = verts @ normals.T
        local_max_S = np.max(projs) / apothem_unit
        if local_max_S > max_proj:
            max_proj = local_max_S
            
    S_required = max(0.0, max_proj)
    
    # Check overlaps
    penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            overlap, pen = _check_overlap(verts_list[i], verts_list[j])
            if overlap:
                penalty += pen
                # Early exit if penalty is already huge? No, need to know total for DE?
                # Actually, if penalty is > S, it's bad.
                
    return S_required + penalty

def _get_S_and_verts(params):
    """Helper to get S and verts for a given param set."""
    p = params.reshape(N, 3)
    xs = p[:, 0]
    ys = p[:, 1]
    angles = p[:, 2]
    
    normals = np.array([
        [0.0, 1.0],
        [math.sqrt(3)/2, 0.5],
        [math.sqrt(3)/2, -0.5],
        [0.0, -1.0],
        [-math.sqrt(3)/2, -0.5],
        [-math.sqrt(3)/2, 0.5]
    ])
    apothem_unit = math.sqrt(3) / 2.0
    
    max_proj = 0.0
    verts_list = []
    for i in range(N):
        verts = _hex_verts(xs[i], ys[i], 1.0, angles[i])
        verts_list.append(verts)
        projs = verts @ normals.T
        local_max_S = np.max(projs) / apothem_unit
        if local_max_S > max_proj:
            max_proj = local_max_S
            
    S_required = max(0.0, max_proj)
    return S_required, verts_list

def _is_valid(verts_list):
    """Check if the configuration is valid (no overlaps)."""
    for i in range(N):
        for j in range(i+1, N):
            overlap, _ = _check_overlap(verts_list[i], verts_list[j])
            if overlap:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Initial guess: Honeycomb
    eps = 0.01
    v1 = np.array([1.5 * (1 + eps), math.sqrt(3)/2 * (1 + eps)])
    v2 = np.array([0.0, math.sqrt(3) * (1 + eps)])
    
    init_params = []
    for i in range(3):
        for j in range(4):
            pos = i * v1 + j * v2
            init_params.append([pos[0], pos[1], 0.0])
            
    init_params = np.array(init_params)
    
    # Center
    cx = init_params[:, 0].mean()
    cy = init_params[:, 1].mean()
    init_params[:, 0] -= cx
    init_params[:, 1] -= cy
    
    bounds = [(-10, 10), (-10, 10), (-180, 180)] * N
    
    best_params = init_params
    best_score = _objective(best_params, deadline)
    
    # Phase 1: Differential Evolution
    # Use a smaller population to be faster
    try:
        de_result = differential_evolution(
            _objective,
            bounds,
            args=(deadline,),
            maxiter=50,
            popsize=10,
            tol=1e-4,
            seed=42,
            init='latinhypercube',
            polish=False # We will polish manually
        )
        if de_result.fun < best_score:
            best_score = de_result.fun
            best_params = de_result.x
    except Exception as e:
        print(f"DE failed: {e}")
        
    # Phase 2: Local Polish with Nelder-Mead
    # Only if we have time and a valid-ish solution
    if time.time() < deadline - 20:
        # Define a local objective that heavily penalizes overlap
        def local_obj(params):
            if time.time() > deadline:
                return 1e10
            S, verts = _get_S_and_verts(params)
            penalty = 0.0
            for i in range(N):
                for j in range(i+1, N):
                    overlap, pen = _check_overlap(verts[i], verts[j])
                    if overlap:
                        penalty += pen
            return S + penalty

        try:
            nm_result = minimize(
                local_obj,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-5}
            )
            if nm_result.fun < best_score:
                best_score = nm_result.fun
                best_params = nm_result.x
        except Exception as e:
            print(f"NM failed: {e}")

    # Final Verification and Formatting
    # Ensure the returned solution is valid
    S_final, verts_final = _get_S_and_verts(best_params)
    if not _is_valid(verts_final):
        # Fallback to initial grid if optimization failed to produce valid result
        S_final, verts_final = _get_S_and_verts(init_params)
        best_params = init_params
        # Re-verify initial is valid (it should be)
        if not _is_valid(verts_final):
             # This should never happen with the eps padding
             pass

    inner = []
    for i in range(N):
        inner.append([best_params[i, 0], best_params[i, 1], best_params[i, 2]])
        
    return inner, (0.0, 0.0), S_final, 0.0

# EVOLVE-BLOCK-END
