import math
import time
import random
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 11
START = time.time()
DEADLINE = 148.0  # Safety margin

def _hex_vertices(cx, cy, s, rot_deg):
    """
    Generate vertices of a regular hexagon with center (cx, cy), side length s, 
    and rotation rot_deg (degrees).
    Vertices are in counter-clockwise order.
    """
    a0 = math.radians(rot_deg)
    pts = []
    for k in range(6):
        a = a0 + math.radians(60 * k)
        pts.append((cx + s * math.cos(a), cy + s * math.sin(a)))
    return pts

def _inner_hex(cx, cy, ang_deg):
    """
    Generate vertices of a unit regular hexagon with center (cx, cy) 
    and rotation ang_deg (degrees).
    """
    return _hex_vertices(cx, cy, 1.0, ang_deg)

def _min_outer_side(inner_params, outer_c=(0.0, 0.0), rot=0.0):
    """
    Calculate the minimum side length of the outer hexagon required to contain 
    all inner hexagons defined by inner_params, given outer center and rotation.
    """
    theta_rad = math.radians(rot)
    # Normals to the sides of the outer hexagon
    normals = []
    for k in range(6):
        alpha = theta_rad + math.radians(30 + k * 60)
        normals.append((math.cos(alpha), math.sin(alpha)))
    
    ox, oy = outer_c
    max_proj = 0.0
    
    for k in range(N):
        x, y, a = inner_params[k]
        pts = _inner_hex(x, y, a)
        for (px, py) in pts:
            for (nx, ny) in normals:
                proj = (px - ox) * nx + (py - oy) * ny
                if proj > max_proj:
                    max_proj = proj
    
    # For a regular hexagon with side length s, the distance from center to side is s * sqrt(3) / 2.
    # So max_proj = s * sqrt(3) / 2  =>  s = 2 * max_proj / sqrt(3)
    return 2.0 * max_proj / math.sqrt(3.0)

def _get_penetration(A, B):
    """
    Calculate the minimum translation distance to separate two convex polygons A and B.
    Returns 0 if they are disjoint or touching.
    Uses Separating Axis Theorem (SAT).
    """
    min_overlap = float('inf')
    
    # Collect all edge normals from both polygons
    edges = []
    for poly in (A, B):
        n = len(poly)
        for i in range(n):
            ax = poly[(i+1) % n][0] - poly[i][0]
            ay = poly[(i+1) % n][1] - poly[i][1]
            # Normal vector
            nx, ny = -ay, ax
            # Normalize
            norm = math.sqrt(nx*nx + ny*ny)
            if norm < 1e-9:
                continue
            nx /= norm
            ny /= norm
            edges.append((nx, ny))
            
    # Check projection on each axis
    for nx, ny in edges:
        minA = min(p[0]*nx + p[1]*ny for p in A)
        maxA = max(p[0]*nx + p[1]*ny for p in A)
        minB = min(p[0]*nx + p[1]*ny for p in B)
        maxB = max(p[0]*nx + p[1]*ny for p in B)
        
        # If separated, return 0
        if maxA < minB or maxB < minA:
            return 0.0
            
        # Calculate overlap on this axis
        overlap = min(maxA, maxB) - max(minA, minB)
        if overlap < min_overlap:
            min_overlap = overlap
            
    if min_overlap == float('inf'):
        # Should not happen for convex polygons
        return 0.0
        
    return min_overlap

def _get_boundary_violation(inner_params, outer_s, outer_c=(0.0, 0.0), rot=0.0):
    """
    Calculate the total distance by which inner hexagon vertices violate the outer hexagon boundary.
    """
    theta_rad = math.radians(rot)
    normals = []
    for k in range(6):
        alpha = theta_rad + math.radians(30 + k * 60)
        normals.append((math.cos(alpha), math.sin(alpha)))
    
    ox, oy = outer_c
    limit = outer_s * math.sqrt(3.0) / 2.0
    
    total_violation = 0.0
    for k in range(N):
        x, y, a = inner_params[k]
        pts = _inner_hex(x, y, a)
        for (px, py) in pts:
            for (nx, ny) in normals:
                proj = (px - ox) * nx + (py - oy) * ny
                if proj > limit:
                    total_violation += (proj - limit)
    
    return total_violation

def _objective(params):
    """
    Objective function for optimization.
    params: array of length 3*N, containing [x1, y1, a1, ..., x11, y11, a11]
    """
    pts = params.reshape(N, 3)
    
    # 1. Calculate the outer side length required to contain all inner hexagons
    # We fix outer center at (0,0) and rotation 0 for simplicity, as the optimal 
    # configuration can be aligned this way.
    outer_s = _min_outer_side(pts)
    
    # 2. Calculate overlap penalty
    overlap_penalty = 0.0
    polys = [_inner_hex(pts[i,0], pts[i,1], pts[i,2]) for i in range(N)]
    for i in range(N):
        for j in range(i+1, N):
            pen = _get_penetration(polys[i], polys[j])
            if pen > 0:
                overlap_penalty += pen * pen
    
    # 3. Calculate boundary violation penalty
    # Note: outer_s is calculated to fit the current points, so boundary violation 
    # should be 0 if we use the calculated outer_s. 
    # However, if we penalize overlaps, we might want to push points inward.
    # Actually, the outer_s is determined by the max projection. 
    # If points overlap, they might be pushed "out" to reduce overlap, increasing outer_s.
    # The objective is to minimize outer_s + penalty.
    # So we don't need a separate boundary penalty if outer_s adapts.
    # But wait, if points are outside a *fixed* outer hexagon, we'd need it.
    # Here, outer_s is a variable dependent on the points.
    # So the cost is just outer_s + overlap_penalty.
    
    # To make the penalty stronger, we scale it
    penalty_weight = 100.0
    total_cost = outer_s + penalty_weight * overlap_penalty
    
    return total_cost

def _valid(inner_params, outer_c=(0.0, 0.0), rot=0.0):
    """
    Strict validity check: no overlaps, all inside.
    """
    outer_s = _min_outer_side(inner_params, outer_c, rot)
    
    # Check overlaps
    polys = [_inner_hex(inner_params[i][0], inner_params[i][1], inner_params[i][2]) for i in range(N)]
    for i in range(N):
        for j in range(i+1, N):
            if _get_penetration(polys[i], polys[j]) > 1e-7:
                return False
    
    # Check boundary (should be satisfied by construction of outer_s, but verify)
    # The _min_outer_side ensures all points are within the limit defined by outer_s.
    # So if we use that outer_s, they are inside.
    
    return True

def _hexagonal_seed():
    """
    Create a hexagonal lattice arrangement of 11 unit hexagons.
    Center at (0,0).
    """
    inner = []
    inner.append([0.0, 0.0, 0.0])
    d = math.sqrt(3.0) * 1.001 # Slight gap
    for k in range(6):
        a = math.radians(30 + k * 60)
        inner.append([d * math.cos(a), d * math.sin(a), 0.0])
    d2 = 2 * d
    for k in range(4):
        a = math.radians(30 + k * 60)
        inner.append([d2 * math.cos(a), d2 * math.sin(a), 0.0])
    return inner

def construct():
    best_params = None
    best_score = float('inf')
    budget = START + DEADLINE - 5.0
    
    # Initial seed
    seed_inner = _hexagonal_seed()
    seed_params = np.array([p for inner in seed_inner for p in inner])
    
    # Evaluate seed
    seed_score = _objective(seed_params)
    if _valid(seed_inner):
        best_params = seed_params
        best_score = seed_score
    
    # Define bounds for DE
    # Coordinates: roughly within [-4, 4]
    # Angles: [0, 60]
    bounds = []
    for _ in range(N):
        bounds.append((-4, 4))
        bounds.append((-4, 4))
        bounds.append((0, 60))
    
    # Run Differential Evolution
    if time.time() < budget - 30:
        try:
            res = differential_evolution(
                _objective,
                bounds,
                seed=42,
                maxiter=100,
                popsize=20,
                tol=1e-5,
                atol=1e-5,
                polish=False,
                init='sobol'
            )
            if res.fun < best_score:
                best_params = res.x
                best_score = res.fun
        except Exception as e:
            pass
    
    # Local Refinement
    if best_params is not None and time.time() < budget - 10:
        # Try Nelder-Mead for refinement
        try:
            res_nm = minimize(
                _objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True}
            )
            if res_nm.fun < best_score:
                best_params = res_nm.x
                best_score = res_nm.fun
        except Exception:
            pass
        
        # Try L-BFGS-B (requires gradient, but we can use numerical)
        if time.time() < budget - 5:
            try:
                res_lbfgs = minimize(
                    _objective,
                    best_params,
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-8}
                )
                if res_lbfgs.fun < best_score:
                    best_params = res_lbfgs.x
                    best_score = res_lbfgs.fun
            except Exception:
                pass

    # Finalize
    if best_params is None:
        # Fallback to seed
        inner = _hexagonal_seed()
        s = _min_outer_side(inner)
        return inner, (0.0, 0.0), s, 0.0
    
    # Convert best_params back to list of lists
    inner = [[float(best_params[i*3]), float(best_params[i*3+1]), float(best_params[i*3+2])] for i in range(N)]
    
    # Calculate final outer side
    s = _min_outer_side(inner)
    
    # Strict validity check
    if _valid(inner):
        return inner, (0.0, 0.0), s, 0.0
    else:
        # If invalid, return the seed (which is valid)
        seed_inner = _hexagonal_seed()
        seed_s = _min_outer_side(seed_inner)
        return seed_inner, (0.0, 0.0), seed_s, 0.0
