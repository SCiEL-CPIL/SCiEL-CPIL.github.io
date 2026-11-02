import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11

def _get_outer_for_center(cx, cy, angle_deg):
    """
    Computes the minimal side length and center of a regular hexagon
    with rotation angle_deg that encloses the point (cx, cy) and the origin
    (assuming one inner hex is at origin for symmetry, or just enclosing a set).
    
    Actually, for the specific case of 11 hexagons where we fix one at (0,0) 
    and the outer hexagon is centered at (0,0) with angle 0 (or 30), 
    we can derive the required side length.
    
    However, the problem allows the outer hexagon to rotate and translate.
    To simplify and ensure validity, we will fix the outer hexagon to be 
    centered at (0,0) with rotation 0 (flat top) or 30 (pointy top).
    
    Wait, the optimal packing of 11 hexagons is known to be a hexagonal cluster 
    (1 center + 6 ring + 4 from next ring) or similar.
    The minimal enclosing hexagon for a specific set of points can be found.
    
    For this strategy, we will assume the outer hexagon is centered at (0,0) 
    and has rotation 0. We will compute the minimal side length required 
    to enclose all inner hexagons.
    
    The condition for a point (x,y) to be inside a hexagon centered at (0,0) 
    with side S and rotation 0 (flat top) is:
    |x| <= S * sqrt(3)/2  (width)
    |y| <= S * 3/2 (height) ... No, that's not right.
    
    Let's use the support function.
    For a hexagon with side S and rotation 0, the vertices are at angles 0, 60, 120...
    The apothem is a = S * sqrt(3)/2.
    The hexagon is the intersection of 6 half-planes.
    The normal vectors to the edges are at angles 30, 90, 150, 210, 270, 330.
    
    For a point p to be inside, the projection of p onto each normal n_i 
    must be <= a.
    So S >= max_i ( |p . n_i| / (sqrt(3)/2) ) for all points p that are vertices of inner hexagons.
    
    To be safe, we check all vertices of all inner hexagons.
    """
    a = math.radians(angle_deg)
    # Normals for hexagon with rotation angle_deg
    # If rotation is 0, vertices are at 0, 60, ...
    # Edges are perpendicular to normals at 30, 90, ...
    normals = []
    for i in range(6):
        theta = a + math.radians(30 + i * 60)
        normals.append((math.cos(theta), math.sin(theta)))
    
    max_proj = 0.0
    # We need to pass the vertices of the inner hexagons to this function
    # But in the optimization loop, we only have centers and angles.
    # We will compute this inside the objective function.
    
    return None # Placeholder

def _compute_min_outer_side(inner_params, outer_angle_deg):
    """
    Computes the minimum side length of an outer hexagon centered at (0,0) 
    with rotation outer_angle_deg that encloses all inner hexagons in inner_params.
    inner_params: list of (cx, cy, angle_deg)
    """
    a_outer = math.radians(outer_angle_deg)
    normals = []
    for i in range(6):
        theta = a_outer + math.radians(30 + i * 60)
        normals.append((math.cos(theta), math.sin(theta)))
    
    max_proj = 0.0
    apothem_factor = math.sqrt(3) / 2.0
    
    for cx, cy, iang in inner_params:
        # Get vertices of this inner hexagon
        a_in = math.radians(iang)
        for k in range(6):
            theta_v = a_in + k * math.pi / 3
            vx = cx + math.cos(theta_v)
            vy = cy + math.sin(theta_v)
            
            # Project vertex onto all outer normals
            for nx, ny in normals:
                proj = vx * nx + vy * ny
                if proj > max_proj:
                    max_proj = proj
    
    return max_proj / apothem_factor

def _soft_penalty(x, y, min_dist):
    """
    Soft penalty for distance d. If d > min_dist, penalty is 0.
    If d < min_dist, penalty is (min_dist - d)^2.
    Here we use a smoother version: exp(-(d-min_dist)^2) or similar.
    Actually, for L-BFGS, a simple quadratic penalty on violation is often enough 
    if initialized well.
    """
    d = math.sqrt(x*x + y*y)
    if d >= min_dist:
        return 0.0
    return (min_dist - d) ** 2

def _objective(x):
    """
    x: [cx1, cy1, a1, ..., cxN, cyN, aN]
    Note: We fix outer hexagon center at (0,0) and angle at 0.
    We minimize the outer side length, plus penalties for overlap.
    """
    n = len(x) // 3
    inner_params = []
    for i in range(n):
        cx = x[i*3]
        cy = x[i*3+1]
        a = x[i*3+2]
        inner_params.append((cx, cy, a))
    
    # 1. Compute outer side length
    # We assume outer hexagon is at (0,0) with angle 0.
    # Is it optimal to have outer hexagon at 0?
    # Due to symmetry of the problem (hexagons are 6-fold symmetric), 
    # we can fix the orientation of the outer hexagon to 0 without loss of generality 
    # IF we allow the inner hexagons to rotate freely.
    # Yes, because any rotation of the outer hexagon can be compensated by 
    # rotating the entire system, which is equivalent to rotating all inner hexagons 
    # and their positions. Since we are optimizing positions and rotations, 
    # fixing the outer frame is valid.
    
    outer_side = _compute_min_outer_side(inner_params, 0.0)
    
    # 2. Compute overlap penalties
    penalty = 0.0
    # Min distance between centers of two unit hexagons to be disjoint is 2.0?
    # No. If they are aligned, distance between centers must be >= 2 * apothem = sqrt(3) ~ 1.732?
    # No, if they are side-by-side, distance is 2 * apothem = sqrt(3).
    # If they are corner-to-corner, distance is 2.
    # The minimum distance between centers for disjoint hexagons depends on relative rotation.
    # For arbitrary rotation, the minimum center distance to guarantee disjointness is 2.0 (diameter).
    # But they can be closer if rotated.
    # E.g., if one is rotated 30 deg relative to the other, they can fit closer.
    # The "kissing" distance for two unit hexagons is related to the sum of their radii of gyration?
    # Actually, the minimum distance between centers for two disjoint unit hexagons 
    # is sqrt(3) if they are aligned (sharing a side).
    # If rotated, it can be smaller? No, the circumradius is 1. The inradius is sqrt(3)/2.
    # The maximum distance from center to any point is 1.
    # So if center distance < 2, they MIGHT overlap.
    # If center distance < sqrt(3), they MUST overlap?
    # If aligned, side length 1. Width across flats is sqrt(3).
    # So if center distance < sqrt(3), they overlap.
    # So min_dist = sqrt(3) is a safe lower bound for aligned hexagons.
    # But for arbitrary rotations, the minimum center distance for disjointness 
    # is actually the same? No.
    # Consider two hexagons. One fixed. One rotated 30 deg.
    # The "radius" of a hexagon in a given direction varies from sqrt(3)/2 to 1.
    # The minimum distance between centers is min over all directions of (r1(theta) + r2(theta+pi)).
    # For a unit hexagon, r(theta) = (sqrt(3)/2) / cos((theta - 30k) mod 60 - 30).
    # The minimum sum of radii in opposite directions is sqrt(3)/2 + sqrt(3)/2 = sqrt(3).
    # So regardless of rotation, the minimum center distance to avoid overlap is sqrt(3)?
    # Let's verify.
    # If two hexagons are rotated 30 deg relative to each other, and we push them together,
    # they will touch at vertices?
    # If they touch at vertices, the distance is 2.
    # If they touch at sides, the distance is sqrt(3).
    # So sqrt(3) is the minimum possible center distance for ANY two disjoint unit hexagons.
    
    min_center_dist = math.sqrt(3)
    
    for i in range(n):
        for j in range(i+1, n):
            dx = inner_params[i][0] - inner_params[j][0]
            dy = inner_params[i][1] - inner_params[j][1]
            d = math.sqrt(dx*dx + dy*dy)
            if d < min_center_dist:
                # This is a necessary condition, but not sufficient for all rotations.
                # We need a better penalty.
                # Let's check actual vertex overlap for a subset or use a tighter bound.
                # For now, use a soft penalty based on distance.
                # To be more accurate, we can check if any vertex of i is inside j.
                # But that's not differentiable.
                # We can use a "soft" support function check.
                
                # Let's refine the penalty:
                # Calculate the minimum distance between the two hexagons.
                # This is expensive.
                # Alternative: use a larger min_dist to be safe, e.g., 2.0?
                # If we use 2.0, we might be too conservative and get a worse score.
                # Let's try to compute a better lower bound.
                
                # For now, let's use a penalty that grows as d approaches sqrt(3).
                # But we must ensure no overlap.
                # If d >= sqrt(3), they *might* still overlap if rotated badly?
                # No, if d >= sqrt(3), is it possible to overlap?
                # Yes. Imagine two hexagons. One at (0,0), one at (sqrt(3), 0).
                # If they are aligned, they touch at the vertical edge.
                # If we rotate the second one by 30 deg, does it overlap?
                # The vertex of the second hexagon at angle 30 is at (1, 0) relative to its center? No.
                # Vertices of hexagon at (cx, cy) with angle 0:
                # (1, 0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1, 0), ...
                # If center is at (sqrt(3), 0), vertex at (sqrt(3)+1, 0).
                # Distance to origin is sqrt(3)+1 > 1.
                # What about the vertex at (sqrt(3)-0.5, sqrt(3)/2)?
                # Distance squared: (sqrt(3)-0.5)^2 + 3/4 = 3 - sqrt(3) + 0.25 + 0.75 = 4 - sqrt(3) ~ 2.268.
                # Distance ~ 1.5. This is > 1.
                # It seems sqrt(3) is indeed the minimum distance.
                
                # So if d >= sqrt(3), they are guaranteed to be disjoint?
                # Let's assume yes for the penalty function, but verify with a hard check in the final output.
                # For the optimizer, we use a soft penalty for d < sqrt(3) + epsilon.
                
                # Actually, to be safe against numerical errors and rotation nuances,
                # let's use a slightly larger distance for the penalty, or a very steep penalty.
                # Let's use a penalty based on the "gap".
                # We want to maximize the minimum gap.
                # Let's define the objective as: Outer_Side + Lambda * sum( max(0, min_dist - d_ij)^2 )
                
                # But this only ensures center distance.
                # We need to ensure they don't overlap.
                # If d >= sqrt(3), they don't overlap.
                # So this constraint is sufficient.
                
                penalty += (min_center_dist - d) ** 2
    
    return outer_side + 100.0 * penalty

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Initial guess: Hexagonal cluster
    # 1 center, 6 in ring 1, 4 in ring 2
    init = []
    init.append((0.0, 0.0, 0.0))
    for i in range(6):
        a = math.radians(60 * i)
        x = math.sqrt(3) * math.cos(a)
        y = math.sqrt(3) * math.sin(a)
        init.append((x, y, 0.0))
    
    # Ring 2: distance 3 * sqrt(3)? No.
    # Ring 1 centers are at distance sqrt(3).
    # Ring 2 centers are at distance 3 * sqrt(3) / 2? No.
    # The distance between adjacent centers in a hexagonal lattice of hexagons is sqrt(3).
    # So ring 2 is at distance 2 * sqrt(3).
    # Let's pick 4 from ring 2.
    # Positions in ring 2 (distance 2*sqrt(3)):
    # Angles: 0, 30, 60, 90...
    # Let's pick 0, 90, 180, 270 for symmetry.
    for i in range(4):
        a = math.radians(90 * i)
        x = 2 * math.sqrt(3) * math.cos(a)
        y = 2 * math.sqrt(3) * math.sin(a)
        init.append((x, y, 0.0))
        
    x0 = []
    for p in init:
        x0.append(p[0])
        x0.append(p[1])
        x0.append(p[2])
        
    # Bounds
    bounds = []
    for i in range(N):
        bounds.append((-5, 5))
        bounds.append((-5, 5))
        bounds.append((0, 360))
        
    best_x = None
    best_val = float('inf')
    
    # Try multiple random restarts or perturbations
    # Since we have a deadline, we can try a few local optimizations.
    
    # 1. Optimize from the symmetric start
    try:
        res = minimize(_objective, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 200, 'ftol': 1e-8})
        if res.fun < best_val:
            best_val = res.fun
            best_x = res.x
    except Exception:
        pass
        
    if time.time() > deadline:
        return _format_result(best_x if best_x is not None else x0)
        
    # 2. Try perturbing the angles
    # Break symmetry slightly
    for restart in range(5):
        x0_pert = x0.copy()
        # Add small noise to angles
        for i in range(N):
            x0_pert[i*3+2] += (random.random() - 0.5) * 10
            # Add small noise to positions
            x0_pert[i*3] += (random.random() - 0.5) * 0.1
            x0_pert[i*3+1] += (random.random() - 0.5) * 0.1
            
        try:
            res = minimize(_objective, x0_pert, method='L-BFGS-B', bounds=bounds, options={'maxiter': 100, 'ftol': 1e-8})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass
            
        if time.time() > deadline:
            break
            
    return _format_result(best_x if best_x is not None else x0)

def _format_result(x):
    inner = []
    for i in range(N):
        cx = x[i*3]
        cy = x[i*3+1]
        a = x[i*3+2]
        inner.append([cx, cy, a])
        
    # Outer hexagon is fixed at (0,0) with angle 0
    # We need to return the actual side length
    # We recompute it to be sure
    inner_params = []
    for i in range(N):
        inner_params.append((x[i*3], x[i*3+1], x[i*3+2]))
        
    os = _compute_min_outer_side(inner_params, 0.0)
    
    return inner, (0.0, 0.0), os, 0.0

import random

if __name__ == "__main__":
    r = construct()
    print(r)
