import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
SQRT3 = math.sqrt(3)
DEADLINE = 140.0  # Safety margin before 150s kill

def get_hex_vertices(cx, cy, side, angle_rad):
    """Return 6 vertices of a regular hexagon."""
    verts = []
    for i in range(6):
        theta = angle_rad + i * math.pi / 3.0
        vx = cx + side * math.cos(theta)
        vy = cy + side * math.sin(theta)
        verts.append((vx, vy))
    return verts

def project_point_on_normal(px, py, nx, ny):
    return px * nx + py * ny

def compute_min_outer_side(inner_hexes, outer_center, outer_angle_deg):
    """
    Compute the minimum outer side length to contain all inner hexagons.
    inner_hexes: list of (x, y, angle_deg)
    outer_center: (x, y)
    outer_angle_deg: rotation of the outer hexagon
    """
    if not inner_hexes:
        return 0.0
    
    ox, oy = outer_center
    # Normals for a hexagon with angle 0 (vertices at 0, 60, ...):
    # Outward normals are at 30, 90, 150, 210, 270, 330 degrees.
    normals = []
    for i in range(6):
        theta = math.radians(30 + i * 60)
        normals.append((math.cos(theta), math.sin(theta)))
    
    max_projection = 0.0
    
    # Rotate outer frame to match outer_angle
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))
    
    for (ix, iy, iangle) in inner_hexes:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        for v in verts:
            # Translate to outer center
            dx = v[0] - ox
            dy = v[1] - oy
            # Rotate points by -outer_angle to align with standard hex axes
            lx = dx * cos_a + dy * sin_a
            ly = -dx * sin_a + dy * cos_a
            
            for nx, ny in normals:
                proj = lx * nx + ly * ny
                if proj > max_projection:
                    max_projection = proj
    
    # S * sqrt(3) / 2 >= max_projection
    # S >= 2 * max_projection / sqrt(3)
    return 2.0 * max_projection / SQRT3

def polygons_overlap(poly1, poly2):
    """
    Check if two convex polygons overlap using Separating Axis Theorem.
    Returns True if they overlap, False otherwise.
    """
    def get_axes(poly):
        axes = []
        for i in range(len(poly)):
            p1 = poly[i]
            p2 = poly[(i + 1) % len(poly)]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            # Normal is perpendicular to edge
            nx = -dy
            ny = dx
            norm = math.sqrt(nx*nx + ny*ny)
            if norm > 1e-9:
                axes.append((nx/norm, ny/norm))
        return axes

    axes = get_axes(poly1) + get_axes(poly2)
    
    for nx, ny in axes:
        min1, max1 = float('inf'), float('-inf')
        min2, max2 = float('inf'), float('-inf')
        
        for p in poly1:
            proj = p[0]*nx + p[1]*ny
            min1 = min(min1, proj)
            max1 = max(max1, proj)
            
        for p in poly2:
            proj = p[0]*nx + p[1]*ny
            min2 = min(min2, proj)
            max2 = max(max2, proj)
            
        # Check for separation
        if max1 < min2 or max2 < min1:
            return False
            
    return True

def compute_overlap_penalty(inner_hexes):
    """
    Compute a smooth penalty for overlapping inner hexagons.
    Uses a simple approximation: if they overlap, return a large value.
    For optimization, we want a gradient. 
    A better approach: calculate the minimum distance between vertices/edges if separated, 
    or penetration depth if overlapping.
    Here we use a simple boolean check for validity in the final result, 
    but for the optimizer, we need a continuous function.
    
    Approximation: Sum of squared distances between all pairs of vertices? No, that's O(N^2 * V^2).
    Better: For each pair, if they overlap, return 100.0. This is non-differentiable.
    
    Let's use a different strategy: 
    The objective function for `minimize` will be the outer side length.
    We will use a penalty method: if overlap is detected, add a large penalty.
    To make it smoother, we can estimate the "overlap area" or "penetration depth".
    
    Simple Penetration Depth Estimate:
    If polygons overlap, find the minimum translation vector (MTV) magnitude.
    """
    penalty = 0.0
    polys = []
    for (ix, iy, iangle) in inner_hexes:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        polys.append(verts)
        
    for i in range(len(polys)):
        for j in range(i+1, len(polys)):
            if polygons_overlap(polys[i], polys[j]):
                # Calculate approximate penetration depth
                # Project all vertices of i onto axes of j and vice versa
                # Find the axis with minimum separation (which is negative for overlap)
                # The magnitude of the minimum separation is the penetration depth.
                
                # Re-use axis logic to find the "most overlapping" axis
                def get_axes(poly):
                    axes = []
                    for k in range(len(poly)):
                        p1 = poly[k]
                        p2 = poly[(k + 1) % len(poly)]
                        dx = p2[0] - p1[0]
                        dy = p2[1] - p1[1]
                        nx = -dy
                        ny = dx
                        norm = math.sqrt(nx*nx + ny*ny)
                        if norm > 1e-9:
                            axes.append((nx/norm, ny/norm))
                    return axes

                axes = get_axes(polys[i]) + get_axes(polys[j])
                min_overlap = float('inf')
                
                for nx, ny in axes:
                    min1, max1 = float('inf'), float('-inf')
                    min2, max2 = float('inf'), float('-inf')
                    
                    for p in polys[i]:
                        proj = p[0]*nx + p[1]*ny
                        min1 = min(min1, proj)
                        max1 = max(max1, proj)
                        
                    for p in polys[j]:
                        proj = p[0]*nx + p[1]*ny
                        min2 = min(min2, proj)
                        max2 = max(max2, proj)
                        
                    # Overlap amount on this axis
                    overlap = min(max1, max2) - max(min1, min2)
                    if overlap > 0:
                        min_overlap = min(min_overlap, overlap)
                    else:
                        # Separated on this axis, so not the MTV axis
                        pass
                
                if min_overlap < float('inf'):
                    penalty += min_overlap * min_overlap * 100.0 # Quadratic penalty
                    # Break early if penalty is huge? No, sum them up.
                    
    return penalty

def objective(params):
    """
    Objective function for the optimizer.
    params: [x1, y1, a1, x2, y2, a2, ..., xN, yN, aN, outer_angle]
    """
    # Extract inner hexagon parameters
    inner_hexes = []
    for i in range(N):
        x = params[i*3]
        y = params[i*3+1]
        a = params[i*3+2]
        inner_hexes.append((x, y, a))
    
    outer_angle = params[N*3]
    
    # Calculate centroid of inner hexagons to serve as outer center
    cx = sum(h[0] for h in inner_hexes) / N
    cy = sum(h[1] for h in inner_hexes) / N
    outer_center = (cx, cy)
    
    side = compute_min_outer_side(inner_hexes, outer_center, outer_angle)
    
    # Calculate overlap penalty
    penalty = compute_overlap_penalty(inner_hexes)
    
    return side + penalty

def construct():
    start_time = time.time()
    
    # 1. Define initial layout: Hexagonal lattice
    # Axial coordinates (q, r)
    axial_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (-1, 2), (-2, 1)
    ]
    
    # Ensure we have exactly N
    if len(axial_coords) > N:
        axial_coords = axial_coords[:N]
        
    centers = []
    for q, r in axial_coords:
        x = 1.5 * q
        y = SQRT3 * (r + q / 2.0)
        centers.append((x, y))
        
    # Initial parameters: x, y, angle for each hexagon, plus outer angle
    initial_params = []
    for x, y in centers:
        initial_params.extend([x, y, 0.0])
    initial_params.append(0.0) # Outer angle
    
    best_side = float('inf')
    best_params = initial_params
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0
    
    # Calculate initial side for reference
    initial_inner = [[x, y, 0.0] for x, y in centers]
    cx_avg = sum(c[0] for c in centers) / len(centers)
    cy_avg = sum(c[1] for c in centers) / len(centers)
    best_side = compute_min_outer_side(initial_inner, (cx_avg, cy_avg), 0.0)
    
    # 2. Optimize using SciPy
    # We run the optimizer for a fixed amount of time, checking for improvements.
    # Since the problem is non-convex, we might get stuck in local minima.
    # We can try multiple restarts or perturbations if time permits.
    
    # Method 'SLSQP' or 'COBYLA'
    # 'L-BFGS-B' is faster but requires gradients. We don't have analytic gradients.
    # Finite difference gradients are expensive for 34 variables.
    # 'COBYLA' uses finite differences and is robust for small N.
    
    try:
        res = minimize(objective, initial_params, method='COBYLA', 
                       options={'maxiter': 200, 'rhobeg': 0.5, 'catol': 1e-6, 'disp': False})
        
        if res.success or res.fun < best_side:
            # Check if the result is valid (no overlap)
            # Extract params
            inner_hexes = []
            for i in range(N):
                x = res.x[i*3]
                y = res.x[i*3+1]
                a = res.x[i*3+2]
                inner_hexes.append((x, y, a))
            
            outer_angle = res.x[N*3]
            cx = sum(h[0] for h in inner_hexes) / N
            cy = sum(h[1] for h in inner_hexes) / N
            outer_center = (cx, cy)
            
            # Verify no overlap
            if compute_overlap_penalty(inner_hexes) < 1e-6:
                side = compute_min_outer_side(inner_hexes, outer_center, outer_angle)
                if side < best_side:
                    best_side = side
                    best_params = res.x
                    best_outer_center = outer_center
                    best_outer_angle = outer_angle
                    best_inner = inner_hexes
            
    except Exception as e:
        pass # If optimization fails, keep initial best
    
    # If time permits, try a second optimization with perturbed initial state
    if time.time() - start_time < DEADLINE - 20:
        # Perturb initial params slightly
        perturbed_params = initial_params.copy()
        np.random.seed(42)
        for i in range(len(perturbed_params)):
            perturbed_params[i] += np.random.uniform(-0.1, 0.1)
            
        try:
            res2 = minimize(objective, perturbed_params, method='COBYLA', 
                           options={'maxiter': 200, 'rhobeg': 0.5, 'catol': 1e-6, 'disp': False})
            
            if res2.fun < best_side:
                inner_hexes = []
                for i in range(N):
                    x = res2.x[i*3]
                    y = res2.x[i*3+1]
                    a = res2.x[i*3+2]
                    inner_hexes.append((x, y, a))
                
                outer_angle = res2.x[N*3]
                cx = sum(h[0] for h in inner_hexes) / N
                cy = sum(h[1] for h in inner_hexes) / N
                outer_center = (cx, cy)
                
                if compute_overlap_penalty(inner_hexes) < 1e-6:
                    side = compute_min_outer_side(inner_hexes, outer_center, outer_angle)
                    if side < best_side:
                        best_side = side
                        best_params = res2.x
                        best_outer_center = outer_center
                        best_outer_angle = outer_angle
                        best_inner = inner_hexes
        except Exception as e:
            pass

    # Prepare final return values
    # Ensure best_inner is defined
    if 'best_inner' not in locals():
        best_inner = initial_inner
        
    # Add tiny epsilon to side to ensure validity
    best_side += 1e-9

    return best_inner, best_outer_center, best_side, best_outer_angle
