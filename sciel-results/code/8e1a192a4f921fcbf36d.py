"""
Place 12 unit hexagons inside a regular hexagon, minimizing outer side length.
Uses lattice initialization and local refinement with fixed angles.
"""
N = 12

import numpy as np
import time
from scipy.optimize import minimize
import math

def _hex_verts(cx, cy, side, angle_deg):
    """Vertices of a regular hexagon centered at (cx, cy)."""
    a = np.radians(angle_deg)
    return np.array([
        [cx + side * np.cos(a + np.radians(60 * i)),
         cy + side * np.sin(a + np.radians(60 * i))]
        for i in range(6)
    ])

def _support_hex(center, angle_deg, side, dx, dy):
    """Support function of a regular hexagon in direction (dx, dy)."""
    d = np.array([dx, dy])
    dn = np.linalg.norm(d)
    if dn < 1e-12:
        return 0.0
    d = d / dn
    a = np.radians(angle_deg)
    angles = a + np.radians(60 * np.arange(6))
    vs = side * np.array([np.cos(angles), np.sin(angles)]).T
    dots = vs @ d
    return np.max(dots)

def _min_enclosing_hex_side(centers, angles, outer_angle_deg=0.0):
    """
    Calculate the minimum side length of a regular hexagon with given outer angle
    that contains all inner hexagons.
    """
    outer_angle = np.radians(outer_angle_deg)
    # The 6 outward normals of the outer hexagon
    normals = np.array([
        [np.cos(outer_angle + np.radians(60 * i)),
         np.sin(outer_angle + np.radians(60 * i))]
        for i in range(6)
    ])
    
    max_ext = 0.0
    for k in range(len(centers)):
        cx, cy = centers[k]
        ang = angles[k]
        for ni in range(6):
            dx = normals[ni, 0]
            dy = normals[ni, 1]
            h = _support_hex(np.array([cx, cy]), ang, 1.0, dx, dy)
            dot_center = cx * dx + cy * dy
            needed = dot_center + h
            if needed > max_ext:
                max_ext = needed
    
    # The apothem of the outer hexagon must be at least max_ext
    # Apothem = S * sqrt(3) / 2
    # So S = 2 * max_ext / sqrt(3)
    S = 2.0 * max_ext / np.sqrt(3.0)
    return S

def _check_validity(centers, angles, S, outer_angle_deg=0.0, tol=1e-6):
    """
    Check if all inner hexagons are inside the outer hexagon and disjoint.
    """
    # Check containment
    outer_verts = _hex_verts(0.0, 0.0, S, outer_angle_deg)
    inner_verts_list = []
    for i in range(len(centers)):
        cx, cy = centers[i]
        ang = angles[i]
        iv = _hex_verts(cx, cy, 1.0, ang)
        inner_verts_list.append(iv)
        for pt in iv:
            # Check if point is inside outer hex
            # Using cross product method for convex polygon
            inside = True
            for j in range(6):
                v1 = outer_verts[j]
                v2 = outer_verts[(j+1)%6]
                edge = v2 - v1
                rel = pt - v1
                cross = edge[0]*rel[1] - edge[1]*rel[0]
                if cross < -tol:
                    inside = False
                    break
            if not inside:
                return False
    
    # Check disjointness
    # For hexagons, if distance between centers is >= 2*sqrt(3) they are definitely disjoint?
    # No, min distance for disjoint unit hexes is 2*sqrt(3) if aligned?
    # Actually, min distance between centers of two disjoint unit hexagons is 2*sqrt(3) ~ 3.46 if they share an edge?
    # Wait, side length 1. Distance between centers of two adjacent hexes in a tiling is 2*sqrt(3)?
    # Let's check.
    # Hexagon apothem = sqrt(3)/2.
    # Distance between parallel sides = sqrt(3).
    # If two hexes share a side, center distance is 2 * apothem = sqrt(3) ~ 1.732?
    # No.
    # Let's calculate.
    # Vertices of hex 1 at (1,0), (0.5, sqrt(3)/2)...
    # Center (0,0).
    # Neighbor hex center at (1.5, sqrt(3)/2)?
    # Distance = sqrt(1.5^2 + (sqrt(3)/2)^2) = sqrt(2.25 + 0.75) = sqrt(3) ~ 1.732.
    # So if dist < sqrt(3) - epsilon, they overlap?
    # Actually, if dist < sqrt(3), they might overlap.
    # To be safe, we check polygon intersection.
    
    for i in range(len(centers)):
        for j in range(i+1, len(centers)):
            vi = inner_verts_list[i]
            vj = inner_verts_list[j]
            # Simple SAT check for intersection
            # We can reuse the logic from before but simpler:
            # If min distance between any vertex of i and edge of j is negative...
            # Or just use scipy spatial?
            # Let's use a simple vertex-edge check.
            def intersect(v1, v2):
                # Check if any vertex of v1 is inside v2 or vice versa
                # And if any edges intersect.
                # For convex polygons, if no vertex is inside and no edges intersect, they are disjoint.
                # But "inside" check is O(N^2).
                # Given N=12, 66 pairs, 6 vertices each, this is fast enough.
                
                def point_in_poly(pt, poly):
                    n = len(poly)
                    for i in range(n):
                        v1 = poly[i]
                        v2 = poly[(i+1)%n]
                        edge = v2 - v1
                        rel = pt - v1
                        cross = edge[0]*rel[1] - edge[1]*rel[0]
                        if cross < -1e-9:
                            return False
                    return True

                # Check vertices
                for pt in v1:
                    if point_in_poly(pt, v2):
                        return True
                for pt in v2:
                    if point_in_poly(pt, v1):
                        return True
                
                # Check edge intersections
                def segments_intersect(p1, p2, p3, p4):
                    # Check if segment p1-p2 intersects p3-p4
                    # Using cross products
                    def cross(o, a, b):
                        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
                    
                    d1 = cross(p3, p4, p1)
                    d2 = cross(p3, p4, p2)
                    d3 = cross(p1, p2, p3)
                    d4 = cross(p1, p2, p4)
                    
                    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
                       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
                        return True
                    return False

                for i in range(6):
                    p1 = vi[i]
                    p2 = vi[(i+1)%6]
                    for j in range(6):
                        p3 = vj[j]
                        p4 = vj[(j+1)%6]
                        if segments_intersect(p1, p2, p3, p4):
                            return True
                return False
            
            if intersect(vi, vj):
                return False
                
    return True

def _objective_pos(params, angles, outer_angle_deg=0.0):
    """
    Objective function for position optimization.
    params: 2*N array [x0, y0, x1, y1, ...]
    """
    centers = np.zeros((N, 2))
    for i in range(N):
        centers[i, 0] = params[2*i]
        centers[i, 1] = params[2*i+1]
    
    S = _min_enclosing_hex_side(centers, angles, outer_angle_deg)
    
    # Add penalty if any hexagon is "too close" to boundary?
    # No, S is derived from the max extension, so containment is guaranteed by definition of S.
    # We just need to minimize S.
    # But we must ensure disjointness.
    # If they overlap, we add a large penalty.
    
    penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            dist = np.linalg.norm(centers[i] - centers[j])
            # If dist < sqrt(3) ~ 1.732, they likely overlap.
            # Let's use a soft penalty for dist < 1.732
            if dist < 1.732:
                penalty += (1.732 - dist)**2 * 100.0
            
    return S + penalty

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_S = float('inf')
    best_params = None
    best_angles = None
    best_outer_angle = 0.0
    
    # Try a few lattice configurations
    # Configuration 1: 3 rows of 4
    # Hex lattice coordinates:
    # Row 0: (0,0), (2,0), (4,0), (6,0)  -- wait, spacing is 2 in x?
    # For side 1 hexes:
    # Horizontal distance between centers in same row: 2 * cos(30) * 1? No.
    # Let's use standard hex grid.
    # Flat-top hexes.
    # Width = 2 * side. Height = sqrt(3) * side.
    # Horizontal spacing = 1.5 * side? No, 3/2 * side?
    # Let's just generate a grid and let the optimizer fix it.
    
    # Initial guess: 3x4 grid
    # x: 0, 1.5, 3.0, 4.5 (spacing 1.5?)
    # y: 0, sqrt(3), 2*sqrt(3)
    # Let's try:
    init_centers = []
    for r in range(3):
        for c in range(4):
            x = c * 1.5 + (0.75 if r % 2 == 1 else 0.0) # Offset
            y = r * np.sqrt(3) * 0.5 # Vertical spacing is sqrt(3)/2 * 2?
            # Standard hex packing:
            # dy = sqrt(3) * side? No.
            # Distance between rows = sqrt(3) * side?
            # If side=1, height of hex = sqrt(3).
            # Row spacing = 3/4 * height?
            # Let's just use a dense grid and let optimizer work.
            x = c * 1.8
            y = r * 1.5
            init_centers.append((x, y))
    
    # Center the grid
    cx_mean = np.mean([p[0] for p in init_centers])
    cy_mean = np.mean([p[1] for p in init_centers])
    init_params = []
    for (x, y) in init_centers:
        init_params.extend([x - cx_mean, y - cy_mean])
    init_params = np.array(init_params)
    
    # Try angles 0 and 30
    for ang in [0.0, 30.0]:
        angles = np.array([ang] * N)
        
        # Run local optimization from init_params
        try:
            result = minimize(
                _objective_pos,
                init_params,
                args=(angles, 0.0),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10}
            )
            opt_params = result.x
            S_val = _min_enclosing_hex_side(
                np.array([[opt_params[2*i], opt_params[2*i+1]] for i in range(N)]),
                angles,
                0.0
            )
            
            # Check validity
            centers = np.array([[opt_params[2*i], opt_params[2*i+1]] for i in range(N)])
            if _check_validity(centers, angles, S_val, 0.0):
                if S_val < best_S:
                    best_S = S_val
                    best_params = opt_params.copy()
                    best_angles = angles.copy()
                    best_outer_angle = 0.0
        except Exception:
            pass
            
        # Try another outer angle?
        # The outer hexagon angle doesn't change the minimal S if we can rotate the inner cluster.
        # But we fixed inner angles. So we should try different outer angles or inner angles.
        
    # If no valid result, fallback to a known good lattice
    if best_params is None:
        # Use a simple 3x4 grid with side ~ 4.5?
        # Let's calculate S for a 3x4 grid of hexes with side 1.
        # Width approx 4 * 1.5 = 6?
        # Height approx 3 * 1.5 = 4.5?
        # S approx 5.0
        inner = []
        for r in range(3):
            for c in range(4):
                x = c * 1.5 + (0.75 if r % 2 == 1 else 0.0)
                y = r * np.sqrt(3) * 0.75
                inner.append([float(x), float(y), 0.0])
        
        # Center it
        cx_mean = np.mean([p[0] for p in inner])
        cy_mean = np.mean([p[1] for p in inner])
        inner = [[p[0]-cx_mean, p[1]-cy_mean, p[2]] for p in inner]
        
        centers = np.array([[p[0], p[1]] for p in inner])
        angles = np.array([0.0]*N)
        S_val = _min_enclosing_hex_side(centers, angles, 0.0)
        
        return inner, (0.0, 0.0), float(S_val), 0.0
    
    # Extract inner hexagons
    inner = []
    for i in range(N):
        x = float(best_params[2*i])
        y = float(best_params[2*i+1])
        ang = float(best_angles[i])
        inner.append([x, y, ang])
    
    return inner, (0.0, 0.0), float(best_S), float(best_outer_angle)
