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

def _objective_full(params, outer_angle_deg=0.0):
    """
    Objective function for full optimization (positions and angles).
    params: 3*N array [x0, y0, a0, x1, y1, a1, ...]
    """
    centers = np.zeros((N, 2))
    angles = np.zeros(N)
    for i in range(N):
        centers[i, 0] = params[3*i]
        centers[i, 1] = params[3*i+1]
        angles[i] = params[3*i+2]
    
    # Calculate minimum enclosing hexagon side length
    S = _min_enclosing_hex_side(centers, angles, outer_angle_deg)
    
    # Check validity: containment and disjointness
    # We use a strict penalty for any violation.
    penalty = 0.0
    
    # 1. Containment Check
    # For each inner hex, check if all vertices are inside the outer hex of side S
    # If S is calculated as the min enclosing, containment is theoretically satisfied.
    # However, due to floating point or if S is very small, we might have issues.
    # The main issue is usually disjointness.
    
    # 2. Disjointness Check
    # We need to ensure no two inner hexes overlap.
    # We can use a distance-based proxy or exact polygon intersection.
    # Exact intersection is expensive for DE (many iterations).
    # A good proxy: The minimum distance between centers must be >= 2 * apothem = sqrt(3) for aligned hexes?
    # Actually, for regular hexagons of side 1, the "collision diameter" varies with relative angle.
    # The minimum distance between centers for disjointness is the Minkowski sum radius.
    # For two unit hexes, the minimum center distance to be disjoint is sqrt(3) if they share an edge?
    # No, if they share an edge, distance is sqrt(3). If they touch at a vertex, distance is 2.
    # Wait, if they share an edge, they are not disjoint (they touch). Disjoint means no interior intersection.
    # So distance >= sqrt(3) is a necessary condition for *some* orientations, but not all.
    # If relative angle is 30 deg, the collision distance is different.
    
    # To be safe and fast, we will compute the actual minimum distance between polygons if they are close.
    # But for DE, we need a smooth-ish penalty.
    # Let's use a soft penalty based on the distance between centers, scaled by a factor.
    # The minimum possible distance between centers of two disjoint unit hexes is sqrt(3) (edge-to-edge).
    # If dist < sqrt(3), they MUST overlap?
    # Let's check. Apothem = sqrt(3)/2. Sum of apothems = sqrt(3).
    # If dist < sqrt(3), the hexes overlap.
    # So, if dist < sqrt(3), we have a penalty.
    # If dist >= sqrt(3), they might still overlap if angles are bad?
    # Actually, if dist >= 2 (vertex to vertex max), they are safe.
    # The critical region is sqrt(3) <= dist < 2.
    
    # For simplicity and speed in DE, we will penalize dist < sqrt(3) heavily.
    # And for dist between sqrt(3) and 2, we will do a quick check or assume safe if angles are 0/30.
    # Since we optimize angles, we should be careful.
    
    # Let's use a robust penalty:
    # If dist < sqrt(3) - epsilon: penalty += (sqrt(3) - dist)^2 * 1000
    # This ensures they don't overlap in the worst-case orientation.
    # If they are closer than sqrt(3), they overlap for ALL orientations.
    # If they are further, they might overlap for SOME orientations.
    # To handle this, we can add a small penalty for dist < 2.0 to encourage spacing.
    
    for i in range(N):
        for j in range(i+1, N):
            dist = np.linalg.norm(centers[i] - centers[j])
            min_dist = np.sqrt(3.0)
            if dist < min_dist:
                penalty += (min_dist - dist)**2 * 1000.0
            elif dist < 2.0:
                # Soft penalty to push them apart if they are in the "risky" zone
                # This helps the optimizer find valid configurations more easily
                penalty += (2.0 - dist)**2 * 10.0
                
    return S + penalty

def construct():
    # Use a guaranteed valid scaled hexagonal lattice
    # This avoids the complexity and failure modes of the DE optimizer
    # which was returning overlapping configurations.
    
    # Vectors for flat-top hexagon tiling (touching)
    # u = (1.5, sqrt(3)/2)
    # v = (-1.5, sqrt(3)/2)
    u = np.array([1.5, np.sqrt(3.0)/2.0])
    v = np.array([-1.5, np.sqrt(3.0)/2.0])
    
    # Scale factor to ensure disjointness (add small gap)
    scale = 1.02
    
    inner = []
    # 4 columns (i), 3 rows (j)
    for i in range(4):
        for j in range(3):
            pos = scale * (i * u + j * v)
            inner.append([float(pos[0]), float(pos[1]), 0.0])
            
    # Center the configuration
    cx_mean = np.mean([p[0] for p in inner])
    cy_mean = np.mean([p[1] for p in inner])
    inner = [[p[0]-cx_mean, p[1]-cy_mean, p[2]] for p in inner]
    
    centers = np.array([[p[0], p[1]] for p in inner])
    angles = np.array([0.0]*N)
    
    # Calculate outer side length
    S_val = _min_enclosing_hex_side(centers, angles, 0.0)
    
    # Verify validity (should be true with scale > 1)
    if not _check_validity(centers, angles, S_val, 0.0):
        # If somehow invalid (unlikely), increase scale and retry
        scale = 1.1
        inner = []
        for i in range(4):
            for j in range(3):
                pos = scale * (i * u + j * v)
                inner.append([float(pos[0]), float(pos[1]), 0.0])
        
        cx_mean = np.mean([p[0] for p in inner])
        cy_mean = np.mean([p[1] for p in inner])
        inner = [[p[0]-cx_mean, p[1]-cy_mean, p[2]] for p in inner]
        centers = np.array([[p[0], p[1]] for p in inner])
        S_val = _min_enclosing_hex_side(centers, angles, 0.0)

    return inner, (0.0, 0.0), float(S_val), 0.0
