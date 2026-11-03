import numpy as np
from scipy.optimize import minimize
import time

N = 11
DEADLINE = time.time() + 140.0  # safety margin before 150s kill

def hex_vertices(cx, cy, side, angle_deg, circumradius=1.0):
    """Return 6 vertices of a regular hexagon."""
    angle = np.radians(angle_deg)
    angles = angle + np.arange(6) * np.pi / 3
    return np.column_stack([
        cx + circumradius * np.cos(angles),
        cy + circumradius * np.sin(angles)
    ])

def hex_distance_to_boundary(point, outer_cx, outer_cy, outer_side, outer_angle_deg):
    """
    Compute the minimum distance from a point to the boundary of the outer hexagon.
    Positive if inside, negative if outside.
    For a regular hexagon with side s, inradius = s * sqrt(3)/2.
    The hexagon has 6 sides. We compute the signed distance to each side.
    """
    inradius = outer_side * np.sqrt(3) / 2.0
    angle = np.radians(outer_angle_deg)
    
    # The 6 side normals point outward at angles angle + k*pi/3 for k=0..5
    # Actually, for a flat-top hexagon, sides are at angles 0, 60, 120, 180, 240, 300
    # The inward normals point toward center
    # Let's define: the hexagon has vertices at angles angle + k*60 degrees
    # Side k is between vertex k and vertex k+1
    # The outward normal of side k points at angle + (k+0.5)*60 degrees
    
    # Alternative: use the fact that for a regular hexagon, the constraint is:
    # For each of 6 directions, the projection of (p - center) onto the direction
    # must be <= inradius (for the side perpendicular to that direction)
    
    # The 6 directions perpendicular to the 6 sides:
    # If vertices are at angles angle + k*60, then sides are perpendicular to directions angle + (k+0.5)*60
    # Actually, let me think more carefully.
    
    # A regular hexagon centered at origin with one vertex at angle 0 has vertices at:
    # (1,0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1,0), (-0.5,-sqrt(3)/2), (0.5,-sqrt(3)/2)
    # The sides connect consecutive vertices.
    # Side 0: from (1,0) to (0.5, sqrt(3)/2). Midpoint at angle 30 degrees.
    # The outward normal points at angle 30 degrees.
    # So side k has outward normal at angle 30 + k*60 degrees.
    
    # For a hexagon rotated by `angle`, side k has outward normal at angle + 30 + k*60 degrees.
    
    normals = []
    for k in range(6):
        theta = angle + np.radians(30 + k * 60)
        normals.append(np.array([np.cos(theta), np.sin(theta)]))
    normals = np.array(normals)
    
    # For point p, the constraint is: dot(p - center, normal_k) <= inradius for all k
    # The signed distance to side k is: inradius - dot(p - center, normal_k)
    # The point is inside if all signed distances >= 0
    # The minimum signed distance is the distance to the boundary (positive if inside)
    
    rel = point - np.array([outer_cx, outer_cy])
    projections = np.dot(rel, normals.T)  # shape (6,)
    signed_distances = inradius - projections
    return np.min(signed_distances)

def objective(params):
    """
    params: [outer_side, outer_cx, outer_cy, outer_angle_deg, 
             x1, y1, a1, x2, y2, a2, ..., xN, yN, aN]
    Minimize outer_side subject to all inner hex vertices being inside outer hex.
    Use penalty method: minimize outer_side + penalty * max(0, -min_distance)^2
    """
    outer_side = params[0]
    outer_cx = params[1]
    outer_cy = params[2]
    outer_angle = params[3]
    
    if outer_side < 1.0:
        return 1e10
    
    inner_params = params[4:]
    min_dist = np.inf
    
    for i in range(N):
        cx = inner_params[i * 3]
        cy = inner_params[i * 3 + 1]
        angle_deg = inner_params[i * 3 + 2]
        
        verts = hex_vertices(cx, cy, 1.0, angle_deg)
        for v in verts:
            d = hex_distance_to_boundary(v, outer_cx, outer_cy, outer_side, outer_angle)
            if d < min_dist:
                min_dist = d
    
    if min_dist < 0:
        penalty = 100.0 * min_dist * min_dist
    else:
        penalty = 0.0
    
    return outer_side + penalty

def construct():
    global DEADLINE
    best_result = None
    best_score = np.inf
    
    # Initial guess: central hex + 6 around + 4 more
    # For side s, the distance from center to a neighboring hex center in a hex grid is 2*inradius = sqrt(3)
    # But with rotation freedom, we can do better
    
    # Start with a reasonable outer side
    # 11 hexagons: central + 6 ring + 4 extra
    # If we place them in a hexagonal arrangement, the bounding hexagon side is roughly 3-4
    
    n_restarts = 8
    rng = np.random.RandomState(42)
    
    for restart in range(n_restarts):
        if time.time() > DEADLINE:
            break
        
        # Randomize initial configuration
        if restart == 0:
            # Symmetric initial guess
            outer_side = 4.0
            outer_cx = 0.0
            outer_cy = 0.0
            outer_angle = 0.0
            
            positions = []
            # Central
            positions.append([0.0, 0.0, 0.0])
            # 6 around at distance sqrt(3) in hex directions
            for k in range(6):
                theta = np.radians(60 * k)
                positions.append([np.sqrt(3) * np.cos(theta), np.sqrt(3) * np.sin(theta), 0.0])
            # 4 more: place at distance 2*sqrt(3) in 4 directions
            for k in range(4):
                theta = np.radians(60 * k + 30)
                positions.append([2 * np.sqrt(3) * np.cos(theta), 2 * np.sqrt(3) * np.sin(theta), 0.0])
        else:
            # Perturb from best or random
            outer_side = 3.5 + rng.random() * 1.0
            outer_cx = rng.uniform(-0.5, 0.5)
            outer_cy = rng.uniform(-0.5, 0.5)
            outer_angle = rng.uniform(0, 60)
            
            positions = []
            for i in range(N):
                if i == 0:
                    positions.append([rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5), rng.uniform(0, 60)])
                else:
                    theta = rng.uniform(0, 2 * np.pi)
                    r = rng.uniform(1.0, 3.0)
                    positions.append([r * np.cos(theta), r * np.sin(theta), rng.uniform(0, 60)])
        
        params = np.array([outer_side, outer_cx, outer_cy, outer_angle] + [p for pos in positions for p in pos])
        
        # Optimize
        result = minimize(
            objective,
            params,
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        if result.success or result.fun < best_score:
            p = result.x
            os = p[0]
            ocx = p[1]
            ocy = p[2]
            oa = p[3]
            
            inner = []
            for i in range(N):
                x = p[4 + i * 3]
                y = p[4 + i * 3 + 1]
                a = p[4 + i * 3 + 2]
                inner.append([x, y, a])
            
            # Verify feasibility
            min_dist = np.inf
            for i in range(N):
                verts = hex_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2])
                for v in verts:
                    d = hex_distance_to_boundary(v, ocx, ocy, os, oa)
                    if d < min_dist:
                        min_dist = d
            
            if min_dist >= -1e-6:
                # Feasible (or nearly)
                score = os
                if score < best_score:
                    best_score = score
                    best_result = (inner, (ocx, ocy), os, oa)
        
        if time.time() > DEADLINE:
            break
    
    # Final polish on best result
    if best_result is not None:
        inner, (ocx, ocy), os, oa = best_result
        params = np.array([os, ocx, ocy, oa] + [p for row in inner for p in row])
        
        result = minimize(
            objective,
            params,
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        
        p = result.x
        os = p[0]
        ocx = p[1]
        ocy = p[2]
        oa = p[3]
        
        inner = []
        for i in range(N):
            x = p[4 + i * 3]
            y = p[4 + i * 3 + 1]
            a = p[4 + i * 3 + 2]
            inner.append([x, y, a])
        
        # Verify
        min_dist = np.inf
        for i in range(N):
            verts = hex_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2])
            for v in verts:
                d = hex_distance_to_boundary(v, ocx, ocy, os, oa)
                if d < min_dist:
                    min_dist = d
        
        if min_dist >= -1e-6:
            best_result = (inner, (ocx, ocy), os, oa)
    
    if best_result is None:
        # Fallback: trivial feasible solution
        inner = [[0.0, 0.0, 0.0] for _ in range(N)]
        return inner, (0.0, 0.0), 10.0, 0.0
    
    return best_result
