import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

N = 12

# Precompute unit hexagon vertices (pointy top)
# Angles: 30, 90, 150, 210, 270, 330 degrees
# Wait, standard "pointy top" has vertex at 90 deg.
# Let's use angles: np.pi/6 + k*np.pi/3
unit_hex_vertices = np.array([
    [np.cos(np.pi/6 + k*np.pi/3), np.sin(np.pi/6 + k*np.pi/3)] for k in range(6)
])

# Outer hexagon normals (perpendicular to edges)
# If vertices are at 30, 90..., edges are between them.
# Normal to edge between 30 and 90 is at 60 deg.
# Normals at: 0, 60, 120, 180, 240, 300?
# Let's check.
# Vertex 0: (cos30, sin30). Vertex 1: (cos90, sin90).
# Midpoint angle 60. Normal points out at 60.
# So normals are at 0, 60, 120, 180, 240, 300?
# Wait, vertex 0 is at 30 deg. Vertex 5 is at 330 deg.
# Edge 5-0 is at bottom? No.
# Let's just calculate normals from the vertices of the outer hexagon (which is same shape, scaled).
# Outer hexagon vertices: S * unit_hex_vertices.
# Edge i is between V[i] and V[(i+1)%6].
# Normal is perpendicular to edge, pointing out.
outer_normals = np.array([
    [np.cos(k*np.pi/3), np.sin(k*np.pi/3)] for k in range(6)
])
# Check: If vertex is at 30 deg, and next at 90 deg.
# Vector V1-V0. Normal should be at 60 deg.
# My normals are at 0, 60, 120...
# Yes, 60 is there.

def get_hex_vertices(center, angle, side=1.0):
    """Return vertices of a hexagon."""
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return center + (unit_hex_vertices * side).T @ rot.T

def point_in_hexagon_dist(point, center, side, angle=0.0):
    """
    Returns signed distance from point to hexagon boundary.
    Negative if inside, positive if outside.
    """
    # Translate point to local coords
    p = point - center
    # Rotate point to align with hexagon
    rot_inv = np.array([[np.cos(-angle), -np.sin(-angle)], [np.sin(-angle), np.cos(-angle)]])
    p_local = rot_inv @ p
    
    # Calculate distance to each of the 6 half-planes
    # Hexagon is intersection of 6 half-planes: n_i . p <= apothem
    apothem = np.sqrt(3) / 2 * side
    # Normals for local hexagon (aligned with axes)
    # If vertices are at 30, 90..., normals are at 0, 60...
    # Let's re-verify normals for the specific vertex set used.
    # Vertices: 30, 90, 150, 210, 270, 330.
    # Edge between 30 and 90. Midpoint 60. Normal 60.
    # Edge between 90 and 150. Midpoint 120. Normal 120.
    # So normals are at 0, 60, 120, 180, 240, 300?
    # Wait. Vertex 330 to Vertex 30. Midpoint 0. Normal 0.
    # Yes.
    dists = np.dot(outer_normals, p_local)
    max_dist = np.max(dists)
    return max_dist - apothem

def polygon_overlap_penalty(hex1_verts, hex2_verts):
    """
    Approximate overlap penalty between two convex polygons.
    Returns 0 if disjoint, >0 if overlapping.
    Uses min distance between vertices and edges.
    """
    min_dist = np.inf
    # Check distance from vertices of hex1 to edges of hex2
    for v in hex1_verts:
        # Distance to each edge of hex2
        for i in range(6):
            p1 = hex2_verts[i]
            p2 = hex2_verts[(i+1)%6]
            # Vector projection
            edge_vec = p2 - p1
            edge_len_sq = np.dot(edge_vec, edge_vec)
            if edge_len_sq < 1e-10: continue
            t = np.dot(v - p1, edge_vec) / edge_len_sq
            t = np.clip(t, 0, 1)
            proj = p1 + t * edge_vec
            d = np.linalg.norm(v - proj)
            if d < min_dist:
                min_dist = d
        # Check if v is inside hex2 (simple check: if all cross products have same sign)
        # Or just use point_in_hexagon_dist?
        # point_in_hexagon_dist gives distance to boundary. If inside, it's negative.
        # But we need to know if it's inside to determine sign.
        # Actually, if min_dist is small, we need to know if they overlap.
        # A simpler robust method for convex polygons:
        # If any vertex of A is inside B, or B is inside A, or edges intersect, they overlap.
        
    # This loop is slow. Let's use a simpler heuristic for the optimizer:
    # Distance between centers minus sum of circumradii is a lower bound on separation.
    # But it's not tight.
    
    # Better: Use the fact that hexagons are convex.
    # If the minimum distance between any vertex of A and any edge of B is > 0,
    # AND the minimum distance between any vertex of B and any edge of A is > 0,
    # AND no edges intersect, they are disjoint.
    
    # For the penalty function, we can approximate the "penetration depth" by:
    # max(0, (r1 + r2) - dist_centers) ? No.
    
    # Let's stick to the vertex-edge distance but optimize it.
    # We only need the penalty if they are likely to overlap.
    # dist_centers < 2 * circumradius (2.0) is a necessary condition for overlap?
    # Circumradius is 1. So if dist_centers > 2.0, they definitely don't overlap.
    # If dist_centers <= 2.0, we check carefully.
    
    # We pass centers as well to this function or calculate them outside.
    # Let's refactor: calculate centers first.
    
    return min_dist # This is just distance, not signed.

def objective(params):
    """
    params: [x1, y1, th1, ..., x12, y12, th12, S]
    """
    S = params[-1]
    if S < 3.0: S = 3.0
    
    penalty = 0.0
    centers = []
    angles = []
    vertices_list = []
    
    for i in range(N):
        x = params[i*3]
        y = params[i*3+1]
        th = params[i*3+2]
        center = np.array([x, y])
        centers.append(center)
        angles.append(th)
        verts = get_hex_vertices(center, th)
        vertices_list.append(verts)
        
        # Containment penalty
        for v in verts:
            d = point_in_hexagon_dist(v, np.array([0,0]), S)
            if d > 0:
                penalty += d**2
                
    # Overlap penalty
    for i in range(N):
        for j in range(i+1, N):
            c1 = centers[i]
            c2 = centers[j]
            dist_c = np.linalg.norm(c1 - c2)
            # Circumradius is 1. If dist > 2.0, no overlap possible.
            if dist_c > 2.0:
                continue
            
            # Check overlap
            # Simple check: if any vertex of i is inside j, or j inside i
            # Or edges intersect.
            # Approximation: min distance between vertices and edges.
            # We'll use a simplified "penetration" estimate:
            # If they are close, assume overlap depth is roughly (2.0 - dist_c) ?
            # This is inaccurate for rotated hexagons.
            
            # Let's do the exact vertex-edge check but only for pairs with dist < 2.0
            min_dist = np.inf
            v1s = vertices_list[i]
            v2s = vertices_list[j]
            
            # Vert 1 to Edge 2
            for v in v1s:
                for k in range(6):
                    p1 = v2s[k]
                    p2 = v2s[(k+1)%6]
                    edge_vec = p2 - p1
                    edge_len_sq = np.dot(edge_vec, edge_vec)
                    if edge_len_sq < 1e-10: continue
                    t = np.dot(v - p1, edge_vec) / edge_len_sq
                    t = np.clip(t, 0, 1)
                    proj = p1 + t * edge_vec
                    d = np.linalg.norm(v - proj)
                    if d < min_dist:
                        min_dist = d
            # Vert 2 to Edge 1
            for v in v2s:
                for k in range(6):
                    p1 = v1s[k]
                    p2 = v1s[(k+1)%6]
                    edge_vec = p2 - p1
                    edge_len_sq = np.dot(edge_vec, edge_vec)
                    if edge_len_sq < 1e-10: continue
                    t = np.dot(v - p1, edge_vec) / edge_len_sq
                    t = np.clip(t, 0, 1)
                    proj = p1 + t * edge_vec
                    d = np.linalg.norm(v - proj)
                    if d < min_dist:
                        min_dist = d
                        
            # If min_dist is 0, they might still be disjoint (touching) or overlapping.
            # To determine overlap, we need to know if they are "inside" each other.
            # If min_dist == 0 and no vertex is inside, they touch.
            # If a vertex is inside, they overlap.
            
            # Check if any vertex of 1 is inside 2
            overlap = False
            for v in v1s:
                if point_in_hexagon_dist(v, c2, 1.0, angles[j]) < 0:
                    overlap = True
                    break
            if not overlap:
                for v in v2s:
                    if point_in_hexagon_dist(v, c1, 1.0, angles[i]) < 0:
                        overlap = True
                        break
                        
            if overlap:
                # Estimate penetration depth
                # Use min_dist from the boundary? 
                # If overlapping, the "distance" we calculated is 0.
                # We need a penalty proportional to how deep the overlap is.
                # Approximation: (2.0 - dist_c) is a lower bound on penetration?
                # Let's use a fixed penalty if overlap is detected, scaled by (2.0 - dist_c).
                penalty += 10.0 * (2.0 - dist_c)**2
            else:
                # Disjoint, but maybe close. No penalty.
                pass
                
    return S + penalty

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin
    
    # Bounds: x, y in [-5, 5], theta in [0, pi/3], S in [3.0, 10.0]
    bounds = []
    for _ in range(N):
        bounds.append((-5.0, 5.0))
        bounds.append((-5.0, 5.0))
        bounds.append((0.0, np.pi/3))
    bounds.append((3.0, 10.0))
    
    # Heuristic initial point
    # Triangular lattice
    init = []
    idx = 0
    # 3 rows of 4
    for row in range(3):
        for col in range(4):
            x = (col - 1.5) * np.sqrt(3)
            y = (row - 1.0) * 1.5
            th = 0.0
            init.extend([x, y, th])
    init.append(5.0) # Initial S
    
    best_result = None
    best_score = np.inf
    
    # Run DE
    # maxiter limited by time
    # We'll run DE with a small maxiter and check time
    try:
        res = differential_evolution(
            objective, bounds, 
            maxiter=200, 
            popsize=15, 
            seed=42,
            init='sobol', # Good for high dim
            polish=True,
            x0=init
        )
        
        if res.success or res.fun < best_score:
            best_score = res.fun
            best_result = res.x
            
    except Exception as e:
        pass
        
    # Check time
    if time.time() > deadline:
        if best_result is None:
             return [[0.0, 0.0, 0.0] for _ in range(N)], (0.0, 0.0), 10.0, 0.0
        # Extract best
        inner = []
        for i in range(N):
            x = best_result[i*3]
            y = best_result[i*3+1]
            th = best_result[i*3+2]
            inner.append([x, y, np.degrees(th)])
        S = best_result[-1]
        return inner, (0.0, 0.0), S, 0.0

    # Polish with Nelder-Mead if time allows
    if best_result is not None:
        try:
            res_nm = minimize(
                objective, best_result, 
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6}
            )
            if res_nm.fun < best_score:
                best_score = res_nm.fun
                best_result = res_nm.x
        except Exception:
            pass
            
    if best_result is None:
        return [[0.0, 0.0, 0.0] for _ in range(N)], (0.0, 0.0), 10.0, 0.0
        
    inner = []
    for i in range(N):
        x = best_result[i*3]
        y = best_result[i*3+1]
        th = best_result[i*3+2]
        inner.append([x, y, np.degrees(th)])
    S = best_result[-1]
    return inner, (0.0, 0.0), S, 0.0

