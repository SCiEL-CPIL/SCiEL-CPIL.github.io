import math
import time
import random
import numpy as np
from itertools import combinations
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 145.0
    best_S = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_angle = 0.0
    best_valid = False

    # Precompute unit hex vertices (pointy-top relative to x-axis for standard math orientation)
    # Vertices at angles 0, 60, 120, 180, 240, 300
    base_verts = np.array([(math.cos(math.radians(60*i)), math.sin(math.radians(60*i))) for i in range(6)])

    def transform_verts(cx, cy, S, angle_deg):
        a = math.radians(angle_deg)
        ca, sa = math.cos(a), math.sin(a)
        # Rotation matrix
        rot = np.array([[ca, -sa], [sa, ca]])
        return (base_verts * S) @ rot.T + np.array([cx, cy])

    def point_in_polygon(p, poly):
        x, y = p
        inside = False
        j = len(poly) - 1
        for i in range(len(poly)):
            xi, yi = poly[i]; xj, yj = poly[j]
            if ((yi > y) != (yj > y)) and (x < (xj-xi)*(y-yi)/(yj-yi) + xi):
                inside = not inside
            j = i
        return inside

    def hexagons_disjoint(va, vb):
        # Fast check: bounding boxes
        min_xa = np.min(va[:, 0]); max_xa = np.max(va[:, 0])
        min_ya = np.min(va[:, 1]); max_ya = np.max(va[:, 1])
        min_xb = np.min(vb[:, 0]); max_xb = np.max(vb[:, 0])
        min_yb = np.min(vb[:, 1]); max_yb = np.max(vb[:, 1])
        if max_xa < min_xb or max_xb < min_xa or max_ya < min_yb or max_yb < min_ya:
            return True
        
        # Check vertex containment
        for v in va:
            if point_in_polygon(v, vb): return False
        for v in vb:
            if point_in_polygon(v, va): return False
            
        # Check edge intersections
        # For convex polygons, if no vertices are inside and bounding boxes overlap, 
        # we strictly need to check edge intersections.
        # However, for lattice-based packings where distance is >= sqrt(3), 
        # they are either touching or disjoint. 
        # Since we are generating lattice points, distance is exactly sqrt(3) for neighbors.
        # Touching is allowed. 
        # If they are not touching, distance > sqrt(3).
        # So for lattice points, if they are distinct, they are disjoint (interiors).
        # This check is mainly for safety if we ever perturb.
        for i in range(6):
            p1 = va[i]; p2 = va[(i+1)%6]
            for j in range(6):
                p3 = vb[j]; p4 = vb[(j+1)%6]
                if segments_intersect(p1, p2, p3, p4):
                    return False
        return True

    def segments_intersect(p1, p2, p3, p4):
        def cross(o, a, b):
            return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
        d1 = cross(p3, p4, p1); d2 = cross(p3, p4, p2)
        d3 = cross(p1, p2, p3); d4 = cross(p1, p2, p4)
        if ((d1>0 and d2<0) or (d1<0 and d2>0)) and ((d3>0 and d4<0) or (d3<0 and d4>0)):
            return True
        return False

    def compute_min_enclosing_hex_size(inner_verts_list, outer_center, outer_angle):
        # inner_verts_list: list of (N, 2) arrays
        # outer_center: (x, y)
        # outer_angle: degrees
        # Returns minimal side length S such that all inner hexes fit inside outer hex centered at outer_center with rotation outer_angle.
        
        # The outer hexagon is defined by 6 half-planes.
        # For a regular hexagon of side S centered at C with rotation A, the apothem is h = S * sqrt(3)/2.
        # The supporting lines are at distance h from the center, with normals at angles A + 30 + 60*k.
        # A point P is inside if dot(P - C, n_k) <= h for all k.
        # So h_min = max over all P, k of dot(P - C, n_k).
        # S_min = h_min / (sqrt(3)/2).
        
        cx, cy = outer_center
        a = math.radians(outer_angle)
        max_h = 0.0
        
        # Normals to the sides of the outer hexagon
        # Side normals are at 30, 90, 150, 210, 270, 330 degrees relative to the hex orientation.
        # If hex vertices are at 0, 60..., sides are at 30, 90...
        # Wait, standard pointy-top hex (vertices at 0, 60...) has flat sides at 90, 150...?
        # Let's check: Vertex at 0 is (1,0). Vertex at 60 is (0.5, sqrt(3)/2).
        # Midpoint of edge 0-60 is (0.75, sqrt(3)/4). Angle is atan2(sqrt(3)/4, 0.75) = 30 deg.
        # So normal to first edge is at 30 degrees.
        # Normals are at 30 + 60*i.
        
        for k in range(6):
            theta = a + math.radians(30 + 60*k)
            nx, ny = math.cos(theta), math.sin(theta)
            
            max_proj = 0.0
            for iv in inner_verts_list:
                # Project all vertices of this inner hex
                # p - C
                diff = iv - np.array([cx, cy])
                projs = diff[:, 0]*nx + diff[:, 1]*ny
                if np.max(projs) > max_proj:
                    max_proj = np.max(projs)
            
            if max_proj > max_h:
                max_h = max_proj
                
        return max_h / (math.sqrt(3) / 2)

    def all_fit(inner, ocx, ocy, S, oangle):
        outer_verts = transform_verts(ocx, ocy, S, oangle)
        inner_verts = [transform_verts(ix, iy, 1.0, ia) for ix, iy, ia in inner]
        # Check containment
        for iv in inner_verts:
            for v in iv:
                if not point_in_polygon(v, outer_verts):
                    return False
        # Check disjointness
        for i in range(len(inner_verts)):
            for j in range(i+1, len(inner_verts)):
                if not hexagons_disjoint(inner_verts[i], inner_verts[j]):
                    return False
        return True

    def update_best(inner, ocx, ocy, S, oangle):
        nonlocal best_S, best_inner, best_center, best_angle, best_valid
        if S < best_S and all_fit(inner, ocx, ocy, S, oangle):
            best_S = S
            best_inner = [list(x) for x in inner]
            best_center = (ocx, ocy)
            best_angle = oangle
            best_valid = True
            return True
        return False

    # --- STRATEGY: Discrete Lattice Enumeration ---
    # We use axial coordinates (q, r) for a hexagonal lattice.
    # Distance between centers of touching hexagons (side 1) is sqrt(3).
    # Pixel coordinates for pointy-top:
    # x = sqrt(3) * (q + r/2)
    # y = 3/2 * r
    # Wait, standard formula:
    # x = size * sqrt(3) * (q + r/2)
    # y = size * 3/2 * r
    # For size=1:
    # x = sqrt(3) * (q + r/2)
    # y = 1.5 * r
    
    def axial_to_pixel(q, r):
        return (math.sqrt(3) * (q + r/2.0), 1.5 * r)

    # Generate a pool of lattice points close to the origin.
    # We need 11 points. The minimal enclosing hexagon for 11 points will likely fit within a radius of ~3-4 lattice steps.
    # Let's generate all points with q^2 + r^2 + q*r <= R^2 (hex distance squared)
    # Hex distance squared = q^2 + r^2 + q*r
    # We'll generate points up to distance 4 or 5.
    
    lattice_points = []
    max_dist_sq = 25 # 5^2
    for q in range(-5, 6):
        for r in range(-5, 6):
            dist_sq = q*q + r*r + q*r
            if dist_sq <= max_dist_sq:
                px, py = axial_to_pixel(q, r)
                lattice_points.append((q, r, px, py))
                
    # Sort by distance to origin to prioritize central clusters
    lattice_points.sort(key=lambda p: p[2]**2 + p[3]**2)
    
    # We need to select 11 points.
    # Instead of checking all C(100, 11), we can restrict to "connected" or "compact" shapes.
    # A simple heuristic: The optimal packing for convex shapes in a convex container is often "compact".
    # We can try all subsets of the K closest points?
    # If we take the 15 closest points, C(15, 11) = 1365. That's small.
    # If we take 20 closest, C(20, 11) = 167960. That's manageable if evaluation is fast.
    # Let's try K=18 closest points. C(18, 11) = 31824.
    
    K = 18
    candidates = lattice_points[:K]
    
    # Precompute vertex arrays for these candidates to speed up evaluation
    cand_verts = [transform_verts(px, py, 1.0, 0.0) for q, r, px, py in candidates]
    
    # To optimize outer rotation, we can sample a few angles or optimize.
    # Given the symmetry of the lattice, the optimal outer hexagon rotation is likely aligned with the lattice (0 or 30 degrees) or offset.
    # Let's test rotations: 0, 10, 20, 30, 40, 50 degrees.
    # Or better: for a specific configuration, the optimal rotation minimizes the max projection.
    # This is a 1D optimization problem for angle.
    
    def optimize_outer_angle(inner_verts_list, center):
        # Minimize S(angle)
        def S_of_angle(angle):
            return compute_min_enclosing_hex_size(inner_verts_list, center, angle)
        
        # Sample 12 angles first to find a good basin
        best_a = 0.0
        best_s = float('inf')
        for a in range(0, 60, 5):
            s = S_of_angle(a)
            if s < best_s:
                best_s = s
                best_a = a
        
        # Refine with Nelder-Mead
        try:
            res = minimize(S_of_angle, [best_a], method='Nelder-Mead', options={'xatol': 1e-4, 'fatol': 1e-4, 'maxiter': 50})
            if res.fun < best_s:
                best_a = res.x[0] % 60.0
                best_s = res.fun
        except:
            pass
        
        return best_s, best_a

    # Enumerate subsets
    # We will iterate through combinations of indices from the candidate list
    # To ensure we don't miss asymmetric but compact shapes, we rely on the "closest points" heuristic.
    # The 11 closest points in a hex lattice form a specific shape (a larger hexagon with corners cut or similar).
    
    # Let's be smarter: 
    # 1. Evaluate the "standard" compact shapes.
    # 2. Do a random search over subsets of a larger pool if time permits.
    
    # First, evaluate all subsets of the 15 closest points.
    # This covers all "very compact" configurations.
    K_small = 15
    pool_small = lattice_points[:K_small]
    verts_small = [transform_verts(px, py, 1.0, 0.0) for q, r, px, py in pool_small]
    
    for combo in combinations(range(K_small), N):
        if time.time() > deadline - 20:
            break
        verts_subset = [verts_small[i] for i in combo]
        
        # Center of mass of the configuration
        all_pts = np.vstack(verts_subset)
        cx = np.mean(all_pts[:, 0])
        cy = np.mean(all_pts[:, 1])
        
        # Translate so COM is at origin for symmetry, then optimize outer hex centered at origin
        # Actually, the outer hex center doesn't have to be the COM, but for symmetric packings it is.
        # Let's assume outer center is (0,0) and inner hexes are shifted to be centered around origin.
        # Shift inner hexes
        shifted_verts = [v - np.array([cx, cy]) for v in verts_subset]
        
        S, angle = optimize_outer_angle(shifted_verts, (0.0, 0.0))
        
        # Verify fit
        inner = []
        for i, v in enumerate(shifted_verts):
            # Recover original relative positions
            # v is shifted. Original was pool_small[combo[i]]
            # We need to return the positions relative to the OUTER hex center (0,0)
            # So the position is just the center of the shifted hex
            # The center of the shifted hex is (pool_small[combo[i]][2]-cx, pool_small[combo[i]][3]-cy)
            px, py = pool_small[combo[i]][2] - cx, pool_small[combo[i]][3] - cy
            inner.append([px, py, 0.0])
            
        if update_best(inner, 0.0, 0.0, S, angle):
            pass # Updated best

    # If we have time, do a broader search with random subsets of a larger pool
    # This helps if the optimal shape is not one of the 15 closest points (e.g. slightly elongated)
    K_large = 25
    pool_large = lattice_points[:K_large]
    verts_large = [transform_verts(px, py, 1.0, 0.0) for q, r, px, py in pool_large]
    
    # Random sampling of subsets
    rng = random.Random(42)
    attempts = 0
    while time.time() < deadline - 10 and attempts < 500:
        attempts += 1
        # Pick 11 random distinct indices
        idxs = rng.sample(range(K_large), N)
        verts_subset = [verts_large[i] for i in idxs]
        
        all_pts = np.vstack(verts_subset)
        cx = np.mean(all_pts[:, 0])
        cy = np.mean(all_pts[:, 1])
        
        shifted_verts = [v - np.array([cx, cy]) for v in verts_subset]
        
        # Quick estimate: bounding box size
        min_x = np.min(all_pts[:, 0]); max_x = np.max(all_pts[:, 0])
        min_y = np.min(all_pts[:, 1]); max_y = np.max(all_pts[:, 1])
        # If bbox is huge, skip detailed optimization
        if (max_x - min_x) > 12 or (max_y - min_y) > 12:
            continue
            
        S, angle = optimize_outer_angle(shifted_verts, (0.0, 0.0))
        
        inner = []
        for i, idx in enumerate(idxs):
            px, py = pool_large[idx][2] - cx, pool_large[idx][3] - cy
            inner.append([px, py, 0.0])
            
        if update_best(inner, 0.0, 0.0, S, angle):
            pass

    if best_valid and best_inner is not None:
        return best_inner, best_center, best_S, best_angle

    # Fallback: Linear array (worst case)
    sp = math.sqrt(3)
    inner = [[(k - N//2)*sp, 0.0, 0.0] for k in range(N)]
    S = compute_min_enclosing_hex_size([transform_verts(p[0], p[1], 1.0, 0.0) for p in inner], (0,0), 0)
    return inner, (0.0, 0.0), S, 0.0
# EVOLVE-BLOCK-END
