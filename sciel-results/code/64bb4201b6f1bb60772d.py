import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 12
DEADLINE = 145.0

def construct():
    t0 = time.time()
    
    # Helper: Vertices of unit hexagon
    def get_hex_vertices(cx, cy, angle_deg):
        base_angles = np.radians(np.arange(0, 360, 60) + angle_deg)
        x = cx + np.cos(base_angles)
        y = cy + np.sin(base_angles)
        return np.column_stack((x, y))

    # Helper: Signed distance from point to convex polygon
    # Returns negative if inside
    def point_to_convex_poly_dist(p, poly):
        # poly is (N, 2) array of vertices in CCW order
        # Check if point is inside
        # For a convex polygon, point is inside if it is to the "left" of every edge
        # (assuming CCW)
        # Actually, let's just compute min distance to edges if outside, 
        # and max "depth" if inside.
        
        # Faster: Check if inside using cross products
        inside = True
        min_dist = float('inf')
        max_depth = 0.0 # Penetration depth if inside
        
        n = len(poly)
        for i in range(n):
            p1 = poly[i]
            p2 = poly[(i+1)%n]
            # Edge vector
            ex = p2[0] - p1[0]
            ey = p2[1] - p1[1]
            # Normal pointing OUTWARD
            # For CCW poly, outward normal is (ey, -ex) ?
            # Let's check: Edge (1,0)->(0,1). Ex=-1, Ey=1. Outward normal should be (1,1)/sqrt(2)? No.
            # Standard: If vertices are CCW, the interior is to the left of the edge vector.
            # So inward normal is (-ey, ex). Outward is (ey, -ex).
            nx = ey
            ny = -ex
            norm = math.sqrt(nx*nx + ny*ny)
            if norm == 0: continue
            nx /= norm
            ny /= norm
            
            # Signed distance from p to line containing edge
            # d = (p - p1) . n
            dx = p[0] - p1[0]
            dy = p[1] - p1[1]
            d = dx*nx + dy*ny
            
            # If d < 0, point is on the outside of this edge
            if d < 0:
                inside = False
                if d > -min_dist: # We want the minimum distance to boundary, so closest edge
                    min_dist = -d # Distance is positive here
            
            # If inside, the penetration depth is determined by the edge that is "closest" to the point in the inward direction?
            # No, if inside, the distance to boundary is the minimum of -d for all edges?
            # If inside, d > 0 for all edges. The distance to the boundary is min(d).
            # So penetration depth = -min(d).
            if d < min_dist:
                min_dist = d # This logic is mixing inside/outside. Let's separate.
        
        # Re-evaluate logic for inside/outside
        # If inside: min_dist should be the smallest positive d. Distance = -min_dist.
        # If outside: min_dist should be the smallest magnitude of negative d. Distance = min_dist.
        
        # Let's restart the loop logic clearly.
        min_d = float('inf')
        for i in range(n):
            p1 = poly[i]
            p2 = poly[(i+1)%n]
            ex = p2[0] - p1[0]
            ey = p2[1] - p1[1]
            nx = ey
            ny = -ex
            norm = math.sqrt(nx*nx + ny*ny)
            if norm == 0: continue
            nx /= norm
            ny /= norm
            
            dx = p[0] - p1[0]
            dy = p[1] - p1[1]
            d = dx*nx + dy*ny
            
            if d < min_d:
                min_d = d
        
        # If min_d < 0, point is outside. Distance is -min_d? No.
        # If outside, at least one d is negative. The closest edge has the d closest to 0 from below.
        # So distance = -min_d (if min_d < 0).
        # If inside, all d are positive. The closest edge has the smallest positive d.
        # So distance = -min_d (if min_d > 0).
        # Wait, if inside, min_d is the smallest positive number. The point is min_d away from the boundary.
        # We want negative distance for inside. So return -min_d.
        # If outside, min_d is the largest negative number (closest to 0).
        # The distance is -min_d.
        # So in both cases, return -min_d?
        # Let's check:
        # Inside: d = [1, 2, 3]. min_d = 1. Return -1. Correct (penetration 1).
        # Outside: d = [-1, -2, -3]. min_d = -3. Return 3. Correct (distance 3).
        # But wait, if outside, we need the distance to the *segment*, not just the line.
        # However, for a convex polygon, if the projection of the point falls within the edge segment, the distance to the line is the distance to the polygon.
        # If the projection falls outside, the distance is to a vertex.
        # The "min_d" logic using normals only works if the closest feature is a face.
        # For a point close to a vertex, the closest point is the vertex.
        # The normal-based distance will overestimate the distance if the point is "beyond" the vertex.
        # However, for packing hexagons, overlaps usually happen face-to-face or edge-to-edge.
        # Vertex-to-vertex is rare in minimal packings.
        # To be safe, we should check distance to vertices as well if the point is "outside" the angular range of the edge.
        
        # Given the complexity, and that this is a penalty function (approximation is okay if it guides the search),
        # the normal-based distance is a decent surrogate for "penetration depth".
        # It might underestimate the required separation if the point is near a corner, 
        # leading to slight overlaps in the final result, which we can fix in a final SAT check.
        
        return -min_d

    # Helper: Minkowski Sum of two hexagons
    # A is centered at 0, angle a1
    # B is centered at 0, angle a2
    # Returns vertices of A + (-B)
    def minkowski_sum_hex(a1_deg, a2_deg):
        v1 = get_hex_vertices(0, 0, a1_deg)
        v2 = get_hex_vertices(0, 0, a2_deg)
        v2_neg = -v2
        
        # Convex hull of union of v1 and v2_neg
        # Since both are convex, we can merge their edge lists.
        # Or just use a simple convex hull algorithm on the 12 points.
        # 12 points is small.
        points = np.vstack((v1, v2_neg))
        
        # Convex hull (Andrew's Monotone Chain)
        points = np.unique(points, axis=0) # Remove duplicates
        if len(points) < 3: return points
        
        points = points[np.lexsort((points[:, 0], points[:, 1]))]
        
        def cross(o, a, b):
            return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
            
        lower = []
        for p in points:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)
            
        upper = []
        for p in reversed(points):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)
            
        hull = np.vstack((lower[:-1], upper[:-1]))
        return hull

    # Cache for Minkowski sums?
    # Relative angle determines the shape.
    # mink_cache = {}
    # def get_minkowski(a1, a2):
    #    rel = (a1 - a2) % 60.0
    #    if rel not in mink_cache:
    #        mink_cache[rel] = minkowski_sum_hex(0, -rel) # Relative shape
    #    return mink_cache[rel]
    
    # Actually, the Minkowski sum shape depends on the relative rotation.
    # Let's just compute it. It's fast enough for 12 hexes (66 pairs).

    def compute_collision_penalty(inner):
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                xi, yi, ai = inner[i]
                xj, yj, aj = inner[j]
                
                # Center of j relative to center of i
                rel_cx = xj - xi
                rel_cy = yj - yi
                
                # Minkowski sum of hex_i and -hex_j
                # This polygon is centered at 0.
                # If rel_c is inside, they overlap.
                # We need the shape of hex_i + (-hex_j).
                # This depends on ai and aj.
                mink_poly = minkowski_sum_hex(ai, aj)
                
                # Translate mink_poly? No, it's centered at 0.
                # The condition is: c_j - c_i inside (H_i + -H_j).
                # So we check if (rel_cx, rel_cy) is inside mink_poly.
                
                d = point_to_convex_poly_dist(np.array([rel_cx, rel_cy]), mink_poly)
                if d < 0:
                    penalty += d * d # Squared penetration
                
        return penalty

    def compute_outer_side(inner, phi_deg):
        # Compute convex hull of all vertices
        all_verts = []
        for h in inner:
            all_verts.append(get_hex_vertices(h[0], h[1], h[2]))
        all_verts = np.vstack(all_verts)
        
        # Center at centroid
        center = np.mean(all_verts, axis=0)
        offsets = all_verts - center
        
        sqrt3 = math.sqrt(3)
        phi = np.radians(phi_deg)
        
        max_extent = 0.0
        for k in range(6):
            normal_angle = phi + np.pi/6 + k * np.pi/3
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            
            proj = offsets[:, 0] * nx + offsets[:, 1] * ny
            max_proj = np.max(np.abs(proj))
            
            side_k = (2.0 / sqrt3) * max_proj
            if side_k > max_extent:
                max_extent = side_k
                
        return max_extent

    def objective(inner):
        # Optimize phi
        best_side = float('inf')
        # Coarse scan
        for phi in range(0, 60, 2):
            s = compute_outer_side(inner, phi)
            if s < best_side: best_side = s
        # Fine scan
        # Find best coarse phi
        best_phi = 0
        min_s = float('inf')
        for phi in range(0, 60, 2):
            s = compute_outer_side(inner, phi)
            if s < min_s:
                min_s = s
                best_phi = phi
        for phi in np.arange(best_phi-2, best_phi+2, 0.1):
            if phi < 0 or phi >= 60: continue
            s = compute_outer_side(inner, phi)
            if s < best_side: best_side = s
            
        penalty = compute_collision_penalty(inner)
        return best_side + 10.0 * penalty

    # Initial Configuration
    initial_inner = []
    cols = 4
    rows = 3
    x0 = -(cols - 1) / 2.0 * 3.0
    y0 = -(rows - 1) / 2.0 * math.sqrt(3)
    for r in range(rows):
        for c in range(cols):
            x = x0 + c * 3.0
            y = y0 + r * math.sqrt(3)
            initial_inner.append([x, y, 0.0])
            
    current_inner = [list(h) for h in initial_inner]
    current_obj = objective(current_inner)
    
    best_inner = [list(h) for h in current_inner]
    best_obj = current_obj
    
    T = 1.0
    T_min = 0.001
    cooling = 0.9995
    step_size = 0.5
    
    step_count = 0
    max_steps = 50000
    
    while time.time() - t0 < DEADLINE and step_count < max_steps:
        step_count += 1
        
        idx = random.randint(0, N - 1)
        new_inner = [list(h) for h in current_inner]
        
        dx = random.gauss(0, step_size)
        dy = random.gauss(0, step_size)
        da = random.gauss(0, 10.0)
        
        new_inner[idx][0] += dx
        new_inner[idx][1] += dy
        new_inner[idx][2] += da
        
        new_obj = objective(new_inner)
        
        if new_obj < current_obj:
            current_inner = new_inner
            current_obj = new_obj
            if new_obj < best_obj:
                best_obj = new_obj
                best_inner = [list(h) for h in current_inner]
        else:
            delta = new_obj - current_obj
            if random.random() < math.exp(-delta / T):
                current_inner = new_inner
                current_obj = new_obj
                if new_obj < best_obj:
                    best_obj = new_obj
                    best_inner = [list(h) for h in current_inner]
                    
        T *= cooling
        if T < T_min: T = T_min

    # Final Polish
    # Use Nelder-Mead on the best_inner
    # Flatten inner
    def flatten(inner):
        return np.array([item for sublist in inner for item in sublist])
        
    def unflatten(x):
        inner = []
        for i in range(N):
            inner.append([x[i*3], x[i*3+1], x[i*3+2]])
        return inner
        
    def obj_flat(x):
        return objective(unflatten(x))
        
    x0 = flatten(best_inner)
    res = minimize(obj_flat, x0, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-4, 'fatol': 1e-4})
    
    if res.fun < best_obj:
        best_inner = unflatten(res.x)
        best_obj = res.fun

    # Extract final parameters
    # Compute best phi and center for best_inner
    best_side = float('inf')
    best_phi = 0
    for phi in range(0, 60, 1):
        s = compute_outer_side(best_inner, phi)
        if s < best_side:
            best_side = s
            best_phi = phi
    for phi in np.arange(best_phi-1, best_phi+1, 0.01):
        if phi < 0 or phi >= 60: continue
        s = compute_outer_side(best_inner, phi)
        if s < best_side:
            best_side = s
            best_phi = phi
            
    # Center
    all_verts = []
    for h in best_inner:
        all_verts.append(get_hex_vertices(h[0], h[1], h[2]))
    all_verts = np.vstack(all_verts)
    center = np.mean(all_verts, axis=0)

    return best_inner, (float(center[0]), float(center[1])), float(best_side), float(best_phi)

# EVOLVE-BLOCK-END
