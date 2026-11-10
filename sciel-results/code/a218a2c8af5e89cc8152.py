import numpy as np
import time
from scipy.optimize import minimize

def _hex_vertices(cx, cy, side, angle_deg):
    a = np.radians(angle_deg)
    angles = a + np.arange(6) * np.pi / 3
    return np.column_stack([
        cx + side * np.cos(angles),
        cy + side * np.sin(angles)
    ])

def _get_all_vertices(positions, angles, side=1.0):
    """positions: (N,2), angles: (N,) in degrees. Returns (N*6, 2)."""
    N = len(positions)
    verts = np.zeros((N * 6, 2))
    for i in range(N):
        verts[i*6:(i+1)*6] = _hex_vertices(positions[i,0], positions[i,1], side, angles[i])
    return verts

def _min_outer_side(vertices):
    """
    Compute minimum side length s of a regular hexagon centered at origin (angle 0)
    that contains all given vertices.
    """
    outward_normals = np.array([
        [np.cos(np.radians(30 + 60*k)), np.sin(np.radians(30 + 60*k))]
        for k in range(6)
    ])
    dots = vertices @ outward_normals.T
    max_dot = np.max(dots, axis=1)
    apothem_min = np.max(max_dot)
    s_min = 2.0 * apothem_min / np.sqrt(3.0)
    return s_min

def _objective(params):
    """
    Objective function for scipy.optimize.minimize.
    params: flattened array of length 3*N (x, y, angle_deg for each hexagon)
    Returns: s_outer + penalty
    """
    N = 12
    pos = params[0:2*N].reshape(N, 2)
    ang = params[2*N:3*N]
    
    # 1. Containment Penalty: Vertices must be inside the "current" outer hexagon
    # We estimate the required outer hexagon size based on current positions.
    # To make this differentiable and guide the optimizer, we compute the max distance
    # from origin for each vertex and compare to a dynamic threshold.
    # Actually, the standard approach is to minimize the max distance (or sum of squares of distances)
    # subject to separation.
    # Let's use a soft penalty for containment in a hexagon of side s_current.
    # But s is determined by the positions.
    # Better: Minimize the max distance from center (proxy for s) + overlap penalty.
    
    verts = _get_all_vertices(pos, ang, 1.0)
    
    # Calculate s needed to contain these vertices
    s_needed = _min_outer_side(verts)
    
    # 2. Overlap Penalty: 
    # For each pair, check if they overlap.
    # A smooth approximation to the overlap area or penetration depth.
    # Penetration depth is hard to compute differentiable.
    # We can use the sum of squared distances between vertices if they are "close", 
    # or use a soft constraint on the distance between centers vs the minimum separation distance.
    # For hexagons of side 1, the minimum distance between centers for non-overlap is 2 (if aligned) 
    # but can be less if rotated. The maximum distance for guaranteed overlap is 0.
    # The "inradius" is sqrt(3)/2. The "circumradius" is 1.
    # A robust soft penalty: sum over pairs of max(0, (d_min - dist)^2)
    # Where d_min is the minimum distance between boundaries.
    # Approximation: d_min ≈ dist_centers - 2 * inradius? No, that's for circles.
    # For hexagons, the "radius" varies with angle.
    # Let's use a simpler heuristic: if centers are closer than 2*inradius (sqrt(3)), 
    # they definitely overlap? No, they can interlock.
    # However, for a dense packing, centers will be close.
    # Let's use the sum of squared distances between all vertices of hex i and hex j 
    # that are "inside" each other? That's complex.
    
    # Alternative: Use the fact that if two convex polygons are separated, 
    # there is a separating axis. If they overlap, they don't.
    # We can compute the maximum penetration depth along the vector connecting centers?
    
    # Let's stick to a robust, vectorizable penalty:
    # Sum of squared "violations" of the minimum separation distance.
    # For two unit hexagons, the minimum distance between centers when they are just touching 
    # (face to face) is sqrt(3). When vertex to vertex it is 2.
    # So if dist < sqrt(3), they MUST overlap (since the inradius is sqrt(3)/2, 
    # two inradii sum to sqrt(3)).
    # If dist > 2, they are definitely separated.
    # Between sqrt(3) and 2, it depends on angle.
    
    # Let's use a penalty that pushes centers apart if they are too close.
    # Penalty = sum_{i<j} max(0, d_threshold - dist_ij)^2
    # Where d_threshold is a lower bound on the distance between centers for non-overlapping hexagons.
    # The minimum distance between centers of two non-overlapping unit hexagons is sqrt(3) 
    # (when faces are parallel and touching).
    # So if dist < sqrt(3), they overlap.
    # If we enforce dist >= sqrt(3) + epsilon, we guarantee no overlap?
    # Wait, if two hexagons are rotated relative to each other, can they be closer than sqrt(3) and not overlap?
    # No, the inradius is the minimum distance from center to edge. 
    # If dist < 2 * inradius = sqrt(3), then the centers are closer than the sum of inradii, 
    # so the hexagons MUST overlap (by the triangle inequality / convexity).
    # So, enforcing dist_ij >= sqrt(3) is a SUFFICIENT condition for non-overlap.
    # It is not necessary (they can be closer if rotated such that vertices point away), 
    # but it is a safe constraint that allows for valid packings, though potentially suboptimal.
    # To get a tighter packing, we need to allow dist < sqrt(3) if the angles permit.
    # However, for a first robust attempt, enforcing dist >= sqrt(3) ensures validity.
    # The optimal packing of 12 hexagons in a hexagon likely has some pairs at exactly sqrt(3) distance.
    
    # Let's refine: Use a soft penalty for dist < sqrt(3).
    # This guarantees validity if the penalty is large enough and the optimizer converges to 0 penalty.
    
    penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(pos[i] - pos[j])
            min_d = np.sqrt(3.0)
            if d < min_d:
                penalty += (min_d - d) ** 2
                
    # Also, we need to ensure they fit in the outer hexagon.
    # The objective is s_needed + lambda * penalty.
    # We want to minimize s_needed. The penalty pushes them apart to satisfy the constraint.
    # But if we just minimize s_needed, they will collapse to the center (d=0, penalty huge).
    # So we minimize s_needed + 100 * penalty.
    
    return s_needed + 100.0 * penalty

def _is_valid(positions, angles, s, tol=1e-6):
    """
    Check if the configuration is valid.
    1. All inner hexagons are separated (SAT).
    2. All inner vertices are inside the outer hexagon.
    """
    N = len(positions)
    
    # Check containment
    verts = _get_all_vertices(positions, angles, 1.0)
    # Outer hexagon vertices
    outer_verts = _hex_vertices(0, 0, s, 0.0)
    
    # Check if each vertex is inside outer hexagon
    # For a convex polygon defined by vertices in CCW order, a point is inside
    # if it is to the left of every edge (cross product >= 0).
    for v in verts:
        inside = True
        for k in range(6):
            p1 = outer_verts[k]
            p2 = outer_verts[(k+1)%6]
            cross = (p2[0] - p1[0]) * (v[1] - p1[1]) - (p2[1] - p1[1]) * (v[0] - p1[0])
            if cross < -tol:
                inside = False
                break
        if not inside:
            return False
            
    # Check separation using SAT
    for i in range(N):
        for j in range(i+1, N):
            v1 = _hex_vertices(positions[i,0], positions[i,1], 1.0, angles[i])
            v2 = _hex_vertices(positions[j,0], positions[j,1], 1.0, angles[j])
            
            normals = []
            for v in [v1, v2]:
                for k in range(6):
                    edge = v[(k+1)%6] - v[k]
                    n = np.array([-edge[1], edge[0]])
                    norm = np.linalg.norm(n)
                    if norm > 1e-12:
                        n = n / norm
                        normals.append(n)
                        
            separated = False
            for n in normals:
                proj1 = v1 @ n
                proj2 = v2 @ n
                min1, max1 = np.min(proj1), np.max(proj1)
                min2, max2 = np.min(proj2), np.max(proj2)
                if max1 < min2 - tol or max2 < min1 - tol:
                    separated = True
                    break
                    
            if not separated:
                return False
                
    return True

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 580.0
    
    N = 12
    best_score = -1.0
    best_result = None
    
    # Initial configurations
    # 1. Honeycomb grid
    # 2. Random
    # 3. Circle packing
    
    def init_honeycomb():
        pos = []
        ang = []
        for row in range(4):
            for col in range(3):
                x = col * 2 * 1.5
                y = row * 1.0 * np.sqrt(3)
                if row % 2 == 1:
                    x += 1.5
                pos.append([x, y])
                ang.append(0.0)
        pos = np.array(pos)
        ang = np.array(ang)
        centroid = np.mean(pos, axis=0)
        pos -= centroid
        return pos, ang

    def init_random():
        pos = np.random.uniform(-2, 2, (N, 2))
        ang = np.random.uniform(0, 60, N)
        centroid = np.mean(pos, axis=0)
        pos -= centroid
        return pos, ang

    def init_circle():
        # Place on a circle
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        r = 2.0
        pos = np.column_stack([r * np.cos(angles), r * np.sin(angles)])
        ang = np.zeros(N)
        return pos, ang

    inits = [init_honeycomb, init_random, init_circle]
    
    # Add more random inits if time permits
    for _ in range(5):
        inits.append(init_random)
        
    for init_func in inits:
        if time.time() > deadline - 60.0:
            break
            
        pos, ang = init_func()
        params = np.concatenate([pos.flatten(), ang])
        
        # Use L-BFGS-B
        try:
            res = minimize(
                _objective, 
                params, 
                method='L-BFGS-B', 
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
        except Exception:
            continue
            
        params_opt = res.x
        pos_opt = params_opt[0:2*N].reshape(N, 2)
        ang_opt = params_opt[2*N:3*N]
        
        # Calculate s for this configuration
        verts = _get_all_vertices(pos_opt, ang_opt, 1.0)
        s_calc = _min_outer_side(verts)
        
        # Validate
        if _is_valid(pos_opt, ang_opt, s_calc):
            score = 1.0 / s_calc
            if score > best_score:
                best_score = score
                best_result = (pos_opt, ang_opt, s_calc)
                
    if best_result is None:
        # Fallback to a known valid config (honeycomb)
        pos, ang = init_honeycomb()
        verts = _get_all_vertices(pos, ang, 1.0)
        s = _min_outer_side(verts)
        best_result = (pos, ang, s)
        best_score = 1.0 / s

    pos_final, ang_final, s_final = best_result
    
    inner_hex_data = np.column_stack([pos_final, ang_final])
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return inner_hex_data, outer_hex_data, s_final
