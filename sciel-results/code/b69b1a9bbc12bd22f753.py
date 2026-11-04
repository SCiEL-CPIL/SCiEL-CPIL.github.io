import math
import time
import random
import numpy as np
from scipy.optimize import differential_evolution

N = 12

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0

    # Lattice vectors for hexagonal packing
    # Distance between centers of touching unit hexagons is sqrt(3)
    # We add a small epsilon to ensure strict disjointness
    eps = 0.01
    d = math.sqrt(3) + eps
    
    u = np.array([d, 0.0])
    v = np.array([d/2.0, d*math.sqrt(3)/2.0])
    
    # Ring 0
    centers = [np.array([0.0, 0.0])]
    
    # Ring 1
    # Neighbors of center
    ring1_vectors = [
        u,
        v,
        v - u,
        -u,
        -v,
        u - v
    ]
    for vec in ring1_vectors:
        centers.append(vec.copy())
        
    # Ring 2 Candidates
    # Neighbors of Ring 1 that are not in Ring 0 or 1
    ring2_candidates = []
    # For each ring1 center, find its 6 neighbors
    # A neighbor of P is P +/- u, P +/- v, P +/- (v-u)
    basis = [u, v, v-u, -u, -v, u-v]
    
    for i in range(6):
        p1 = ring1_vectors[i]
        for b in basis:
            p2 = p1 + b
            # Check if p2 is in Ring 0 or Ring 1
            is_r0 = np.allclose(p2, np.array([0.0, 0.0]))
            is_r1 = False
            for p1j in ring1_vectors:
                if np.allclose(p2, p1j):
                    is_r1 = True
                    break
            if not is_r0 and not is_r1:
                # Avoid duplicates in candidates list
                duplicate = False
                for pc in ring2_candidates:
                    if np.allclose(p2, pc):
                        duplicate = True
                        break
                if not duplicate:
                    ring2_candidates.append(p2)
                    
    # Sort Ring 2 candidates by distance to origin
    ring2_candidates.sort(key=lambda p: np.linalg.norm(p))
    
    # Pick 5 closest
    for i in range(5):
        centers.append(ring2_candidates[i].copy())
        
    # Now we have 12 centers
    # Verify count
    assert len(centers) == 12
    
    # Convert to list of [x, y, angle]
    # Rotate hexagons by 30 degrees so they are 'flat-top' relative to the x-axis,
    # allowing them to touch at distance sqrt(3) along the x-axis.
    inner = [[c[0], c[1], 30.0] for c in centers]
    
    # Function to get vertices of a hexagon
    def get_hex_vertices(cx, cy, side, angle_deg):
        angle = math.radians(angle_deg)
        vertices = []
        for i in range(6):
            theta = angle + i * math.pi / 3
            x = cx + side * math.cos(theta)
            y = cy + side * math.sin(theta)
            vertices.append((x, y))
        return vertices

    # Collect all vertices
    all_verts = []
    for x, y, a in inner:
        verts = get_hex_vertices(x, y, 1.0, a)
        all_verts.extend(verts)
        
    # Find minimum outer hexagon side
    # The outer hexagon is centered at the centroid of the inner cluster?
    # Or we can optimize the center too.
    # For symmetric clusters, the centroid is a good guess.
    cx = sum(p[0] for p in all_verts) / len(all_verts)
    cy = sum(p[1] for p in all_verts) / len(all_verts)
    
    best_side = float('inf')
    best_angle = 0.0
    
    # Scan outer hexagon angle
    # The hexagon is defined by 6 half planes.
    # Normal angles: theta + 30 + 60*k
    # Condition: dot(p - c, n) <= S * sqrt(3) / 2
    # S >= 2/sqrt(3) * max(dot(p-c, n))
    
    for theta_deg in np.linspace(0, 60, 61):
        theta = math.radians(theta_deg)
        max_proj = 0.0
        for k in range(6):
            normal_angle = theta + math.pi/6 + k * math.pi/3
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            for px, py in all_verts:
                proj = (px - cx) * nx + (py - cy) * ny
                if proj > max_proj:
                    max_proj = proj
        s = 2.0 * max_proj / math.sqrt(3)
        if s < best_side:
            best_side = s
            best_angle = theta_deg
            
    outer_center = (cx, cy)
    outer_side = best_side
    outer_angle = best_angle
    
    # Now try to improve with DE optimization
    # Create objective function
    def objective(params):
        # params: 36 values = 12*(x, y, angle_deg)
        # Extract positions and angles
        positions = []
        angles = []
        for i in range(12):
            x = params[i*3]
            y = params[i*3+1]
            angle = params[i*3+2]
            positions.append((x, y))
            angles.append(angle)
        
        # Get all vertices
        all_v = []
        for i in range(12):
            verts = get_hex_vertices(positions[i][0], positions[i][1], 1.0, angles[i])
            all_v.extend(verts)
        
        # Compute centroid
        cx_opt = sum(p[0] for p in all_v) / len(all_v)
        cy_opt = sum(p[1] for p in all_v) / len(all_v)
        
        # Find minimum enclosing hexagon side
        best_s = float('inf')
        for theta_deg in np.linspace(0, 60, 13):  # 13 samples for speed
            theta = math.radians(theta_deg)
            max_proj = 0.0
            for k in range(6):
                normal_angle = theta + math.pi/6 + k * math.pi/3
                nx = math.cos(normal_angle)
                ny = math.sin(normal_angle)
                for px, py in all_v:
                    proj = (px - cx_opt) * nx + (py - cy_opt) * ny
                    if proj > max_proj:
                        max_proj = proj
            s = 2.0 * max_proj / math.sqrt(3)
            if s < best_s:
                best_s = s
        
        # Check for overlaps using SAT
        overlap_penalty = 0.0
        for i in range(12):
            for j in range(i+1, 12):
                # Get vertices of hex i and j
                verts_i = get_hex_vertices(positions[i][0], positions[i][1], 1.0, angles[i])
                verts_j = get_hex_vertices(positions[j][0], positions[j][1], 1.0, angles[j])
                
                # SAT test
                overlapping = True
                for verts in [verts_i, verts_j]:
                    for k in range(6):
                        p1 = verts[k]
                        p2 = verts[(k+1) % 6]
                        # Edge vector
                        ex = p2[0] - p1[0]
                        ey = p2[1] - p1[1]
                        # Normal
                        nx = -ey
                        ny = ex
                        norm = math.sqrt(nx*nx + ny*ny)
                        if norm < 1e-10:
                            continue
                        nx /= norm
                        ny /= norm
                        
                        # Project both hexagons
                        min_i = float('inf')
                        max_i = float('-inf')
                        for px, py in verts_i:
                            proj = px * nx + py * ny
                            if proj < min_i: min_i = proj
                            if proj > max_i: max_i = proj
                        
                        min_j = float('inf')
                        max_j = float('-inf')
                        for px, py in verts_j:
                            proj = px * nx + py * ny
                            if proj < min_j: min_j = proj
                            if proj > max_j: max_j = proj
                        
                        # Check if projections overlap
                        if max_i <= min_j or max_j <= min_i:
                            overlapping = False
                            break
                    if not overlapping:
                        break
                if overlapping:
                    overlap_penalty += 100.0
        
        return best_s + overlap_penalty
    
    # Define bounds
    bounds = []
    for i in range(12):
        bounds.append((-5, 5))  # x
        bounds.append((-5, 5))  # y
        bounds.append((0, 60))  # angle (0-60 degrees is sufficient due to symmetry)
    
    # Initial population from lattice solution
    init_params = []
    for i in range(12):
        init_params.append(centers[i][0])
        init_params.append(centers[i][1])
        init_params.append(30.0)
    
    # Run DE if time permits
    remaining_time = deadline - time.time()
    if remaining_time > 30:  # Need at least 30s for DE
        try:
            result = differential_evolution(
                objective, 
                bounds, 
                maxiter=50, 
                popsize=15,
                tol=1e-4,
                seed=42,
                init='latinhypercube'
            )
            if result.fun < outer_side + 1e-6:
                # Update best solution
                new_params = result.x
                inner = []
                for i in range(12):
                    inner.append([new_params[i*3], new_params[i*3+1], new_params[i*3+2]])
                
                # Recompute outer hexagon
                all_verts = []
                for x, y, a in inner:
                    verts = get_hex_vertices(x, y, 1.0, a)
                    all_verts.extend(verts)
                
                cx = sum(p[0] for p in all_verts) / len(all_verts)
                cy = sum(p[1] for p in all_verts) / len(all_verts)
                
                best_side = float('inf')
                best_angle = 0.0
                for theta_deg in np.linspace(0, 60, 61):
                    theta = math.radians(theta_deg)
                    max_proj = 0.0
                    for k in range(6):
                        normal_angle = theta + math.pi/6 + k * math.pi/3
                        nx = math.cos(normal_angle)
                        ny = math.sin(normal_angle)
                        for px, py in all_verts:
                            proj = (px - cx) * nx + (py - cy) * ny
                            if proj > max_proj:
                                max_proj = proj
                    s = 2.0 * max_proj / math.sqrt(3)
                    if s < best_side:
                        best_side = s
                        best_angle = theta_deg
                
                outer_center = (cx, cy)
                outer_side = best_side
                outer_angle = best_angle
        except Exception:
            pass
    
    return inner, outer_center, outer_side, outer_angle
# EVOLVE-BLOCK-END
