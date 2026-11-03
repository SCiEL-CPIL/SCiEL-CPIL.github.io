import math
import time
import numpy as np

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    # Precompute unit hexagon vertices for angle 0 (pointy-top, vertex at 0 deg)
    # Vertices at 0, 60, 120, 180, 240, 300 degrees
    unit_verts_0 = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])

    def get_hex_verts(cx, cy, side, angle_deg):
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        verts = unit_verts_0 @ rot.T * side
        verts += np.array([cx, cy])
        return verts

    def calc_min_outer_side(positions, angles, outer_angle=0.0):
        """
        Calculate the minimum side length of a regular hexagon centered at (0,0) 
        with rotation outer_angle that contains all inner hexagons.
        """
        theta_rad = math.radians(outer_angle)
        normals = []
        for k in range(6):
            ang = theta_rad + math.pi/6 + k * math.pi/3
            normals.append([math.cos(ang), math.sin(ang)])
        normals = np.array(normals) # (6, 2)
        
        max_proj = 0.0
        
        for i in range(N):
            cx, cy = positions[i]
            ang = angles[i]
            verts = get_hex_verts(cx, cy, 1.0, ang)
            projs = verts @ normals.T # (6, 6)
            max_proj_i = np.max(projs)
            if max_proj_i > max_proj:
                max_proj = max_proj_i
        
        return max_proj * 2.0 / math.sqrt(3)

    def check_disjoint_fast(poly1, poly2):
        """
        Fast check for disjointness using SAT.
        Returns True if disjoint (or touching), False if overlapping.
        """
        def project_polygon(poly, axis):
            proj = poly @ axis
            return np.min(proj), np.max(proj)
            
        axes = []
        for i in range(6):
            # Edge 1
            edge1 = poly1[(i+1)%6] - poly1[i]
            normal1 = np.array([-edge1[1], edge1[0]])
            axes.append(normal1)
            # Edge 2
            edge2 = poly2[(i+1)%6] - poly2[i]
            normal2 = np.array([-edge2[1], edge2[0]])
            axes.append(normal2)
            
        for axis in axes:
            norm = np.linalg.norm(axis)
            if norm < 1e-9: continue
            axis = axis / norm
            min1, max1 = project_polygon(poly1, axis)
            min2, max2 = project_polygon(poly2, axis)
            if max1 < min2 or max2 < min1:
                return True # Separating axis found, disjoint
                
        return False # No separating axis, overlapping

    # --- PBD Simulation Logic ---
    
    # State: positions (N, 2), angles (N,)
    # We will simulate N rigid hexagons.
    
    # Initial guess: Honeycomb lattice
    sqrt3 = math.sqrt(3)
    lattice_points = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            x = i * 1.5
            y = i * sqrt3 / 2.0 + j * sqrt3
            lattice_points.append((x, y))
            
    lattice_points.sort(key=lambda p: p[0]**2 + p[1]**2)
    init_positions = np.array(lattice_points[:12])
    init_positions -= np.mean(init_positions, axis=0)
    init_angles = np.zeros(N)
    
    # We will try a few different initial configurations and boundary shrinkage rates
    best_score = float('inf')
    best_final_state = None
    
    # Candidate initial states
    initial_states = []
    # 1. Honeycomb
    initial_states.append((init_positions.copy(), init_angles.copy()))
    
    # 2. Circle packing (rough)
    for r in [1.5, 2.0, 2.5]:
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        # Place in a ring
        for i in range(N):
            theta = 2 * math.pi * i / N
            pos[i] = [r * math.cos(theta), r * math.sin(theta)]
        initial_states.append((pos.copy(), ang.copy()))
        
    # 3. Random jittered honeycomb
    np.random.seed(42)
    for _ in range(3):
        pos = init_positions.copy()
        pos += np.random.normal(0, 0.5, (N, 2))
        ang = np.random.uniform(-10, 10, N)
        initial_states.append((pos.copy(), ang.copy()))

    for init_pos, init_ang in initial_states:
        if time.time() > deadline - 20:
            break
            
        positions = init_pos.copy()
        angles = init_ang.copy()
        
        # PBD Parameters
        num_iterations = 2000
        num_substeps = 10 # Iterations per shrinkage step
        damping = 0.9
        
        # Start with a large boundary and shrink it
        # Initial boundary side length
        current_side = calc_min_outer_side(positions, angles, 0.0) * 1.2
        target_side = 3.0 # Lower bound guess, will be refined
        
        # We will shrink the boundary from current_side down to a small value
        # The "pressure" comes from the boundary constraint
        
        for iter_idx in range(num_iterations):
            if time.time() > deadline - 5:
                break
                
            # Shrink boundary
            # Linear shrinkage towards a lower bound, but we don't know the exact bound.
            # Let's shrink towards the current min possible side length of the cluster?
            # No, PBD works by projecting out of a container.
            # We want to minimize the container size.
            # Strategy: Start with a large container, shrink it slowly.
            # If the hexagons can fit, they stay inside. If not, they get pushed out (which increases size).
            # So we shrink the target size, and let the hexagons settle.
            
            # Simple strategy: Shrink the boundary side length by a small factor each step
            shrink_factor = 0.995
            current_side *= shrink_factor
            
            # Perform substeps of PBD
            for _ in range(num_substeps):
                # 1. Compute new positions (velocity update is skipped for simplicity, we use position only)
                # Actually, standard PBD updates position based on constraint satisfaction.
                
                # 2. Satisfy constraints
                # a. Boundary Constraint: All vertices must be inside the hexagon of side 'current_side'
                # b. Non-penetration: No two hexagons overlap.
                
                # We will iterate vertex projections.
                # This is expensive if done naively. 
                # Optimization: Only check pairs that are close.
                
                # For each hexagon, get vertices
                verts_list = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
                
                # Boundary Projection
                # Normals for boundary
                normals = np.array([[math.cos(math.radians(30 + k*60)), math.sin(math.radians(30 + k*60))] for k in range(6)])
                # Support distance for unit hexagon is 1.0 (circumradius).
                # The hexagon must be inside the boundary.
                # Condition: max( (v - center) . n ) <= side * sqrt(3)/2 - 1.0 ?
                # No, the distance from center to edge of boundary is apothem = side * sqrt(3)/2.
                # The distance from center of inner hex to its farthest vertex in direction n is support function.
                # For a regular hexagon of side 1, support function h(n) depends on angle.
                # Max projection of unit hexagon onto normal n is:
                # If n is aligned with a vertex normal (30, 90, ...), h = 1.0
                # If n is aligned with an edge normal (0, 60, ...), h = sqrt(3)/2
                # Actually, the support function of a regular hexagon of side 1 is:
                # h(theta) = max(cos(theta), cos(60-theta), cos(120-theta)...)
                # It's easier to just project vertices.
                
                # We want: max( (v) . n ) <= apothem_boundary
                # where v are vertices of the inner hex.
                # If violated, we must move the center of the inner hex in direction -n.
                
                for i in range(N):
                    verts = verts_list[i]
                    # Check each boundary normal
                    for n in normals:
                        proj = np.max(verts @ n)
                        limit = current_side * math.sqrt(3) / 2.0
                        if proj > limit:
                            # Push back
                            delta = proj - limit
                            # Move center in direction -n
                            positions[i] -= delta * n
                            # Update verts
                            verts_list[i] = get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i])
                            
                # Non-penetration Projection
                # For each pair (i, j), if overlapping, push apart.
                # We can use a simple heuristic: find the minimum translation vector (MTV) or just push along center-to-center.
                # Center-to-center is robust but not optimal.
                # Let's try to find a separating axis and push along it.
                
                for i in range(N):
                    for j in range(i+1, N):
                        # Quick check: distance between centers
                        d_center = np.linalg.norm(positions[i] - positions[j])
                        # Min distance for non-overlap of two unit hexagons aligned is sqrt(3) ~ 1.732
                        # With rotation, it can be closer? No, circumradius is 1.
                        # If d_center < 2.0, they might overlap.
                        if d_center < 2.5:
                            poly1 = verts_list[i]
                            poly2 = verts_list[j]
                            
                            # Check SAT
                            axes = []
                            for k in range(6):
                                edge1 = poly1[(k+1)%6] - poly1[k]
                                normal1 = np.array([-edge1[1], edge1[0]])
                                axes.append(normal1)
                                edge2 = poly2[(k+1)%6] - poly2[k]
                                normal2 = np.array([-edge2[1], edge2[0]])
                                axes.append(normal2)
                            
                            min_overlap = float('inf')
                            min_axis = None
                            
                            for axis in axes:
                                norm = np.linalg.norm(axis)
                                if norm < 1e-9: continue
                                axis = axis / norm
                                
                                min1, max1 = np.min(poly1 @ axis), np.max(poly1 @ axis)
                                min2, max2 = np.min(poly2 @ axis), np.max(poly2 @ axis)
                                
                                if max1 < min2 or max2 < min1:
                                    break # Disjoint
                                
                                # Calculate overlap depth
                                overlap = min(max1, max2) - max(min1, min2)
                                if overlap < min_overlap:
                                    min_overlap = overlap
                                    # Ensure axis points from i to j
                                    center_diff = positions[j] - positions[i]
                                    if np.dot(center_diff, axis) < 0:
                                        axis = -axis
                                    min_axis = axis
                            
                            if min_overlap > 0 and min_axis is not None:
                                # Push i and j apart by min_overlap / 2 each
                                # But we must respect the direction.
                                # min_axis points from i to j.
                                # So i should move by -min_axis * min_overlap/2
                                # j should move by min_axis * min_overlap/2
                                push = min_overlap * 0.5
                                positions[i] -= min_axis * push
                                positions[j] += min_axis * push
                                
                                # Update verts
                                verts_list[i] = get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i])
                                verts_list[j] = get_hex_verts(positions[j][0], positions[j][1], 1.0, angles[j])

                # Damping / Relaxation
                # In PBD, we usually update velocities. Here we just adjust positions.
                # To prevent oscillation, we could blend with previous positions, but simple projection is often stable enough for this.
                
            # End substeps
            
            # Check validity and score
            # We need to ensure the final state is disjoint.
            # The projection loop should enforce this, but numerical errors might creep in.
            # Let's verify.
            
            # Optimization: Don't check every iteration, only every 10th.
            if iter_idx % 10 == 0:
                polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
                valid = True
                for i in range(N):
                    for j in range(i+1, N):
                        if not check_disjoint_fast(polys[i], polys[j]):
                            valid = False
                            break
                    if not valid: break
                
                if valid:
                    score = calc_min_outer_side(positions, angles, 0.0)
                    if score < best_score:
                        best_score = score
                        best_final_state = (positions.copy(), angles.copy())
                        
    # If no valid state found (should not happen with PBD), fallback to initial
    if best_final_state is None:
        # Use the last iterated state from the last initial state
        # Or just return the honeycomb
        best_final_state = (init_positions, init_angles)
        best_score = calc_min_outer_side(init_positions, init_angles, 0.0)

    final_positions, final_angles = best_final_state
    final_side = calc_min_outer_side(final_positions, final_angles, 0.0)
    
    inner = []
    for i in range(N):
        inner.append([final_positions[i][0], final_positions[i][1], final_angles[i]])
    
    return inner, (0.0, 0.0), final_side, 0.0

