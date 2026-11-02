import numpy as np
import time
import random

N = 12
SQRT3 = np.sqrt(3.0)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    def hexagon_vertices(cx, cy, side, angle_deg):
        angle_rad = np.radians(angle_deg)
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def get_hexagon_normals(angle_deg):
        """
        Returns 6 unit normals for the hexagon edges.
        For a regular hexagon, the normals are perpendicular to the edges.
        If vertices are at angles theta, theta+60..., the edge midpoints are at theta+30, theta+90...
        So normals are at angles theta+30, theta+90, ...
        """
        angle_rad = np.radians(angle_deg)
        # Normals are offset by 30 degrees (pi/6) from vertex angles
        normal_angles = angle_rad + np.pi/6 + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            np.cos(normal_angles),
            np.sin(normal_angles)
        ])

    def sat_distance(v1, n1, v2, n2):
        """
        Computes the minimum separation distance between two convex hexagons.
        Positive if disjoint, negative if overlapping.
        v1, v2: (6, 2) arrays of vertices.
        n1, n2: (6, 2) arrays of unit normals.
        """
        min_sep = float('inf')
        
        # Check axes from hexagon 1
        for i in range(6):
            n = n1[i]
            proj1 = np.dot(v1, n)
            proj2 = np.dot(v2, n)
            sep = np.min(proj2) - np.max(proj1)
            if sep < min_sep:
                min_sep = sep
            if sep > 0:
                # If we find a separating axis, we can return early if we want, 
                # but for distance we need the min over all axes? 
                # Actually, for disjoint polygons, the distance is the min positive sep.
                # For overlapping, it's the max negative sep (least overlap).
                # Wait, standard SAT for distance:
                # If disjoint: distance = min(sep) over all axes where sep > 0? No.
                # If disjoint, there is at least one axis with sep > 0. The distance is the MINIMUM of these positive seps?
                # No, the distance between two disjoint convex polygons is the minimum over all axes of the separation.
                # If they are disjoint, ALL axes have sep >= 0? No, only some.
                # Actually, if disjoint, min_sep over ALL axes will be positive? 
                # Let's verify. If disjoint, there is a gap. The gap is bounded by the min projection difference on the separating axis.
                # However, other axes might have smaller positive projections.
                # Yes, the Euclidean distance is the minimum of the support function differences over all directions.
                # So we must check all 12 axes and take the minimum.
                pass
        
        # Check axes from hexagon 2
        for i in range(6):
            n = n2[i]
            proj1 = np.dot(v1, n)
            proj2 = np.dot(v2, n)
            sep = np.min(proj2) - np.max(proj1)
            if sep < min_sep:
                min_sep = sep
        
        return min_sep

    def required_outer_side(positions, angles):
        """
        Compute the exact side length of the smallest regular hexagon (angle 0, centered at origin)
        that contains all unit hexagons at given positions and angles.
        """
        # The outer hexagon has side S, angle 0.
        # Its support function in direction n (unit vector) is S * h_outer(n).
        # For a regular hexagon of side S, the distance from center to side (apothem) is S * SQRT3 / 2.
        # The normals of the outer hexagon are at 30, 90, 150, ...
        # Let's compute the required S by checking the 6 directions of the outer hexagon.
        # Actually, to contain a set of points P, S must satisfy:
        # max_{p in P} (p . n_k) <= S * h_outer(n_k) for all k=0..5
        # h_outer(n_k) is the projection of the outer hexagon vertices onto n_k.
        # For the outer hexagon (angle 0), vertices are at 0, 60, ...
        # Normal n_k is at 30 + 60k.
        # The projection of the outer hexagon onto its own normal n_k is the apothem: S * SQRT3 / 2.
        # Wait, is that sufficient?
        # The condition for a polygon A to contain polygon B is that for all normals n of A,
        # h_B(n) <= h_A(n).
        # Since both are regular hexagons, checking the 6 normals of the outer hexagon is sufficient?
        # Yes, because the support function of a regular hexagon is convex and piecewise linear,
        # and the "corners" of the support function align with the normals.
        
        max_proj = 0.0
        # Normals of outer hexagon (angle 0)
        outer_normals = get_hexagon_normals(0.0)
        
        for k in range(6):
            n = outer_normals[k]
            # For each inner hexagon, compute max projection of its vertices onto n
            for i in range(N):
                cx, cy = positions[i]
                ang = angles[i]
                # Vertices of inner hexagon i
                # We don't need to construct full vertices, just project.
                # Projection of a regular hexagon of side 1 at (cx,cy) angle ang onto n:
                # h_i(n) = (cx*n[0] + cy*n[1]) + h_unit(n - rotation(ang))
                # Actually, easier to just compute vertices since N=12 is small.
                verts = hexagon_vertices(cx, cy, 1.0, ang)
                proj = np.dot(verts, n)
                m = np.max(proj)
                if m > max_proj:
                    max_proj = m
            
        # The apothem of the outer hexagon is S * SQRT3 / 2.
        # So max_proj <= S * SQRT3 / 2
        # S >= 2 * max_proj / SQRT3
        return 2.0 * max_proj / SQRT3

    def evaluate_solution(positions, angles):
        """
        Returns (side_length, min_clearance).
        min_clearance is the minimum SAT distance between any pair.
        """
        side = required_outer_side(positions, angles)
        
        min_clearance = float('inf')
        verts_list = []
        normals_list = []
        for i in range(N):
            verts_list.append(hexagon_vertices(positions[i, 0], positions[i, 1], 1.0, angles[i]))
            normals_list.append(get_hexagon_normals(angles[i]))
        
        for i in range(N):
            for j in range(i + 1, N):
                dist = sat_distance(verts_list[i], normals_list[i], verts_list[j], normals_list[j])
                if dist < min_clearance:
                    min_clearance = dist
                # Early exit if we find a good clearance? No, we need min.
        
        return side, min_clearance

    def objective(params):
        """
        params: [x1, y1, a1, x2, y2, a2, ...]
        """
        positions = params[0::3].reshape(-1, 2) # Wait, this is wrong.
        # Let's use a flat array: x1, y1, a1, x2, y2, a2...
        positions = np.array([[params[i*3], params[i*3+1]] for i in range(N)])
        angles = np.array([params[i*3+2] for i in range(N)])
        
        side, min_clearance = evaluate_solution(positions, angles)
        
        # Penalty for overlap
        # We want min_clearance >= 0.
        # Penalty = max(0, -min_clearance)^2
        penalty = 0.0
        if min_clearance < 0:
            penalty = (-min_clearance) ** 2
        
        return side + 100.0 * penalty

    def simulated_annealing(initial_positions, initial_angles, time_budget):
        current_positions = initial_positions.copy()
        current_angles = initial_angles.copy()
        current_side, current_clearance = evaluate_solution(current_positions, current_angles)
        
        best_positions = current_positions.copy()
        best_angles = current_angles.copy()
        best_side = current_side
        best_clearance = current_clearance
        
        # If initial is invalid, we still track it, but we prefer valid ones.
        # Score for comparison: side + penalty
        def get_score(pos, ang, side, clr):
            pen = 0.0 if clr >= 0 else (-clr)**2
            return side + 100.0 * pen

        current_score = get_score(current_positions, current_angles, current_side, current_clearance)
        best_score = current_score
        
        T = 1.0
        T_min = 0.01
        cooling_rate = 0.999
        step_size = 0.5
        
        while time.time() < time_budget and T > T_min:
            # Generate neighbor
            new_positions = current_positions.copy()
            new_angles = current_angles.copy()
            
            # Randomly pick a few hexagons to move/rotate
            num_moves = random.randint(1, 3)
            for _ in range(num_moves):
                idx = random.randint(0, N-1)
                new_positions[idx] += np.random.normal(0, step_size, 2)
                new_angles[idx] += np.random.normal(0, step_size)
            
            new_side, new_clearance = evaluate_solution(new_positions, new_angles)
            new_score = get_score(new_positions, new_angles, new_side, new_clearance)
            
            delta = new_score - current_score
            
            if delta < 0 or random.random() < np.exp(-delta / T):
                current_positions = new_positions
                current_angles = new_angles
                current_side = new_side
                current_clearance = new_clearance
                current_score = new_score
                
                if new_score < best_score:
                    best_score = new_score
                    best_positions = new_positions.copy()
                    best_angles = new_angles.copy()
                    best_side = new_side
                    best_clearance = new_clearance
            else:
                # Rejected
                pass
            
            T *= cooling_rate
            
            # Adapt step size? Keep it simple for now.
            # If we are stuck, maybe increase step size?
            
        return best_positions, best_angles, best_side, best_clearance

    # Initial configurations
    inits = []
    
    # 1. Honeycomb lattice (valid, compact)
    # Hexagonal grid coordinates (axial)
    coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1)
    ]
    init_pos = []
    init_ang = []
    for q, r in coords:
        x = SQRT3 * (q + r / 2.0)
        y = 1.5 * r
        init_pos.append([x, y])
        init_ang.append(0.0)
    inits.append((np.array(init_pos), np.array(init_ang)))
    
    # 2. Circular ring
    init_pos2 = []
    init_ang2 = []
    for i in range(N):
        angle = 2 * np.pi * i / N
        r = 2.5
        init_pos2.append([r * np.cos(angle), r * np.sin(angle)])
        init_ang2.append(angle * 180 / np.pi) # Align radially?
    inits.append((np.array(init_pos2), np.array(init_ang2)))
    
    # 3. Random
    random.seed(42)
    init_pos3 = np.random.uniform(-4, 4, (N, 2))
    init_ang3 = np.random.uniform(0, 60, N)
    inits.append((init_pos3, init_ang3))
    
    # 4. Random with rotation
    random.seed(123)
    init_pos4 = np.random.uniform(-4, 4, (N, 2))
    init_ang4 = np.random.uniform(0, 360, N)
    inits.append((init_pos4, init_ang4))
    
    best_global_side = float('inf')
    best_global_pos = None
    best_global_ang = None
    
    # Allocate time to each init
    # We have ~140s. 4 inits. ~30s each.
    time_per_init = 30.0
    
    for i, (init_p, init_a) in enumerate(inits):
        if time.time() > deadline:
            break
        
        budget = min(time_per_init, deadline - time.time())
        if budget < 1.0:
            break
            
        pos, ang, side, clr = simulated_annealing(init_p, init_a, start_time + budget)
        
        # Check if valid
        if clr >= -1e-5: # Allow tiny floating point error
            if side < best_global_side:
                best_global_side = side
                best_global_pos = pos
                best_global_ang = ang
        else:
            # If invalid, we might still want to keep it if it's very close to valid?
            # No, the task requires disjoint. An invalid solution is score 0 (or penalty).
            # The evaluator measures "raw value". If invalid, it might return a huge number or error.
            # The prompt says: "an invalid result... scores 0".
            # So we MUST return a valid configuration.
            # If SA fails to find a valid one, we should fall back to a known valid one (like the honeycomb).
            
            # However, if the honeycomb itself is valid, we should have found it.
            # Let's ensure we always have a valid fallback.
            if best_global_pos is None:
                # Use the first init if it's valid
                side_chk, clr_chk = evaluate_solution(init_p, init_a)
                if clr_chk >= -1e-5:
                    best_global_side = side_chk
                    best_global_pos = init_p
                    best_global_ang = init_a

    # Final check: if we still don't have a valid solution, force one.
    if best_global_pos is None:
        # Use the honeycomb, and scale it up if necessary to be disjoint?
        # Honeycomb is disjoint by construction (touching).
        # Touching is usually considered "disjoint" in packing problems (interiors disjoint).
        # SAT distance 0 means touching.
        # Let's assume touching is valid.
        pos, ang, side, clr = evaluate_solution(inits[0][0], inits[0][1])
        if clr < 0:
            # It shouldn't be, but if it is, scale up
            scale = 1.1
            while clr < 0 and scale < 10:
                pos_scaled = inits[0][0] * scale
                pos, ang, side, clr = evaluate_solution(pos_scaled, inits[0][1])
                scale += 0.1
        best_global_side = side
        best_global_pos = pos
        best_global_ang = ang

    # Return format:
    # inner: list of (x, y, angle_degrees)
    # center: (x, y)
    # side: float
    # angle: float (outer hexagon rotation, we used 0)
    
    inner = []
    for i in range(N):
        inner.append([float(best_global_pos[i, 0]), float(best_global_pos[i, 1]), float(best_global_ang[i])])
        
    return inner, (0.0, 0.0), float(best_global_side), 0.0

# EVOLVE-BLOCK-END
