import numpy as np
import time
import math

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    # Precompute constants
    SQRT3 = math.sqrt(3.0)
    
    # Vertices of a unit hexagon centered at (0,0) with angle 0
    # Angles: 0, 60, 120, 180, 240, 300 degrees
    unit_hex_verts = np.array([
        [1.0, 0.0],
        [0.5, SQRT3 / 2.0],
        [-0.5, SQRT3 / 2.0],
        [-1.0, 0.0],
        [-0.5, -SQRT3 / 2.0],
        [0.5, -SQRT3 / 2.0]
    ])

    def get_hex_vertices(cx, cy):
        """
        Returns vertices of a unit hexagon centered at (cx, cy).
        Assumes angle 0 (aligned with outer hexagon).
        """
        return unit_hex_verts + np.array([cx, cy])

    def min_distance_between_hexagons(pos1, pos2):
        """
        Calculate the minimum distance between two unit hexagons at pos1 and pos2.
        If they overlap, this returns a negative value (approximation of penetration depth).
        We use a fast SAT-like check or just bounding box first.
        
        For speed, we use a simplified SAT that only checks if they are separated.
        If not separated, we estimate overlap.
        However, for SA, we just need a penalty. 
        A robust way: check all pairs of vertices vs edges? No, too slow.
        
        Let's use the fact that hexagons are convex.
        We can compute the distance between two convex polygons.
        Since N is small (12), we can afford a slightly more expensive check per pair.
        
        Efficient SAT for distance:
        1. Check separating axes (normals of edges of both polygons).
        2. If a separating axis is found, the distance is the gap on that axis.
        3. If no separating axis, they overlap. We can return 0 or a small negative penalty.
           For SA, returning 0 when overlapping is sufficient if the penalty is high enough,
           but it creates a flat landscape. 
           
        Alternative: Use the distance between the two sets of vertices? 
        min_{v1 in P1, v2 in P2} ||v1 - v2||.
        This is O(36) per pair. 66 pairs * 36 = ~2400 ops. Very fast.
        But min vertex distance is NOT the same as polygon distance if edges cross.
        However, for convex polygons, the minimum distance is either vertex-vertex or vertex-edge.
        
        Let's implement a robust polygon distance function.
        """
        v1 = pos1 + unit_hex_verts
        v2 = pos2 + unit_hex_verts
        
        # 1. Check Separating Axes
        # Normals of v1
        for i in range(6):
            j = (i + 1) % 6
            edge = v1[j] - v1[i]
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-9: continue
            normal /= norm
            
            proj1 = v1.dot(normal)
            proj2 = v2.dot(normal)
            
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            if max1 < min2:
                return min2 - max1
            elif max2 < min1:
                return min1 - max2
        
        # Normals of v2
        for i in range(6):
            j = (i + 1) % 6
            edge = v2[j] - v2[i]
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-9: continue
            normal /= norm
            
            proj1 = v1.dot(normal)
            proj2 = v2.dot(normal)
            
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            if max1 < min2:
                return min2 - max1
            elif max2 < min1:
                return min1 - max2
        
        # If no separating axis, they overlap.
        # Return 0. The penalty will handle the rest.
        # To help SA escape, we could estimate penetration, but 0 is standard.
        return 0.0

    def calculate_outer_side(positions):
        """
        Calculate the minimum side length S of a regular hexagon (angle 0, centered at origin)
        that contains all vertices of the inner hexagons.
        """
        max_s = 0.0
        for i in range(N):
            x, y = positions[i]
            verts = get_hex_vertices(x, y)
            for vx, vy in verts:
                # Constraints for hexagon side S (angle 0):
                # |x| <= S
                # |y| * 2/sqrt(3) <= S  -> |y| <= S * sqrt(3)/2
                # |x| + |y|/sqrt(3) <= S
                
                s1 = abs(vx)
                s2 = abs(vy) * 2.0 / SQRT3
                s3 = abs(vx) + abs(vy) / SQRT3
                
                s_req = max(s1, s2, s3)
                if s_req > max_s:
                    max_s = s_req
        return max_s

    def objective(positions):
        """
        Returns the cost: Outer Side Length + Penalty for overlaps.
        """
        # Check overlaps
        penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                dist = min_distance_between_hexagons(positions[i], positions[j])
                if dist < 0:
                    # Overlap. Penalty proportional to how much they are "not separated".
                    # Since dist is 0 on overlap, we need a measure of violation.
                    # Actually, my min_distance returns 0 on overlap. 
                    # This makes the penalty flat. 
                    # Let's modify min_distance to return negative distance if overlapping?
                    # SAT doesn't easily give negative distance (penetration depth).
                    # 
                    # Workaround: If dist == 0, they overlap. 
                    # We can use the distance between centers as a proxy? No.
                    # Let's just add a fixed penalty for any overlap.
                    # Or better: check if they are "too close".
                    # If dist == 0, penalty += 1.0.
                    penalty += 10.0
                elif dist < 1e-6:
                    # Touching or very close. Allow small tolerance, but discourage.
                    penalty += (1e-6 - dist) * 100.0 if dist < 1e-6 else 0.0
                    
        # Calculate outer size
        s_outer = calculate_outer_side(positions)
        
        return s_outer + penalty

    # Initialization: Honeycomb pattern
    # This is a valid, dense packing.
    # Centers in axial coordinates (q, r)
    # x = sqrt(3) * (q + r/2)
    # y = 1.5 * r
    
    coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1)
    ]
    
    positions = np.array([
        [SQRT3 * (q + r / 2.0), 1.5 * r] for q, r in coords
    ])
    
    current_cost = objective(positions)
    best_positions = positions.copy()
    best_cost = current_cost
    
    # Simulated Annealing
    T = 5.0  # Initial temperature
    T_min = 0.01
    cooling_rate = 0.9995
    max_iterations = 50000
    
    # Random number generator for reproducibility
    rng = np.random.default_rng(42)
    
    for i in range(max_iterations):
        if time.time() > deadline:
            break
            
        # Generate neighbor
        new_positions = positions.copy()
        idx = rng.integers(0, N)
        # Move one hexagon
        dx = rng.normal(0, 0.5)
        dy = rng.normal(0, 0.5)
        new_positions[idx] += np.array([dx, dy])
        
        new_cost = objective(new_positions)
        
        # Acceptance probability
        delta = new_cost - current_cost
        if delta < 0 or rng.random() < math.exp(-delta / T):
            positions = new_positions
            current_cost = new_cost
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_positions = positions.copy()
                
        T *= cooling_rate
        if T < T_min:
            T = T_min
            
    # Final check and cleanup
    # The best_cost might still have penalty if we didn't fully converge.
    # We should return the best valid configuration found.
    # If best_cost > calculate_outer_side(best_positions), it had overlap.
    # Let's try to "relax" the best solution if it has overlap.
    
    # Check if best is valid
    def is_valid(pos):
        for i in range(N):
            for j in range(i + 1, N):
                d = min_distance_between_hexagons(pos[i], pos[j])
                if d < -1e-5: # Overlap
                    return False
        return True

    if not is_valid(best_positions):
        # Try a few more iterations with lower temperature or just return the initial honeycomb if it's better
        # The honeycomb is always valid.
        honeycomb_cost = calculate_outer_side(positions) # Wait, positions is current, not initial
        # Let's re-evaluate the initial honeycomb
        init_coords = [
            (0, 0),
            (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
            (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1)
        ]
        init_pos = np.array([
            [SQRT3 * (q + r / 2.0), 1.5 * r] for q, r in init_coords
        ])
        init_cost = calculate_outer_side(init_pos)
        
        if init_cost < best_cost:
            best_positions = init_pos
            best_cost = init_cost
        else:
            # Try to fix best_positions with a quick local search
            # Just use best_positions if it's valid, otherwise init
            # Actually, if best_cost includes penalty, it's higher than the true size.
            # So comparing best_cost (with penalty) to init_cost (no penalty) is fair if penalty is large.
            # But if best_positions is invalid, its "true" cost is unknown (infinite).
            # So we should only accept best_positions if it is valid.
            
            # Let's do a strict check
            if is_valid(best_positions):
                final_cost = calculate_outer_side(best_positions)
                if final_cost < init_cost:
                    # Keep best
                    pass
                else:
                    best_positions = init_pos
                    best_cost = init_cost
            else:
                best_positions = init_pos
                best_cost = init_cost

    # Construct return values
    inner = []
    for i in range(N):
        x, y = best_positions[i]
        inner.append([float(x), float(y), 0.0])
        
    outer_side = float(best_cost)
    
    return inner, (0.0, 0.0), outer_side, 0.0

# EVOLVE-BLOCK-END
