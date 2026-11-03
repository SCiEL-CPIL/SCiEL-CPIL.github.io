"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    # Helper: get vertices of a regular hexagon with side length 1
    def hex_vertices(cx, cy, angle_deg):
        angle_rad = math.radians(angle_deg)
        verts = []
        for i in range(6):
            a = angle_rad + i * math.pi / 3
            x = cx + math.cos(a)
            y = cy + math.sin(a)
            verts.append((x, y))
        return np.array(verts)
    
    # Helper: check if two hexagons overlap using SAT (Separating Axis Theorem)
    def hexagons_overlap(v1, v2):
        """Check if two convex hexagons overlap using SAT."""
        # Precompute axes for both hexagons
        axes = []
        for i in range(6):
            edge = v1[i] - v1[(i+1) % 6]
            normal = np.array([-edge[1], edge[0]])
            axes.append(normal)
            edge2 = v2[i] - v2[(i+1) % 6]
            normal2 = np.array([-edge2[1], edge2[0]])
            axes.append(normal2)
        
        for axis in axes:
            proj1 = np.dot(v1, axis)
            proj2 = np.dot(v2, axis)
            min1, max1 = proj1.min(), proj1.max()
            min2, max2 = proj2.min(), proj2.max()
            if max1 < min2 or max2 < min1:
                return False
        return True
    
    # Helper: compute minimum outer hexagon side length needed to contain all inner hexagons
    def compute_min_side(positions, angles, outer_angle_deg):
        """
        positions: list of [x, y] for each inner hexagon
        angles: list of rotation angles for each inner hexagon
        outer_angle_deg: rotation of outer hexagon
        Returns the minimum side length of outer hexagon to contain all inner hexagons.
        """
        all_verts = []
        for i in range(N):
            cx, cy = positions[i]
            a = angles[i]
            verts = hex_vertices(cx, cy, a)
            all_verts.append(verts)
        
        all_verts = np.vstack(all_verts)
        
        theta = math.radians(outer_angle_deg)
        normals = []
        for k in range(6):
            a = theta + k * math.pi / 3
            normals.append(np.array([math.cos(a), math.sin(a)]))
        
        max_proj = 0
        for normal in normals:
            proj = np.dot(all_verts, normal)
            max_proj = max(max_proj, proj.max())
        
        side = 2 * max_proj / math.sqrt(3)
        return side
    
    # Objective: given positions, angles, outer_angle, return (side_length, is_valid)
    def evaluate(positions, angles, outer_angle_deg):
        # Check overlaps
        verts = [hex_vertices(p[0], p[1], a) for p, a in zip(positions, angles)]
        for i in range(N):
            for j in range(i+1, N):
                if hexagons_overlap(verts[i], verts[j]):
                    return None, False
        side = compute_min_side(positions, angles, outer_angle_deg)
        return side, True

    # Generate a valid initial honeycomb layout
    def get_initial_honeycomb(spacing=2.0):
        positions = []
        angles = [0.0] * N
        # 4 rows of 3
        for row in range(4):
            for col in range(3):
                x = col * spacing * math.sqrt(3) / 2 + (row % 2) * spacing * math.sqrt(3) / 4
                y = row * spacing
                positions.append([x, y])
        # Center around origin
        cx = sum(p[0] for p in positions) / N
        cy = sum(p[1] for p in positions) / N
        positions = [[x - cx, y - cy] for x, y in positions]
        return positions, angles

    # Stochastic Local Search (Simulated Annealing style)
    def stochastic_local_search(positions, angles, outer_angle, max_iters=5000):
        current_side, valid = evaluate(positions, angles, outer_angle)
        if not valid:
            # Try to fix by scaling up slightly if invalid
            for scale in [1.01, 1.05, 1.1, 1.2]:
                scaled_pos = [[x*scale, y*scale] for x, y in positions]
                current_side, valid = evaluate(scaled_pos, angles, outer_angle)
                if valid:
                    positions = scaled_pos
                    break
            if not valid:
                return positions, angles, outer_angle, None

        best_side = current_side
        best_positions = [p[:] for p in positions]
        best_angles = angles[:]
        best_outer = outer_angle
        
        temperature = 1.0
        cooling_rate = 0.999
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Randomly perturb one hexagon
            i = np.random.randint(N)
            new_positions = [p[:] for p in positions]
            new_angles = angles[:]
            
            # Perturb position
            new_positions[i][0] += np.random.normal(0, 0.2)
            new_positions[i][1] += np.random.normal(0, 0.2)
            
            # Perturb angle
            new_angles[i] += np.random.normal(0, 0.1)
            
            # Perturb outer angle occasionally
            if np.random.random() < 0.05:
                new_outer = outer_angle + np.random.normal(0, 0.5)
            else:
                new_outer = outer_angle
            
            new_side, new_valid = evaluate(new_positions, new_angles, new_outer)
            
            if new_valid:
                if new_side < best_side:
                    best_side = new_side
                    best_positions = [p[:] for p in new_positions]
                    best_angles = new_angles[:]
                    best_outer = new_outer
                
                # Accept if better, or with probability exp(-delta/T)
                if new_side < current_side:
                    current_side = new_side
                    positions = new_positions
                    angles = new_angles
                    outer_angle = new_outer
                else:
                    delta = new_side - current_side
                    prob = math.exp(-delta / temperature)
                    if np.random.random() < prob:
                        current_side = new_side
                        positions = new_positions
                        angles = new_angles
                        outer_angle = new_outer
            else:
                # Reject invalid moves, but maybe accept if we are stuck? 
                # For now, just reject to maintain validity mostly.
                pass
                
            temperature *= cooling_rate
            if temperature < 0.01:
                temperature = 0.01
                
        return best_positions, best_angles, best_outer, best_side

    # Main logic
    np.random.seed(42)
    
    best_side = 1e10
    best_positions = None
    best_angles = None
    best_outer_angle = 0.0
    
    # Start with a few different honeycomb spacings
    for restart in range(5):
        if time.time() > deadline - 10:
            break
        
        spacing = 2.0 + restart * 0.1
        positions, angles = get_initial_honeycomb(spacing)
        outer_angle = 0.0
        
        # Run stochastic local search
        positions, angles, outer_angle, side = stochastic_local_search(positions, angles, outer_angle, max_iters=2000)
        
        if side is not None and side < best_side:
            best_side = side
            best_positions = positions
            best_angles = angles
            best_outer_angle = outer_angle
            
    # If we didn't find a valid solution, use a large fallback
    if best_positions is None:
        positions, angles = get_initial_honeycomb(3.0)
        outer_angle = 0.0
        best_side = compute_min_side(positions, angles, outer_angle)
        best_positions = positions
        best_angles = angles
        best_outer_angle = outer_angle

    # Final verification and cleanup
    side, valid = evaluate(best_positions, best_angles, best_outer_angle)
    if not valid:
        # Fallback to a very large hexagon
        best_positions = get_initial_honeycomb(5.0)[0]
        best_angles = get_initial_honeycomb(5.0)[1]
        best_outer_angle = 0.0
        best_side = compute_min_side(best_positions, best_angles, best_outer_angle)
        
    # Build output
    inner = [[best_positions[i][0], best_positions[i][1], best_angles[i]] for i in range(N)]
    center = (0.0, 0.0)
    side = compute_min_side(best_positions, best_angles, best_outer_angle)
    
    return inner, center, side, best_outer_angle
# EVOLVE-BLOCK-END
