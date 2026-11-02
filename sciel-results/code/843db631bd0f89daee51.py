import numpy as np
import time
import math

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    def hexagon_vertices(cx, cy, side, angle_deg):
        angle_rad = np.radians(angle_deg)
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def required_outer_side(positions):
        """
        Compute the exact side length of the smallest regular hexagon (angle 0, centered at origin)
        that contains all unit hexagons at given positions.
        """
        verts = np.zeros((N, 6, 2))
        for i in range(N):
            verts[i] = hexagon_vertices(positions[i, 0], positions[i, 1], 1.0, 0.0)
        verts_flat = verts.reshape(-1, 2)
        
        x = verts_flat[:, 0]
        y = verts_flat[:, 1]
        sqrt3 = math.sqrt(3.0)
        
        L1 = x + y / sqrt3
        L2 = y * 2.0 / sqrt3
        L3 = -x + y / sqrt3
        L4 = -x - y / sqrt3
        L5 = -y * 2.0 / sqrt3
        L6 = x - y / sqrt3
        
        return np.max(np.stack([L1, L2, L3, L4, L5, L6], axis=1))

    def check_validity(positions, tol=1e-6):
        """
        Check if all hexagons are pairwise disjoint using SAT.
        """
        vertices_list = []
        for i in range(N):
            x, y = positions[i]
            verts = hexagon_vertices(x, y, 1.0, 0.0)
            vertices_list.append(verts)
        
        for i in range(N):
            for j in range(i + 1, N):
                p1 = vertices_list[i]
                p2 = vertices_list[j]
                
                # SAT check
                disjoint = False
                # Check separating axes from hexagon 1
                for i_sat in range(6):
                    j_sat = (i_sat + 1) % 6
                    edge = p1[j_sat] - p1[i_sat]
                    normal = np.array([-edge[1], edge[0]])
                    norm = np.linalg.norm(normal)
                    if norm < 1e-10:
                        continue
                    normal /= norm
                    proj1 = p1.dot(normal)
                    proj2 = p2.dot(normal)
                    if np.max(proj1) < np.min(proj2) - tol or np.max(proj2) < np.min(proj1) - tol:
                        disjoint = True
                        break
                
                if not disjoint:
                    # Check separating axes from hexagon 2
                    for i_sat in range(6):
                        j_sat = (i_sat + 1) % 6
                        edge = p2[j_sat] - p2[i_sat]
                        normal = np.array([-edge[1], edge[0]])
                        norm = np.linalg.norm(normal)
                        if norm < 1e-10:
                            continue
                        normal /= norm
                        proj1 = p1.dot(normal)
                        proj2 = p2.dot(normal)
                        if np.max(proj1) < np.min(proj2) - tol or np.max(proj2) < np.min(proj1) - tol:
                            disjoint = True
                            break
                    if not disjoint:
                        return False
        return True

    def overlap_penalty(positions, hard_threshold=2.0, soft_threshold=1.8):
        """
        Calculate a penalty for overlaps. 
        Uses a smooth function based on center distance.
        """
        penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                d = positions[i] - positions[j]
                dist = np.linalg.norm(d)
                if dist < hard_threshold:
                    # Quadratic penalty that goes to 0 at hard_threshold
                    diff = hard_threshold - dist
                    penalty += diff * diff
        return penalty

    def simulated_annealing():
        # Initialize with hexagonal lattice
        coords = [
            (0, 0),
            (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
            (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1)
        ]
        init_pos = []
        for q, r in coords:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            init_pos.append([x, y])
        
        current_pos = np.array(init_pos, dtype=float)
        current_side = required_outer_side(current_pos)
        current_penalty = overlap_penalty(current_pos)
        
        # Check initial validity
        if not check_validity(current_pos):
             # If invalid, scale up slightly or just accept and let SA fix it? 
             # Lattice is valid.
             pass

        best_pos = current_pos.copy()
        best_side = current_side
        best_penalty = current_penalty
        
        T = 1.0
        T_min = 1e-6
        alpha = 0.9995 # Cooling rate
        
        steps = 0
        max_steps = 100000
        
        rng = np.random.default_rng(42)
        
        while T > T_min and steps < max_steps:
            if time.time() > deadline - 5.0:
                break
                
            steps += 1
            
            # Generate move: pick a random hexagon and move it
            idx = rng.integers(0, N)
            # Move size scales with temperature
            move_size = 0.2 * T + 0.01
            dx = rng.normal(0, move_size)
            dy = rng.normal(0, move_size)
            
            new_pos = current_pos.copy()
            new_pos[idx][0] += dx
            new_pos[idx][1] += dy
            
            new_side = required_outer_side(new_pos)
            new_penalty = overlap_penalty(new_pos)
            
            # Objective: minimize side length, heavily penalize overlaps
            # We want valid configs (penalty=0) with minimal side.
            # If penalty > 0, we effectively have a large cost.
            # Let's use: Cost = side + 100 * penalty
            current_cost = current_side + 100.0 * current_penalty
            new_cost = new_side + 100.0 * new_penalty
            
            delta_cost = new_cost - current_cost
            
            # Accept if better, or with probability exp(-delta/T) if worse
            if delta_cost < 0 or rng.random() < math.exp(-delta_cost / T):
                current_pos = new_pos
                current_side = new_side
                current_penalty = new_penalty
                
                if current_penalty == 0.0 and current_side < best_side:
                    best_pos = current_pos.copy()
                    best_side = current_side
                    best_penalty = current_penalty
            
            T *= alpha
            
            # Occasionally reset T if stuck? Or just cool.
            # Let's add a restart mechanism if no improvement for a while?
            # For simplicity, just cool.
            
        # Final check
        if check_validity(best_pos, tol=1e-5):
            return best_pos, best_side
        else:
            # If best is invalid (shouldn't happen if we only update best when penalty==0),
            # try to return the current one if valid, else scale up
            if check_validity(current_pos, tol=1e-5):
                 return current_pos, current_side
            return best_pos, best_side

    # Run SA
    best_positions, best_side = simulated_annealing()
    
    # If SA didn't find a valid solution or failed, fallback to lattice
    if not check_validity(best_positions, tol=1e-5):
        coords = [
            (0, 0),
            (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
            (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1)
        ]
        init_pos = []
        for q, r in coords:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            init_pos.append([x, y])
        best_positions = np.array(init_pos, dtype=float)
        best_side = required_outer_side(best_positions)

    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), 0.0])
        
    return inner, (0.0, 0.0), float(best_side), 0.0

# EVOLVE-BLOCK-END
