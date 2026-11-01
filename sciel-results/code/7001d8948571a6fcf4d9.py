import numpy as np
import time
from scipy.optimize import minimize

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
        sqrt3 = np.sqrt(3.0)
        
        # Support function directions for a regular hexagon of side S centered at origin
        # The width of a regular hexagon of side S in the 6 normal directions is S*sqrt(3).
        # The distance from center to side (apothem) is S*sqrt(3)/2.
        # The projection of a point p onto the normal n_k must be <= S*sqrt(3)/2.
        # So S >= 2/sqrt(3) * max_k (p . n_k).
        # The normals for a hexagon with vertices at 0, 60, 120... are at 30, 90, 150...
        # n_0 = (cos(30), sin(30)) = (sqrt(3)/2, 1/2)
        # n_1 = (cos(90), sin(90)) = (0, 1)
        # n_2 = (cos(150), sin(150)) = (-sqrt(3)/2, 1/2)
        # n_3 = (cos(210), sin(210)) = (-sqrt(3)/2, -1/2)
        # n_4 = (cos(270), sin(270)) = (0, -1)
        # n_5 = (cos(330), sin(330)) = (sqrt(3)/2, -1/2)
        
        # Let's verify the linear forms used in the original code:
        # L1 = x + y/sqrt3. 
        # p . n_0 = x*sqrt(3)/2 + y/2.
        # L1 = (2/sqrt(3)) * (x*sqrt(3)/2 + y/2) = (2/sqrt(3)) * (p . n_0).
        # So S = max(L1, ..., L6). This is correct.
        
        L1 = x + y / sqrt3
        L2 = y * 2.0 / sqrt3
        L3 = -x + y / sqrt3
        L4 = -x - y / sqrt3
        L5 = -y * 2.0 / sqrt3
        L6 = x - y / sqrt3
        
        all_L = np.stack([L1, L2, L3, L4, L5, L6], axis=1)
        return np.max(all_L)

    def check_validity(positions):
        """
        Check if all hexagons are pairwise disjoint using SAT.
        Optimized for speed.
        """
        # Precompute vertices for all hexagons
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
                # Check edges of p1
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
                    if np.max(proj1) < np.min(proj2) - 1e-5 or np.max(proj2) < np.min(proj1) - 1e-5:
                        disjoint = True
                        break
                
                if not disjoint:
                    # Check edges of p2
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
                        if np.max(proj1) < np.min(proj2) - 1e-5 or np.max(proj2) < np.min(proj1) - 1e-5:
                            disjoint = True
                            break
                    if not disjoint:
                        return False
        return True

    def simulated_annealing(init_positions, max_iters=5000, T_start=1.0, T_end=0.01):
        """
        Simulated annealing with hard validity constraint.
        Moves that result in invalid (overlapping) configurations are always rejected.
        """
        current = init_positions.copy()
        if not check_validity(current):
            return None, float('inf')
        
        current_side = required_outer_side(current)
        best = current.copy()
        best_side = current_side
        
        for i in range(max_iters):
            if time.time() > deadline:
                break
            
            # Generate a new position by perturbing one hexagon
            idx = np.random.randint(N)
            step_size = 0.5 * (T_start - T_end) / T_start + T_end
            new_pos = current.copy()
            new_pos[idx, 0] += np.random.uniform(-step_size, step_size)
            new_pos[idx, 1] += np.random.uniform(-step_size, step_size)
            
            if not check_validity(new_pos):
                continue
            
            new_side = required_outer_side(new_pos)
            delta = new_side - current_side
            
            if delta < 0 or np.random.random() < np.exp(-delta / T_start):
                current = new_pos
                current_side = new_side
                if current_side < best_side:
                    best = current.copy()
                    best_side = current_side
            
            # Cool down
            T_start = T_start * 0.999 + T_end * 0.001
            
        return best, best_side

    # Initial configurations
    configs = []
    
    # 1. Hexagonal lattice (compact)
    # A 12-hexagon cluster can be formed by a central hexagon surrounded by 6, 
    # and then adding 5 more around the perimeter.
    # Let's use a standard hexagonal grid coordinate system (axial).
    # Center (0,0)
    # Ring 1: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # Ring 2 (partial): (2,0), (1,1), (0,2), (-1,2), (-2,1)
    coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1)
    ]
    init1 = []
    for q, r in coords:
        x = np.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        init1.append([x, y])
    configs.append(np.array(init1))
    
    # 2. Circular arrangement
    init2 = np.zeros((N, 2))
    for i in range(N):
        angle = 2 * np.pi * i / N
        r = 2.5
        init2[i, 0] = r * np.cos(angle)
        init2[i, 1] = r * np.sin(angle)
    configs.append(init2)
    
    # 3. Random
    np.random.seed(42)
    init3 = np.random.uniform(-5, 5, (N, 2))
    configs.append(init3)
    
    # 4. Another random seed
    np.random.seed(123)
    init4 = np.random.uniform(-5, 5, (N, 2))
    configs.append(init4)

    best_positions = None
    best_side = float('inf')
    
    for init in configs:
        if time.time() > deadline:
            break
        
        # Ensure initial config is valid; if not, spread out
        if not check_validity(init):
            for scale in [1.2, 1.5, 2.0, 2.5, 3.0]:
                scaled = init * scale
                if check_validity(scaled):
                    init = scaled
                    break
            else:
                continue
        
        # Run Simulated Annealing
        try:
            pos, side = simulated_annealing(init, max_iters=5000, T_start=1.0, T_end=0.01)
            if pos is not None and side < best_side:
                best_side = side
                best_positions = pos
        except Exception as e:
            pass

    # Fallback
    if best_positions is None:
        best_positions = configs[0]
        best_side = required_outer_side(best_positions)
    
    # Final validation
    if not check_validity(best_positions):
        # Try to find a valid config
        for init in configs:
            pos = init.copy()
            for scale in [1.0, 1.2, 1.5, 2.0, 2.5, 3.0]:
                scaled = pos * scale
                if check_validity(scaled):
                    best_positions = scaled
                    best_side = required_outer_side(best_positions)
                    break
            if check_validity(best_positions):
                break
        else:
            # Last resort: spread out linearly
            best_positions = np.array([[i * 3.0, 0.0] for i in range(N)])
            best_side = required_outer_side(best_positions)
    
    # Calculate final exact side
    final_side = required_outer_side(best_positions)
    
    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), 0.0])
        
    return inner, (0.0, 0.0), float(final_side), 0.0

# EVOLVE-BLOCK-END
