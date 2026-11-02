import numpy as np
import time
import math

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130.0

    def hexagon_vertices(cx, cy, side, angle_deg):
        angle_rad = np.radians(angle_deg)
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def get_all_vertices(pose):
        verts = np.zeros((N, 6, 2))
        for i in range(N):
            verts[i] = hexagon_vertices(pose[i, 0], pose[i, 1], 1.0, pose[i, 2])
        return verts

    def required_outer_side(positions):
        verts = get_all_vertices(positions)
        verts_flat = verts.reshape(-1, 2)
        x = verts_flat[:, 0]
        y = verts_flat[:, 1]
        sqrt3 = np.sqrt(3.0)
        
        L1 = x + y / sqrt3
        L2 = y * 2.0 / sqrt3
        L3 = -x + y / sqrt3
        L4 = -x - y / sqrt3
        L5 = -y * 2.0 / sqrt3
        L6 = x - y / sqrt3
        
        all_L = np.stack([L1, L2, L3, L4, L5, L6], axis=1)
        return np.max(all_L)

    def check_validity(positions):
        vertices_list = []
        for i in range(N):
            x, y, ang = positions[i]
            verts = hexagon_vertices(x, y, 1.0, ang)
            vertices_list.append(verts)
        
        for i in range(N):
            for j in range(i + 1, N):
                p1 = vertices_list[i]
                p2 = vertices_list[j]
                
                disjoint = False
                # Check separating axes from p1
                for k in range(6):
                    v1 = p1[k]
                    v2 = p1[(k+1)%6]
                    edge = v2 - v1
                    normal = np.array([-edge[1], edge[0]])
                    norm = np.linalg.norm(normal)
                    if norm < 1e-12:
                        continue
                    normal /= norm
                    proj1 = p1.dot(normal)
                    proj2 = p2.dot(normal)
                    if np.max(proj1) < np.min(proj2) - 1e-7 or np.max(proj2) < np.min(proj1) - 1e-7:
                        disjoint = True
                        break
                
                if not disjoint:
                    # Check separating axes from p2
                    for k in range(6):
                        v1 = p2[k]
                        v2 = p2[(k+1)%6]
                        edge = v2 - v1
                        normal = np.array([-edge[1], edge[0]])
                        norm = np.linalg.norm(normal)
                        if norm < 1e-12:
                            continue
                        normal /= norm
                        proj1 = p1.dot(normal)
                        proj2 = p2.dot(normal)
                        if np.max(proj1) < np.min(proj2) - 1e-7 or np.max(proj2) < np.min(proj1) - 1e-7:
                            disjoint = True
                            break
                    if not disjoint:
                        return False
        return True

    def simulated_annealing(init_positions, max_iters=100000, T_start=3.0, T_end=0.01):
        current = init_positions.copy()
        
        # Ensure initial validity
        if not check_validity(current):
            for _ in range(50):
                # Random restarts if invalid
                current = np.random.uniform(-4, 4, (N, 3))
                if check_validity(current):
                    break
            else:
                # Fallback to a known valid configuration (spread out)
                current = init_positions.copy()
                for i in range(N):
                    current[i, 0] *= 2.0
                    current[i, 1] *= 2.0
                if not check_validity(current):
                    return None, None

        current_side = required_outer_side(current)
        best = current.copy()
        best_side = current_side
        no_improve = 0
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Adaptive temperature
            T = T_start * (T_end / T_start) ** (it / max_iters)
            
            # Choose a random hexagon to move
            idx = np.random.randint(N)
            
            # Random perturbation
            dx = np.random.normal(0, 0.3)
            dy = np.random.normal(0, 0.3)
            dangle = np.random.uniform(-15, 15)
            
            new_pos = current.copy()
            new_pos[idx, 0] += dx
            new_pos[idx, 1] += dy
            new_pos[idx, 2] += dangle
            
            # Check validity
            if not check_validity(new_pos):
                continue
            
            new_side = required_outer_side(new_pos)
            delta = new_side - current_side
            
            # Accept if better or with probability exp(-delta/T)
            if delta < 0 or np.random.random() < np.exp(-delta / max(T, 1e-9)):
                current = new_pos
                current_side = new_side
                if current_side < best_side:
                    best = current.copy()
                    best_side = current_side
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1
                
            # Optional: restart logic if stuck
            if no_improve > 5000 and it > max_iters // 4:
                # Perturb current state more significantly
                idxs = np.random.choice(N, size=N, replace=False)
                for idx in idxs:
                    current[idx, 0] += np.random.normal(0, 1.0)
                    current[idx, 1] += np.random.normal(0, 1.0)
                    current[idx, 2] += np.random.uniform(-30, 30)
                if check_validity(current):
                    current_side = required_outer_side(current)
                else:
                    # Revert
                    current = best.copy()
                    current_side = best_side
                no_improve = 0

        return best, best_side

    # Initial configurations
    configs = []
    
    # 1. Hexagonal lattice (compact)
    coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1)
    ]
    init1 = []
    for q, r in coords:
        x = np.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        init1.append([x, y, 0.0])
    configs.append(np.array(init1))
    
    # 2. Circular arrangement
    init2 = np.zeros((N, 3))
    for i in range(N):
        angle = 2 * np.pi * i / N
        r = 2.5
        init2[i, 0] = r * np.cos(angle)
        init2[i, 1] = r * np.sin(angle)
        init2[i, 2] = 0.0
    configs.append(init2)
    
    # 3. Random
    np.random.seed(42)
    init3 = np.random.uniform(-4, 4, (N, 3))
    configs.append(init3)
    
    best_positions = None
    best_side = float('inf')
    
    for init in configs:
        if time.time() > deadline:
            break
        
        # Ensure initial config is valid; if not, try to fix
        if not check_validity(init):
            for scale in [1.5, 2.0, 2.5, 3.0]:
                scaled = init.copy()
                scaled[:, 0] *= scale
                scaled[:, 1] *= scale
                if check_validity(scaled):
                    init = scaled
                    break
            else:
                continue
        
        result_pos, result_side = simulated_annealing(init)
        
        if result_pos is not None and result_side < best_side:
            best_side = result_side
            best_positions = result_pos
    
    # Fallback
    if best_positions is None:
        # Use a simple linear arrangement
        best_positions = np.array([[i * 2.5, 0.0, 0.0] for i in range(N)])
        best_side = required_outer_side(best_positions)
    
    # Final validation
    if not check_validity(best_positions):
        # Last resort: spread out linearly with large gap
        best_positions = np.array([[i * 4.0, 0.0, 0.0] for i in range(N)])
        best_side = required_outer_side(best_positions)
    
    # Calculate final exact side
    final_side = required_outer_side(best_positions)
    
    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), float(best_positions[i, 2])])
        
    return inner, (0.0, 0.0), float(final_side), 0.0

# EVOLVE-BLOCK-END
