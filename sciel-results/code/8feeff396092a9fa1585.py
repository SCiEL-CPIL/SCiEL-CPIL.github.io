import math
import time
import numpy as np
import random

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 135.0
    
    # Precompute unit hexagon vertices for angle 0 (pointy-top orientation)
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
        theta_rad = math.radians(outer_angle)
        normals = []
        for k in range(6):
            ang = theta_rad + math.pi/6 + k * math.pi/3
            normals.append([math.cos(ang), math.sin(ang)])
        normals = np.array(normals)
        
        max_proj = 0.0
        
        for i in range(N):
            cx, cy = positions[i]
            ang = angles[i]
            verts = get_hex_verts(cx, cy, 1.0, ang)
            projs = verts @ normals.T
            max_proj_i = np.max(projs)
            if max_proj_i > max_proj:
                max_proj = max_proj_i
        
        return max_proj * 2.0 / math.sqrt(3)
    
    def sat_gap(poly1, poly2):
        min_gap = float('inf')
        
        for i in range(6):
            edge1 = poly1[(i+1)%6] - poly1[i]
            normal1 = np.array([-edge1[1], edge1[0]])
            norm = np.linalg.norm(normal1)
            if norm < 1e-9: continue
            axis = normal1 / norm
            
            proj1 = poly1 @ axis
            proj2 = poly2 @ axis
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            gap = max(min1 - max2, min2 - max1)
            if gap < 0:
                min_gap = min(min_gap, gap)
            else:
                return gap
                
        for i in range(6):
            edge2 = poly2[(i+1)%6] - poly2[i]
            normal2 = np.array([-edge2[1], edge2[0]])
            norm = np.linalg.norm(normal2)
            if norm < 1e-9: continue
            axis = normal2 / norm
            
            proj1 = poly1 @ axis
            proj2 = poly2 @ axis
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            gap = max(min1 - max2, min2 - max1)
            if gap < 0:
                min_gap = min(min_gap, gap)
            else:
                return gap
                
        return min_gap
    
    def is_valid(positions, angles, tol=1e-5):
        polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
        for i in range(N):
            for j in range(i+1, N):
                if sat_gap(polys[i], polys[j]) < -tol:
                    return False
        return True
    
    def compute_penalty(params, penalty_weight=100.0):
        positions = params[:2*N].reshape(N, 2)
        angles = params[2*N:]
        side_len = calc_min_outer_side(positions, angles, 0.0)
        
        penalty = 0.0
        polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
        for i in range(N):
            for j in range(i+1, N):
                gap = sat_gap(polys[i], polys[j])
                if gap < 0:
                    penalty += penalty_weight * gap**2
        
        return side_len + penalty
    
    # Generate initial configurations using different lattice rotations
    sqrt3 = math.sqrt(3)
    
    def generate_lattice_init(rotation_angle=0.0):
        lattice_points = []
        for i in range(-3, 4):
            for j in range(-3, 4):
                x = i * 1.5
                y = i * sqrt3 / 2.0 + j * sqrt3
                lattice_points.append((x, y))
        
        # Sort by distance from origin
        lattice_points.sort(key=lambda p: p[0]**2 + p[1]**2)
        init_positions = np.array(lattice_points[:12])
        
        # Apply rotation
        rot_angle = math.radians(rotation_angle)
        cos_r = math.cos(rot_angle)
        sin_r = math.sin(rot_angle)
        rot_matrix = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
        init_positions = init_positions @ rot_matrix.T
        
        # Center
        init_positions -= np.mean(init_positions, axis=0)
        
        return init_positions
    
    # Multiple initial configurations with different rotations
    init_configs = []
    for rot in [0, 15, 30, 45, 60, 75, 90]:
        pos = generate_lattice_init(rot)
        ang = np.zeros(N)
        init_configs.append(np.concatenate([pos.flatten(), ang]))
    
    # Also add a ring configuration
    ring_pos = np.zeros((N, 2))
    for i in range(N):
        angle = i * 2 * math.pi / N
        ring_pos[i] = [2.0 * math.cos(angle), 2.0 * math.sin(angle)]
    ring_ang = np.zeros(N)
    init_configs.append(np.concatenate([ring_pos.flatten(), ring_ang]))
    
    # Simulated annealing with multiple restarts
    np.random.seed(42)
    random.seed(42)
    
    best_params = init_configs[0].copy()
    best_score = compute_penalty(init_configs[0])
    
    # Phase 1: Coarse search with simulated annealing
    T_init = 0.5
    cooling_rate = 0.9995
    step_size_init = 0.3
    max_iter_per_restart = 30000
    
    for restart_idx, x0 in enumerate(init_configs):
        if time.time() > deadline - 20:
            break
        
        current_params = x0.copy()
        current_score = compute_penalty(current_params)
        
        if current_score < best_score:
            best_score = current_score
            best_params = current_params.copy()
        
        T = T_init
        step_size = step_size_init
        
        for iteration in range(max_iter_per_restart):
            if time.time() > deadline - 20:
                break
            
            # Generate perturbation
            new_params = current_params.copy()
            
            # Randomly perturb 1-3 hexagons
            n_perturb = random.randint(1, 3)
            for _ in range(n_perturb):
                idx = random.randint(0, N-1)
                new_params[2*idx] += np.random.randn() * step_size
                new_params[2*idx+1] += np.random.randn() * step_size
                new_params[2*N + idx] += np.random.randn() * step_size * 30
            
            new_score = compute_penalty(new_params)
            
            # Acceptance criterion
            if new_score < current_score:
                current_params = new_params
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_params = current_params.copy()
            else:
                delta = new_score - current_score
                if T > 0:
                    acceptance_prob = math.exp(-delta / T)
                    if random.random() < acceptance_prob:
                        current_params = new_params
                        current_score = new_score
            
            # Cool down
            T *= cooling_rate
            
            # Reduce step size slowly
            if iteration % 500 == 0:
                step_size *= 0.95
    
    # Phase 2: Fine-tuning with smaller steps
    if time.time() < deadline - 10:
        current_params = best_params.copy()
        current_score = best_score
        
        T = 0.1
        cooling_rate = 0.999
        step_size = 0.05
        max_iter = 20000
        
        for iteration in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            new_params = current_params.copy()
            
            n_perturb = random.randint(1, 2)
            for _ in range(n_perturb):
                idx = random.randint(0, N-1)
                new_params[2*idx] += np.random.randn() * step_size
                new_params[2*idx+1] += np.random.randn() * step_size
                new_params[2*N + idx] += np.random.randn() * step_size * 15
            
            new_score = compute_penalty(new_params)
            
            if new_score < current_score:
                current_params = new_params
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_params = current_params.copy()
            else:
                delta = new_score - current_score
                if T > 0:
                    acceptance_prob = math.exp(-delta / T)
                    if random.random() < acceptance_prob:
                        current_params = new_params
                        current_score = new_score
            
            T *= cooling_rate
            if iteration % 300 == 0:
                step_size *= 0.95
    
    # Phase 3: Very fine polish
    if time.time() < deadline - 5:
        current_params = best_params.copy()
        current_score = best_score
        
        T = 0.01
        cooling_rate = 0.998
        step_size = 0.01
        max_iter = 10000
        
        for iteration in range(max_iter):
            if time.time() > deadline - 2:
                break
            
            new_params = current_params.copy()
            
            idx = random.randint(0, N-1)
            new_params[2*idx] += np.random.randn() * step_size
            new_params[2*idx+1] += np.random.randn() * step_size
            new_params[2*N + idx] += np.random.randn() * step_size * 5
            
            new_score = compute_penalty(new_params)
            
            if new_score < current_score:
                current_params = new_params
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_params = current_params.copy()
            else:
                delta = new_score - current_score
                if T > 0:
                    acceptance_prob = math.exp(-delta / T)
                    if random.random() < acceptance_prob:
                        current_params = new_params
                        current_score = new_score
            
            T *= cooling_rate
    
    # Check if best solution is valid
    final_positions = best_params[:2*N].reshape(N, 2)
    final_angles = best_params[2*N:]
    
    if not is_valid(final_positions, final_angles, tol=1e-5):
        # Try to repair by using a simpler initial configuration
        fallback_positions = generate_lattice_init(0)
        fallback_angles = np.zeros(N)
        final_positions = fallback_positions
        final_angles = fallback_angles
    
    final_side = calc_min_outer_side(final_positions, final_angles, 0.0)
    
    inner = []
    for i in range(N):
        inner.append([final_positions[i][0], final_positions[i][1], final_angles[i]])
    
    return inner, (0.0, 0.0), final_side, 0.0

if __name__ == "__main__":
    construct()
