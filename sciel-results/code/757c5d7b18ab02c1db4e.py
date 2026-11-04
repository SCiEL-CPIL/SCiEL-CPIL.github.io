import math
import time
import numpy as np
import random
from scipy.optimize import minimize

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    sqrt3 = math.sqrt(3)
    
    # Precompute unit hexagon vertices (pointy-top, vertex at 0 deg)
    unit_verts_0 = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])
    
    def get_hex_verts(cx, cy, side, angle_rad):
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        verts = unit_verts_0 @ rot.T * side
        verts += np.array([cx, cy])
        return verts
    
    def calc_min_outer_side(positions, angles, outer_angle_rad=0.0):
        """
        Calculate the minimum side length of a regular hexagon centered at (0,0) 
        with rotation outer_angle_rad that contains all inner hexagons.
        """
        theta_rad = outer_angle_rad
        normals = []
        for k in range(6):
            ang = theta_rad + math.pi/6 + k * math.pi/3
            normals.append([math.cos(ang), math.sin(ang)])
        normals = np.array(normals)  # (6, 2)
        
        max_proj = 0.0
        
        for i in range(N):
            cx, cy = positions[i]
            ang = angles[i]
            verts = get_hex_verts(cx, cy, 1.0, ang)
            projs = verts @ normals.T  # (6, 6)
            max_proj_i = np.max(projs)
            if max_proj_i > max_proj:
                max_proj = max_proj_i
        
        return max_proj * 2.0 / math.sqrt(3)
    
    def sat_gap(poly1, poly2):
        """
        Compute the minimum gap (positive if separated, negative if overlapping)
        between two convex polygons using SAT.
        """
        min_gap = float('inf')
        
        # Axes from poly1
        for i in range(6):
            edge = poly1[(i+1)%6] - poly1[i]
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-9:
                continue
            axis = normal / norm
            
            proj1 = poly1 @ axis
            proj2 = poly2 @ axis
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            gap = max(min1 - max2, min2 - min1)
            if gap < 0:
                min_gap = min(min_gap, gap)
            else:
                return gap  # Separated
        
        # Axes from poly2
        for i in range(6):
            edge = poly2[(i+1)%6] - poly2[i]
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-9:
                continue
            axis = normal / norm
            
            proj1 = poly1 @ axis
            proj2 = poly2 @ axis
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            gap = max(min1 - max2, min2 - max1)
            if gap < 0:
                min_gap = min(min_gap, gap)
            else:
                return gap  # Separated
        
        return min_gap
    
    def is_valid_config(positions, angles, tol=1e-5):
        polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
        for i in range(N):
            for j in range(i+1, N):
                if sat_gap(polys[i], polys[j]) < -tol:
                    return False
        return True
    
    def objective(params, penalty_weight=50.0):
        if time.time() > deadline - 5:
            return 1e6
        positions = params[:2*N].reshape(N, 2)
        angles = params[2*N:]
        
        polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                gap = sat_gap(polys[i], polys[j])
                if gap < 0:
                    penalty += penalty_weight * gap**2
        
        base_score = calc_min_outer_side(positions, angles, 0.0)
        return base_score + penalty
    
    # --- Generate diverse initial configurations ---
    
    def generate_honeycomb_config(radius):
        """Generate hexagons in a honeycomb pattern."""
        positions = []
        for q in range(-radius, radius+1):
            for r in range(-radius, radius+1):
                if abs(q + r) <= radius:
                    x = 1.5 * q
                    y = sqrt3 * (r + 0.5 * q)
                    positions.append((x, y))
        positions.sort(key=lambda p: p[0]**2 + p[1]**2)
        return np.array(positions[:N])
    
    def generate_ring_config():
        """Generate hexagons in a ring pattern around center."""
        positions = []
        # Center
        positions.append((0, 0))
        # First ring: 6 hexagons at distance sqrt(3)
        for k in range(6):
            ang = k * math.pi / 3
            x = sqrt3 * math.cos(ang)
            y = sqrt3 * math.sin(ang)
            positions.append((x, y))
        # Second ring: 6 more hexagons at distance 2*sqrt(3)
        for k in range(6):
            ang = k * math.pi / 3 + math.pi/6
            x = 2 * sqrt3 * math.cos(ang)
            y = 2 * sqrt3 * math.sin(ang)
            positions.append((x, y))
        return np.array(positions[:N])
    
    def generate_random_config(seed=None):
        """Generate random configuration."""
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random.RandomState()
        
        positions = []
        angles = []
        for i in range(N):
            r = rng.uniform(0, 4)
            theta = rng.uniform(0, 2*math.pi)
            x = r * math.cos(theta)
            y = r * math.sin(theta)
            positions.append((x, y))
            angles.append(rng.uniform(0, 2*math.pi))
        
        pos_arr = np.array(positions)
        pos_arr -= np.mean(pos_arr, axis=0)
        return pos_arr, np.array(angles)
    
    def generate_grid_config():
        """Generate hexagons in a grid pattern."""
        positions = []
        for i in range(-2, 3):
            for j in range(-2, 3):
                x = i * 1.5
                y = i * sqrt3 / 2 + j * sqrt3
                positions.append((x, y))
        positions.sort(key=lambda p: p[0]**2 + p[1]**2)
        return np.array(positions[:N])
    
    # Generate diverse initial configurations
    initial_configs = []
    
    # Honeycomb patterns with different radii
    for radius in [2, 3]:
        pos = generate_honeycomb_config(radius)
        pos -= np.mean(pos, axis=0)
        initial_configs.append((pos, np.zeros(N)))
    
    # Ring pattern
    pos = generate_ring_config()
    pos -= np.mean(pos, axis=0)
    initial_configs.append((pos, np.zeros(N)))
    
    # Grid pattern
    pos = generate_grid_config()
    pos -= np.mean(pos, axis=0)
    initial_configs.append((pos, np.zeros(N)))
    
    # Random configurations
    for seed in range(10):
        pos, ang = generate_random_config(seed)
        initial_configs.append((pos, ang))
    
    best_score = float('inf')
    best_params = None
    
    # Evaluate each initial configuration with optimization
    for idx, (init_pos, init_ang) in enumerate(initial_configs):
        if time.time() > deadline - 30:
            break
        
        x0 = np.concatenate([init_pos.flatten(), init_ang])
        
        # Try multiple optimization methods
        methods = [
            ('Nelder-Mead', {'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}),
            ('Powell', {'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8}),
        ]
        
        for method, options in methods:
            if time.time() > deadline - 20:
                break
            
            try:
                res = minimize(objective, x0, method=method, options=options)
                if res.fun < best_score:
                    new_positions = res.x[:2*N].reshape(N, 2)
                    new_angles = res.x[2*N:]
                    if is_valid_config(new_positions, new_angles):
                        final_side = calc_min_outer_side(new_positions, new_angles, 0.0)
                        if final_side < best_score:
                            best_score = final_side
                            best_params = res.x.copy()
            except Exception:
                pass
    
    # If we still don't have a good result, try a more intensive search
    if best_params is None or best_score > 5.0:
        # Use the best initial configuration as starting point
        if best_params is None:
            x0 = np.concatenate([generate_honeycomb_config(2).flatten(), np.zeros(N)])
        else:
            x0 = best_params.copy()
        
        # Run simulated annealing for longer
        current_params = x0.copy()
        current_score = objective(current_params)
        current_best = best_score if best_params is not None else current_score
        current_best_params = best_params.copy() if best_params is not None else current_params.copy()
        
        T = 0.3
        cooling_rate = 0.9999
        step_size = 0.2
        max_iter = 30000
        
        np.random.seed(123)
        random.seed(123)
        
        for iteration in range(max_iter):
            if time.time() > deadline - 10:
                break
            
            # Generate perturbation
            new_params = current_params.copy()
            
            # Randomly perturb 1-2 hexagons
            n_perturb = random.randint(1, 2)
            for _ in range(n_perturb):
                idx = random.randint(0, N-1)
                new_params[2*idx] += np.random.randn() * step_size
                new_params[2*idx+1] += np.random.randn() * step_size
                new_params[2*N + idx] += np.random.randn() * step_size * 0.3  # Radians
                
            new_score = objective(new_params)
            
            # Acceptance criterion
            if new_score < current_score:
                current_params = new_params
                current_score = new_score
                # Check if valid and better than best
                if current_score < current_best:
                    new_positions = current_params[:2*N].reshape(N, 2)
                    new_angles = current_params[2*N:]
                    if is_valid_config(new_positions, new_angles):
                        current_best = calc_min_outer_side(new_positions, new_angles, 0.0)
                        current_best_params = current_params.copy()
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
                step_size *= 0.97
        
        best_score = current_best
        best_params = current_best_params
    
    # Final local refinement
    if best_params is not None:
        try:
            res = minimize(
                objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True}
            )
            if res.fun < best_score:
                new_positions = res.x[:2*N].reshape(N, 2)
                new_angles = res.x[2*N:]
                if is_valid_config(new_positions, new_angles):
                    final_side = calc_min_outer_side(new_positions, new_angles, 0.0)
                    if final_side < best_score:
                        best_score = final_side
                        best_params = res.x
        except Exception:
            pass
    
    # Final validation and output
    final_positions = best_params[:2*N].reshape(N, 2)
    final_angles = best_params[2*N:]
    final_side = calc_min_outer_side(final_positions, final_angles, 0.0)
    
    inner = []
    for i in range(N):
        angle_deg = math.degrees(final_angles[i])
        inner.append([final_positions[i][0], final_positions[i][1], angle_deg])
    
    return inner, (0.0, 0.0), final_side, 0.0

if __name__ == "__main__":
    construct()
