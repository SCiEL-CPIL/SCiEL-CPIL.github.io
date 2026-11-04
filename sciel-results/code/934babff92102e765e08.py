import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    # Precompute unit hexagon vertices for angle 0 (flat-top)
    # Vertices at 0, 60, ..., 300 degrees
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
        # Normals for the outer hexagon sides
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

    def find_best_outer_angle(positions, angles, coarse=False):
        """
        Find the optimal rotation of the outer hexagon that minimizes its side length.
        """
        best_side = float('inf')
        best_angle = 0.0
        
        if coarse:
            test_angles = np.linspace(0, 60, 13)
        else:
            test_angles = np.linspace(0, 60, 181)

        for test_angle in test_angles:
            side = calc_min_outer_side(positions, angles, test_angle)
            if side < best_side:
                best_side = side
                best_angle = test_angle
                
        return best_side, best_angle

    def sat_gap(poly1, poly2):
        """
        Returns the minimum separation distance between two convex polygons.
        If they overlap, returns a negative value (approximate penetration depth).
        If they are separated, returns the positive distance.
        """
        min_gap = float('inf')
        
        # Axes from poly1 edges
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
            
            # Gap on this axis
            gap = max(min1 - max2, min2 - max1)
            if gap < 0:
                min_gap = min(min_gap, gap)
            else:
                return gap
                
        # Axes from poly2 edges
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

    # --- PHASE 1: GENERATE INITIAL POPULATION FOR DIFFERENTIAL EVOLUTION ---
    # We create diverse starting points using different lattice topologies and random perturbations.
    
    sqrt3 = math.sqrt(3)
    
    def generate_lattice_config(scale=1.0, rot_deg=0.0, angle_offset=0.0, noise=0.0):
        """
        Generate a configuration based on a hexagonal lattice.
        """
        base_v1 = np.array([1.5 * scale, sqrt3/2 * scale])
        base_v2 = np.array([0, sqrt3 * scale])
        
        points = []
        for i in range(-4, 5):
            for j in range(-4, 5):
                p = i * base_v1 + j * base_v2
                points.append(p)
        
        # Sort by distance from origin
        points.sort(key=lambda p: np.dot(p, p))
        
        # Take the first 12
        selected = np.array(points[:12])
        
        # Center the cluster
        centroid = np.mean(selected, axis=0)
        selected -= centroid
        
        # Rotate
        if rot_deg != 0:
            rot_rad = math.radians(rot_deg)
            rot_mat = np.array([[math.cos(rot_rad), -math.sin(rot_rad)], 
                                [math.sin(rot_rad), math.cos(rot_rad)]])
            selected = selected @ rot_mat.T
            
        # Add noise
        if noise > 0:
            selected += np.random.randn(N, 2) * noise
        
        # Assign angles
        angles = np.full(N, rot_deg + angle_offset)
        
        return selected, angles

    # Create initial population
    pop_size = 40
    population = []
    
    # Some structured starts
    for _ in range(5):
        scale = np.random.uniform(0.95, 1.05)
        rot = np.random.choice([0, 30, 60, 90])
        pos, ang = generate_lattice_config(scale=scale, rot_deg=rot, noise=0.05)
        population.append(np.concatenate([pos.flatten(), ang]))
        
    # Some random starts
    for _ in range(pop_size - 5):
        # Random positions in a circle
        pos = np.random.randn(N, 2) * 3.0
        # Random angles
        ang = np.random.uniform(0, 60, N)
        population.append(np.concatenate([pos.flatten(), ang]))
        
    population = np.array(population)
    
    # --- PHASE 2: DIFFERENTIAL EVOLUTION ---
    # We implement a simple Differential Evolution loop.
    
    def de_objective(params):
        """
        Objective function for DE. Returns the outer side length plus a penalty for overlap.
        """
        positions = params[:2*N].reshape(N, 2)
        angles = params[2*N:]
        
        # Calculate outer side length (using a coarse outer angle search for speed)
        side_len, _ = find_best_outer_angle(positions, angles, coarse=True)
        
        # Calculate overlap penalty
        penalty = 0.0
        polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
        for i in range(N):
            for j in range(i+1, N):
                gap = sat_gap(polys[i], polys[j])
                if gap < 0:
                    penalty += 1000.0 * gap**2
                    
        return side_len + penalty

    # DE parameters
    F = 0.8  # Scaling factor
    CR = 0.9  # Crossover rate
    max_iter = 200
    best_individual = None
    best_fitness = float('inf')
    
    # Evaluate initial population
    fitnesses = np.array([de_objective(ind) for ind in population])
    best_idx = np.argmin(fitnesses)
    best_individual = population[best_idx].copy()
    best_fitness = fitnesses[best_idx]
    
    # DE loop
    for gen in range(max_iter):
        if time.time() > deadline - 20:
            break
            
        for i in range(pop_size):
            # Select three distinct random individuals
            r1, r2, r3 = np.random.choice(pop_size, 3, replace=False)
            
            # Create mutant vector
            mutant = population[r1] + F * (population[r2] - population[r3])
            
            # Crossover
            trial = population[i].copy()
            for j in range(len(mutant)):
                if np.random.random() < CR or j == np.random.randint(len(mutant)):
                    trial[j] = mutant[j]
            
            # Bounds: positions in [-10, 10], angles in [0, 60]
            trial[:2*N] = np.clip(trial[:2*N], -10, 10)
            trial[2*N:] = np.clip(trial[2*N:], 0, 60)
            
            # Evaluate trial
            trial_fitness = de_objective(trial)
            
            # Selection
            if trial_fitness < fitnesses[i]:
                population[i] = trial
                fitnesses[i] = trial_fitness
                
                if trial_fitness < best_fitness:
                    best_fitness = trial_fitness
                    best_individual = trial.copy()
                    
    # --- PHASE 3: LOCAL POLISHING ---
    # Use SLSQP or Nelder-Mead to refine the best individual found by DE.
    
    best_pos = best_individual[:2*N].reshape(N, 2)
    best_ang = best_individual[2*N:]
    
    # Check if valid
    if not is_valid(best_pos, best_ang, tol=1e-5):
        # If DE failed to find a valid config, fall back to the best lattice config
        print("DE failed to find valid config, using lattice fallback")
        pos, ang = generate_lattice_config(scale=1.0, rot_deg=0, noise=0)
        best_pos = pos
        best_ang = ang
    else:
        # Polish with Nelder-Mead
        def obj_penalty(params):
            positions = params[:2*N].reshape(N, 2)
            angles = params[2*N:]
            side_len, _ = find_best_outer_angle(positions, angles, coarse=True)
            penalty = 0.0
            polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
            for i in range(N):
                for j in range(i+1, N):
                    gap = sat_gap(polys[i], polys[j])
                    if gap < 0:
                        penalty += 1000.0 * gap**2
            return side_len + penalty

        try:
            res = minimize(
                obj_penalty, 
                best_individual, 
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True, 'disp': False}
            )
            
            pos = res.x[:2*N].reshape(N, 2)
            ang = res.x[2*N:]
            
            if is_valid(pos, ang, tol=1e-5):
                score, _ = find_best_outer_angle(pos, ang, coarse=True)
                if score < best_fitness:
                    best_pos = pos
                    best_ang = ang
        except Exception:
            pass

    # Finalize
    final_positions = best_pos
    final_angles = best_ang
    
    # Find the precise best outer angle for the final configuration
    final_side, final_outer_angle = find_best_outer_angle(final_positions, final_angles, coarse=False)
    
    inner = []
    for i in range(N):
        inner.append([final_positions[i][0], final_positions[i][1], final_angles[i]])
    
    return inner, (0.0, 0.0), final_side, final_outer_angle

if __name__ == "__main__":
    construct()
