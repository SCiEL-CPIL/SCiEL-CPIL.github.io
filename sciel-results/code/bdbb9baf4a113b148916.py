import numpy as np
import random

N = 26

def construct():
    # PSO Parameters
    num_particles = 100
    num_iterations = 200
    w_init = 0.9
    w_end = 0.4
    c1 = 1.5 # Cognitive
    c2 = 1.5 # Social
    
    # Initialize particles
    particles = np.zeros((num_particles, 3 * N))
    velocities = np.zeros((num_particles, 3 * N))
    
    # Initialize positions: Start with a grid-like structure
    side = int(np.ceil(np.sqrt(N)))
    grid_xs = []
    grid_ys = []
    for i in range(side):
        for j in range(side):
            if len(grid_xs) < N:
                grid_xs.append((i + 0.5) / side)
                grid_ys.append((j + 0.5) / side)
                
    base_r = 0.5 / side
    
    for i in range(num_particles):
        for j in range(N):
            particles[i, 3*j] = grid_xs[j] + np.random.uniform(-0.05, 0.05)
            particles[i, 3*j+1] = grid_ys[j] + np.random.uniform(-0.05, 0.05)
            particles[i, 3*j+2] = base_r + np.random.uniform(-0.02, 0.02)
            
            particles[i, 3*j] = np.clip(particles[i, 3*j], 0.0, 1.0)
            particles[i, 3*j+1] = np.clip(particles[i, 3*j+1], 0.0, 1.0)
            particles[i, 3*j+2] = np.clip(particles[i, 3*j+2], 0.01, 0.5)
            
    pbest = particles.copy()
    pbest_fitness = np.zeros(num_particles)
    
    gbest = None
    gbest_fitness = -np.inf
    
    # Vectorized Fitness function for batch
    def get_fitness_batch(pos_batch):
        x = pos_batch[:, 0::3]
        y = pos_batch[:, 1::3]
        r = pos_batch[:, 2::3]
        
        # Boundary violations
        viol_x_low = np.maximum(0, r - x)
        viol_x_high = np.maximum(0, x - (1 - r))
        viol_y_low = np.maximum(0, r - y)
        viol_y_high = np.maximum(0, y - (1 - r))
        
        boundary_penalty = np.sum(viol_x_low**2 + viol_x_high**2 + viol_y_low**2 + viol_y_high**2, axis=1)
        
        # Overlap violations
        x_diff = x[:, :, None] - x[:, None, :]
        y_diff = y[:, :, None] - y[:, None, :]
        dist_sq = x_diff**2 + y_diff**2
        
        r_sum = r[:, :, None] + r[:, None, :]
        
        dist = np.sqrt(dist_sq + 1e-12)
        
        overlap = np.maximum(0, r_sum - dist)
        
        overlap_penalty = np.sum(overlap**2, axis=(1, 2))
        
        penalty = boundary_penalty + overlap_penalty
        
        score = np.sum(r, axis=1) - 10.0 * penalty
        return score

    # Initialize fitness
    initial_fitness = get_fitness_batch(particles)
    pbest_fitness = initial_fitness
    gbest_idx = np.argmax(pbest_fitness)
    gbest_fitness = pbest_fitness[gbest_idx]
    gbest = particles[gbest_idx].copy()

    for it in range(num_iterations):
        # Decay w
        w = w_init - (w_init - w_end) * (it / num_iterations)
        
        # Vectorized update
        r1 = np.random.rand(num_particles, 3 * N)
        r2 = np.random.rand(num_particles, 3 * N)
        
        velocities = (w * velocities + 
                      c1 * r1 * (pbest - particles) + 
                      c2 * r2 * (gbest[None, :] - particles))
        
        particles += velocities
        
        # Clamp
        particles[:, 0::3] = np.clip(particles[:, 0::3], 0.0, 1.0)
        particles[:, 1::3] = np.clip(particles[:, 1::3], 0.0, 1.0)
        particles[:, 2::3] = np.clip(particles[:, 2::3], 0.01, 0.5)
        
        # Calculate fitness for all particles
        fitness = get_fitness_batch(particles)
        
        # Update pbest
        mask = fitness > pbest_fitness
        pbest_fitness[mask] = fitness[mask]
        pbest[mask] = particles[mask]
        
        # Update gbest
        current_gbest_idx = np.argmax(fitness)
        if fitness[current_gbest_idx] > gbest_fitness:
            gbest_fitness = fitness[current_gbest_idx]
            gbest = particles[current_gbest_idx].copy()
            
    # Extract best result
    x = gbest[0::3]
    y = gbest[1::3]
    r = gbest[2::3]
    
    # Final strict validity check
    valid = True
    for i in range(N):
        if x[i] < r[i] or x[i] > 1 - r[i] or y[i] < r[i] or y[i] > 1 - r[i]:
            valid = False
            break
        for j in range(i + 1, N):
            dist_sq = (x[i] - x[j])**2 + (y[i] - y[j])**2
            min_dist = r[i] + r[j]
            if dist_sq < min_dist * min_dist - 1e-9:
                valid = False
                break
        if not valid:
            break
            
    if not valid:
        # Repair by scaling down radii
        scale = 1.0
        while not valid and scale > 0.1:
            r_new = r * scale
            valid = True
            for i in range(N):
                if x[i] < r_new[i] or x[i] > 1 - r_new[i] or y[i] < r_new[i] or y[i] > 1 - r_new[i]:
                    valid = False
                    break
                for j in range(i + 1, N):
                    dist_sq = (x[i] - x[j])**2 + (y[i] - y[j])**2
                    min_dist = r_new[i] + r_new[j]
                    if dist_sq < min_dist * min_dist - 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                scale *= 0.95
            r = r_new
            
    if valid:
        return np.column_stack((x, y, r))
    else:
        # Fallback: Grid
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) < N:
                    cx = (i + 0.5) / side
                    cy = (j + 0.5) / side
                    r = 0.5 / side - 0.01
                    cells.append([cx, cy, r])
        return np.array(cells[:N])
