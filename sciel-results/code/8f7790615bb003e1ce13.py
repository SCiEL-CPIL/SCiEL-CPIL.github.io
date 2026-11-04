import numpy as np
import random

N = 26

def construct():
    # PSO Parameters
    num_particles = 150
    num_iterations = 200
    w_init = 0.9
    w_end = 0.4
    c1 = 1.5 # Cognitive
    c2 = 1.5 # Social
    
    # Initialize particles
    particles = np.zeros((num_particles, 3 * N))
    velocities = np.zeros((num_particles, 3 * N))
    
    # Initialize positions
    for i in range(num_particles):
        for j in range(N):
            particles[i, 3*j] = np.random.uniform(0.1, 0.9)
            particles[i, 3*j+1] = np.random.uniform(0.1, 0.9)
            particles[i, 3*j+2] = np.random.uniform(0.05, 0.15)
            
    pbest = particles.copy()
    pbest_fitness = np.zeros(num_particles)
    
    gbest = None
    gbest_fitness = -np.inf
    
    # Vectorized Fitness function
    def get_fitness(pos):
        x = pos[0::3]
        y = pos[1::3]
        r = pos[2::3]
        
        # Boundary penalty
        pen_left = np.maximum(0, r - x)**2
        pen_right = np.maximum(0, x - (1 - r))**2
        pen_bottom = np.maximum(0, r - y)**2
        pen_top = np.maximum(0, y - (1 - r))**2
        boundary_penalty = np.sum(pen_left) + np.sum(pen_right) + np.sum(pen_bottom) + np.sum(pen_top)
        
        # Overlap penalty
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist_sq = dx**2 + dy**2
        min_dist = r[:, None] + r[None, :]
        
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        
        dist_sq_masked = dist_sq[mask]
        min_dist_masked = min_dist[mask]
        
        overlap_mask = dist_sq_masked < min_dist_masked**2
        
        if np.any(overlap_mask):
            ds = dist_sq_masked[overlap_mask]
            md = min_dist_masked[overlap_mask]
            dist = np.sqrt(ds)
            overlap = md - dist
            overlap_penalty = np.sum(overlap**2)
        else:
            overlap_penalty = 0.0
            
        penalty = boundary_penalty + overlap_penalty
        score = np.sum(r) - 100.0 * penalty
        return score

    # Initialize fitness
    for i in range(num_particles):
        pbest_fitness[i] = get_fitness(particles[i])
        if pbest_fitness[i] > gbest_fitness:
            gbest_fitness = pbest_fitness[i]
            gbest = particles[i].copy()

    for it in range(num_iterations):
        # Decay w
        w = w_init - (w_init - w_end) * (it / num_iterations)
        
        for i in range(num_particles):
            r1 = np.random.rand(3 * N)
            r2 = np.random.rand(3 * N)
            
            velocities[i] = (w * velocities[i] + 
                             c1 * r1 * (pbest[i] - particles[i]) + 
                             c2 * r2 * (gbest - particles[i]))
            
            particles[i] += velocities[i]
            
            # Clamp
            particles[i, 0::3] = np.clip(particles[i, 0::3], 0.0, 1.0)
            particles[i, 1::3] = np.clip(particles[i, 1::3], 0.0, 1.0)
            particles[i, 2::3] = np.clip(particles[i, 2::3], 0.01, 0.5)
            
            fitness = get_fitness(particles[i])
            
            if fitness > pbest_fitness[i]:
                pbest_fitness[i] = fitness
                pbest[i] = particles[i].copy()
                
                if fitness > gbest_fitness:
                    gbest_fitness = fitness
                    gbest = particles[i].copy()
                    
    # Extract best result
    x = gbest[0::3].copy()
    y = gbest[1::3].copy()
    r = gbest[2::3].copy()
    
    # Local Refinement: Iterative Projection and Radius Growth
    for _ in range(100):
        # Step 1: Fix boundaries
        for i in range(N):
            if x[i] < r[i]: x[i] = r[i]
            elif x[i] > 1 - r[i]: x[i] = 1 - r[i]
            if y[i] < r[i]: y[i] = r[i]
            elif y[i] > 1 - r[i]: y[i] = 1 - r[i]
            
        # Step 2: Fix overlaps (push apart)
        for _iter in range(10):
            moved = False
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[j] - x[i]
                    dy = y[j] - y[i]
                    dist_sq = dx*dx + dy*dy
                    min_dist = r[i] + r[j]
                    if dist_sq < min_dist * min_dist:
                        dist = np.sqrt(dist_sq)
                        if dist < 1e-8:
                            x[j] += 0.001
                            y[j] += 0.001
                            moved = True
                            continue
                        overlap = min_dist - dist
                        push_x = (overlap * dx) / (2 * dist)
                        push_y = (overlap * dy) / (2 * dist)
                        x[i] -= push_x
                        y[i] -= push_y
                        x[j] += push_x
                        y[j] += push_y
                        moved = True
                        x[i] = np.clip(x[i], r[i], 1 - r[i])
                        y[i] = np.clip(y[i], r[i], 1 - r[i])
                        x[j] = np.clip(x[j], r[j], 1 - r[j])
                        y[j] = np.clip(y[j], r[j], 1 - r[j])
            if not moved:
                break
        
        # Step 3: Try to grow radii
        growth = 0.001
        r += growth
        r = np.clip(r, 0.01, 0.5)
        
    # Final strict validity check
    valid = True
    for i in range(N):
        if x[i] < r[i] - 1e-9 or x[i] > 1 - r[i] + 1e-9 or y[i] < r[i] - 1e-9 or y[i] > 1 - r[i] + 1e-9:
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
            
    if valid:
        return np.column_stack((x, y, r))
    else:
        # Fallback: Try to find a valid subset or use grid
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
