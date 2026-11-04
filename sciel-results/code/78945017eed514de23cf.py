import numpy as np
import random

N = 26

def construct():
    # PSO Parameters
    num_particles = 100
    num_iterations = 300
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
    
    # Fitness function with smooth penalty
    def get_fitness(pos):
        x = pos[0::3]
        y = pos[1::3]
        r = pos[2::3]
        
        penalty = 0.0
        
        # Boundary violations
        for i in range(N):
            if x[i] < r[i]:
                penalty += (r[i] - x[i])**2
            elif x[i] > 1 - r[i]:
                penalty += (x[i] - (1 - r[i]))**2
            if y[i] < r[i]:
                penalty += (r[i] - y[i])**2
            elif y[i] > 1 - r[i]:
                penalty += (y[i] - (1 - r[i]))**2
        
        # Overlap violations
        for i in range(N):
            for j in range(i + 1, N):
                dist_sq = (x[i] - x[j])**2 + (y[i] - y[j])**2
                min_dist = r[i] + r[j]
                if dist_sq < min_dist * min_dist:
                    dist = np.sqrt(dist_sq)
                    overlap = min_dist - dist
                    penalty += overlap**2
        
        score = np.sum(r) - 10.0 * penalty
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
