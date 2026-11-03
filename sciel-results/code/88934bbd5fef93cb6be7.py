"""
Strategy: Discrete Genetic Algorithm on a Grid + Local Refinement
Instead of continuous force-directed methods, we model the packing as a discrete problem.
1. Define a high-resolution grid (e.g., 60x60) covering the square.
2. Represent a solution as a set of 200 grid indices.
3. Use a Genetic Algorithm (GA) to evolve the population:
   - Fitness: Minimum distance (pairwise and to walls) on the grid.
   - Selection: Tournament selection.
   - Crossover: Uniform crossover.
   - Mutation: Randomly swap a circle's position with another grid point or shift slightly.
4. After GA converges or time runs out, take the best discrete solution.
5. Convert grid indices to continuous coordinates.
6. Run a short, high-precision local refinement (gradient descent on soft-min) to maximize the radius.
"""

import numpy as np
import time
import math

N = 200
GRID_SIZE = 60  # 60x60 grid -> 3600 possible positions. 200 circles.

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    # 1. Grid Setup
    # Coordinates from -0.5 to 0.5
    # We want points strictly inside.
    # Let's use 60 points.
    # x_i = -0.5 + (i + 0.5) * (1.0 / GRID_SIZE)
    # This centers the grid points nicely.
    grid_coords = np.linspace(-0.5 + 0.5/GRID_SIZE, 0.5 - 0.5/GRID_SIZE, GRID_SIZE)
    
    # Create meshgrid of all possible positions
    gx, gy = np.meshgrid(grid_coords, grid_coords)
    all_points = np.column_stack((gx.ravel(), gy.ravel()))
    num_grid_points = len(all_points)
    
    # Precompute wall distances for each grid point
    # d_wall = min(0.5 - x, 0.5 + x, 0.5 - y, 0.5 + y) = 0.5 - max(|x|, |y|) ? No.
    # d_wall = min(0.5 - x, 0.5 - (-x), 0.5 - y, 0.5 - (-y)) = 0.5 - max(|x|, |y|) is wrong for square.
    # For square [-0.5, 0.5]:
    # dist to left wall: x - (-0.5) = x + 0.5
    # dist to right wall: 0.5 - x
    # dist to bottom wall: y + 0.5
    # dist to top wall: 0.5 - y
    # min_wall = min(x+0.5, 0.5-x, y+0.5, 0.5-y)
    wall_dists = np.min(np.stack([
        all_points[:, 0] + 0.5,
        0.5 - all_points[:, 0],
        all_points[:, 1] + 0.5,
        0.5 - all_points[:, 1]
    ]), axis=0)
    
    # Pairwise distance matrix between all grid points (num_grid_points x num_grid_points)
    # This is 3600x3600 ~ 13M floats ~ 100MB. Might be too big?
    # Let's check memory. 13M * 8 bytes = 104MB. Acceptable.
    # But computing it might take time.
    # Alternatively, compute on the fly for fitness?
    # GA evaluates many fitnesses. Precomputing is better if memory allows.
    # Let's try to be smarter: We only need distances between selected points.
    # But in GA, the set changes.
    # Let's stick to on-the-fly for fitness to save memory and initialization time, 
    # but optimize the fitness function.
    
    def compute_fitness(indices):
        """
        indices: array of shape (N,) containing indices into all_points.
        Returns: a score proportional to the radius.
        We want to maximize min(min_pair_dist/2, min_wall_dist).
        """
        pts = all_points[indices]
        
        # Wall distances
        min_wall = np.min(wall_dists[indices])
        
        # Pairwise distances
        # Use cdist for speed
        from scipy.spatial.distance import cdist
        dist_mat = cdist(pts, pts)
        np.fill_diagonal(dist_mat, np.inf)
        min_pair = np.min(dist_mat)
        
        r = min(min_wall, min_pair / 2.0)
        return r

    # GA Parameters
    pop_size = 50
    max_generations = 200
    mutation_rate = 0.1
    crossover_rate = 0.8
    
    # Initialize Population
    # Start with a few diverse random packings and one hex-like grid
    population = []
    
    # 1. Random initialization
    for _ in range(pop_size - 2):
        # Randomly select N unique grid points
        # To avoid clustering, we can prefer points that are far apart, but random is a good start
        # Actually, pure random on 3600 points for 200 circles will have many overlaps.
        # Fitness will be low (near 0).
        idx = np.random.choice(num_grid_points, N, replace=False)
        population.append(idx)
        
    # 2. Hexagonal grid initialization (mapped to nearest grid point)
    def hex_init():
        # Generate hex points
        d = 1.0 / 15.0 # Rough spacing
        pts = []
        for row in range(15):
            y = -0.5 + d/2 + row * d * math.sqrt(3)/2
            if y > 0.5: break
            x_offset = 0 if row % 2 == 0 else d/2
            x = -0.5 + d/2 + x_offset
            while x <= 0.5:
                if x >= -0.5:
                    pts.append([x, y])
                x += d
        pts = np.array(pts)
        if len(pts) < N:
             # Pad with random
             while len(pts) < N:
                 p = np.random.uniform(-0.5, 0.5, 2)
                 pts = np.vstack([pts, [p]])
        else:
             pts = pts[:N]
        
        # Snap to grid
        indices = []
        for p in pts:
            # Find closest grid point
            dists = np.sum((all_points - p)**2, axis=1)
            idx = np.argmin(dists)
            indices.append(idx)
            
        # Ensure uniqueness? 
        # If two points snap to same grid cell, we have a collision (dist 0).
        # We need to resolve collisions.
        # Simple approach: if duplicate, pick a random different neighbor.
        unique_indices = []
        used = set()
        for idx in indices:
            if idx not in used:
                unique_indices.append(idx)
                used.add(idx)
            else:
                # Find a nearby free point
                # Just pick random for now, GA will fix
                free = np.setdiff1d(np.arange(num_grid_points), list(used))
                if len(free) > 0:
                    new_idx = np.random.choice(free)
                    unique_indices.append(new_idx)
                    used.add(new_idx)
                    
        while len(unique_indices) < N:
             free = np.setdiff1d(np.arange(num_grid_points), list(used))
             if len(free) == 0: break
             new_idx = np.random.choice(free)
             unique_indices.append(new_idx)
             used.add(new_idx)
             
        return np.array(unique_indices[:N])

    population.append(hex_init())
    population.append(hex_init()) # Two hex inits

    # Evaluate initial fitness
    fitnesses = np.array([compute_fitness(ind) for ind in population])
    
    best_indices = population[np.argmax(fitnesses)]
    best_fitness = np.max(fitnesses)
    
    # GA Loop
    gen = 0
    while time.time() < deadline - 20.0: # Leave 20s for refinement
        if gen >= max_generations: break
        
        new_population = []
        
        # Elitism: Keep top 2
        top_indices = np.argsort(fitnesses)[-2:]
        for idx in top_indices:
            new_population.append(population[idx].copy())
            
        # Fill rest
        while len(new_population) < pop_size:
            # Selection
            # Tournament selection
            def tournament_select(pop, fit, k=3):
                candidates = np.random.choice(len(pop), k, replace=False)
                best_cand = candidates[np.argmax(fit[candidates])]
                return pop[best_cand].copy()
            
            parent1 = tournament_select(population, fitnesses)
            parent2 = tournament_select(population, fitnesses)
            
            # Crossover
            if np.random.random() < crossover_rate:
                # Uniform crossover
                mask = np.random.random(N) < 0.5
                child = np.where(mask, parent1, parent2)
            else:
                child = parent1.copy()
                
            # Mutation
            # With probability mutation_rate, mutate a random position
            # Mutation: swap with a random grid point? Or shift to a neighbor?
            # Swapping with a random point is a large jump.
            # Let's do: pick a random index in the child, and replace it with a random neighbor or random point.
            # To maintain uniqueness, we need to handle collisions.
            # Strategy: 
            # 1. Pick a random position i in the child.
            # 2. Pick a new grid point j (random or neighbor).
            # 3. If j is already in child, swap i and j? No, just replace i with j and remove duplicate?
            #    If we replace i with j, and j was at k, then k becomes empty? 
            #    We need a permutation.
            #    So, swap i and k if j is at k.
            #    If j is not in child, just replace i with j. But then we have duplicate j? No, i is replaced.
            #    Wait, if i is replaced by j, and j was not in child, we are fine.
            #    If j was in child at k, we have two j's. We should swap i and k.
            
            if np.random.random() < mutation_rate:
                i = np.random.randint(N)
                # Choose a mutation type
                if np.random.random() < 0.5:
                    # Small step: change to a neighbor
                    # Get current point coords
                    curr_idx = child[i]
                    curr_pt = all_points[curr_idx]
                    # Pick a random direction
                    dx = np.random.choice([-1, 0, 1])
                    dy = np.random.choice([-1, 0, 1])
                    if dx == 0 and dy == 0:
                        continue
                    # Find closest grid point to curr_pt + (dx, dy) * step
                    # Step size is 1/GRID_SIZE
                    target_pt = curr_pt + np.array([dx, dy]) * (1.0 / GRID_SIZE)
                    # Clamp to bounds? Grid points are already bounded.
                    # Find closest grid point
                    dists = np.sum((all_points - target_pt)**2, axis=1)
                    new_idx = np.argmin(dists)
                    
                    # Check if new_idx is already in child
                    if new_idx in child:
                        # Find index of new_idx
                        k = np.where(child == new_idx)[0][0]
                        # Swap i and k
                        child[i], child[k] = child[k], child[i]
                    else:
                        # Just replace
                        child[i] = new_idx
                        
                else:
                    # Large step: random point
                    new_idx = np.random.randint(num_grid_points)
                    if new_idx in child:
                        k = np.where(child == new_idx)[0][0]
                        child[i], child[k] = child[k], child[i]
                    else:
                        child[i] = new_idx

            new_population.append(child)
            
        # Evaluate new population
        population = new_population
        fitnesses = np.array([compute_fitness(ind) for ind in population])
        
        current_best = np.max(fitnesses)
        if current_best > best_fitness:
            best_fitness = current_best
            best_indices = population[np.argmax(fitnesses)]
            
        gen += 1
        
    # 2. Local Refinement
    # Convert best_indices to continuous coordinates
    centers = all_points[best_indices].copy()
    
    # Check time
    if time.time() > deadline - 10.0:
        return centers
        
    # Run a short gradient descent / force directed refinement
    # This recovers the precision lost by the grid.
    
    def refine(centers, time_limit=10.0):
        from scipy.spatial.distance import cdist
        current_r = compute_fitness(np.array([np.argmin(np.sum((all_points-c)**2, axis=1)) for c in centers])) # Approx
        # Better to just compute on continuous
        def compute_r_cont(c):
            wall_dists = np.min(np.stack([
                c[:, 0] + 0.5,
                0.5 - c[:, 0],
                c[:, 1] + 0.5,
                0.5 - c[:, 1]
            ]), axis=0)
            min_wall = np.min(wall_dists)
            dist_mat = cdist(c, c)
            np.fill_diagonal(dist_mat, np.inf)
            min_pair = np.min(dist_mat)
            return min(min_wall, min_pair / 2.0)
            
        best_c = centers.copy()
        best_r = compute_r_cont(centers)
        
        dt = 0.005
        temp = 0.1
        start = time.time()
        
        while time.time() - start < time_limit:
            # Forces
            forces = np.zeros_like(centers)
            dist_mat = cdist(centers, centers)
            np.fill_diagonal(dist_mat, np.inf)
            dist_mat_safe = np.maximum(dist_mat, 1e-6)
            
            # Repulsion
            force_mag = 1.0 / (dist_mat_safe ** 2)
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dir_vec = diff / dist_mat_safe[..., np.newaxis]
            forces += np.sum(force_mag[..., np.newaxis] * dir_vec, axis=1)
            
            # Wall repulsion
            wall_dist_x = 0.5 - np.abs(centers[:, 0])
            wall_dist_y = 0.5 - np.abs(centers[:, 1])
            wall_dist_x_safe = np.maximum(wall_dist_x, 1e-6)
            wall_dist_y_safe = np.maximum(wall_dist_y, 1e-6)
            
            sign_x = np.sign(centers[:, 0])
            sign_y = np.sign(centers[:, 1])
            
            forces[:, 0] += 10.0 * sign_x / (wall_dist_x_safe ** 3)
            forces[:, 1] += 10.0 * sign_y / (wall_dist_y_safe ** 3)
            
            # Update
            max_f = np.max(np.linalg.norm(forces, axis=1))
            if max_f > 1e4:
                forces /= max_f * 1e4
                
            centers += forces * dt * temp
            centers[:, 0] = np.clip(centers[:, 0], -0.5, 0.5)
            centers[:, 1] = np.clip(centers[:, 1], -0.5, 0.5)
            
            new_r = compute_r_cont(centers)
            if new_r > best_r:
                best_r = new_r
                best_c = centers.copy()
            else:
                # Revert if worse? Or just keep going?
                # Let's keep going but cool down
                temp *= 0.99
                
            if time.time() > deadline - 2.0:
                break
                
        return best_c, best_r

    refined_centers, refined_r = refine(centers, time_limit=15.0)
    
    # Final validity check
    if not np.all(np.isfinite(refined_centers)):
        return centers
        
    return refined_centers

# EVOLVE-BLOCK-END
