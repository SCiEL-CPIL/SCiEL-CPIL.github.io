import numpy as np
import time
import sys

N = 50

def construct():
    start_time = time.time()
    time_limit = 140.0  # Safety margin for 150s limit
    
    # 1. Initialization: Fibonacci sphere distribution
    # This provides a good uniform starting point, better than a grid.
    n = N
    phi = np.pi * (3. - np.sqrt(5.))  # Golden angle
    z = np.linspace(1, -1, n)
    z *= 0.9 # Shrink slightly to avoid corners initially
    r = np.sqrt(1.0 - z**2)
    theta = phi * np.arange(n)
    
    # Generate points on sphere
    x = np.cos(theta) * r
    y = np.sin(theta) * r
    pts = np.column_stack((x, y, z))
    
    # Scale to fit inside [-0.5, 0.5]^3
    # Max radius of sphere is 1. We want max coordinate to be < 0.5.
    # If we scale by 0.4, max coord is 0.4.
    scale = 0.4
    pts = pts * scale
    
    # 2. Simulated Annealing
    # Objective: Maximize min( min_{i<j} ||p_i - p_j||, min_i min_k (0.5 - |p_ik|) )
    
    def calculate_score(positions):
        # Calculate pairwise distances
        # Using broadcasting for vectorized calculation
        # dist[i, j] = ||p_i - p_j||
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        
        # Minimum pairwise distance (excluding diagonal)
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        # Distance to walls
        # Wall distance for point p is min(0.5 - p_x, 0.5 - p_y, 0.5 - p_z, 0.5 + p_x, 0.5 + p_y, 0.5 + p_z)
        # Which is min(0.5 - |p_x|, 0.5 - |p_y|, 0.5 - |p_z|)
        wall_dists = 0.5 - np.abs(positions)
        min_wall_dist = np.min(wall_dists)
        
        return np.min([min_pair_dist, min_wall_dist])

    current_pos = pts.copy()
    current_score = calculate_score(current_pos)
    best_pos = current_pos.copy()
    best_score = current_score
    
    # Annealing parameters
    T = 0.05  # Initial temperature
    alpha = 0.999  # Cooling rate
    min_T = 1e-6
    
    # We want to maximize score. 
    # Energy E = -Score. Minimize E.
    
    step_size = 0.05
    
    while T > min_T:
        # Check time
        if time.time() - start_time > time_limit:
            break
            
        # Generate a random move
        # Pick a random point and a random axis
        i = np.random.randint(N)
        axis = np.random.randint(3)
        
        # Gaussian perturbation
        delta = np.random.normal(0, step_size)
        
        new_pos = current_pos.copy()
        new_pos[i, axis] += delta
        
        # Ensure points stay inside the cube (hard constraint for validity, though score handles it)
        # Actually, we can let the score handle it, but keeping them inside helps.
        # The score function returns a small number if outside, so SA will naturally push them back.
        # However, to be safe and efficient, we clip. Clipping changes the "distance" calculation?
        # No, if we clip, the point is on the boundary. The wall distance is 0.
        # The score will drop to 0. This is a valid penalty.
        # Let's not clip to allow "escaping" via low scores, but maybe it's better to clip to prevent flying out.
        # Let's clip to [-0.5, 0.5].
        new_pos = np.clip(new_pos, -0.5, 0.5)
        
        new_score = calculate_score(new_pos)
        
        # Calculate delta in "Energy" (we want to maximize score, so delta_E = -(new_score - current_score))
        delta_E = current_score - new_score
        
        # Acceptance criterion
        if delta_E < 0 or np.random.rand() < np.exp(-delta_E / T):
            current_pos = new_pos
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_pos = current_pos.copy()
        
        # Cool down
        T *= alpha
        
        # Adaptive step size: if we haven't improved in a while, reduce step size?
        # Or increase? Usually reduce to fine-tune.
        # Let's reduce step size as T decreases.
        step_size = 0.05 * (T / 0.05)
        
        if step_size < 1e-4:
            step_size = 1e-4

    # Final local optimization: Coordinate Descent
    # Try to move each point slightly to improve the minimum distance.
    # This is a greedy refinement.
    
    def local_refine(positions, iterations=500):
        pos = positions.copy()
        score = calculate_score(pos)
        for _ in range(iterations):
            if time.time() - start_time > time_limit:
                break
            improved = False
            for i in range(N):
                for axis in range(3):
                    # Try small moves in + and - direction
                    for direction in [1, -1]:
                        trial = pos.copy()
                        # Binary search for the max move that improves score? 
                        # Or just fixed small steps.
                        # Let's try a few step sizes
                        for step in [0.01, 0.005, 0.001]:
                            trial[i, axis] += direction * step
                            trial = np.clip(trial, -0.5, 0.5)
                            trial_score = calculate_score(trial)
                            if trial_score > score:
                                pos = trial
                                score = trial_score
                                improved = True
            if not improved:
                break
        return pos, score

    best_pos, best_score = local_refine(best_pos)
    
    # Ensure the result is valid (finite, inside cube)
    if not np.all(np.isfinite(best_pos)):
        # Fallback to initial grid if something went wrong
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N]
        
    return best_pos

if __name__ == "__main__":
    # Quick test to ensure it runs
    res = construct()
    print(res.shape)
