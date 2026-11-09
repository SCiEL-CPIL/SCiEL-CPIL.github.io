import numpy as np
import time

N = 100

def _compute_radius(centres):
    # centres: (N, 3)
    # Returns min radius
    # Pairwise
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    # Boundary
    # dist to boundary is 0.5 - |c|
    min_bound = np.min(0.5 - np.abs(centres))
    
    return min(min_pair, min_bound) / 2.0

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    # 1. Initialization
    # Use a dense grid or FCC
    # Let's try a 5x5x4 grid = 100 points exactly.
    # Spacing: 1/5 = 0.2.
    # Radius approx 0.1.
    # Better: FCC lattice.
    
    def init_fcc():
        # FCC lattice points
        # Generate a large enough block
        # Standard FCC: (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)
        # Scaled to fit in [-0.5, 0.5]
        # Side length of unit cell in FCC is 'a'.
        # We want to fit 100 spheres.
        # Volume ~ 1.
        # Let's just use a grid for simplicity and robustness, then optimize.
        # A 5x5x4 grid is a good start.
        xs = np.linspace(-0.4, 0.4, 5)
        ys = np.linspace(-0.4, 0.4, 5)
        zs = np.linspace(-0.35, 0.35, 4)
        grid = np.array([(x, y, z) for z in zs for y in ys for x in xs])
        return grid[:N]
        
    # Actually, a better init is to take the first N points of a high-density grid
    # and let the optimizer do the work.
    
    best_centres = init_fcc()
    best_r = _compute_radius(best_centres)
    
    current_centres = best_centres.copy()
    current_r = best_r
    
    # Optimization Loop
    step_size = 0.01
    noise = 0.001
    
    while time.time() < deadline:
        # Calculate forces
        # We want to increase the minimum distance.
        # Identify the tightest constraints.
        
        # Pairwise distances
        diff = current_centres[:, np.newaxis, :] - current_centres[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        # Boundary distances
        bound_dists = 0.5 - np.abs(current_centres) # (N, 3)
        
        # Find global min
        min_pair_val = np.min(dists)
        min_bound_val = np.min(bound_dists)
        
        current_min = min(min_pair_val, min_bound_val)
        
        # If we improved, update best
        if current_min / 2.0 > best_r:
            best_r = current_min / 2.0
            best_centres = current_centres.copy()
            
        # Generate forces
        forces = np.zeros_like(current_centres)
        
        # Pairwise forces: push apart
        # Only consider pairs with distance < some threshold (e.g., 1.5 * min)
        # To be efficient, we can just sum all repulsive forces weighted by 1/d^2 or similar
        # But for max-min, pushing the *closest* ones is key.
        
        # Let's use a gradient ascent on soft min for stability
        # Objective: -log(sum(exp(-beta * d)))
        # Grad w.r.t c_i: sum_{j} (exp(-beta d_ij) / sum_k exp(-beta d_ik)) * (c_i - c_j) / d_ij * beta
        
        beta = 100.0 # Sharpness
        
        # Soft min gradient
        # For each i, sum over j != i
        # This is O(N^2)
        
        # We can vectorize the pairwise part.
        # exp_term = np.exp(-beta * dists)
        # np.fill_diagonal(exp_term, 0)
        # sums = exp_term.sum(axis=1) # Sum of weights for each i
        # weights = exp_term / sums[:, np.newaxis] # (N, N) weights for each i,j
        # grad_pair = np.sum(weights[:, :, np.newaxis] * (current_centres[:, np.newaxis, :] - current_centres[np.newaxis, :, :]) / dists[:, :, np.newaxis], axis=1)
        # grad_pair *= beta
        
        # Boundary gradient
        # d(0.5 - |c|)/dc = -sign(c)
        # We want to maximize 0.5 - |c|, so gradient is -sign(c)
        # But we want to move in direction of increasing distance to boundary.
        # If c > 0, dist = 0.5 - c, grad = -1. Move -1 * (-1) = +1? No.
        # We want to move c towards 0.
        # If c > 0, move c down (negative direction).
        # If c < 0, move c up (positive direction).
        # So force is -sign(c).
        
        # Let's combine.
        # The soft min handles the "min" nature smoothly.
        
        exp_term = np.exp(-beta * dists)
        np.fill_diagonal(exp_term, 0)
        sums = exp_term.sum(axis=1)
        # Avoid division by zero
        sums[sums == 0] = 1.0
        weights = exp_term / sums[:, np.newaxis]
        
        # Vector difference
        vec_diff = current_centres[:, np.newaxis, :] - current_centres[np.newaxis, :, :]
        # Safe divide
        safe_dists = np.where(dists == 0, 1.0, dists)
        
        grad_pair = np.sum(weights[:, :, np.newaxis] * (vec_diff / safe_dists[:, :, np.newaxis]), axis=1)
        grad_pair *= beta
        
        # Boundary
        grad_bound = -np.sign(current_centres) * beta # Scale similarly
        
        total_grad = grad_pair + grad_bound
        
        # Normalize gradient to have unit length? Or use fixed step?
        # Fixed step is safer.
        step = 0.01
        
        # Add noise
        noise_vec = np.random.normal(0, noise, size=current_centres.shape)
        
        new_centres = current_centres + step * total_grad + noise_vec
        
        # Clip to cube
        new_centres = np.clip(new_centres, -0.5, 0.5)
        
        # Check new radius
        new_r = _compute_radius(new_centres)
        
        if new_r > current_r:
            current_centres = new_centres
            current_r = new_r
            # Decrease noise slightly?
            noise = max(noise * 0.99, 0.0001)
        else:
            # If we didn't improve, we might be stuck.
            # Increase noise occasionally
            if np.random.rand() < 0.05:
                noise = min(noise * 1.2, 0.05)
                # Or take a random step
                rand_step = np.random.normal(0, 0.05, size=current_centres.shape)
                trial = np.clip(current_centres + rand_step, -0.5, 0.5)
                trial_r = _compute_radius(trial)
                if trial_r > current_r:
                    current_centres = trial
                    current_r = trial_r
            else:
                # Try smaller step or backtrack
                # For now, just keep current
                pass
                
    return best_centres
