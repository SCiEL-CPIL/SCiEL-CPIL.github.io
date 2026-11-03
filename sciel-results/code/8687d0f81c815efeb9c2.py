"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    # New Approach: Phyllotaxis Initialization + Gradient Descent
    # This avoids the hexagonal grid symmetry traps and SA stochasticity.
    
    # 1. Initialization: Phyllotaxis Spiral
    # Distributes points uniformly in area, good for packing.
    def init_psyllotaxis(n):
        points = []
        golden_angle = np.pi * (3 - np.sqrt(5)) # ~137.5 degrees
        for i in range(n):
            r = np.sqrt(i / n)
            theta = i * golden_angle
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            points.append([x, y])
        return np.array(points)

    centers = init_psyllotaxis(N)
    
    # 2. Optimization: Gradient Descent on Overlap Energy
    # We want to maximize r. We will start with a conservative r and increase it.
    # Energy function:
    # E = sum_{i<j} max(0, 2r - d_ij)^2 + lambda * sum_i max(0, |c_i| + r - 1)^2
    # We minimize E. If E -> 0, we have a valid packing for radius r.
    
    def compute_energy_and_grad(centers, r, lambda_boundary=10.0):
        n = len(centers)
        energy = 0.0
        grad = np.zeros_like(centers)
        
        # Pairwise repulsion
        # Vectorized calculation for efficiency
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        
        # Avoid division by zero
        mask = dists > 1e-10
        inv_dists = np.zeros_like(dists)
        inv_dists[mask] = 1.0 / dists[mask]
        
        # Overlap amount: 2r - d_ij. Positive if overlapping.
        overlap = 2 * r - dists
        # Only consider overlaps
        overlap_mask = overlap > 0
        # Set diagonal to 0 to avoid self-interaction
        np.fill_diagonal(overlap_mask, False)
        
        # Gradient contribution from pairs
        # dE/dc_k = sum_{j != k} 2 * (2r - d_kj) * (c_k - c_j) / d_kj
        unit_vecs = diff * inv_dists[:, :, np.newaxis]
        weights = 2 * np.where(overlap_mask, overlap, 0)
        
        # Gradient is sum over j of weights[i, j] * unit_vecs[i, j]
        grad_pairs = np.sum(weights[:, :, np.newaxis] * unit_vecs, axis=1)
        
        # Boundary term
        radii = np.linalg.norm(centers, axis=1)
        boundary_viol = radii + r - 1
        boundary_mask = boundary_viol > 0
        boundary_weights = 2 * lambda_boundary * np.where(boundary_mask, boundary_viol, 0)
        
        # Gradient for boundary: boundary_weights_i * (c_i / |c_i|)
        inv_radii = np.where(radii > 1e-10, 1.0 / radii, 0.0)
        unit_radii = centers * inv_radii[:, np.newaxis]
        
        grad_bound = boundary_weights[:, np.newaxis] * unit_radii
        
        total_grad = grad_pairs + grad_bound
        
        # Calculate Energy
        e_pairs = np.sum(np.where(overlap_mask, overlap**2, 0))
        e_bound = lambda_boundary * np.sum(np.where(boundary_mask, boundary_viol**2, 0))
        total_energy = e_pairs + e_bound
        
        return total_energy, total_grad

    def optimize(centers, r, max_iter=2000, lr_init=0.01, lr_decay=0.999):
        lr = lr_init
        best_centers = centers.copy()
        best_energy = np.inf
        stuck_count = 0
        
        for i in range(max_iter):
            energy, grad = compute_energy_and_grad(centers, r)
            
            if energy < best_energy:
                best_energy = energy
                best_centers = centers.copy()
                stuck_count = 0
            else:
                stuck_count += 1
                
            # If energy is very low, we can stop
            if energy < 1e-10:
                break
                
            # If stuck, add noise
            if stuck_count > 50:
                noise = np.random.normal(0, 0.01, size=centers.shape)
                centers += noise
                stuck_count = 0
                lr = max(lr, 0.005)
            
            # Gradient Step
            new_centers = centers - lr * grad
            
            # Check if new centers are valid (finite)
            if not np.all(np.isfinite(new_centers)):
                lr *= 0.5
                continue
                
            new_energy, _ = compute_energy_and_grad(new_centers, r)
            
            if new_energy < energy:
                centers = new_centers
                lr = min(lr * 1.05, 0.05) # Slight increase if improving, capped
            else:
                lr *= 0.8 # Reduce if not improving
                
            # Decay lr slowly
            lr *= lr_decay
            
            if lr < 1e-8:
                break
                
        return best_centers, best_energy

    # Strategy: Start with a radius slightly below the known best and increase.
    # Known best: 0.1258254895304
    # Start lower to ensure convergence, then push up.
    r_current = 0.120
    step = 0.0001
    max_attempts = 500
    
    best_valid_r = 0
    best_valid_centers = centers
    
    for _ in range(max_attempts):
        # Try to optimize for r_current
        opt_centers, final_energy = optimize(centers, r_current, max_iter=1000)
        
        if final_energy < 1e-6:
            # Valid packing found
            best_valid_r = r_current
            best_valid_centers = opt_centers
            centers = opt_centers # Use this as start for next iteration
            r_current += step
            # Reduce step size as we get closer to optimum for finer tuning
            if r_current > 0.125:
                step = 0.00002
        else:
            # Failed to find valid packing for this r
            break

    return best_valid_centers

# EVOLVE-BLOCK-END
