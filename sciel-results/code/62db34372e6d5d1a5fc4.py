"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    # Known best radius
    r_best = 0.1258254895304
    r = r_best
    
    # Initialize with a hexagonal lattice
    def init_hex():
        # Approximate spacing for 50 circles in unit circle
        # Area of unit circle = pi. Area of 50 circles = 50 * pi * r^2.
        # Density ~ 0.9 for hex packing.
        # r ~ sqrt(1 / (50 * 0.9)) ~ 0.14. But boundary effects reduce this.
        # Let's just use a grid and filter.
        spacing = 2 * r_best
        points = []
        k = 5
        for j in range(-k, k+1):
            for i in range(-k, k+1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3) / 2
                if x*x + y*y <= (1 - r_best)**2:
                    points.append((x, y))
        
        # If we don't have enough points, add random ones
        while len(points) < N:
            x = np.random.uniform(-1, 1)
            y = np.random.uniform(-1, 1)
            if x*x + y*y <= (1 - r_best)**2:
                points.append((x, y))
        
        points = np.array(points[:N])
        return points

    centers = init_hex()
    
    # Simulated Annealing to minimize overlaps for a given radius
    def sa_optimize(centers, r, max_iter=20000):
        T = 0.01
        T_min = 1e-6
        cooling = 0.995
        best_centers = centers.copy()
        best_energy = np.inf
        
        for i in range(max_iter):
            # Calculate current energy
            # Vectorized distance calculation
            # This is O(N^2), which is 2500 ops. Fine for 20k iters.
            
            # Move one random circle
            idx = np.random.randint(N)
            old_pos = centers[idx].copy()
            
            # Random perturbation
            angle = np.random.uniform(0, 2 * np.pi)
            dist = np.random.uniform(0, 0.1) # Step size
            centers[idx] += dist * np.array([np.cos(angle), np.sin(angle)])
            
            # Ensure inside container (hard constraint for efficiency, or include in energy)
            # It's better to include in energy to allow temporary violations
            # But for speed, we can clamp? No, let energy handle it.
            
            # Calculate Energy
            # Pairwise distances
            # To save time, we can calculate energy difference only for pairs involving idx
            # d_ij changes only if i==idx or j==idx
            # E = sum(max(0, 2r - d)^2)
            
            # Let's just recalculate full energy for simplicity first, optimize if needed
            # N=50, 50*49/2 = 1225 pairs.
            # 20000 iters * 1225 pairs * ~10 flops = 245 million flops. Might be slow in Python loop.
            # We need vectorized energy calculation.
            
            # Vectorized Energy Calculation
            # c: (N, 2)
            # diff: (N, N, 2)
            # dist: (N, N)
            # This creates a 50x50x2 array. 2500*2 = 5000 floats. Very small.
            
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            
            # Overlap term
            # We only care about i < j
            # Use triu_indices
            iu, ju = np.triu_indices(N, k=1)
            d_ij = dists[iu, ju]
            overlaps = np.maximum(0, 2 * r - d_ij)
            e_pairs = np.sum(overlaps ** 2)
            
            # Boundary term
            radii = np.linalg.norm(centers, axis=1)
            b_viol = np.maximum(0, radii + r - 1)
            e_bound = np.sum(b_viol ** 2)
            
            current_energy = e_pairs + 10 * e_bound # Weight boundary heavily
            
            # Acceptance
            if current_energy < best_energy:
                best_energy = current_energy
                best_centers = centers.copy()
                
            # Metropolis criterion
            if current_energy <= best_energy:
                pass # Accepted
            else:
                delta_e = current_energy - best_energy
                if np.random.rand() < np.exp(-delta_e / T):
                    pass # Accepted
                else:
                    # Reject, revert
                    centers[idx] = old_pos
            
            # Cool down
            T *= cooling
            if T < T_min:
                break
                
        return best_centers, best_energy

    # Iteratively increase radius
    current_r = r_best
    current_centers = centers
    
    # First, try to stabilize at current best
    current_centers, e = sa_optimize(current_centers, current_r, max_iter=5000)
    
    # Try to increase radius
    step = 0.0001
    for _ in range(200): # Max attempts
        new_r = current_r + step
        # If step is too big, it will fail. We can binary search or just linear scan.
        # Linear scan with small step is safer.
        
        # Run SA for a few iterations to see if it can fit
        # If energy stays low, we accept the new radius.
        # We need to check if a valid packing exists.
        
        # Let's run SA for 2000 iters
        test_centers, test_energy = sa_optimize(current_centers, new_r, max_iter=2000)
        
        # If test_energy is close to 0, we have a valid packing
        if test_energy < 1e-5:
            current_r = new_r
            current_centers = test_centers
            # Reset step to smaller value to fine-tune
            step = 0.00005
        else:
            # Failed to fit, break or reduce step?
            # If we fail, we should stop increasing.
            break
            
    return current_centers

# EVOLVE-BLOCK-END
