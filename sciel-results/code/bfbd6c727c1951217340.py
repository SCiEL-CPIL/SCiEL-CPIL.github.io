import numpy as np
import time

N = 50
DEADLINE = 115.0 # Safety margin against 120s limit

def compute_radius(c):
    """
    Compute the maximum possible radius for a given configuration c (N, 2).
    Radius is limited by:
    1. Half the minimum distance between any two centers.
    2. The minimum distance from any center to the square boundary [-0.5, 0.5].
    """
    # Pairwise distances
    # diff[i, j] = c[i] - c[j]
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Mask diagonal (self-distances) to infinity
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Wall distances
    # Distance to x=-0.5 is c[:,0] + 0.5
    # Distance to x=0.5 is 0.5 - c[:,0]
    # Distance to y=-0.5 is c[:,1] + 0.5
    # Distance to y=0.5 is 0.5 - c[:,1]
    walls = np.minimum.reduce([
        c[:, 0] + 0.5, 
        0.5 - c[:, 0], 
        c[:, 1] + 0.5, 
        0.5 - c[:, 1]
    ])
    min_wall_dist = np.min(walls)
    
    return 0.5 * min(min_pair_dist, min_wall_dist)

def construct():
    start_time = time.time()
    
    # Initial configurations
    # 1. Hexagonal lattice (good starting point for dense packing)
    # 2. Grid
    # 3. Random
    
    def make_hex_grid(n):
        # Approximate hex packing
        # Side length 1. 
        # Let's try to fit as many as possible in a hex pattern.
        # For N=50, a 7x7 hex grid has 49. 
        # We can adjust spacing.
        # Alternatively, just use a perturbed grid which is safer.
        # Let's stick to a robust grid + random + perturbed.
        m = int(np.ceil(np.sqrt(n)))
        # Create a grid
        xs = np.linspace(-0.5 + 1.0/(2*m), 0.5 - 1.0/(2*m), m)
        ys = np.linspace(-0.5 + 1.0/(2*m), 0.5 - 1.0/(2*m), m)
        grid = np.array([(x, y) for y in ys for x in xs])[:n]
        return grid

    initializations = []
    
    # 1. Grid
    grid = make_hex_grid(N)
    initializations.append(grid)
    
    # 2. Perturbed Grids
    for _ in range(5):
        initializations.append(grid + np.random.uniform(-0.05, 0.05, (N, 2)))
        
    # 3. Random
    for _ in range(5):
        initializations.append(np.random.uniform(-0.45, 0.45, (N, 2)))

    best_radius = 0.0
    best_c = initializations[0]
    
    # Optimization parameters
    # We use a force-directed approach with simulated annealing style acceptance.
    # Force is proportional to (1/d^2) * direction.
    # We want to maximize min_dist.
    
    total_budget = DEADLINE
    # We will iterate through initializations and allocate time to each.
    # Or, we can just run a single long optimization on the best starting point.
    # Better: Run optimization on multiple starts, keep the best.
    
    num_starts = len(initializations)
    time_per_start = (DEADLINE - (time.time() - start_time)) / num_starts
    
    for c0 in initializations:
        if time.time() - start_time > DEADLINE:
            break
            
        c = c0.copy()
        current_radius = compute_radius(c)
        
        # If this start is already bad, skip heavy optimization?
        # No, let's optimize it.
        
        # Simulated Annealing / Force Relaxation
        # Parameters
        max_iter = 5000
        temp = 0.05 # Initial temperature for random kicks
        cool_rate = 0.999
        step_size = 0.01
        
        for i in range(max_iter):
            if time.time() - start_time > DEADLINE:
                break
                
            # 1. Calculate Forces
            # diff[i, j] = c[i] - c[j]
            diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            
            # Avoid division by zero
            dists[dist < 1e-10] = 1e-10 # Note: dists is 2D, need to handle diag
            np.fill_diagonal(dists, np.inf)
            
            # Force magnitude: 1/d^2 (soft repulsion)
            # Only consider pairs closer than some cutoff? 
            # For N=50, all pairs are relevant.
            # We use a soft potential to make it differentiable and smooth.
            # F = -grad U. U ~ 1/d. F ~ 1/d^2.
            
            # Vectorized force calculation
            # f_ij = (c_i - c_j) / dist^2
            # We need sum over j for each i.
            
            # Mask diagonal
            mask = ~np.eye(N, dtype=bool)
            safe_dists = np.where(mask, dists, 1.0)
            
            # Force vector from j on i: (c_i - c_j) / dist^2
            # We want to push i away from j.
            # force_ij = (c_i - c_j) / dist_ij^2
            # But we only want to push if they are close? 
            # Actually, for packing, we want to maximize the MIN distance.
            # A common heuristic is to apply repulsive forces from ALL neighbors,
            # weighted by distance, to spread them out.
            
            # Let's use a simple repulsive force:
            # F_i = sum_j (c_i - c_j) / dist_ij^2
            # This tends to expand the configuration.
            
            # However, we also need to handle walls.
            # Wall forces: push inward if too close to wall.
            
            # Calculate pair forces
            # diff is (N, N, 2)
            # dists is (N, N)
            # We divide diff by dists^2
            # dists_sq = dists**2
            # dists_sq[~mask] = 1.0 # Avoid div by 0 on diag
            
            dists_sq = dists ** 2
            dists_sq[~mask] = 1.0 # Set diag to 1 so it doesn't contribute 0/0 or inf
            
            # Force from pairs: (c_i - c_j) / dist^2
            # This is (N, N, 2)
            pair_forces = diff / dists_sq[:, :, np.newaxis]
            
            # Sum over j (axis=1) to get net force on i
            net_pair_force = np.sum(pair_forces, axis=1)
            
            # Wall forces
            # If c[i,0] < -0.5 + r, push right.
            # We don't know r exactly, but we can use a fixed "repulsion distance" 
            # or simply clamp. 
            # A better way: define a potential energy that goes to infinity at walls.
            # Or, simply check if the circle would overlap the wall given the current 
            # estimated radius? 
            # Simpler: Add a force that pushes away from walls if distance < threshold.
            # Let threshold be 0.1 (approx radius).
            
            wall_force = np.zeros_like(c)
            # X walls
            # Left wall (-0.5): if c_x < -0.5 + thresh, force += (thresh - (c_x - (-0.5))) 
            # Let's use a linear spring near walls.
            thresh = 0.1
            # Left
            mask_left = c[:, 0] < -0.5 + thresh
            wall_force[mask_left, 0] += (-0.5 + thresh - c[mask_left, 0])
            # Right
            mask_right = c[:, 0] > 0.5 - thresh
            wall_force[mask_right, 0] -= (c[mask_right, 0] - (0.5 - thresh))
            # Bottom
            mask_bottom = c[:, 1] < -0.5 + thresh
            wall_force[mask_bottom, 1] += (-0.5 + thresh - c[mask_bottom, 1])
            # Top
            mask_top = c[:, 1] > 0.5 - thresh
            wall_force[mask_top, 1] -= (c[mask_top, 1] - (0.5 - thresh))
            
            total_force = net_pair_force + wall_force
            
            # Normalize forces to prevent large jumps
            force_mags = np.linalg.norm(total_force, axis=1)
            max_force = np.max(force_mags)
            if max_force > 0:
                # Scale forces to have max magnitude 1
                total_force = total_force * (1.0 / (max_force + 1e-10))
            
            # 2. Update positions
            # Add random noise (annealing)
            noise = np.random.normal(0, temp, (N, 2))
            
            new_c = c + step_size * (total_force + noise)
            
            # Keep inside bounds (hard constraint)
            new_c = np.clip(new_c, -0.5, 0.5)
            
            # 3. Evaluate
            new_radius = compute_radius(new_c)
            
            # Acceptance criterion
            # If new_radius > current_radius, accept.
            # If new_radius < current_radius, accept with probability exp((new - cur) / temp)
            # Note: We are maximizing radius.
            # Delta = new_radius - current_radius
            
            if new_radius > current_radius:
                c = new_c
                current_radius = new_radius
            else:
                delta = new_radius - current_radius
                # delta is negative
                # prob = exp(delta / temp)
                # If delta is large negative, prob is ~0
                if temp > 1e-6:
                    prob = np.exp(delta / temp)
                    if np.random.random() < prob:
                        c = new_c
                        current_radius = new_radius
            
            # Cool down
            temp *= cool_rate
            
            # Update global best
            if current_radius > best_radius:
                best_radius = current_radius
                best_c = c.copy()

    # Final polish: Local gradient ascent on the best configuration found
    # Use a simple coordinate descent or small step force relaxation without noise
    if time.time() - start_time < DEADLINE - 5:
        c = best_c.copy()
        current_radius = compute_radius(c)
        step_size = 0.005
        for _ in range(2000):
            if time.time() - start_time > DEADLINE - 5:
                break
            
            # Recalculate forces
            diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            mask = ~np.eye(N, dtype=bool)
            dists_sq = dists ** 2
            dists_sq[~mask] = 1.0
            pair_forces = diff / dists_sq[:, :, np.newaxis]
            net_pair_force = np.sum(pair_forces, axis=1)
            
            # Wall forces (same as before)
            wall_force = np.zeros_like(c)
            thresh = 0.1
            mask_left = c[:, 0] < -0.5 + thresh
            wall_force[mask_left, 0] += (-0.5 + thresh - c[mask_left, 0])
            mask_right = c[:, 0] > 0.5 - thresh
            wall_force[mask_right, 0] -= (c[mask_right, 0] - (0.5 - thresh))
            mask_bottom = c[:, 1] < -0.5 + thresh
            wall_force[mask_bottom, 1] += (-0.5 + thresh - c[mask_bottom, 1])
            mask_top = c[:, 1] > 0.5 - thresh
            wall_force[mask_top, 1] -= (c[mask_top, 1] - (0.5 - thresh))
            
            total_force = net_pair_force + wall_force
            force_mags = np.linalg.norm(total_force, axis=1)
            max_force = np.max(force_mags)
            if max_force > 0:
                total_force = total_force * (1.0 / (max_force + 1e-10))
            
            new_c = c + step_size * total_force
            new_c = np.clip(new_c, -0.5, 0.5)
            
            new_radius = compute_radius(new_c)
            if new_radius > current_radius:
                c = new_c
                current_radius = new_radius
                if current_radius > best_radius:
                    best_radius = current_radius
                    best_c = c.copy()
            else:
                # If stuck, break early? Or just stop?
                # Let's just continue, maybe it escapes a tiny local dip?
                # Actually, if no improvement, we can break to save time.
                pass

    return best_c

# Dummy call to ensure syntax is correct in module load (optional, but good for checking)
# construct() 
