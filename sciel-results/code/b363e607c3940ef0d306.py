"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n = N
    # 1. Initialization: Jittered Grid
    # Use a grid that is slightly perturbed to break symmetry
    m = int(np.ceil(n ** (1.0 / 3.0)))
    # Create grid coordinates in [-0.5, 0.5]
    # We need exactly n points. A 5x5x4 = 100 grid is perfect.
    # Let's use 5x5x4.
    nx, ny, nz = 5, 5, 4
    xs = np.linspace(-0.4, 0.4, nx)
    ys = np.linspace(-0.4, 0.4, ny)
    zs = np.linspace(-0.4, 0.4, nz)
    
    grid = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    pts = grid[:n].copy()
    
    # Add small random jitter to break perfect symmetry
    np.random.seed(42)
    pts += np.random.normal(0, 0.01, size=(n, 3))
    
    # 2. Physics-Based Relaxation
    # We will simulate the system with a target radius r that grows over time.
    # The system tries to minimize overlaps for the current r.
    
    # Parameters
    lr = 0.05  # Learning rate
    r_start = 0.05
    r_end = 0.12
    growth_steps = 500
    relax_steps_per_growth = 100
    
    current_r = r_start
    
    # Helper to compute gradient of overlap energy
    def get_forces(pts, r):
        # pts: (n, 3)
        # Returns: (n, 3) forces
        forces = np.zeros_like(pts)
        
        # Pairwise repulsion
        # diff[i, j, k] = pts[i, k] - pts[j, k]
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        
        # Avoid division by zero
        np.fill_diagonal(dists, 1e-9)
        
        # Overlap amount: max(0, 2r - dist)
        # We use a smooth approximation or just linear penalty for gradient
        # Gradient of 0.5 * (max(0, 2r - d))^2 w.r.t pts
        # If d < 2r: grad = (2r - d) * (pts_i - pts_j) / d
        # Note: The force on i is in direction (pts_i - pts_j)
        
        overlap = np.maximum(0, 2 * r - dists)
        
        # Compute force contributions
        # F_ij = overlap_ij * (pts_i - pts_j) / dist_ij
        # We can compute this as:
        # factor[i, j] = overlap[i, j] / dists[i, j]
        # Force on i = sum_j factor[i, j] * (pts_i - pts_j)
        
        factors = overlap / dists
        # Set diagonal to 0
        np.fill_diagonal(factors, 0)
        
        # Sum over j for each i
        # force_i = sum_j factors[i, j] * (pts[i] - pts[j])
        #         = sum_j factors[i, j] * pts[i] - sum_j factors[i, j] * pts[j]
        #         = pts[i] * sum_j factors[i, j] - (factors @ pts)[i]
        
        sum_factors = np.sum(factors, axis=1)
        force_pair = pts * sum_factors[:, np.newaxis] - (factors @ pts)
        
        # Boundary forces
        # We want to push points back if they are closer than r to the wall.
        # Wall at 0.5: if pts[i, k] > 0.5 - r, push left.
        # Wall at -0.5: if pts[i, k] < -0.5 + r, push right.
        
        # Upper bound violation
        upper_viol = np.maximum(0, pts - (0.5 - r))
        # Lower bound violation
        lower_viol = np.maximum(0, (-0.5 + r) - pts)
        
        force_bound = -upper_viol + lower_viol
        
        total_forces = force_pair + force_bound
        return total_forces

    # Main relaxation loop
    for step in range(growth_steps):
        # Increase radius
        progress = step / growth_steps
        # Smoothly increase r
        current_r = r_start + (r_end - r_start) * progress
        
        # Relax for a few steps
        for _ in range(relax_steps_per_growth):
            forces = get_forces(pts, current_r)
            
            # Adaptive learning rate: scale down if forces are huge?
            # Simple approach: fixed lr, but clip forces
            max_force = np.max(np.abs(forces))
            if max_force > 1e-6:
                forces = forces / max_force * 0.1 # Normalize and scale
            
            pts += lr * forces
            
            # Project into the valid domain [-0.5, 0.5]
            # Actually, we should project into [-0.5+r, 0.5-r] to prevent deep penetration
            # But the force handles this. Hard clipping is safe.
            pts = np.clip(pts, -0.5, 0.5)
            
    # Final refinement with higher r and smaller steps
    # Try to push r a bit higher to see if it holds
    for step in range(100):
        current_r += 0.0001
        for _ in range(50):
            forces = get_forces(pts, current_r)
            max_force = np.max(np.abs(forces))
            if max_force > 1e-8:
                forces = forces / max_force * 0.05
            pts += 0.01 * forces
            pts = np.clip(pts, -0.5, 0.5)
            
    # Calculate the actual achieved radius
    def calc_radius(pts):
        max_abs = np.max(np.abs(pts))
        boundary_r = 0.5 - max_abs
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pairwise = np.min(dists)
        pairwise_r = min_pairwise / 2.0
        return min(boundary_r, pairwise_r)
    
    final_r = calc_radius(pts)
    
    # If the result is worse than a simple grid, fallback?
    # The grid radius is approx 0.1 (for 5x5x4, dist is 0.2, r=0.1).
    # Our method should do better.
    
    return pts
# EVOLVE-BLOCK-END
