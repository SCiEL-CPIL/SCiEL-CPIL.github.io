"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Construct centers of 300 equal circles in the unit disk using a physics-based simulation.
    We simulate soft-body collisions to pack the circles as tightly as possible.
    """
    # Parameters for the simulation
    dt = 0.005          # Time step
    damping = 0.98      # Velocity damping
    repulsion_strength = 50.0  # Strength of repulsive force between circles
    repulsion_power = 2        # Power of the repulsive force (Hooke's law like)
    boundary_repulsion = 100.0 # Strength of repulsive force from boundary
    max_iterations = 2000      # Number of simulation steps
    target_radius = 0.05       # Initial guess for circle radius, will be adjusted by packing
    
    # Initialize positions in a random or grid-like pattern
    # A spiral or grid initialization helps convergence
    np.random.seed(42)
    positions = np.zeros((N, 2))
    velocities = np.zeros((N, 2))
    
    # Simple initialization: Random points inside a smaller circle
    # Or a grid. Let's use a dense grid in a square, then map to circle?
    # Better: Random uniform in unit disk, but clustered.
    # Let's use a Fibonacci sphere-like distribution projected to 2D or just random with jitter.
    # Random is okay if we run enough steps.
    angles = np.random.uniform(0, 2 * np.pi, N)
    radii = np.sqrt(np.random.uniform(0, 1, N))
    positions[:, 0] = radii * np.cos(angles)
    positions[:, 1] = radii * np.sin(angles)
    
    # Scale down initial positions to avoid immediate boundary collisions
    positions *= 0.8
    
    # We will iteratively find the maximum radius.
    # Instead of fixing radius, we let the physics find the equilibrium spacing.
    # The "radius" is effectively determined by the minimum distance at equilibrium.
    # To maximize radius, we want to minimize the potential energy of the system
    # where potential energy is high when circles are too close.
    
    # We simulate forces based on a hypothetical radius r.
    # Actually, a simpler heuristic:
    # 1. Simulate with a "soft" constraint that circles try to maintain a certain distance.
    # 2. The optimal packing will settle into a configuration where the minimum distance is maximized.
    
    # Let's use a force model:
    # F_repel = k * (r_target - d)^2 if d < r_target else 0
    # But we don't know r_target.
    # Alternative: Use a repulsive force that is always active but weak at large distances?
    # No, standard approach: 
    # We assume a radius r. We simulate. If they overlap, they push apart.
    # We want to find the largest r such that a valid packing exists.
    # Since we can't do binary search easily in one shot without many runs, 
    # we can use a "pressure" model.
    
    # Let's use a different strategy: 
    # Simulate with a fixed "desired" separation distance.
    # Start with a small separation, pack, then increase separation?
    # Or, just simulate with a strong repulsion that mimics hard spheres of radius r.
    
    # Let's try a binary search on the radius r, using the physics sim to check feasibility.
    # But 120s limit. Physics sim is fast.
    
    def simulate(r, max_iter=500):
        """
        Simulate packing of circles of radius r.
        Returns True if a stable non-overlapping configuration is found, False otherwise.
        Also returns the positions if successful.
        """
        pos = positions.copy()
        vel = np.zeros_like(pos)
        
        # If initial positions overlap significantly, the sim might explode.
        # We should start from a configuration known to be valid for a smaller r,
        # or just accept that we might need to restart.
        
        for step in range(max_iter):
            forces = np.zeros_like(pos)
            
            # Pairwise repulsion
            # Vectorized distance calculation
            diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
            dist = np.linalg.norm(diff, axis=2)
            
            # Avoid division by zero
            dist[dist < 1e-9] = 1e-9
            
            # We only care about pairs closer than 2*r
            # Force magnitude: k * (2*r - dist) if dist < 2*r
            # Direction: diff / dist
            # F_ij = k * (2*r - dist_ij) * (p_i - p_j) / dist_ij
            # Note: (p_i - p_j) is -diff_ij if diff = p_i - p_j? 
            # diff[i,j] = p_i - p_j. 
            # Force on i from j should push i away from j. 
            # Direction is (p_i - p_j) / dist.
            # So F_i += sum_j k * (2*r - dist_ij) * (p_i - p_j) / dist_ij
            
            # Optimization: Only compute for dist < 2*r
            mask = dist < 2 * r
            # Exclude self
            np.fill_diagonal(mask, False)
            
            # Compute forces
            # diff is (N, N, 2). dist is (N, N).
            # We need to sum over j for each i.
            
            # Create a force matrix
            # F[i] = sum_j Force_ij
            # Force_ij vector = k * (2*r - dist[i,j]) * diff[i,j] / dist[i,j]
            # Only if mask[i,j] is True.
            
            # To vectorize:
            # factor = k * (2*r - dist) / dist
            # factor[~mask] = 0
            # factor[i,i] = 0
            factor = np.zeros_like(dist)
            factor[mask] = repulsion_strength * (2 * r - dist[mask]) / dist[mask]
            
            # forces[i] = sum_j factor[i,j] * diff[i,j]
            # diff is (N, N, 2). factor is (N, N).
            # forces = np.sum(factor[:, :, np.newaxis] * diff, axis=1)
            
            forces += np.sum(factor[:, :, np.newaxis] * diff, axis=1)
            
            # Boundary repulsion
            # If point is too close to boundary (dist > 1 - r), push inwards.
            # dist_from_center = np.linalg.norm(pos, axis=1)
            # limit = 1 - r
            # overflow = np.maximum(0, dist_from_center - limit)
            # force_boundary = -boundary_repulsion * overflow * pos / dist_from_center
            
            dist_center = np.linalg.norm(pos, axis=1)
            limit = 1.0 - r
            overflow = np.maximum(0, dist_center - limit)
            # Avoid div by zero
            safe_dist = np.where(dist_center > 1e-9, dist_center, 1e-9)
            boundary_force = -boundary_repulsion * overflow / safe_dist
            # Vector form
            forces += boundary_force[:, np.newaxis] * pos
            
            # Update velocities and positions
            vel = vel * damping + forces * dt
            pos = pos + vel * dt
            
            # Check for stability: if max velocity is small, we might be converged
            # But we also need to check if overlaps are resolved.
            # For simplicity, we run fixed steps.
            
        # Check final configuration
        final_dist = np.linalg.norm(pos[:, np.newaxis, :] - pos[np.newaxis, :, :], axis=2)
        np.fill_diagonal(final_dist, np.inf)
        min_pair_dist = np.min(final_dist)
        
        # Check boundary
        final_center_dist = np.linalg.norm(pos, axis=1)
        max_center_dist = np.max(final_center_dist)
        
        # Valid if min_pair_dist >= 2*r and max_center_dist <= 1-r
        # Allow small numerical tolerance
        tol = 1e-5
        if min_pair_dist >= 2 * r - tol and max_center_dist <= 1 - r + tol:
            return True, pos
        else:
            return False, pos

    # Binary search for the maximum radius
    # Lower bound: 0.01
    # Upper bound: 0.1 (known best is ~0.053)
    low_r = 0.01
    high_r = 0.08
    
    best_pos = positions
    best_r = low_r
    
    # We want to find the max r that is feasible.
    # However, the simulation is stochastic and might fail to converge for feasible r.
    # To mitigate this, we can run multiple attempts for a given r.
    # But time limit is 120s.
    
    # Let's try a few values manually first to gauge performance.
    # Actually, let's just run the simulation for a specific r that is likely close to optimal.
    # The known best is 0.05315.
    # Let's try to find the max r via a coarse binary search with limited iterations.
    
    # To save time, we will use a smaller number of iterations in the inner loop
    # and rely on the fact that if it converges, it's likely valid.
    
    # Refined Strategy:
    # 1. Start with r = 0.05
    # 2. Simulate. If success, try larger r.
    # 3. If fail, try smaller r.
    
    # We will do a binary search with ~10 steps.
    # Each step runs simulation for ~1000 steps.
    # 10 * 1000 * (N^2 operations) = 10000 * 90000 = 900M ops?
    # N=300. N^2 = 90,000.
    # 1000 steps * 90,000 = 90M ops per sim.
    # 10 sims = 900M ops.
    # Python loop overhead is high. Vectorized NumPy is fast but 900M ops might take seconds.
    # 120s limit is generous.
    
    # Let's optimize the simulation loop.
    # We can reduce N^2 cost by only checking neighbors? 
    # For 300 points, N^2 is manageable in NumPy if we are careful.
    
    # Let's adjust parameters for speed and accuracy.
    max_iter_sim = 1500
    dt = 0.002
    damping = 0.99
    
    def simulate_fast(r):
        pos = positions.copy()
        vel = np.zeros_like(pos)
        
        # Pre-allocate
        for step in range(max_iter_sim):
            diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
            dist = np.linalg.norm(diff, axis=2)
            
            # Mask for interactions
            # We only need to consider pairs within 2*r
            # But we compute all distances anyway for simplicity, mask later.
            # To save time, we could use a spatial hash, but N=300 is small.
            
            mask = dist < 2 * r
            np.fill_diagonal(mask, False)
            
            # Forces
            # factor = k * (2*r - dist) / dist
            factor = np.zeros_like(dist)
            # Only calculate where mask is true
            # np.where is faster than boolean indexing for small arrays?
            # Let's use np.where
            # Note: dist can be 0 on diagonal, but mask is false there.
            # We need to avoid div by 0 in the calculation before masking.
            # So we calculate factor only where mask is true?
            # Or set dist to 1 where not mask?
            
            # Safe dist
            safe_dist = np.where(mask, dist, 1.0)
            factor = np.where(mask, repulsion_strength * (2 * r - safe_dist) / safe_dist, 0.0)
            
            forces = np.sum(factor[:, :, np.newaxis] * diff, axis=1)
            
            # Boundary
            dist_center = np.linalg.norm(pos, axis=1)
            limit = 1.0 - r
            overflow = np.maximum(0, dist_center - limit)
            safe_dist_c = np.where(dist_center > 1e-9, dist_center, 1e-9)
            boundary_force = -boundary_repulsion * overflow / safe_dist_c
            forces += boundary_force[:, np.newaxis] * pos
            
            # Integration
            vel = vel * damping + forces * dt
            pos = pos + vel * dt
            
        # Check validity
        final_dist = np.linalg.norm(pos[:, np.newaxis, :] - pos[np.newaxis, :, :], axis=2)
        np.fill_diagonal(final_dist, np.inf)
        min_pair_dist = np.min(final_dist)
        max_center_dist = np.max(np.linalg.norm(pos, axis=1))
        
        tol = 1e-4
        if min_pair_dist >= 2 * r - tol and max_center_dist <= 1 - r + tol:
            return True, pos
        return False, pos

    # Binary Search
    lo = 0.01
    hi = 0.08
    best_pos = positions
    best_r = lo
    
    # Run binary search
    for _ in range(10):
        mid = (lo + hi) / 2
        success, pos = simulate_fast(mid)
        if success:
            best_r = mid
            best_pos = pos
            lo = mid
        else:
            hi = mid
            
    return best_pos
# EVOLVE-BLOCK-END
