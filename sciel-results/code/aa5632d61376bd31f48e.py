"""
Strategy: Global Energy Minimization via Simulated Annealing.
Instead of Lloyd's algorithm which relies on Voronoi cell centroids (often leading to grid-like local minima),
we treat the sphere packing as an energy minimization problem.
Energy = Sum of Repulsive Pair Potentials (1/d^p) + Wall Penalties.
We use Simulated Annealing to escape local minima and find a global or near-global minimum,
which corresponds to a packing with large minimum distances (and thus large radius).
"""
import numpy as np
import time

N = 50

def _compute_radius(centers):
    """Compute the maximum common radius for a given set of centers."""
    # Wall distances: distance from center to nearest wall
    wall_dists = 0.5 - np.max(np.abs(centers), axis=1)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    
    # Minimum distance from each center to any other center
    min_pair_dist = np.min(dists, axis=1)
    
    # The radius for each sphere is limited by the wall and half the nearest neighbor distance
    radii = np.minimum(wall_dists, min_pair_dist / 2.0)
    
    # The common radius is the minimum over all spheres
    return np.min(radii)

def _energy(centers, p=8):
    """
    Compute the total energy of the configuration.
    Lower energy is better.
    Energy consists of:
    1. Pair repulsion: Sum(1/d_ij^p) for all pairs.
    2. Wall repulsion: Sum(1/wall_i^p) for all spheres, where wall_i is distance to nearest wall.
    """
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    
    # Avoid division by zero (should be handled by inf diagonal, but just in case)
    # Mask upper triangle to avoid double counting
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu]
    
    # Repulsive potential
    # If distance is very small, energy is huge
    # Using p=8 makes the potential stiff, encouraging uniform spacing
    pair_energy = np.sum(1.0 / (pair_dists ** p))
    
    # Wall distances
    wall_dists = 0.5 - np.max(np.abs(centers), axis=1)
    # If a sphere is outside or on the wall, wall_dist <= 0.
    # We need a penalty that goes to infinity as wall_dist -> 0.
    # To avoid NaNs or inf, we clip wall_dists to a small positive epsilon for the potential calculation
    # but the optimizer should avoid this region.
    eps = 1e-6
    safe_wall_dists = np.maximum(wall_dists, eps)
    
    wall_energy = np.sum(1.0 / (safe_wall_dists ** p))
    
    return pair_energy + wall_energy

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    # 1. Initialization
    # Start with a random configuration inside the cube, slightly inset to avoid immediate wall collisions
    rng = np.random.default_rng(42)
    
    # We try multiple restarts with different initial temperatures and seeds if time permits
    best_centers = None
    best_radius = -np.inf
    best_energy = np.inf
    
    # Number of restarts depends on time. Let's aim for a few high-quality runs.
    # Each run: SA with specific schedule.
    
    max_restarts = 5
    # Estimate time per restart: ~20-30 seconds for 50 points in 3D with ~5000 steps?
    # We will adapt the number of steps based on time.
    
    for restart in range(max_restarts):
        if time.time() > deadline - 20:  # Leave 20s for final checks
            break
            
        # Initialize
        # Random points in [-0.4, 0.4] to start away from walls
        centers = rng.uniform(-0.4, 0.4, size=(N, 3))
        
        # Simulated Annealing Parameters
        # Temperature schedule: T = T0 * alpha^iter
        # T0 should be high enough to allow large moves initially.
        # Max step size should be comparable to the average spacing.
        # Average spacing ~ 1 / N^(1/3) ~ 1/3.6 ~ 0.27
        # Let's start with T0 such that kT is comparable to typical energy changes.
        
        T0 = 0.05
        T_end = 0.0001
        alpha = (T_end / T0) ** (1.0 / 5000) # 5000 steps
        
        current_energy = _energy(centers)
        
        # Step size schedule
        # Start with large steps, decrease over time
        step_scale = 0.1
        
        # Pre-compute indices for pair energy updates if we were doing incremental, 
        # but for simplicity and robustness in Python, we recompute full energy.
        # N=50 -> 1225 pairs. Recomputing is fast enough in NumPy.
        
        steps = 0
        max_steps = 8000 # Adjust based on time
        
        while steps < max_steps:
            if time.time() > deadline - 10:
                break
                
            # Generate a move: pick a random sphere and move it in a random direction
            i = rng.integers(0, N)
            direction = rng.normal(size=3)
            direction /= np.linalg.norm(direction)
            
            # Step size proportional to temperature
            # As T drops, steps get smaller
            step_size = T * 10.0 # Scaling factor
            # Also scale down step size as we progress in the run to refine
            progress = steps / max_steps
            effective_step = step_size * (1.0 - 0.5 * progress)
            
            delta = direction * effective_step
            
            new_centers = centers.copy()
            new_centers[i] += delta
            
            # Check if new position is within bounds (with small margin)
            # If it goes outside, we can either reject immediately or let the wall penalty handle it.
            # Letting the penalty handle it is smoother but might allow temporary "tunneling" through walls.
            # For strict constraint satisfaction, it's better to project back or reject.
            # However, high T allows exploring the boundary.
            # Let's just compute energy. The wall penalty will be huge if outside.
            
            new_energy = _energy(new_centers)
            
            # Metropolis Criterion
            dE = new_energy - current_energy
            
            if dE < 0:
                # Accept
                centers = new_centers
                current_energy = new_energy
            else:
                # Accept with probability exp(-dE / T)
                if T > 1e-9:
                    prob = np.exp(-dE / T)
                    if rng.random() < prob:
                        centers = new_centers
                        current_energy = new_energy
            
            # Update temperature
            T *= alpha
            steps += 1
            
            # Occasionally, if stuck (energy not improving for a while), we could do a "kick"
            # but standard SA usually handles this with the schedule.
            
        # End of SA run
        # Polish the result with a local optimizer to find the exact local minimum
        # We use a simple gradient descent or Nelder-Mead on the energy function
        # Or just rely on the low-T SA.
        
        # Let's do a quick local refinement using scipy if time permits
        if time.time() < deadline - 5:
            try:
                from scipy.optimize import minimize
                
                # Flatten centers
                x0 = centers.flatten()
                
                # Objective function for scipy
                def obj(x):
                    c = x.reshape(N, 3)
                    return _energy(c)
                    
                # Gradient is not easily available without significant code, so use Nelder-Mead
                # Nelder-Mead doesn't require gradients and works well for small N (150 vars)
                # But 150 dims is high for Nelder-Mead.
                # Let's stick to the SA result which is usually very close to the local min at low T.
                # Alternatively, a few steps of L-BFGS-B on the smooth energy?
                # The energy function is differentiable except at d=0 or wall=0.
                # Let's try a simple manual gradient step refinement.
                
                # Actually, the SA at T_end=0.0001 is already quite refined.
                # Let's just check the radius.
                
                pass
            except Exception:
                pass
        
        radius = _compute_radius(centers)
        
        if radius > best_radius:
            best_radius = radius
            best_centers = centers.copy()
            best_energy = current_energy
            
        # Optional: If we have time, do a second pass with the best found centers as a seed?
        # No, better to do independent restarts to explore different basins.
        
    # Final safety check
    if best_centers is None:
        # Fallback: Grid
        m = 4
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        indices = rng.choice(len(grid), size=N, replace=False)
        best_centers = grid[indices].copy()
        
    return best_centers
