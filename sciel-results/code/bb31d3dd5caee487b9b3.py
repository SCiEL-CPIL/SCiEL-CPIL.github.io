import numpy as np
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    # Simulated Annealing approach
    # Objective: Minimize sum of 1/d^k for pairs and walls
    # This effectively maximizes the minimum distance.
    
    # Parameters
    k_power = 20  # High power to create steep potential wells
    cutoff_dist = 0.3 # Only consider pairs closer than this for energy calculation (optimization)
    # Note: The true minimum distance will be ~0.22, so 0.3 is a safe cutoff for significant contributions
    # 1/0.3^20 is very small, so we can ignore pairs further than this.
    
    # Initial State: FCC Lattice (from seed) with slight perturbation
    a = 0.315
    v1 = np.array([0, 0.5, 0.5]) * a
    v2 = np.array([0.5, 0, 0.5]) * a
    v3 = np.array([0.5, 0.5, 0]) * a
    
    points = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            for k in range(-4, 5):
                p = i * v1 + j * v2 + k * v3
                if -0.5 <= p[0] <= 0.5 and -0.5 <= p[1] <= 0.5 and -0.5 <= p[2] <= 0.5:
                    points.append(p)
    
    points = np.array(points)
    
    if len(points) < N:
        # Fallback to random if lattice fails (unlikely)
        points = np.random.uniform(-0.45, 0.45, (N, 3))
        
    # Select N points
    if len(points) > N:
        # Random selection to break symmetry
        idx = np.random.choice(len(points), N, replace=False)
        centers = points[idx].copy()
    else:
        centers = points.copy()
        if len(centers) < N:
            # Pad with random
            extra = np.random.uniform(-0.45, 0.45, (N - len(centers), 3))
            centers = np.vstack([centers, extra])

    # Grid for neighbor search
    # Cube size 1.0, [-0.5, 0.5]
    # Grid cells size: cutoff_dist / 2 ? No, let's use a fixed grid size.
    # If cutoff is 0.3, a grid cell size of 0.3 means we check 27 neighbors.
    grid_size = 0.3
    n_cells = int(np.ceil(1.0 / grid_size))
    # Map coordinates to grid indices
    # coord in [-0.5, 0.5] -> idx in [0, n_cells-1]
    # idx = floor((coord + 0.5) / grid_size)
    
    def get_neighbors(centers, grid_size, n_cells):
        """Returns a list of lists of neighbor indices for each particle."""
        # This is expensive to rebuild every step. 
        # Alternative: Just compute full pairwise distances if N=100?
        # 100^2 = 10000 distances. 
        # In Python/NumPy, computing 10000 distances is ~0.1ms.
        # SA might run 100,000 steps. 100,000 * 0.1ms = 10 seconds.
        # This is acceptable and much simpler to implement correctly than dynamic grid updates.
        # Let's stick to full pairwise for simplicity and robustness, relying on NumPy vectorization.
        pass

    # Energy Function
    def calc_energy(centers):
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, 1.0) # Avoid div by 0
        
        # Mask for cutoff
        # We only care about d < cutoff_dist for the high power term?
        # 1/d^k drops off fast.
        # For d > 0.5, 1/d^20 is tiny.
        # Let's just compute all, it's fast enough for N=100.
        dists = np.maximum(dists, 1e-6)
        pair_energy = np.sum(1.0 / np.power(dists, k_power))
        
        # Wall energy
        # Distance to walls: min(0.5 - |x|, 0.5 - |y|, 0.5 - |z|)
        dx = 0.5 - np.abs(centers[:, 0])
        dy = 0.5 - np.abs(centers[:, 1])
        dz = 0.5 - np.abs(centers[:, 2])
        wall_dists = np.minimum(np.minimum(dx, dy), dz)
        wall_dists = np.maximum(wall_dists, 1e-6)
        wall_energy = np.sum(1.0 / np.power(wall_dists, k_power))
        
        return pair_energy + wall_energy

    def calc_radius(centers):
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, 1.0)
        min_dist = np.min(dists)
        
        dx = 0.5 - np.abs(centers[:, 0])
        dy = 0.5 - np.abs(centers[:, 1])
        dz = 0.5 - np.abs(centers[:, 2])
        min_wall = np.min(np.minimum(np.minimum(dx, dy), dz))
        
        return 0.5 * min(min_dist, min_wall)

    # Simulated Annealing
    current_centers = centers.copy()
    current_energy = calc_energy(current_centers)
    best_centers = current_centers.copy()
    best_radius = calc_radius(current_centers)
    
    T_start = 0.05
    T_end = 1e-5
    n_steps = 200000
    cooling_factor = (T_end / T_start) ** (1.0 / n_steps)
    
    step_size = 0.05
    
    rng = np.random.default_rng(42)
    
    start_time = time.time()
    max_time = 110.0 # Leave 10s buffer
    
    for step in range(n_steps):
        if time.time() - start_time > max_time:
            break
            
        T = T_start * (cooling_factor ** step)
        
        # Pick a random particle
        i = rng.integers(0, N)
        
        # Generate a random displacement
        # Gaussian step
        delta = rng.normal(0, step_size, 3)
        
        new_centers = current_centers.copy()
        new_centers[i] += delta
        
        # Keep inside bounds (hard constraint)
        # If it goes outside, we can either reject or clamp.
        # Clamping changes the energy landscape. Rejecting is safer for SA.
        # But if we reject too often, we waste steps.
        # Let's clamp to [-0.5, 0.5] and let the wall penalty handle the rest?
        # No, the task says "inside the cube". If we clamp, it's on the boundary.
        # The energy function penalizes being close to the wall.
        # If we move it outside, the wall distance becomes negative -> abs makes it positive?
        # 0.5 - |x| becomes negative if |x| > 0.5.
        # 1/negative^k is undefined or complex if k is odd, or positive if even?
        # We used max(dists, 1e-6). If dist is negative, max makes it 1e-6 -> huge energy.
        # So moving outside creates huge energy. SA will naturally avoid it.
        # However, for numerical stability, let's clamp the position to [-0.5, 0.5].
        # This effectively adds an infinite wall.
        new_centers[i] = np.clip(new_centers[i], -0.5, 0.5)
        
        # Calculate new energy
        new_energy = calc_energy(new_centers)
        
        delta_E = new_energy - current_energy
        
        if delta_E < 0 or rng.random() < np.exp(-delta_E / T):
            current_centers = new_centers
            current_energy = new_energy
            
            # Check if we found a better radius
            # We don't need to check every step, but checking every 100 steps is fine.
            if step % 100 == 0:
                r = calc_radius(current_centers)
                if r > best_radius:
                    best_radius = r
                    best_centers = current_centers.copy()
        
        # Adaptive step size?
        # If T is low, we can reduce step size to fine-tune.
        # Simple: step_size = T * 100 ?
        # Let's scale step size with T to help convergence.
        step_size = max(0.001, T * 50)

    # Final polish: A few iterations of local optimization (L-BFGS) on the best SA result?
    # This combines the global search of SA with local precision of L-BFGS.
    # It's a hybrid approach, but the core is SA.
    
    from scipy.optimize import minimize
    
    def objective(x):
        c = x.reshape(N, 3)
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, 1.0)
        dists = np.maximum(dists, 1e-9)
        pair_term = np.sum(1.0 / np.power(dists, k_power))
        
        dx = 0.5 - np.abs(c[:, 0])
        dy = 0.5 - np.abs(c[:, 1])
        dz = 0.5 - np.abs(c[:, 2])
        b_dists = np.minimum(np.minimum(dx, dy), dz)
        b_dists = np.maximum(b_dists, 1e-9)
        bound_term = np.sum(1.0 / np.power(b_dists, k_power))
        
        return pair_term + bound_term

    bounds = [(-0.5, 0.5)] * (N * 3)
    
    try:
        res = minimize(objective, best_centers.flatten(), method='L-BFGS-B', bounds=bounds, options={'maxiter': 50, 'ftol': 1e-12})
        if res.success:
            final_centers = res.x.reshape(N, 3)
            r_final = calc_radius(final_centers)
            if r_final > best_radius:
                best_centers = final_centers
                best_radius = r_final
    except Exception:
        pass

    return best_centers
# EVOLVE-BLOCK-END
