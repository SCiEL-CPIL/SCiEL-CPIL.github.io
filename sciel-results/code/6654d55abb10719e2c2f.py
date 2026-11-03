"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """Place 50 equal spheres in unit cube [-0.5, 0.5]^3 to maximize minimum radius using Simulated Annealing."""
    
    def compute_min_radius(positions):
        """Compute the minimum radius given positions of shape (N, 3)."""
        positions = np.asarray(positions).reshape(-1, 3)
        # Pairwise distances
        # Using broadcasting for O(N^2) but vectorized
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=-1))
        # Set diagonal to infinity to ignore self-distance
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
        min_r_pairs = min_dist / 2.0
        
        # Wall distances: distance from each center to nearest wall
        walls = 0.5 - np.abs(positions)
        min_wall = np.min(walls)
        
        return min(min_r_pairs, min_wall)

    def simulated_annealing(initial_pos, n_steps=50000, T_start=0.1, T_end=0.001, cooling_rate=0.9999):
        """
        Run Simulated Annealing to optimize positions.
        """
        current_pos = initial_pos.copy()
        current_r = compute_min_radius(current_pos)
        
        best_pos = current_pos.copy()
        best_r = current_r
        
        # Pre-compute indices for random selection
        indices = np.arange(N)
        
        for i in range(n_steps):
            T = T_start * (cooling_rate ** i)
            if T < T_end:
                T = T_end
            
            # Move a random subset of particles (e.g., 10%)
            n_move = max(1, N // 10)
            move_indices = np.random.choice(indices, size=n_move, replace=False)
            
            # Generate random displacement
            # Scale displacement by temperature to allow larger jumps when hot
            scale = 0.05 * (T / T_start) + 0.005
            displacements = np.random.normal(0, scale, size=(n_move, 3))
            
            new_pos = current_pos.copy()
            new_pos[move_indices] += displacements
            
            # Clip to boundaries
            new_pos = np.clip(new_pos, -0.5, 0.5)
            
            new_r = compute_min_radius(new_pos)
            
            # Metropolis criterion
            delta_r = new_r - current_r
            if delta_r > 0:
                # Accept if better
                current_pos = new_pos
                current_r = new_r
                if current_r > best_r:
                    best_r = current_r
                    best_pos = current_pos.copy()
            else:
                # Accept worse with probability exp(delta_r / T)
                # Note: delta_r is negative, T is positive
                # We want to maximize r, so we minimize -r.
                # Standard SA minimizes energy. Here energy = -r.
                # delta_energy = -new_r - (-current_r) = current_r - new_r = -delta_r
                # P = exp(-delta_energy / T) = exp(delta_r / T)
                prob = np.exp(delta_r / T)
                if np.random.rand() < prob:
                    current_pos = new_pos
                    current_r = new_r
        
        return best_pos, best_r

    # Run multiple restarts
    best_overall_pos = None
    best_overall_r = -1.0
    
    np.random.seed(42)
    
    # Initial configurations: Random, Grid-like
    initializations = []
    
    # 1. Random
    initializations.append(np.random.uniform(-0.45, 0.45, size=(N, 3)))
    
    # 2. Grid-like
    m = int(np.ceil(N ** (1.0 / 3.0)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    initializations.append(grid[:N].copy() + np.random.uniform(-0.02, 0.02, size=(N, 3)))
    
    # 3. Another random
    initializations.append(np.random.uniform(-0.45, 0.45, size=(N, 3)))
    
    for i, init_pos in enumerate(initializations):
        # Adjust steps based on time budget, but 50k is fast enough for N=50
        # We can run 3 restarts with 50k steps each.
        best_pos, best_r = simulated_annealing(init_pos, n_steps=50000)
        print(f"SA Restart {i}: radius = {best_r:.10f}")
        
        if best_r > best_overall_r:
            best_overall_r = best_r
            best_overall_pos = best_pos
    
    # Final refinement: A few more steps on the best found
    if best_overall_pos is not None:
        # Run a short, low-temperature SA to polish
        best_overall_pos, best_overall_r = simulated_annealing(
            best_overall_pos, n_steps=10000, T_start=0.01, T_end=0.0001, cooling_rate=0.9999
        )
        print(f"Polished radius: {best_overall_r:.10f}")
        
        # Ensure strictly inside
        best_overall_pos = np.clip(best_overall_pos, -0.499999, 0.499999)
        
    return best_overall_pos
# EVOLVE-BLOCK-END
