"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist


def construct():
    """Place 50 equal spheres in unit cube [-0.5, 0.5]^3 to maximize minimum radius."""
    
    def compute_min_radius(positions):
        """Compute the minimum radius given positions of shape (N, 3)."""
        positions = np.asarray(positions).reshape(-1, 3)
        # Pairwise distances
        dists = pdist(positions, 'euclidean')
        min_dist = np.min(dists) if len(dists) > 0 else 1.0
        min_r_pairs = min_dist / 2.0
        
        # Wall distances: distance from each center to nearest wall
        walls = 0.5 - np.abs(positions)
        min_wall = np.min(walls)
        
        return min(min_r_pairs, min_wall)
    
    def objective_direct(positions):
        """Direct objective: negative of the actual minimum radius."""
        return -compute_min_radius(positions)
    
    def objective_energy(positions):
        """
        Smooth objective: Sum of inverse squared distances (pairwise and to walls).
        Minimizing this spreads points out.
        """
        positions = positions.reshape(-1, 3)
        
        # Pairwise distances
        dists = pdist(positions, 'euclidean')
        # Add epsilon to avoid division by zero
        dists = dists + 1e-9
        pair_energy = np.sum(1.0 / (dists ** 2))
        
        # Wall distances
        # For each point, distance to each of the 6 walls
        # x walls: 0.5 - x, 0.5 + x
        # y walls: 0.5 - y, 0.5 + y
        # z walls: 0.5 - z, 0.5 + z
        walls = np.abs(positions)
        wall_dists = 0.5 - walls
        # We want to keep them away from walls, so we add penalty for small wall dists
        # wall_dists can be negative if outside, but we use bounds in optimizer.
        # To be safe, clip to small positive value
        wall_dists = np.clip(wall_dists, 1e-9, None)
        wall_energy = np.sum(1.0 / (wall_dists ** 2))
        
        return pair_energy + 10.0 * wall_energy # Weight walls higher to keep them inside effectively

    def make_initial_grid():
        """Create a reasonable initial grid configuration."""
        m = int(np.ceil(N ** (1.0 / 3.0)))
        # Create m x m x m grid, take first N points
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N].copy()

    def make_bcc_like():
        """Create a body-centered cubic inspired arrangement."""
        positions = make_initial_grid()
        np.random.seed(123)
        positions += np.random.uniform(-0.02, 0.02, size=positions.shape)
        return positions

    def make_random():
        """Create random initial positions."""
        return np.random.uniform(-0.45, 0.45, size=(N, 3))

    def polish(positions, maxiter=200):
        """Refine using the direct objective."""
        bounds = [(-0.499999, 0.499999)] * (N * 3)
        try:
            res = minimize(objective_direct, positions.flatten(), method='L-BFGS-B',
                          bounds=bounds, options={'maxiter': maxiter, 'ftol': 1e-12})
            pos_opt = res.x.reshape(-1, 3)
            pos_opt = np.clip(pos_opt, -0.499999, 0.499999)
            return pos_opt
        except:
            return positions

    best_positions = None
    best_radius = -1.0
    
    np.random.seed(42)
    
    # Strategy: Use Energy Minimization to find good basins, then Polish with Direct Objective.
    # This is robust because Energy is smooth, while Min-Radius is not.
    
    initializations = [
        ("grid", make_initial_grid()),
        ("bcc", make_bcc_like()),
        ("rand1", make_random()),
        ("rand2", make_random()),
        ("rand3", make_random()),
    ]
    
    for name, init_pos in initializations:
        # Step 1: Minimize Energy
        bounds = [(-0.499999, 0.499999)] * (N * 3)
        try:
            res_energy = minimize(objective_energy, init_pos.flatten(), method='L-BFGS-B',
                                  bounds=bounds, options={'maxiter': 300, 'ftol': 1e-12})
            pos_energy = res_energy.x.reshape(-1, 3)
            pos_energy = np.clip(pos_energy, -0.499999, 0.499999)
        except Exception as e:
            pos_energy = init_pos
            
        # Step 2: Polish with Direct Objective
        pos_final = polish(pos_energy, maxiter=200)
        
        r = compute_min_radius(pos_final)
        print(f"{name} result: radius = {r:.10f}")
        
        if r > best_radius:
            best_radius = r
            best_positions = pos_final

    # Additional random restarts if time permits (simple loop)
    for i in range(5):
        init_pos = make_random()
        bounds = [(-0.499999, 0.499999)] * (N * 3)
        try:
            res_energy = minimize(objective_energy, init_pos.flatten(), method='L-BFGS-B',
                                  bounds=bounds, options={'maxiter': 100, 'ftol': 1e-12})
            pos_energy = res_energy.x.reshape(-1, 3)
            pos_energy = np.clip(pos_energy, -0.499999, 0.499999)
            pos_final = polish(pos_energy, maxiter=100)
            r = compute_min_radius(pos_final)
            if r > best_radius:
                best_radius = r
                best_positions = pos_final
        except:
            pass

    # Final clip
    if best_positions is not None:
        best_positions = np.clip(best_positions, -0.499999, 0.499999)
    
    print(f"Final best radius: {best_radius:.10f}")
    
    return best_positions
# EVOLVE-BLOCK-END
