"""Improved packing for 100 equal spheres in a unit cube."""
import numpy as np

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a packing of N spheres in a unit cube using Simulated Annealing
    with a structured grid initialization.
    """
    # Reproducibility
    rng = np.random.default_rng(42)
    
    # Step 1: Structured Initialization
    # Create a 5x5x5 grid of cells. 125 cells total.
    # We need 100 spheres. We will pick 100 unique cells and place centers at their midpoints.
    grid_size = 5
    cell_size = 1.0 / grid_size
    cell_centers = []
    for x in range(grid_size):
        for y in range(grid_size):
            for z in range(grid_size):
                cx = -0.5 + (x + 0.5) * cell_size
                cy = -0.5 + (y + 0.5) * cell_size
                cz = -0.5 + (z + 0.5) * cell_size
                cell_centers.append([cx, cy, cz])
    cell_centers = np.array(cell_centers)
    
    # Select 100 random unique indices
    indices = rng.choice(len(cell_centers), size=N, replace=False)
    centers = cell_centers[indices].copy()
    
    # Add small random jitter to break symmetry
    centers += rng.normal(0, 0.01, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    
    # Step 2: Simulated Annealing with Optimized Distance Updates
    def calculate_radius_and_D(pts, D=None):
        """
        Calculate radius and update distance matrix D.
        If D is None, compute full D.
        """
        if D is None:
            diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            D = np.sqrt(np.sum(diffs**2, axis=2))
            np.fill_diagonal(D, np.inf)
        
        min_pair = np.min(D)
        
        # Wall clearance
        wall_clearances = np.min(0.5 - np.abs(pts), axis=1)
        min_wall = np.min(wall_clearances)
        
        return min(min_pair / 2.0, min_wall), D

    # Initialize Distance Matrix
    best_centers = centers.copy()
    best_radius, D = calculate_radius_and_D(best_centers, None)
    
    # SA Parameters
    T_start = 0.01
    T_end = 0.0001
    n_iterations = 15000
    base_step_size = 0.05
    
    current_centers = best_centers.copy()
    current_radius = best_radius
    current_D = D.copy()
    
    for i in range(n_iterations):
        # Cool down
        T = T_start * (T_end / T_start) ** (i / n_iterations)
        
        # Pick a random sphere
        idx = rng.integers(N)
        
        # Generate random move
        # Step size decreases slightly over time
        move_step = base_step_size * (0.5 ** (i / n_iterations))
        delta = rng.normal(0, move_step, 3)
        
        new_center = current_centers[idx].copy()
        new_center += delta
        new_center = np.clip(new_center, -0.4999, 0.4999)
        
        # Update centers and distance matrix efficiently
        temp_centers = current_centers.copy()
        temp_centers[idx] = new_center
        
        # Update D for the moved sphere
        # We need to update row and col idx in current_D
        # To avoid modifying current_D if we reject, we can compute the new min directly
        # or copy D. Copying D (100x100) is cheap.
        temp_D = current_D.copy()
        
        # Update row and col
        diffs = new_center - current_centers
        dists = np.sqrt(np.sum(diffs**2, axis=1))
        temp_D[idx, :] = dists
        temp_D[:, idx] = dists
        temp_D[idx, idx] = np.inf
        
        # Calculate new radius
        min_pair = np.min(temp_D)
        wall_clearances = np.min(0.5 - np.abs(temp_centers), axis=1)
        min_wall = np.min(wall_clearances)
        new_radius = min(min_pair / 2.0, min_wall)
        
        # Acceptance criterion
        if new_radius >= current_radius:
            current_centers = temp_centers
            current_radius = new_radius
            current_D = temp_D
            if current_radius > best_radius:
                best_radius = current_radius
                best_centers = current_centers.copy()
        else:
            # Accept with probability exp((new - old) / T)
            delta_r = new_radius - current_radius
            prob = np.exp(delta_r / T)
            if rng.random() < prob:
                current_centers = temp_centers
                current_radius = new_radius
                current_D = temp_D
    
    return best_centers
# EVOLVE-BLOCK-END
