"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def construct():
    """
    Construct centers of N=100 equal non-overlapping spheres in the unit cube [-0.5, 0.5]^3
    to maximize the common radius.
    
    Strategy:
    1. Initialize with a well-distributed starting configuration (e.g., jittered grid or random).
    2. Use simulated annealing or gradient-based optimization to maximize the minimum distance
       between any pair of centers and between any center and the cube boundary.
    3. The objective is to maximize r = min(min_{i<j} ||c_i - c_j||/2, min_i min_k (0.5 - |c_ik|)).
    
    We use a two-stage approach:
    - Stage 1: Simulated annealing to escape local minima.
    - Stage 2: Fine-tuning with gradient-based methods (if feasible) or continued SA.
    
    Given the 120s time limit, we use a robust SA with adaptive cooling.
    """
    rng = np.random.default_rng(42)
    
    # Initial configuration: jittered grid for good spatial distribution
    # Use a grid of roughly N^(1/3) points per dimension, with jitter
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    # Create grid points
    ticks = (np.arange(m) + 0.5) / m
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    centers = grid[:N].copy()
    
    # Add small random jitter to break symmetry and help optimization
    centers += rng.uniform(-0.01, 0.01, size=centers.shape)
    
    # Clip to stay within cube
    centers = np.clip(centers, -0.5, 0.5)
    
    def compute_radius(centers):
        """Compute the maximum radius for given centers."""
        # Minimum distance between pairs
        if len(centers) < 2:
            min_pair_dist = 1.0
        else:
            # Use scipy's cdist for efficiency
            from scipy.spatial.distance import cdist
            dists = cdist(centers, centers)
            # Set diagonal to inf
            np.fill_diagonal(dists, np.inf)
            min_pair_dist = np.min(dists)
        
        # Minimum distance to boundary
        # For each center, distance to each face
        dist_to_faces = np.minimum(
            0.5 - np.abs(centers),
            axis=1
        )
        min_boundary_dist = np.min(dist_to_faces)
        
        # Radius is half the minimum of pair distance and boundary distance
        # Wait, the radius r must satisfy:
        # r <= ||c_i - c_j|| / 2 for all i < j
        # r <= 0.5 - |c_ik| for all i, k
        # So r = min(min_pair_dist / 2, min_boundary_dist)
        
        return min(min_pair_dist / 2.0, min_boundary_dist)
    
    def energy(centers):
        """Negative of the radius, to minimize."""
        return -compute_radius(centers)
    
    # Simulated annealing
    T = 0.01  # Initial temperature
    T_min = 1e-6
    alpha = 0.9995  # Cooling rate
    max_iter = 50000
    current_energy = energy(centers)
    best_centers = centers.copy()
    best_energy = current_energy
    
    for i in range(max_iter):
        # Propose a move: perturb a random center
        idx = rng.integers(0, N)
        new_center = centers[idx].copy()
        new_center += rng.normal(0, T * 0.5, size=3)
        new_center = np.clip(new_center, -0.5, 0.5)
        
        old_center = centers[idx].copy()
        centers[idx] = new_center
        
        new_energy = energy(centers)
        
        # Accept or reject
        if new_energy < current_energy or rng.random() < np.exp((current_energy - new_energy) / T):
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_centers = centers.copy()
        else:
            centers[idx] = old_center
        
        # Cool down
        T *= alpha
        if T < T_min:
            T = T_min
    
    return best_centers
# EVOLVE-BLOCK-END
