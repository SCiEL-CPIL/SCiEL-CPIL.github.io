"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
import time

def _generate_probes(count=20000):
    """Generate a dense set of probe points on the unit sphere using Fibonacci spiral."""
    i = np.arange(count) + 0.5
    z = 1.0 - 2.0 * i / count
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _compute_covering_radius(points, probes):
    """
    Compute the covering radius (max angular distance to nearest point) 
    using a KDTree for efficient nearest neighbor search.
    """
    tree = cKDTree(points)
    dists, _ = tree.query(probes, k=1)
    return np.max(dists)

def _perturb_points(points, sigma):
    """
    Create a perturbed copy of the points by adding Gaussian noise 
    and re-normalizing to the unit sphere.
    """
    noise = np.random.normal(0, sigma, size=points.shape)
    new_points = points + noise
    norms = np.linalg.norm(new_points, axis=1, keepdims=True)
    # Avoid division by zero
    norms[norms < 1e-10] = 1.0
    new_points /= norms
    return new_points

def construct():
    start_time = time.time()
    time_limit = 115.0  # Leave 5s buffer
    
    # 1. Generate Probes
    # 20,000 probes provide a good balance between accuracy and speed.
    # The angular spacing is roughly sqrt(4*pi / 20000) ~ 0.025 rad ~ 1.4 degrees.
    # This is sufficient to estimate the radius to ~0.1 degrees.
    probes = _generate_probes(20000)
    
    # 2. Initial Configuration: Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    current_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    current_radius = _compute_covering_radius(current_points, probes)
    best_points = current_points.copy()
    best_radius = current_radius
    
    # 3. Simulated Annealing Parameters
    # Start with a larger perturbation to explore, cool down to refine.
    # The scale of perturbation should be related to the desired movement.
    # A small angle like 0.1 rad (5.7 deg) is a good start.
    sigma_start = 0.1
    sigma_end = 0.001
    num_iterations = 50000 # We will loop until time limit or this count
    
    # Temperature schedule
    T_start = 0.05
    T_end = 0.001
    cooling_factor = (T_end / T_start) ** (1.0 / num_iterations)
    
    rng = np.random.RandomState(42)
    
    for i_iter in range(num_iterations):
        if time.time() - start_time > time_limit:
            break
            
        # Current Temperature
        T = T_start * (cooling_factor ** i_iter)
        
        # Perturb
        # We can perturb all points or a subset. Perturbing all is simpler.
        # As we cool, we reduce the perturbation size.
        current_sigma = sigma_start * (sigma_end / sigma_start) ** (i_iter / num_iterations)
        new_points = _perturb_points(current_points, current_sigma)
        
        # Evaluate
        new_radius = _compute_covering_radius(new_points, probes)
        delta_E = new_radius - current_radius
        
        # Acceptance Criteria
        if delta_E < 0 or rng.random() < np.exp(-delta_E / T):
            current_points = new_points
            current_radius = new_radius
            
            if current_radius < best_radius:
                best_radius = current_radius
                best_points = current_points.copy()
        
        # Optional: Occasionally restart or jitter if stuck? 
        # SA with cooling should handle it.
        
    # Final Normalization
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    best_points = best_points / norms
    
    return best_points
# EVOLVE-BLOCK-END
