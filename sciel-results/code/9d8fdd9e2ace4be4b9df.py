"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def _fibonacci_sphere(n):
    """Generate points on a sphere using the Fibonacci lattice."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _energy(points):
    """Calculate the logarithmic energy. Returns inf if invalid."""
    if not np.all(np.isfinite(points)):
        return float('inf')
    if points.shape != (N, 3):
        return float('inf')
    
    # Check normalization
    norms = np.linalg.norm(points, axis=1)
    if np.any(norms < 1e-9):
        return float('inf')
    if np.any(np.abs(norms - 1.0) > 1e-6):
        return float('inf')

    # Check minimum distance
    dists = np.linalg.norm(points[:, np.newaxis, :] - points[np.newaxis, :, :], axis=-1)
    # Mask diagonal
    np.fill_diagonal(dists, 1.0)
    min_dist = np.min(dists)
    if min_dist < 1e-9:
        return float('inf')
        
    # Calculate energy
    # E = sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    # We only need upper triangle
    iu = np.triu_indices(N, k=1)
    d = dists[iu]
    return -np.sum(np.log(d))

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # Initial points
    best_points = _fibonacci_sphere(N)
    best_energy = _energy(best_points)
    
    if not np.isfinite(best_energy):
         return best_points

    current_points = best_points.copy()
    current_energy = best_energy
    
    # Simulated Annealing parameters
    T = 1.0  # Initial temperature
    T_min = 1e-6
    cooling = 0.9995
    
    # We will use random perturbations. 
    # To respect the sphere constraint, we can generate random vectors and add them, then normalize.
    # Or rotate points slightly.
    
    while time.time() < deadline:
        # Perturb a random subset of points
        n_perturb = np.random.randint(1, 5)
        idx = np.random.choice(N, n_perturb, replace=False)
        
        # Generate random perturbations
        # We want to move points on the sphere. 
        # A simple way: add random noise and renormalize.
        # However, large noise can cause points to collide or be far off.
        # Better: small rotations or small steps in tangent space.
        
        # Let's use small steps in Cartesian space, then normalize.
        # Step size should be related to T.
        step_size = 0.1 * T
        noise = np.random.normal(0, step_size, size=(n_perturb, 3))
        
        new_points = current_points.copy()
        new_points[idx] += noise
        
        # Normalize
        norms = np.linalg.norm(new_points, axis=1)
        new_points /= norms[:, np.newaxis]
        
        new_energy = _energy(new_points)
        
        if np.isfinite(new_energy):
            delta_e = new_energy - current_energy
            # Accept if better, or with probability exp(-delta_e / T)
            if delta_e < 0 or np.random.random() < math.exp(-delta_e / T):
                current_points = new_points
                current_energy = new_energy
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_points = current_points.copy()
        
        # Cool down
        T *= cooling
        if T < T_min:
            T = T_min
    
    return best_points
# EVOLVE-BLOCK-END
