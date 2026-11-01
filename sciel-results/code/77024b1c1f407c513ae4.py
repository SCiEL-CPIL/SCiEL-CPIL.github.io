"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random

def _energy(points):
    """
    Calculate the logarithmic energy.
    E = sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    """
    if not np.all(np.isfinite(points)):
        return float('inf')
    
    # Compute pairwise distances
    # dists[i, j] = ||p_i - p_j||
    dists = np.linalg.norm(points[:, np.newaxis, :] - points[np.newaxis, :, :], axis=-1)
    
    # Avoid division by zero on diagonal
    np.fill_diagonal(dists, 1.0)
    
    # Check for invalid close points
    min_dist = np.min(dists)
    if min_dist < 1e-9:
        return float('inf')
        
    # Energy
    # Only sum upper triangle to avoid double counting
    iu = np.triu_indices(N, k=1)
    d = dists[iu]
    # Handle any zero distances that might slip through (though min_dist check should catch)
    d[d < 1e-9] = 1e-9
    energy = -np.sum(np.log(d))
    return energy

def _fibonacci_sphere(n):
    """Generate points on a sphere using the Fibonacci lattice."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(n):
    """Generate random points on the sphere."""
    pts = np.random.randn(n, 3)
    norms = np.linalg.norm(pts, axis=1)
    return pts / norms[:, np.newaxis]

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # Initialize with Fibonacci sphere, which is a good starting point
    points = _fibonacci_sphere(N)
    current_energy = _energy(points)
    best_points = points.copy()
    best_energy = current_energy
    
    # Simulated Annealing parameters
    # Initial temperature should be high enough to accept most moves
    # Energy scale is roughly ~1000. Let's start T around 10-20.
    T_start = 20.0
    T_end = 0.01
    # Cooling schedule: geometric or linear? Geometric is standard.
    # T_k = T_start * (T_end/T_start)^k
    # We need to estimate number of steps.
    # Let's run as many steps as possible until deadline.
    # But we need a fixed number of steps to define the cooling curve?
    # Or we can use a time-based cooling.
    # Time-based is safer.
    
    # Let's use a time-based cooling: T(t) = T_start * exp(-lambda * t)
    # Or simpler: T(t) = T_start * (1 - t/T_total)^power
    # Let's stick to a fixed number of iterations if we can estimate time per iteration.
    # N=100, distance calculation is O(N^2) = 10000 ops. Very fast.
    # Python overhead might dominate.
    # Let's try to do ~100,000 to 500,000 iterations.
    
    # To be safe, let's do a fixed number of iterations that we know fits in time,
    # or check time every few iterations.
    
    max_iters = 200000
    # Pre-calculate cooling factor
    # T_k = T_start * (T_end/T_start)^(k/max_iters)
    cool_factor = (T_end / T_start) ** (1.0 / max_iters)
    
    T = T_start
    rng = np.random.default_rng(42)
    
    for k in range(max_iters):
        if time.time() > deadline:
            break
            
        # Check time every 1000 iterations to avoid overhead
        if k % 1000 == 0 and time.time() > deadline - 2.0:
            break
            
        # Generate a random move
        # Strategy: Pick a random point and perturb it slightly on the sphere
        i = rng.integers(0, N)
        
        # Generate a random tangent direction
        # p_i is current point.
        # Random vector v.
        v = rng.normal(size=3)
        # Project v to tangent space at p_i
        # v_tan = v - (v . p_i) * p_i
        p_i = points[i]
        dot = np.dot(v, p_i)
        v_tan = v - dot * p_i
        
        # Normalize v_tan
        norm_v = np.linalg.norm(v_tan)
        if norm_v < 1e-10:
            continue
        v_tan /= norm_v
        
        # Step size: adaptive or fixed?
        # Fixed step size is simpler.
        # Typical distance between neighbors in Fibonacci is ~0.17 (1/sqrt(N)).
        # Let's try a step size of 0.05 to 0.1.
        step_size = 0.08
        
        # New point: p_i_new = normalize(p_i + step_size * v_tan)
        p_i_new = p_i + step_size * v_tan
        p_i_new /= np.linalg.norm(p_i_new)
        
        # Calculate delta energy
        # E_new - E_old
        # E = - sum log(d_ij)
        # Change involves only distances from i to all j != i
        # d_ij_old = ||p_i - p_j||
        # d_ij_new = ||p_i_new - p_j||
        
        # Vectorized calculation of delta
        # diffs_old = p_i - points
        # diffs_new = p_i_new - points
        # dists_old = norm(diffs_old)
        # dists_new = norm(diffs_new)
        # mask j != i
        
        diffs_old = p_i - points
        diffs_new = p_i_new - points
        
        dists_old = np.linalg.norm(diffs_old, axis=1)
        dists_new = np.linalg.norm(diffs_new, axis=1)
        
        # Mask out i
        mask = np.ones(N, dtype=bool)
        mask[i] = False
        
        # Ensure no zeros
        dists_old[mask] = np.maximum(dists_old[mask], 1e-12)
        dists_new[mask] = np.maximum(dists_new[mask], 1e-12)
        
        # Delta E = - sum log(d_new) - (- sum log(d_old)) = sum log(d_old) - sum log(d_new)
        # = sum (log(d_old) - log(d_new))
        delta_e = np.sum(np.log(dists_old[mask]) - np.log(dists_new[mask]))
        
        # Accept if delta_e < 0 or with probability exp(-delta_e / T)
        if delta_e < 0 or rng.random() < math.exp(-delta_e / T):
            points[i] = p_i_new
            current_energy += delta_e
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_points = points.copy()
                
        # Cool down
        T *= cool_factor
        
    return best_points
# EVOLVE-BLOCK-END
