"""Simulated Annealing for logarithmic energy minimization on the sphere."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import math

def _fibonacci_sphere(n):
    """Generate a Fibonacci spiral distribution on the unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, 1.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _energy_delta(pts, i, new_x_i):
    """Calculate the change in energy if point i is moved to new_x_i."""
    # pts is the current array, new_x_i is the new position for point i
    # Energy = sum_{j<k} log(1/|x_j - x_k|) = -sum_{j<k} log(|x_j - x_k|)
    # Terms involving i are sum_{j != i} -log(|x_i - x_j|)
    
    # Distances from i to all others
    # diffs[j] = x_j - x_i
    diffs = pts - pts[i] 
    dists = np.linalg.norm(diffs, axis=1)
    dists[i] = 1.0 # Ignore self
    
    old_energy_part = -np.sum(np.log(dists))
    
    new_diffs = pts - new_x_i
    new_dists = np.linalg.norm(new_diffs, axis=1)
    new_dists[i] = 1.0
    
    new_energy_part = -np.sum(np.log(new_dists))
    
    return new_energy_part - old_energy_part

def construct():
    pts = _fibonacci_sphere(N).astype(np.float64)
    
    # SA Parameters
    T_start = 10.0
    T_end = 0.0001
    num_steps = 200000
    cool_factor = (T_end / T_start) ** (1.0 / num_steps)
    
    T = T_start
    rng = np.random.default_rng(42)
    
    for step in range(num_steps):
        # Pick a random point
        i = rng.integers(0, N)
        
        # Generate a random perturbation
        v = rng.standard_normal(3)
        
        # Project to tangent space at pts[i]
        x_i = pts[i]
        v_t = v - np.dot(v, x_i) * x_i
        
        norm_v_t = np.linalg.norm(v_t)
        if norm_v_t < 1e-12:
            continue
        v_t = v_t / norm_v_t
        
        # Step size scales with sqrt(T) to allow larger moves at high T
        step_size = 0.3 * math.sqrt(T)
        
        new_x_i = x_i + step_size * v_t
        
        # Renormalize to sphere
        norm_new = np.linalg.norm(new_x_i)
        if norm_new < 1e-12:
             continue
        new_x_i = new_x_i / norm_new
        
        # Calculate energy delta
        delta_E = _energy_delta(pts, i, new_x_i)
        
        # Acceptance criterion
        if delta_E < 0 or rng.random() < math.exp(-delta_E / T):
            pts[i] = new_x_i
            
        # Cool down
        T *= cool_factor
        
    # Final cleanup
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / norms
    
    assert np.all(np.isfinite(pts))
    assert pts.shape == (N, 3)
    
    return pts
# EVOLVE-BLOCK-END
