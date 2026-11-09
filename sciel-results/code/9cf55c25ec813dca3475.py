"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import math
import random


def _energy(pts):
    """Compute logarithmic energy: sum_{i<j} log(1/|x_i - x_j|)"""
    n = pts.shape[0]
    # Vectorized distance calculation
    # diff[i, j] = pts[i] - pts[j]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    
    # We need sum over i < j of -log(sqrt(dist_sq[i, j]))
    # = -0.5 * sum_{i<j} log(dist_sq[i, j])
    
    # Get upper triangle indices
    i_idx, j_idx = np.triu_indices(n, k=1)
    d_sq = dist_sq[i_idx, j_idx]
    
    # Avoid log(0)
    d_sq = np.maximum(d_sq, 1e-300)
    
    E = -0.5 * np.sum(np.log(d_sq))
    return E


def _random_point_on_sphere():
    """Generate a random point on the unit sphere."""
    v = np.random.randn(3)
    return v / np.linalg.norm(v)


def construct():
    # Set random seed for reproducibility if needed, but usually we want exploration.
    # The task says "score is seed/value", implying the evaluator might run multiple seeds or just one.
    # We will use a fixed seed for internal randomness to ensure consistent behavior if called multiple times,
    # or rely on the system. Let's use a fixed seed for the algorithm's internal noise to be deterministic per call.
    np.random.seed(42)
    random.seed(42)
    
    best_pts = None
    best_energy = float('inf')
    
    # Try multiple initializations to avoid poor local minima
    initializations = []
    
    # 1. Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    fib_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    initializations.append(fib_pts / np.linalg.norm(fib_pts, axis=1, keepdims=True))
    
    # 2. Random Points
    rand_pts = np.random.randn(N, 3)
    initializations.append(rand_pts / np.linalg.norm(rand_pts, axis=1, keepdims=True))
    
    # 3. Grid-like (simple lat/long)
    # 10x10 grid
    lat = np.linspace(-np.pi/2, np.pi/2, 10)
    lon = np.linspace(0, 2*np.pi, 10, endpoint=False)
    lat_grid, lon_grid = np.meshgrid(lat, lon)
    lat_flat = lat_grid.flatten()
    lon_flat = lon_grid.flatten()
    x = np.cos(lat_flat) * np.cos(lon_flat)
    y = np.cos(lat_flat) * np.sin(lon_flat)
    z = np.sin(lat_flat)
    grid_pts = np.column_stack([x, y, z])
    initializations.append(grid_pts / np.linalg.norm(grid_pts, axis=1, keepdims=True))

    # Simulated Annealing Parameters
    T_start = 10.0
    T_end = 1e-6
    alpha = 0.999 # Cooling rate
    n_iterations = 20000
    
    for init_pts in initializations:
        pts = init_pts.copy()
        current_energy = _energy(pts)
        
        if current_energy < best_energy:
            best_energy = current_energy
            best_pts = pts.copy()
            
        T = T_start
        for it in range(n_iterations):
            # Pick a random point to move
            idx = np.random.randint(N)
            
            # Generate a small random perturbation
            # Step size should decrease with temperature or time
            step_size = 0.1 * T / T_start # Simple scaling
            
            # Random direction in 3D
            delta = np.random.randn(3) * step_size
            
            new_point = pts[idx] + delta
            
            # Project to sphere
            norm = np.linalg.norm(new_point)
            if norm < 1e-10:
                new_point = _random_point_on_sphere()
            else:
                new_point = new_point / norm
            
            # Calculate energy change
            # For efficiency, we can calculate delta E directly, but _energy is fast enough for N=100
            # Let's just recalculate for simplicity and robustness.
            # Optimization: Only recalculate if we accept? No, we need it for acceptance.
            
            old_pts = pts.copy()
            pts[idx] = new_point
            new_energy = _energy(pts)
            
            delta_E = new_energy - current_energy
            
            # Metropolis criterion
            if delta_E < 0 or random.random() < math.exp(-delta_E / T):
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_pts = pts.copy()
            else:
                pts[idx] = old_pts[idx]
                
            # Cool down
            T *= alpha
            if T < T_end:
                T = T_end
                
    return best_pts
# EVOLVE-BLOCK-END
