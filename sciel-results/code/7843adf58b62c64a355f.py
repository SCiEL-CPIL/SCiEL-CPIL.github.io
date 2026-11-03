import numpy as np
import time
import math

N = 150

def _energy(points):
    # points: (N, 3)
    # Compute all pairwise distances
    # d_ij = ||p_i - p_j||
    # E = sum_{i<j} log(1/d_ij)
    
    # Vectorized distance calculation
    # diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    # dists = np.linalg.norm(diff, axis=2)
    
    # To avoid log(0), we assume points are distinct.
    # We only need upper triangle.
    i, j = np.triu_indices(N, k=1)
    dists = np.linalg.norm(points[i] - points[j], axis=1)
    
    # Clip to avoid log(0) if points coincide (shouldn't happen with SA, but safety)
    dists = np.clip(dists, 1e-12, None)
    
    return -np.sum(np.log(dists))

def construct():
    start_time = time.time()
    deadline = 140.0  # 10s margin
    
    # 1. Initialization: Fibonacci Sphere
    # i = np.arange(N) + 0.5
    # z = 1.0 - 2.0 * i / N
    # phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    # r = np.sqrt(1.0 - z * z)
    # points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Alternative Init: Random + Quick Relaxation?
    # Fibonacci is usually better for start.
    
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    current_energy = _energy(points)
    best_points = points.copy()
    best_energy = current_energy
    
    # Precompute indices for energy calculation to speed up loop
    i_idx, j_idx = np.triu_indices(N, k=1)
    
    # SA Parameters
    T_start = 1.0
    T_end = 1e-6
    steps = 200000  # Try to run as many as possible
    
    for step in range(steps):
        if time.time() > deadline:
            break
            
        # Cooling schedule
        # Linear or Exponential? Exponential is common.
        # T = T_start * (T_end / T_start) ** (step / steps)
        T = T_start * (T_end / T_start) ** (step / steps)
        
        # Perturbation
        # Choose a random point
        idx = np.random.randint(N)
        
        # Generate random displacement
        # Step size should scale with T?
        # Let's use a fixed small step for stability, or scale slightly.
        step_size = 0.1 * (T / T_start + 0.1) # Keep it from being too small too early?
        # Actually, constant step size often works well if T is handled correctly.
        # Let's try step_size = 0.05
        disp = np.random.normal(0, 0.1, 3)
        
        new_point = points[idx] + disp
        norm = np.linalg.norm(new_point)
        if norm < 1e-9:
            continue
        new_point /= norm
        
        # Calculate energy change
        # We only need to update terms involving idx
        # Old energy contribution of idx: sum_{j != idx} log(1/d(idx, j))
        # New energy contribution of idx: sum_{j != idx} log(1/d(new_point, j))
        
        # Points other than idx:
        mask = np.ones(N, dtype=bool)
        mask[idx] = False
        others = points[mask]
        
        # Distances from old point to others
        old_dists = np.linalg.norm(others - points[idx], axis=1)
        old_dists = np.clip(old_dists, 1e-12, None)
        old_contrib = -np.sum(np.log(old_dists))
        
        # Distances from new point to others
        new_dists = np.linalg.norm(others - new_point, axis=1)
        new_dists = np.clip(new_dists, 1e-12, None)
        new_contrib = -np.sum(np.log(new_dists))
        
        delta_E = new_contrib - old_contrib
        
        # Metropolis Acceptance
        if delta_E < 0 or np.random.random() < math.exp(-delta_E / T):
            points[idx] = new_point
            current_energy += delta_E
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_points = points.copy()
        
        # Optional: If we are stuck, maybe restart? 
        # For now, rely on SA cooling.
        
    return best_points
