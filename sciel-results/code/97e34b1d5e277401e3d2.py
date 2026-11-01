import numpy as np
import time

try:
    import science_chemistry
except ImportError:
    science_chemistry = None

N = 55
TIME_LIMIT = 110.0

def calculate_energy(positions):
    n = positions.shape[0]
    if n < 2:
        return 0.0
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    inv_r6 = 1.0 / (dist_sq ** 3)
    inv_r12 = inv_r6 ** 2
    np.fill_diagonal(inv_r6, 0.0)
    np.fill_diagonal(inv_r12, 0.0)
    return 4.0 * (np.sum(inv_r12) - np.sum(inv_r6)) / 2.0

def construct():
    if science_chemistry is None:
        rng = np.random.default_rng(42)
        pos = rng.uniform(-3, 3, size=(N, 3))
        return pos

    best_energy = np.inf
    best_positions = None
    start_time = time.time()
    rng = np.random.default_rng()
    
    while time.time() - start_time < TIME_LIMIT:
        # Generate random positions in a sphere of radius R
        # R=5 is a reasonable guess for N=55
        # Use rejection sampling or just uniform in box and filter?
        # Box is simpler. [-3, 3] volume 216.
        # 55 atoms, min dist 0.5.
        # Probability of overlap is low.
        positions = rng.uniform(-3.0, 3.0, size=(N, 3))
        
        try:
            relaxed = science_chemistry.relax(positions)
        except Exception:
            continue
        
        if relaxed is None:
            continue
            
        # Check validity
        valid = True
        # Vectorized min distance check
        diff = relaxed[:, np.newaxis, :] - relaxed[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
        
        if min_dist < 0.5:
            continue
            
        energy = calculate_energy(relaxed)
        
        if energy < best_energy:
            best_energy = energy
            best_positions = relaxed.copy()
            
    if best_positions is None:
        # Fallback
        rng = np.random.default_rng(0)
        pos = rng.uniform(-3.0, 3.0, size=(N, 3))
        return pos
        
    return best_positions
