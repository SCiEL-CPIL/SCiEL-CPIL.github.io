import numpy as np
import time

try:
    import science_chemistry
except ImportError:
    science_chemistry = None

N = 55
R_BOUND = 3.5
MIN_DIST = 0.5

def compute_energy(positions):
    n = len(positions)
    if n < 2:
        return 0.0
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    iu = np.triu_indices(n, k=1)
    d2 = dist_sq[iu]
    d6 = 1.0 / d2
    d6 = d6 * d6 * d6
    d12 = d6 * d6
    energy = 4.0 * np.sum(d12 - d6)
    return energy

def generate_random_cluster():
    while True:
        u = np.random.rand(N, 3)
        theta = np.arccos(2 * u[:, 0] - 1)
        phi = 2 * np.pi * u[:, 1]
        r = R_BOUND * np.cbrt(u[:, 2])
        
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        
        positions = np.column_stack((x, y, z))
        
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dist_sq, 1.0)
        min_dist_sq = np.min(dist_sq)
        
        if min_dist_sq >= MIN_DIST**2:
            positions -= np.mean(positions, axis=0)
            return positions

def construct():
    if science_chemistry is None:
        return generate_random_cluster()

    best_energy = np.inf
    best_positions = None
    start_time = time.time()
    time_limit = 110.0
    
    max_restarts = 50
    
    for i in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
            
        pos = generate_random_cluster()
        
        try:
            relaxed_pos = science_chemistry.relax(pos)
        except Exception:
            continue
            
        diff = relaxed_pos[:, np.newaxis, :] - relaxed_pos[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dist_sq, 1.0)
        if np.min(dist_sq) < MIN_DIST**2:
            continue
            
        energy = compute_energy(relaxed_pos)
        
        if energy < best_energy:
            best_energy = energy
            best_positions = relaxed_pos

    if best_positions is None:
        return generate_random_cluster()
        
    return best_positions
