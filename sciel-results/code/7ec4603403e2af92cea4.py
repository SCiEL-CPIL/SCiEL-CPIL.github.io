import numpy as np
import time

N = 300

def _hex_lattice_init(n):
    """Generate a hexagonal lattice point cloud inside the unit circle."""
    points = []
    s = np.sqrt(2 * np.pi / (n * np.sqrt(3)))
    max_r = 1.0
    k = int(max_r / s) + 2
    
    for _ in range(10): # Retry loop for spacing
        points = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * s
                y = j * s * np.sqrt(3) / 2
                if np.hypot(x, y) <= 1.0:
                    points.append([x, y])
        points = np.array(points)
        if len(points) >= n:
            break
        s *= 0.95
        k = int(max_r / s) + 2
        
    if len(points) > n:
        dists = np.hypot(points[:, 0], points[:, 1])
        idx = np.argsort(dists)[:n]
        points = points[idx]
        
    return points

def _relax(positions, r, steps, deadline):
    """
    Relax positions to fit circles of radius r.
    """
    pos = positions.copy()
    n = len(pos)
    step_size = 0.05
    temp = 0.01
    
    for step in range(steps):
        if time.time() > deadline:
            break
            
        # Compute pairwise distances
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        
        inv_dists = np.zeros_like(dists)
        np.divide(1.0, dists, out=inv_dists, where=dists > 1e-6)
        
        dir_vec = diff * inv_dists[..., np.newaxis]
        
        magnitudes = np.zeros_like(dists)
        np.subtract(2 * r, dists, out=magnitudes, where=dists < 2 * r)
        magnitudes[magnitudes < 0] = 0.0
        
        forces = np.sum(dir_vec * magnitudes[..., np.newaxis], axis=1)
        
        # Boundary force
        radii = np.linalg.norm(pos, axis=1)
        boundary_violation = np.maximum(0, radii - (1 - r))
        safe_radii = np.where(radii > 1e-6, radii, 1.0)
        boundary_forces = -pos / safe_radii[:, np.newaxis] * boundary_violation[:, np.newaxis]
        
        forces += boundary_forces
        
        noise = np.random.normal(0, temp, size=pos.shape)
        pos += step_size * forces + noise
        
        # Anneal
        temp *= 0.99
        step_size *= 0.999
        
    return pos

def _check_validity(positions, r):
    radii = np.linalg.norm(positions, axis=1)
    if np.any(radii > 1 - r + 1e-9):
        return False
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    if np.any(dists < 2 * r - 1e-9):
        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    pos = _hex_lattice_init(N)
    
    def calc_radius(p):
        radii = np.linalg.norm(p, axis=1)
        r_bound = 1 - np.max(radii)
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        r_pair = np.min(dists) / 2
        return min(r_bound, r_pair)
        
    r_current = calc_radius(pos)
    r_low = r_current
    r_high = r_current * 1.05
    
    best_pos = pos.copy()
    best_r = r_current
    
    while time.time() < deadline:
        r_mid = (r_low + r_high) / 2
        
        # Use remaining time to determine steps
        time_left = deadline - time.time()
        # Estimate time per step ~ 0.01s for N=300? 
        # Let's just use a fixed number of steps that fits in ~10s
        steps = 500
        
        new_pos = _relax(best_pos, r_mid, steps, deadline)
        
        if _check_validity(new_pos, r_mid):
            r_low = r_mid
            best_pos = new_pos.copy()
            best_r = r_mid
        else:
            r_high = r_mid
            
        if r_high - r_low < 1e-7:
            break
            
    return best_pos
