"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    t0 = time.time()
    deadline = t0 + 115.0  # safety margin before 120s limit
    
    # Initial grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    centers = grid[:N].copy().astype(np.float64)
    
    def compute_score(c):
        """Compute the minimum radius (score) for given centers."""
        # Wall distances
        wall_dist = np.min(np.abs(c), axis=1)
        wall_r = 0.5 - wall_dist
        # Pair distances
        diffs = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        return min(wall_r.min(), min_pair)
    
    best_centers = centers.copy()
    best_score = compute_score(centers)
    
    # Force-directed optimization with velocity-Verlet
    def compute_forces(c):
        """Compute repulsive forces between spheres and wall forces."""
        n = len(c)
        forces = np.zeros_like(c)
        
        # Pair repulsion: F = k / d^3 in direction away from neighbor
        for i in range(n):
            for j in range(i + 1, n):
                d_vec = c[i] - c[j]
                d = np.linalg.norm(d_vec)
                if d < 1e-10:
                    # Random nudge for coincident points
                    d_vec = np.random.randn(3) * 1e-6
                    d = np.linalg.norm(d_vec)
                if d > 0:
                    f_mag = 1.0 / (d * d * d)
                    f_vec = f_mag * d_vec / d
                    forces[i] += f_vec
                    forces[j] -= f_vec
        
        # Wall repulsion: push back if too close to wall
        for i in range(n):
            for k in range(3):
                # Distance to wall
                dist_to_wall = 0.5 - abs(c[i, k])
                if dist_to_wall < 0.1:
                    # Strong repulsion from wall
                    f_wall = 100.0 / (dist_to_wall + 1e-6)
                    sign = 1.0 if c[i, k] >= 0 else -1.0
                    forces[i, k] += sign * f_wall
        
        return forces
    
    # Velocity-Verlet simulation
    v = np.zeros_like(centers)
    dt = 0.001
    damping = 0.99
    temp = 0.01  # temperature for random kicks
    
    n_iter = 0
    while time.time() < deadline:
        # Compute forces
        F = compute_forces(centers)
        
        # Velocity-Verlet update
        v = v * damping + F * dt
        centers = centers + v * dt
        
        # Add small random perturbation to escape local minima
        if n_iter % 50 == 0:
            centers += np.random.randn(*centers.shape) * temp
        
        # Clamp to cube
        centers = np.clip(centers, -0.49, 0.49)
        
        n_iter += 1
        
        # Check score periodically
        if n_iter % 100 == 0:
            s = compute_score(centers)
            if s > best_score:
                best_score = s
                best_centers = centers.copy()
        
        # Adaptive: reduce temperature and dt over time
        temp *= 0.999
        if n_iter % 1000 == 0:
            dt *= 0.95
            damping *= 0.999
    
    # Final polish: local optimization using scipy
    from scipy.optimize import minimize
    
    def neg_score(c):
        c = c.reshape(N, 3)
        return -compute_score(c)
    
    # Try a few local optimizations from best position
    for _ in range(3):
        if time.time() > deadline - 10:
            break
        res = minimize(neg_score, best_centers.flatten(), method='Nelder-Mead',
                      options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12})
        if res.success or -res.fun > best_score:
            candidate = res.x.reshape(N, 3)
            s = compute_score(candidate)
            if s > best_score:
                best_score = s
                best_centers = candidate.copy()
    
    # Ensure validity: no centers outside cube, no coincident centers
    best_centers = np.clip(best_centers, -0.4999, 0.4999)
    
    return best_centers
# EVOLVE-BLOCK-END
