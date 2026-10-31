import numpy as np
from scipy.optimize import minimize
import time

N = 38
DEADLINE = 110.0  # 10s margin before 120s limit

def _lj_energy(pos):
    """Compute LJ energy for positions pos (N, 3)."""
    # Compute all pair distances
    # pos: (N, 3)
    # diff: (N, N, 3)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    # Avoid division by zero for i==j
    dist_sq[np.diag_indices_from(dist_sq)] = np.inf
    dist = np.sqrt(dist_sq)
    
    # LJ potential: 4 * (r^-12 - r^-6)
    # Using r^-6 to reduce operations
    inv_r6 = 1.0 / (dist**6)
    inv_r12 = inv_r6 * inv_r6
    potential = 4.0 * (inv_r12 - inv_r6)
    
    # Sum over all pairs (each pair counted twice in the matrix, so divide by 2)
    return 0.5 * np.sum(potential)

def _lj_force(pos):
    """Compute LJ force for positions pos (N, 3)."""
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    dist_sq[np.diag_indices_from(dist_sq)] = np.inf
    dist = np.sqrt(dist_sq)
    
    # Force is -dV/dr.
    # V = 4(r^-12 - r^-6)
    # dV/dr = 4(-12r^-13 + 6r^-7) = -24r^-13 + 24r^-7
    # F = -dV/dr * (r_vec / r) = (24r^-13 - 24r^-7) * (r_vec / r)
    # F = 24 * (r^-14 - r^-8) * r_vec
    # Alternatively: F_ij = 48 * (r^-14 - 0.5 * r^-8) * r_vec ?
    # Let's stick to scalar potential gradient.
    # grad V = 4 * (-12 r^-14 + 6 r^-8) * r_vec
    # Force = -grad V = 4 * (12 r^-14 - 6 r^-8) * r_vec = 24 * (2 r^-14 - r^-8) * r_vec
    
    inv_r2 = 1.0 / dist_sq
    inv_r4 = inv_r2 * inv_r2
    inv_r6 = inv_r4 * inv_r2
    inv_r8 = inv_r6 * inv_r2
    inv_r14 = inv_r8 * inv_r6 * inv_r2 # r^-14 = r^-8 * r^-6
    
    # Coefficient for vector r_vec
    # F_vec = 24 * (2 * r^-14 - r^-8) * r_vec
    coeff = 24.0 * (2.0 * inv_r14 - inv_r8)
    # coeff shape (N, N), diff shape (N, N, 3)
    forces = coeff[:, :, np.newaxis] * diff
    # Sum over neighbors
    return np.sum(forces, axis=1)

def construct():
    start_time = time.time()
    
    # 1. Generate FCC lattice points
    # We need at least 38 points close to origin.
    # Truncated octahedron for N=38 is the global min.
    # Generating a box of FCC points and selecting closest to origin is a robust way to get the correct topology.
    
    # FCC points: (i, j, k) where i+j+k is even.
    # Scale: nearest neighbor distance is sqrt(2) in this integer grid.
    # Target nearest neighbor distance: 2^(1/6).
    # Scale factor: 2^(1/6) / sqrt(2) = 2^(-1/3).
    
    max_range = 5 # Should be enough to get 38+ points
    pts = []
    for i in range(-max_range, max_range + 1):
        for j in range(-max_range, max_range + 1):
            for k in range(-max_range, max_range + 1):
                if (i + j + k) % 2 == 0:
                    pts.append([i, j, k])
    
    pts = np.array(pts, dtype=float)
    # Scale to LJ units
    scale = 2.0 ** (-1.0 / 3.0)
    pts = pts * scale
    
    # Center at origin (it already is) and sort by distance to origin
    dists = np.sum(pts**2, axis=1)
    indices = np.argsort(dists)
    initial_pos = pts[indices[:N]]
    
    # Check validity: atoms must be at least 0.5 apart.
    # In LJ units, min distance is 2^(1/6) ~ 1.12, so this is fine.
    
    # 2. Relax using L-BFGS-B
    # We have a deadline.
    # L-BFGS is usually fast for this size.
    
    def energy_func(x):
        return _lj_energy(x)
        
    def force_func(x):
        # minimize expects gradient (negative force)
        return -_lj_force(x)
        
    # Initial guess is already very close to global min.
    # Run minimize.
    try:
        res = minimize(
            energy_func, 
            initial_pos, 
            method='L-BFGS-B', 
            jac=force_func,
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        # Check time
        if time.time() - start_time > DEADLINE:
             # Return best found so far? res.x is the last one.
             # If it timed out, we might be in a bad state, but L-BFGS usually converges fast.
             pass
             
        final_pos = res.x
        final_energy = _lj_energy(final_pos)
        
        # Optional: A final polish or check if we hit the deadline.
        # Given the start is the global min structure, one run should suffice.
        
        return final_pos
        
    except Exception as e:
        # Fallback to initial position if optimization fails
        return initial_pos
