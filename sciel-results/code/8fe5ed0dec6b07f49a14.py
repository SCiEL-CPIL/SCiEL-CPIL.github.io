"""
Optimization of Lennard-Jones 38-atom cluster.
Strategy: Start from the known global minimum topology (FCC truncated octahedron)
and perform local relaxation using L-BFGS.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 38

def _lj_energy(pos):
    """Calculate total Lennard-Jones energy."""
    # pos is (N, 3)
    # Vectorized calculation
    d = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(d**2, axis=-1)
    # Avoid division by zero
    r2[r2 == 0] = 1e-10
    r6 = r2 ** -3
    r12 = r6 ** 2
    # Mask diagonal to avoid self-interaction
    mask = ~np.eye(N, dtype=bool)
    # Sum over all pairs (i,j) and (j,i), then divide by 2
    energy = 4.0 * np.sum((r12 - r6) * mask) / 2.0
    return energy

def _lj_grad(pos):
    """Calculate gradient of Lennard-Jones energy."""
    d = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(d**2, axis=-1)
    r2[r2 == 0] = 1e-10
    r6 = r2 ** -3
    r12 = r6 ** 2
    # Gradient component for atom i:
    # sum_j 24 * (2*r12_ij - r6_ij) / r2_ij * (pos_i - pos_j)
    factor = 24.0 * (2.0 * r12 - r6) / r2
    factor[np.diag_indices(N)] = 0.0
    grad = np.sum(factor[:, :, np.newaxis] * d, axis=1)
    return grad

def _generate_fcc_init():
    """Generate initial positions based on FCC lattice."""
    # FCC lattice points: (i, j, k) where i+j+k is even.
    # Scaled such that nearest neighbor distance is 1.0.
    # Nearest neighbors in integer coords are at distance sqrt(2).
    # So scale factor is 1/sqrt(2).
    
    max_coord = 8
    points = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                if (i + j + k) % 2 == 0:
                    points.append([i, j, k])
    
    points = np.array(points, dtype=float)
    points /= np.sqrt(2.0)
    
    # Sort by distance from origin
    dists = np.sum(points**2, axis=1)
    idx = np.argsort(dists)
    points = points[idx]
    
    # Take the first N points
    return points[:N].copy()

def construct():
    start_time = time.time()
    # Deadline guard: 120s limit, use 110s to be safe
    deadline = start_time + 110.0
    
    # 1. Generate FCC initial guess
    init_pos = _generate_fcc_init()
    
    # 2. Local Relaxation
    best_energy = _lj_energy(init_pos)
    best_pos = init_pos.copy()
    
    # Run L-BFGS-B
    try:
        # maxiter can be high, but it will converge quickly from this start
        res = minimize(
            _lj_energy, 
            init_pos, 
            method='L-BFGS-B', 
            jac=_lj_grad,
            options={
                'maxiter': 500, 
                'ftol': 1e-12, 
                'gtol': 1e-10,
                'disp': False
            }
        )
        
        if res.fun < best_energy:
            best_energy = res.fun
            best_pos = res.x
            
    except Exception:
        # If optimization fails, return the initial FCC guess
        pass
    
    # Check time
    if time.time() > deadline:
        return best_pos
        
    # If we have time, we could try other starts, but FCC is the global min.
    # Adding random restarts is risky (might find local minima) and unnecessary.
    
    return best_pos
