"""LJ38 cluster global minimum search using Basin Hopping."""
import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial.distance import cdist

N = 38

def lj_energy_and_grad(pos):
    """Compute Lennard-Jones energy and gradient for N atoms."""
    # pos: (N, 3) array
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]  # (N, N, 3)
    r2 = np.sum(diff ** 2, axis=2)  # (N, N)
    # Avoid division by zero
    np.fill_diagonal(r2, 1.0)
    r6 = r2 ** 3
    r12 = r6 ** 2
    
    # Energy: sum over i<j of 4*(r^-12 - r^-6)
    # Using the full matrix and dividing by 2
    energy = 4.0 * (np.sum(1.0 / r12) - np.sum(1.0 / r6)) - N * 4.0  # subtract diagonal contribution
    # Actually, let's be more careful. The diagonal has r2=1, so r6=1, r12=1, term = 4*(1-1)=0.
    # So the sum over all pairs (including i=j) gives 2 * sum_{i<j} 4*(r^-12 - r^-6).
    # So energy = 0.5 * 4.0 * (sum_{i!=j} (r^-12 - r^-6))
    # Let's recompute:
    # mask for i != j
    mask = ~np.eye(N, dtype=bool)
    energy = 4.0 * 0.5 * (np.sum(1.0 / r12[mask]) - np.sum(1.0 / r6[mask]))
    
    # Gradient: dE/dr_i = sum_{j != i} 4 * (-12 * r^-14 * (r_i - r_j)_k / r + 6 * r^-8 * (r_i - r_j)_k / r)
    # = sum_{j != i} 4 * (r_i - r_j) * (-12 * r^-14 + 6 * r^-8) / r
    # Note: r = sqrt(r2)
    r = np.sqrt(r2)
    # grad_ij = 4 * (-12 * r^-14 + 6 * r^-8) * diff_ij / r
    # = 4 * diff_ij * (-12 / r^14 + 6 / r^8)
    coeff = 4.0 * (-12.0 / (r2 ** 7) + 6.0 / (r2 ** 4))  # (N, N)
    np.fill_diagonal(coeff, 0.0)
    grad = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    
    return energy, grad

def check_validity(pos):
    """Check if all atoms are at least 0.5 apart."""
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1.0)
    return np.all(r2 >= 0.25)  # 0.5^2 = 0.25

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin
    
    # Initial position: FCC truncated octahedron-like structure
    # For N=38, a common low-energy structure is based on FCC
    # Let's start with a reasonable guess and refine
    
    # Generate a grid and pick points
    # Better: start from the known approximate global minimum structure
    # FCC lattice with spacing ~1.122 (2^(1/6))
    a = 2.0 ** (1.0/6.0)  # ~1.1225
    
    # Create a larger FCC grid and pick 38 closest to center
    # FCC basis: (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)
    # In units of a
    
    # Generate points in a cube
    n = 4
    points = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                # FCC basis vectors
                points.append([i*a, j*a, k*a])
                points.append([(i+0.5)*a, (j+0.5)*a, k*a])
                points.append([(i+0.5)*a, j*a, (k+0.5)*a])
                points.append([i*a, (j+0.5)*a, (k+0.5)*a])
    
    points = np.array(points)
    
    # Center the points
    center = points.mean(axis=0)
    points = points - center
    
    # Sort by distance to center and pick 38
    dists = np.sum(points ** 2, axis=1)
    idx = np.argsort(dists)[:N]
    pos = points[idx].copy()
    
    # Relax the initial structure
    res = opt.minimize(lj_energy_and_grad, pos, jac=True, method='L-BFGS-B', 
                       options={'maxiter': 500, 'gtol': 1e-10, 'ftol': 1e-12})
    pos = res.x
    best_energy = res.fun
    best_pos = pos.copy()
    
    # Basin Hopping
    np.random.seed(42)
    step_size = 0.5
    max_iter = 200
    
    for i in range(max_iter):
        if time.time() > deadline - 5.0:
            break
        
        # Random perturbation
        noise = np.random.normal(0, step_size, size=pos.shape)
        pos_perturbed = pos + noise
        
        # Relax
        try:
            res = opt.minimize(lj_energy_and_grad, pos_perturbed, jac=True, method='L-BFGS-B',
                               options={'maxiter': 300, 'gtol': 1e-10, 'ftol': 1e-12})
            new_energy = res.fun
            new_pos = res.x
            
            # Check validity
            if check_validity(new_pos):
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_pos = new_pos.copy()
                    pos = new_pos.copy()
                    # Reduce step size on success
                    step_size = max(0.1, step_size * 0.9)
                else:
                    # Metropolis acceptance
                    if np.random.rand() < np.exp(-new_energy / 0.05):
                        pos = new_pos.copy()
            else:
                # Invalid, try smaller perturbation
                step_size = max(0.1, step_size * 0.9)
        except Exception:
            continue
    
    # Final high-precision polish
    if time.time() < deadline - 2.0:
        res = opt.minimize(lj_energy_and_grad, best_pos, jac=True, method='L-BFGS-B',
                           options={'maxiter': 1000, 'gtol': 1e-12, 'ftol': 1e-15})
        if check_validity(res.x) and res.fun < best_energy:
            best_pos = res.x
            best_energy = res.fun
    
    return best_pos
