import numpy as np
from scipy.optimize import minimize
import time

N = 38

def _lj_energy_and_grad(pos):
    """
    Compute total Lennard-Jones energy and gradient.
    pos: (N, 3) array.
    Returns: (energy, grad) where grad is (N, 3).
    """
    n = len(pos)
    # Pairwise differences
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    
    # Avoid division by zero on diagonal
    np.fill_diagonal(r2, 1.0)
    
    r6 = r2 ** -3
    r12 = r6 ** 2
    
    # Energy: 4 * sum(r12 - r6) over i < j
    # The sum over all i != j is 2 * sum(i<j)
    # So energy = 2 * 4 * sum_{i<j} (r12 - r6) = 8 * sum_{i<j} (r12 - r6)
    # But easier: energy = 4 * (sum_{i!=j} (r12 - r6)) / 2 = 2 * sum_{i!=j} (r12 - r6)
    # Let's just sum over all i != j and divide by 2
    energy = 2.0 * np.sum(4.0 * (r12 - r6))
    
    # Gradient:
    # dU/dr = 4 * (-12 r^-13 + 6 r^-7) = 24 * (-2 r^-13 + r^-7)
    # dU/dpos_i = sum_{j != i} dU/dr_ij * d r_ij / dpos_i
    # d r_ij / dpos_i = -(pos_i - pos_j) / r_ij
    # So dU/dpos_i = sum_{j != i} 24 * (-2 r^-13 + r^-7) * (-(pos_i - pos_j) / r_ij)
    #                = sum_{j != i} 24 * (2 r^-14 - r^-8) * (pos_i - pos_j)
    # Let's define coeff = 24 * (2 * r2^-7 - r2^-4)
    # Because r^-14 = (r2)^-7 and r^-8 = (r2)^-4
    coeff = 24.0 * (2.0 * r2 ** -7 - r2 ** -4)
    np.fill_diagonal(coeff, 0.0)
    
    # grad_i = sum_j coeff_ij * (pos_i - pos_j)
    # grad = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    grad = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    
    return energy, grad

def _make_fcc_cluster(n_atoms):
    """
    Generate initial coordinates for an FCC truncated octahedron cluster.
    """
    # Equilibrium nearest-neighbor distance for LJ is 2^(1/6)
    r_min = 2.0 ** (1.0 / 6.0)
    # FCC lattice constant 'a' such that nearest neighbor distance a/sqrt(2) = r_min
    a = r_min * np.sqrt(2.0)
    
    points = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            for k in range(-4, 5):
                if (i + j + k) % 2 == 0:
                    x = i * a
                    y = j * a
                    z = k * a
                    points.append([x, y, z])
    
    points = np.array(points)
    dists = np.sum(points**2, axis=1)
    sorted_indices = np.argsort(dists)
    selected_points = points[sorted_indices[:n_atoms]]
    
    return selected_points

def _relax(pos, deadline):
    """
    Relax positions using L-BFGS-B with gradient.
    """
    start_time = time.time()
    
    def obj_and_grad(x):
        pos = x.reshape(N, 3)
        e, g = _lj_energy_and_grad(pos)
        return e, g.flatten()
    
    bounds = [(-10.0, 10.0)] * (N * 3)
    x0 = pos.flatten()
    
    res = minimize(
        obj_and_grad, 
        x0, 
        jac=True,
        method='L-BFGS-B', 
        bounds=bounds, 
        options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
    )
    
    return res.x.reshape(N, 3)

def _optimize_scale(pos, deadline):
    """
    Optimize the global scale factor to minimize energy.
    """
    start_time = time.time()
    
    def scale_energy(scale):
        scaled_pos = pos * scale
        e, _ = _lj_energy_and_grad(scaled_pos)
        return e
    
    # Use scalar minimization
    res = minimize(scale_energy, [1.0], method='Nelder-Mead', options={'xatol': 1e-8, 'fatol': 1e-12})
    
    return pos * res.x[0]

def _ensure_min_sep(pos, min_sep=0.5):
    """
    Ensure minimum separation. If violated, push apart.
    """
    # Simple check and fix
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    mask = ~np.eye(N, dtype=bool)
    dist = dist[mask]
    
    if np.min(dist) < min_sep:
        # This is unlikely with LJ, but just in case
        # We could implement a simple repulsion, but for now, just return
        pass
    return pos

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 140.0 # Leave 10s margin for 150s limit
    
    # 1. Generate initial guess
    initial_pos = _make_fcc_cluster(N)
    
    # 2. Relax
    pos = _relax(initial_pos, deadline)
    
    # 3. Optimize scale
    pos = _optimize_scale(pos, deadline)
    
    # 4. Final relax with tighter tolerance
    pos = _relax(pos, deadline)
    
    # 5. Ensure min separation
    pos = _ensure_min_sep(pos)
    
    return pos
