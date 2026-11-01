import numpy as np
import time
import scipy.optimize as opt
import itertools

N = 55

def _lj_energy_and_grad(pos):
    """
    Compute the Lennard-Jones energy and gradient for a set of positions.
    pos: (N, 3) array
    """
    N = pos.shape[0]
    # Compute all pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    # Avoid division by zero
    dist_sq[dist_sq == 0] = 1e-12
    
    # Compute r^-6 and r^-12
    r_inv6 = dist_sq ** -3
    r_inv12 = r_inv6 ** 2
    
    # Energy: 4 * sum(r^-12 - r^-6)
    # Sum over all pairs i < j
    energy = 4 * (np.sum(r_inv12) - np.sum(r_inv6)) / 2
    
    # Gradient: dE/dr_i = 4 * sum_j ( -12 r_ij^-14 + 6 r_ij^-8 ) * (r_i - r_j)
    # dE/dx_i = 4 * sum_j ( -12 r_ij^-12 * r_ij^-2 + 6 r_ij^-6 * r_ij^-2 ) * (x_i - x_j)
    #         = 4 * sum_j ( -12 r_inv12 * dist_sq^-1 + 6 r_inv6 * dist_sq^-1 ) * (x_i - x_j)
    #         = 4 * sum_j ( -12 r_inv12 / dist_sq + 6 r_inv6 / dist_sq ) * (x_i - x_j)
    
    # Factor = 4 * ( -12 * r_inv12 + 6 * r_inv6 ) / dist_sq
    factor = 4 * (-12 * r_inv12 + 6 * r_inv6) / dist_sq
    # Set diagonal to 0
    np.fill_diagonal(factor, 0)
    
    # Gradient for each atom i: sum_j factor[i,j] * (pos[i] - pos[j])
    # grad[i] = sum_j factor[i,j] * diff[i,j]
    grad = np.sum(factor[:, :, np.newaxis] * diff, axis=1)
    
    return energy, grad

def _relax(pos):
    """
    Relax the positions using L-BFGS-B.
    """
    result = opt.minimize(
        _lj_energy_and_grad,
        pos,
        jac=True,
        method='L-BFGS-B',
        options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
    )
    return result.x, result.fun

def _generate_icosahedron_guess():
    """
    Generate a guess for the 55-atom Mackay icosahedron.
    1 atom at center.
    12 atoms at vertices of an icosahedron.
    42 atoms on a sphere.
    """
    phi = (1 + np.sqrt(5)) / 2
    
    # 12 vertices of a regular icosahedron
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    
    # Normalize to unit length
    norms = np.linalg.norm(verts, axis=1)
    verts = verts / norms[:, np.newaxis]
    
    # Scale to radius R1
    R1 = 1.0
    inner_shell = verts * R1
    
    # 42 atoms on a sphere of radius R2
    # Use Fibonacci sphere for even distribution
    R2 = 1.5
    n_outer = 42
    golden_angle = np.pi * (3 - np.sqrt(5))
    outer_shell = np.zeros((n_outer, 3))
    for i in range(n_outer):
        z = 1 - (i / (n_outer - 1)) * 2
        radius = np.sqrt(1 - z**2)
        theta = golden_angle * i
        x = np.cos(theta) * radius
        y = np.sin(theta) * radius
        outer_shell[i] = [x, y, z] * R2
    
    # Combine: 1 center, 12 inner, 42 outer
    center = np.array([[0, 0, 0]])
    pos = np.vstack([center, inner_shell, outer_shell])
    
    return pos

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    # Initial guess
    pos = _generate_icosahedron_guess()
    
    # Relax
    pos, energy = _relax(pos)
    
    # Check time
    if time.time() > deadline:
        return pos
    
    # Try a few random restarts to ensure we find the global minimum
    best_pos = pos
    best_energy = energy
    
    for i in range(10):
        if time.time() > deadline:
            break
            
        # Perturb the best position
        perturbed = best_pos + np.random.normal(0, 0.1, size=best_pos.shape)
        
        # Relax
        pos, energy = _relax(perturbed)
        
        if energy < best_energy:
            best_pos = pos
            best_energy = energy
            
    return best_pos
