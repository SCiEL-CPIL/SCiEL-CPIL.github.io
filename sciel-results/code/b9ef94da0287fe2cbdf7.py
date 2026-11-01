"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 55

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import itertools

# Deadline guard
START_TIME = time.time()
TIME_LIMIT = 110.0  # Safety margin before 120s kill

def lj_energy(pos):
    """Compute total Lennard-Jones energy."""
    n = len(pos)
    # Compute all pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    # Avoid division by zero (diagonal)
    np.fill_diagonal(r2, 1.0)
    r = np.sqrt(r2)
    r6 = r ** 6
    r12 = r6 ** 2
    # Energy per pair: 4 * (r^-12 - r^-6)
    pair_energy = 4.0 * (1.0 / r12 - 1.0 / r6)
    # Sum over all pairs (upper triangle only, exclude diagonal)
    iu = np.triu_indices(n, k=1)
    return np.sum(pair_energy[iu])

def lj_gradient(pos):
    """Compute gradient of LJ energy."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1.0)
    r6 = r2 ** 3
    r12 = r6 ** 2
    # Force on atom i from atom j: 4 * (12/r^13 * r_vec - 6/r^7 * r_vec)
    # = 4 * r_vec * (12/r^13 - 6/r^7)
    # Gradient = -Force
    # Let's compute: for each pair (i,j), contribution to grad_i
    # E = sum_{i<j} 4*(r^-12 - r^-6)
    # dE/dx_i = sum_{j!=i} 4 * (-12*r^-13 + 6*r^-7) * (x_i - x_j)/r
    # = sum_{j!=i} 4 * (x_i - x_j) * (-12/r^14 + 6/r^8)
    # = sum_{j!=i} 4 * (x_i - x_j) / r * (-12/r^13 + 6/r^7)
    # = sum_{j!=i} 4 * (x_i - x_j) * (-12/r^14 + 6/r^8)
    
    # More carefully:
    # E_pair = 4*(r^-12 - r^-6)
    # dE_pair/dx_i = 4*(-12*r^-13*dr/dx_i + 6*r^-7*dr/dx_i)
    # dr/dx_i = (x_i - x_j)/r
    # So dE_pair/dx_i = 4*(-12*r^-13 + 6*r^-7) * (x_i - x_j)/r
    # = 4*(-12*r^-14 + 6*r^-8) * (x_i - x_j)
    
    coeff = 4.0 * (-12.0 / (r2 ** 7) + 6.0 / (r2 ** 4))
    # coeff shape: (n, n)
    # Set diagonal to 0
    np.fill_diagonal(coeff, 0.0)
    # grad_i = sum_j coeff[i,j] * (x_i - x_j)
    # = sum_j coeff[i,j] * x_i - sum_j coeff[i,j] * x_j
    # = x_i * sum_j coeff[i,j] - sum_j coeff[i,j] * x_j
    row_sum = np.sum(coeff, axis=1, keepdims=True)
    grad = row_sum * pos - coeff @ pos
    return grad

def mackay_icosahedron_55(r1=1.12, r2=2.05, r3=2.5):
    """Generate Mackay icosahedron with 55 atoms (1 + 12 + 30 + 12)."""
    # Structure:
    # 1 center
    # 12 inner shell (vertices of icosahedron)
    # 30 middle shell (edge midpoints)
    # 12 outer shell (vertices of icosahedron, same directions as inner)
    # This 1-12-30-12 structure is a known symmetric local minimum for N=55.
    
    phi = (1 + np.sqrt(5)) / 2
    r_circ = np.sqrt(1 + phi**2)
    
    vertices = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    vertices = vertices / r_circ
    
    # Find edges
    d01 = np.linalg.norm(vertices[0] - vertices[1])
    edge_len = d01
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(vertices[i] - vertices[j])
            if np.isclose(d, edge_len, atol=1e-9):
                edges.append((i, j))
    
    edge_midpoints = np.array([(vertices[i] + vertices[j]) / 2 for i, j in edges])
    edge_mid_unit = edge_midpoints / np.linalg.norm(edge_midpoints, axis=1, keepdims=True)
    
    positions = np.zeros((55, 3))
    positions[0] = [0, 0, 0]
    positions[1:13] = r1 * vertices
    positions[13:43] = r2 * edge_mid_unit
    positions[43:55] = r3 * vertices  # Outer shell uses same vertex directions
    
    return positions

def construct():
    global START_TIME
    START_TIME = time.time()
    
    # Initial guess for radii
    r1_init = 1.12
    r2_init = 2.05
    r3_init = 2.5
    
    pos = mackay_icosahedron_55(r1_init, r2_init, r3_init)
    
    # Optimize the radii r1, r2, r3 using the symmetric structure
    def symmetric_energy(params):
        r1_val, r2_val, r3_val = params
        p = mackay_icosahedron_55(r1_val, r2_val, r3_val)
        return lj_energy(p)
    
    # Optimize radii
    try:
        res = minimize(symmetric_energy, [r1_init, r2_init, r3_init], method='Nelder-Mead',
                      options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10})
        r1_opt, r2_opt, r3_opt = res.x
        pos = mackay_icosahedron_55(r1_opt, r2_opt, r3_opt)
    except Exception:
        pass
    
    best_pos = pos.copy()
    best_energy = lj_energy(pos)
    
    # Now do full Cartesian relaxation
    if time.time() - START_TIME < TIME_LIMIT:
        try:
            res = minimize(lj_energy, pos, jac=lj_gradient, method='L-BFGS-B',
                         options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12})
            if res.success or res.fun < best_energy:
                best_pos = res.x
                best_energy = res.fun
        except Exception:
            pass
    
    # Try a few perturbations and relaxations if time permits
    if time.time() - START_TIME < TIME_LIMIT - 30:
        rng = np.random.default_rng(42)
        for trial in range(5):
            if time.time() - START_TIME >= TIME_LIMIT - 10:
                break
            # Perturb the current best
            perturbed = best_pos + rng.normal(0, 0.05, size=(55, 3))
            try:
                res = minimize(lj_energy, perturbed, jac=lj_gradient, method='L-BFGS-B',
                             options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-11})
                if res.fun < best_energy:
                    best_energy = res.fun
                    best_pos = res.x
            except Exception:
                pass
    
    # Final check: ensure minimum distance >= 0.5
    diff = best_pos[:, np.newaxis, :] - best_pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, 1e10)
    min_dist = np.min(dists)
    
    if min_dist < 0.5:
        # Scale up slightly
        scale = 0.5 / min_dist * 1.01
        best_pos = best_pos * scale
        best_energy = lj_energy(best_pos)
    
    return best_pos
# EVOLVE-BLOCK-END
