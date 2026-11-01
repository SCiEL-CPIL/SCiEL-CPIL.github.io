import numpy as np
from scipy.optimize import minimize
import time

N = 55
DEADLINE = 145.0  # Safety margin before 150s kill

def _lj_energy_grad(pos):
    """Compute LJ energy and gradient for N atoms. Returns (energy, grad)."""
    pos = pos.reshape(N, 3)
    # Pairwise differences
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]  # (N, N, 3)
    dist_sq = np.sum(diff**2, axis=2)  # (N, N)
    
    # Avoid division by zero on diagonal
    np.fill_diagonal(dist_sq, 1.0)
    
    r_inv2 = 1.0 / dist_sq
    r_inv6 = r_inv2 ** 3
    r_inv12 = r_inv6 ** 2
    
    # Energy: 4 * sum_{i<j} (r^-12 - r^-6)
    # Using full matrix and dividing by 2
    pair_energy = 4.0 * (r_inv12 - r_inv6)
    np.fill_diagonal(pair_energy, 0.0)
    energy = np.sum(pair_energy) / 2.0
    
    # Gradient: dE/dr_i = 4 * sum_{j != i} (-12*r_i^-13 + 6*r_i^-7) * (r_i - r_j) / r
    # = 4 * sum_{j != i} (-12*r^-12 + 6*r^-6) / r^2 * (r_i - r_j)
    # = 4 * sum_{j != i} (-12*r^-14 + 6*r^-8) * (r_i - r_j)
    # Let's compute: coeff_{ij} = 4 * (-12 * r_inv12 + 6 * r_inv6) * r_inv2
    # Then grad_i = sum_j coeff_{ij} * (pos_i - pos_j)
    
    coeff = 4.0 * (-12.0 * r_inv12 + 6.0 * r_inv6) * r_inv2
    np.fill_diagonal(coeff, 0.0)
    
    # grad_i = sum_j coeff_{ij} * (pos_i - pos_j)
    # = sum_j coeff_{ij} * pos_i - sum_j coeff_{ij} * pos_j
    # = pos_i * sum_j coeff_{ij} - sum_j coeff_{ij} * pos_j
    
    coeff_sum = np.sum(coeff, axis=1)  # (N,)
    grad = coeff_sum[:, np.newaxis] * pos - (coeff @ pos)
    
    return energy, grad.reshape(-1)

def construct():
    """
    Constructs the 55-atom Mackay icosahedron by optimizing the radial 
    coordinates of the inner and outer shells, then performs a full 
    local relaxation to refine the geometry.
    """
    # Golden ratio
    phi = (1 + np.sqrt(5)) / 2.0
    
    # 12 vertices of a regular icosahedron
    # (0, +/-1, +/-phi), (+/-1, +/-phi, 0), (+/-phi, 0, +/-1)
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    
    # Normalize vertices to unit length
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    
    # Find edge directions (midpoints of adjacent vertices)
    # Adjacent vertices have dot product cos(theta) where cos(theta) = phi / (2+phi)
    target_dot = phi / (2 + phi)
    edge_dirs = []
    for i in range(12):
        for j in range(i + 1, 12):
            if np.abs(np.dot(verts[i], verts[j]) - target_dot) < 1e-6:
                edge_dir = verts[i] + verts[j]
                edge_dir = edge_dir / np.linalg.norm(edge_dir)
                edge_dirs.append(edge_dir)
    
    edge_dirs = np.array(edge_dirs)
    # There should be 30 edges
    assert len(edge_dirs) == 30, f"Expected 30 edges, got {len(edge_dirs)}"

    def positions_from_params(params):
        r1, r2, r3 = params
        # 1 center
        p0 = np.array([0.0, 0.0, 0.0])
        # 12 inner shell (vertices)
        p_inner = verts * r1
        # 12 outer shell vertices
        p_outer_v = verts * r2
        # 30 outer shell edges
        p_outer_e = edge_dirs * r3
        
        points = np.vstack([p0, p_inner, p_outer_v, p_outer_e])
        return points

    def energy(params):
        points = positions_from_params(params)
        e, _ = _lj_energy_grad(points)
        return e

    # Initial guess for radii
    # r1 ~ 1.12 (min of LJ)
    # r2, r3 ~ 2.0-2.5
    x0 = np.array([1.12, 2.2, 2.1])
    
    # Check deadline before shell optimization
    t_start = time.time()
    
    # Optimize shell radii
    try:
        result = minimize(energy, x0, method='L-BFGS-B', 
                         options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
        best_params = result.x
    except Exception:
        best_params = x0

    final_points = positions_from_params(best_params)
    
    # Ensure atoms are at least 0.5 apart
    diff = final_points[:, np.newaxis, :] - final_points[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    min_dist = np.sqrt(np.min(dist_sq))
    
    if min_dist < 0.5:
        scale = 0.5 / min_dist
        final_points = final_points * scale * 1.1

    # Check deadline before full relaxation
    if time.time() - t_start > DEADLINE - 30:
        return final_points

    # Full local relaxation of the 55 atoms with analytical gradient
    def full_energy_and_grad(pos_flat):
        pos = pos_flat.reshape(N, 3)
        e, g = _lj_energy_grad(pos)
        return e, g

    # Use L-BFGS-B for the full relaxation
    try:
        result_relax = minimize(full_energy_and_grad, final_points.flatten(), method='L-BFGS-B', 
                                 options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50})
        if result_relax.success:
            final_points = result_relax.x.reshape(N, 3)
    except Exception:
        pass

    # Check deadline before high-precision polish
    if time.time() - t_start > DEADLINE - 15:
        return final_points

    # High-precision polish pass with even tighter tolerances
    try:
        result_polish = minimize(full_energy_and_grad, final_points.flatten(), method='L-BFGS-B', 
                                  options={'maxiter': 200, 'ftol': 1e-16, 'gtol': 1e-12, 'maxls': 30})
        if result_polish.success:
            final_points = result_polish.x.reshape(N, 3)
    except Exception:
        pass

    # Final check: ensure minimum distance constraint
    diff = final_points[:, np.newaxis, :] - final_points[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    min_dist = np.sqrt(np.min(dist_sq))
    
    if min_dist < 0.5:
        scale = 0.5 / min_dist
        final_points = final_points * scale * 1.05

    return final_points
