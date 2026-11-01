import numpy as np
from scipy.optimize import minimize
import time

N = 55
_DEADLINE = 140.0  # Safety margin before 150s wall-clock kill

def construct():
    """
    Constructs the 55-atom Mackay icosahedron by optimizing the radial 
    coordinates of the inner and outer shells, then performs a full 
    local relaxation to refine the geometry.
    """
    t_start = time.time()
    
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

    def _lj_energy_and_grad(points):
        """Compute LJ energy and gradient w.r.t. point positions."""
        n = len(points)
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        
        # Mask diagonal
        mask = ~np.eye(n, dtype=bool)
        # Avoid division by zero
        dist_sq_safe = np.where(mask, dist_sq, 1.0)
        dist_sq_safe = np.maximum(dist_sq_safe, 1e-10)
        
        r_inv6 = dist_sq_safe ** -3.0
        r_inv12 = r_inv6 ** 2
        
        # Energy: 4 * sum over unique pairs of (r^-12 - r^-6)
        # Using full matrix and dividing by 2
        energy = 4.0 * np.sum(np.where(mask, r_inv12 - r_inv6, 0.0)) / 2.0
        
        # Gradient: dE/dx_i = sum_j 4*(-12*r^-14 + 6*r^-8) * (x_i - x_j) / r
        # In terms of dist_sq: dE/dx_i = sum_j 4*(-12*r^-14 + 6*r^-8) * (x_i - x_j) / r
        # = sum_j 4*(-12 * r^-14 + 6 * r^-8) * diff[i,j] / r
        # r = sqrt(dist_sq), so 1/r = 1/sqrt(dist_sq)
        # -12*r^-14 + 6*r^-8 = -12*(dist_sq)^-7 + 6*(dist_sq)^-4
        # Multiply by 1/sqrt(dist_sq) = (dist_sq)^-0.5
        # Total coefficient: 4 * (-12*(dist_sq)^-7.5 + 6*(dist_sq)^-4.5)
        coeff = 4.0 * (-12.0 * dist_sq_safe ** -7.5 + 6.0 * dist_sq_safe ** -4.5)
        coeff = np.where(mask, coeff, 0.0)
        
        # grad[i] = sum_j coeff[i,j] * diff[i,j]
        grad = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
        
        return energy, grad

    def energy(params):
        points = positions_from_params(params)
        e, _ = _lj_energy_and_grad(points)
        return e

    def energy_grad(params):
        points = positions_from_params(params)
        e, grad = _lj_energy_and_grad(points)
        
        # Map gradient back to params [r1, r2, r3]
        # grad[0] is center (unused for params)
        # grad[1:13] are inner shell -> dE/dr1
        # grad[13:25] are outer vertices -> dE/dr2
        # grad[25:55] are outer edges -> dE/dr3
        
        # dr1: dE/dr1 = sum_{inner} dot(grad_inner, verts)
        g_r1 = np.sum(grad[1:13] * verts)
        # dr2: dE/dr2 = sum_{outer_v} dot(grad_outer_v, verts)
        g_r2 = np.sum(grad[13:25] * verts)
        # dr3: dE/dr3 = sum_{outer_e} dot(grad_outer_e, edge_dirs)
        g_r3 = np.sum(grad[25:55] * edge_dirs)
        
        return e, np.array([g_r1, g_r2, g_r3])

    # Initial guess for radii
    # r1 ~ 1.12 (min of LJ)
    # r2, r3 ~ 2.0-2.5
    x0 = np.array([1.12, 2.2, 2.1])
    
    # Optimize with analytical gradient
    best_params = x0
    try:
        result = minimize(energy, x0, jac=energy_grad, method='L-BFGS-B', 
                         options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10})
        if result.success and result.fun < energy(x0):
            best_params = result.x
    except Exception:
        pass
    
    # Check time
    if time.time() - t_start > _DEADLINE - 30:
        final_points = positions_from_params(best_params)
        # Quick minimum distance check
        diff = final_points[:, np.newaxis, :] - final_points[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dist_sq, np.inf)
        min_dist = np.sqrt(np.min(dist_sq))
        if min_dist < 0.5:
            scale = 0.5 / min_dist
            final_points = final_points * scale * 1.1
        return final_points

    final_points = positions_from_params(best_params)
    
    # Ensure atoms are at least 0.5 apart
    # Check minimum distance
    diff = final_points[:, np.newaxis, :] - final_points[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    min_dist = np.sqrt(np.min(dist_sq))
    
    if min_dist < 0.5:
        # This should not happen for the icosahedron, but just in case
        # Fall back to a simple expansion
        scale = 0.5 / min_dist
        final_points = final_points * scale * 1.1 # Add margin

    # Full local relaxation of the 55 atoms with analytical gradient
    def full_energy(pos):
        pos = pos.reshape(N, 3)
        e, _ = _lj_energy_and_grad(pos)
        return e

    def full_energy_grad(pos):
        pos = pos.reshape(N, 3)
        e, grad = _lj_energy_and_grad(pos)
        return e, grad.flatten()

    # Use L-BFGS-B for the final polish with tight tolerances
    try:
        result_relax = minimize(full_energy, final_points.flatten(), jac=full_energy_grad, 
                                method='L-BFGS-B', 
                                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12})
        if result_relax.success and result_relax.fun < full_energy(final_points.flatten()):
            final_points = result_relax.x.reshape(N, 3)
    except Exception:
        pass # Keep the previous points if relaxation fails
    
    # High-precision polish pass if time permits
    if time.time() - t_start < _DEADLINE - 10:
        try:
            result_polish = minimize(full_energy, final_points.flatten(), jac=full_energy_grad,
                                     method='L-BFGS-B',
                                     options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-14})
            if result_polish.success and result_polish.fun < full_energy(final_points.flatten()):
                final_points = result_polish.x.reshape(N, 3)
        except Exception:
            pass
    
    # Final minimum distance check
    diff = final_points[:, np.newaxis, :] - final_points[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    min_dist = np.sqrt(np.min(dist_sq))
    if min_dist < 0.5:
        scale = 0.5 / min_dist
        final_points = final_points * scale * 1.05

    return final_points
