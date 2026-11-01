import numpy as np
from scipy.optimize import minimize
import time

N = 55
START_TIME = time.time()
TIME_LIMIT = 110.0  # Safety margin

def construct():
    """
    Constructs the 55-atom Mackay icosahedron by optimizing the radial 
    coordinates of the inner and outer shells, then performs a full local relaxation.
    """
    # Golden ratio
    phi = (1 + np.sqrt(5)) / 2.0
    
    # 12 vertices of a regular icosahedron
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    
    # Normalize vertices to unit length
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    
    # Find edge directions (midpoints of adjacent vertices)
    target_dot = phi / (2 + phi)
    edge_dirs = []
    for i in range(12):
        for j in range(i + 1, 12):
            if np.abs(np.dot(verts[i], verts[j]) - target_dot) < 1e-6:
                edge_dir = verts[i] + verts[j]
                edge_dir = edge_dir / np.linalg.norm(edge_dir)
                edge_dirs.append(edge_dir)
    
    edge_dirs = np.array(edge_dirs)
    assert len(edge_dirs) == 30, f"Expected 30 edges, got {len(edge_dirs)}"

    def positions_from_params(params):
        r1, r2, r3 = params
        p0 = np.array([0.0, 0.0, 0.0])
        p_inner = verts * r1
        p_outer_v = verts * r2
        p_outer_e = edge_dirs * r3
        points = np.vstack([p0, p_inner, p_outer_v, p_outer_e])
        return points

    def energy_radial(params):
        points = positions_from_params(params)
        return lj_energy(points)

    def lj_energy(points):
        n = len(points)
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        mask = ~np.eye(n, dtype=bool)
        dist_sq_masked = dist_sq[mask]
        dist_sq_masked = np.maximum(dist_sq_masked, 1e-10)
        r_inv6 = dist_sq_masked ** -3.0
        r_inv12 = r_inv6 ** 2
        return 4.0 * np.sum(r_inv12 - r_inv6) / 2.0

    def lj_grad(points):
        n = len(points)
        grad = np.zeros_like(points)
        # Vectorized gradient calculation
        # diff[i,j] = points[i] - points[j]
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        # Avoid division by zero
        dist_sq = np.maximum(dist_sq, 1e-10)
        
        # r = sqrt(dist_sq)
        # term = 24 * (r^-9 - 2*r^-15) * (points[i] - points[j])
        # r^-9 = dist_sq^-4.5
        # r^-15 = dist_sq^-7.5
        r_inv9 = dist_sq ** -4.5
        r_inv15 = dist_sq ** -7.5
        coeff = 24.0 * (r_inv9 - 2.0 * r_inv15)
        
        # coeff is (N, N). We need to sum over j for each i.
        # grad[i] = sum_j coeff[i,j] * diff[i,j]
        # Using einsum or matmul
        # diff is (N, N, 3). coeff is (N, N).
        # We want (N, 3)
        # grad[i,k] = sum_j coeff[i,j] * diff[i,j,k]
        grad = np.einsum('ij,ijk->ik', coeff, diff)
        
        # The diagonal terms (i=j) are 0 because diff[i,i]=0, so coeff[i,i] * 0 = 0.
        # However, the formula derived was for i != j.
        # The sum in the gradient is over j != i.
        # The einsum includes j=i, but diff[i,i]=0, so it's fine.
        
        return grad

    # 1. Optimize radial parameters
    x0 = np.array([1.12, 2.2, 2.1])
    
    # Time check
    elapsed = time.time() - START_TIME
    if elapsed > TIME_LIMIT * 0.5:
        # Not enough time for full relax, return radial optimized
        best_params = x0
        final_points = positions_from_params(best_params)
        return final_points

    try:
        result_radial = minimize(energy_radial, x0, method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8})
        best_params = result_radial.x
    except Exception:
        best_params = x0

    initial_points = positions_from_params(best_params)
    
    # 2. Full relaxation
    elapsed = time.time() - START_TIME
    if elapsed > TIME_LIMIT * 0.8:
        # Not enough time for full relax
        return initial_points

    try:
        result_full = minimize(
            lambda x: lj_energy(x.reshape(N, 3)),
            initial_points.flatten(),
            jac=lambda x: lj_grad(x.reshape(N, 3)).flatten(),
            method='L-BFGS-B',
            options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        final_points = result_full.x.reshape(N, 3)
    except Exception:
        final_points = initial_points

    # Check minimum distance
    diff = final_points[:, np.newaxis, :] - final_points[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    min_dist = np.sqrt(np.min(dist_sq))
    
    if min_dist < 0.5:
        scale = 0.5 / min_dist
        final_points = final_points * scale * 1.1

    return final_points
