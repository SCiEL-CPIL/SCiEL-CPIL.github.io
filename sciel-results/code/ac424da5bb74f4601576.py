import numpy as np
import time
from scipy.optimize import minimize

N = 38
START_TIME = time.time()
TIME_LIMIT = 110.0

def lj_energy_and_grad(pos):
    # pos: (N, 3)
    N = pos.shape[0]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=-1)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    
    # r^-6 and r^-12
    inv_dist_6 = dist**-6
    inv_dist_12 = inv_dist_6**2
    
    # Energy term per pair (i,j) where i!=j
    # E_pair = 4 * (r^-12 - r^-6)
    # Total E = 0.5 * sum(E_pair)
    
    # Gradient
    # grad_i = sum_{j!=i} (24*r^-7 - 48*r^-13) * (r_i - r_j) / r
    # = sum_{j!=i} (24*r^-6 - 48*r^-12) * (r_i - r_j) / r^2 * r ?
    # Let's use: grad_i = sum_{j!=i} (24*inv_dist_6 - 48*inv_dist_12) * (diff[i,j] / dist_sq[i,j])
    # Because (r_i - r_j)/r * (1/r^6) = (r_i - r_j) / r^7 = (r_i - r_j) * r^-7.
    # My formula: (24r^-7 - 48r^-13) * (r_i - r_j)/r = 24r^-6(r_i-r_j) - 48r^-12(r_i-r_j)
    # = (24r^-6 - 48r^-12) * (r_i - r_j)
    # Wait, the unit vector is (r_i-r_j)/r.
    # So it is (24r^-7 - 48r^-13) * (r_i-r_j)/r = (24r^-6 - 48r^-12) * (r_i-r_j) / r^2 * r?
    # No.
    # (A/r) * (v/r) = A*v/r^2.
    # Here A = 24r^-7 - 48r^-13.
    # So term = (24r^-7 - 48r^-13) * (r_i-r_j) / r = (24r^-6 - 48r^-12) * (r_i-r_j) / r^2 * r?
    # Let's stick to:
    # coeff = 24 * inv_dist_6 - 48 * inv_dist_12
    # grad_contrib = coeff * diff / dist_sq  # This is coeff * (r_i-r_j) / r^2
    # But we need coeff * (r_i-r_j) / r.
    # So grad_contrib = (coeff / dist) * diff?
    # coeff = 24r^-7 - 48r^-13.
    # coeff / r = 24r^-8 - 48r^-14.
    # This is getting messy.
    
    # Let's use the standard form:
    # F_ij = 24 * (2*r^-13 - r^-7) * (r_i - r_j) / r^2 ? No.
    # Force on i due to j:
    # F = -dE/dr * dr/dri = -(-48r^-13 + 24r^-7) * (ri-rj)/r
    # = (48r^-13 - 24r^-7) * (ri-rj)/r
    # = (48r^-12 - 24r^-6) * (ri-rj) / r^2
    # = 24 * (2*r^-12 - r^-6) * (ri-rj) / r^2
    
    # So Gradient = sum_j 24 * (2*inv_dist_12 - inv_dist_6) * diff / dist_sq
    
    energy = 2.0 * np.sum(4.0 * (inv_dist_12 - inv_dist_6)) / 2.0 # 4*sum / 2
    # Actually E = 0.5 * sum(4*(r^-12 - r^-6)) = 2 * sum(r^-12 - r^-6)
    energy = 2.0 * np.sum(inv_dist_12 - inv_dist_6)
    # Zero out diagonal contribution to energy (it's 0 anyway if we filled diag with 1)
    # But inv_dist_6 on diag is 1. inv_dist_12 is 1. 1-1=0. So it's fine.
    
    grad_factor = 24.0 * (2.0 * inv_dist_12 - inv_dist_6)
    # grad_factor shape (N, N)
    # grad[i] = sum_j grad_factor[i,j] * diff[i,j] / dist_sq[i,j]
    # diff[i,j] = r_i - r_j
    # dist_sq[i,j] = r_ij^2
    
    # We can compute this as:
    # weighted_diff = grad_factor[:, :, np.newaxis] * diff / dist_sq[:, :, np.newaxis]
    # grad = np.sum(weighted_diff, axis=1)
    
    # Handle division by zero on diagonal (dist_sq=1, diff=0, so 0/1=0. Fine.)
    weighted_diff = grad_factor[:, :, np.newaxis] * diff / dist_sq[:, :, np.newaxis]
    grad = np.sum(weighted_diff, axis=1)
    
    return energy, grad

def generate_fcc(n_atoms):
    a = np.sqrt(2.0)
    pts = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            for k in range(-3, 4):
                pts.append((i, j, k))
                pts.append((i + 0.5, j + 0.5, k))
                pts.append((i + 0.5, j, k + 0.5))
                pts.append((i, j + 0.5, k + 0.5))
    pts = np.array(pts) * a
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)
    return pts[idx[:n_atoms]]

def construct():
    best_energy = np.inf
    best_pos = None
    
    starts = []
    fcc_pos = generate_fcc(N)
    starts.append(fcc_pos)
    
    np.random.seed(42)
    for sigma in [0.01, 0.05, 0.1]:
        noisy = fcc_pos + np.random.normal(0, sigma, fcc_pos.shape)
        starts.append(noisy)
        
    for start_pos in starts:
        if time.time() - START_TIME > TIME_LIMIT:
            break
            
        res = minimize(lj_energy_and_grad, start_pos, method='L-BFGS-B', jac=True, options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
        
        if res.success:
            e = res.fun
            if e < best_energy:
                best_energy = e
                best_pos = res.x
                
    if best_pos is None:
        side = int(np.ceil(N ** (1 / 3)))
        points = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
        points.sort(key=lambda p: (p[0] - (side - 1) / 2) ** 2 + (p[1] - (side - 1) / 2) ** 2 + (p[2] - (side - 1) / 2) ** 2)
        best_pos = np.array(points[:N], dtype=float) * 1.12

    return best_pos
