"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def hex_lattice_init():
    # Generate a hex lattice
    s = 0.11
    while True:
        k = int(1.0 / s) + 2
        rows = [((i + 0.5 * (j % 2)) * s, j * s * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        mask = radius <= 1.0 - s / 2.0
        points = points[mask]
        
        if len(points) >= N:
            points = points[np.argsort(radius, kind="stable")[:N]]
            break
        s *= 0.98
        
    np.random.seed(42)
    points += np.random.normal(0, 1e-6, points.shape)
    return points

def calc_score(points):
    dists = np.linalg.norm(points, axis=1)
    boundary_min = np.min(1.0 - dists)
    
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_pair = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists_pair, np.inf)
    pair_min = np.min(dists_pair) / 2.0
    
    return min(boundary_min, pair_min)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    points = hex_lattice_init()
    current_r = calc_score(points)
    
    def objective_and_grad(x, r_target):
        pts = x.reshape(-1, 2)
        dists = np.linalg.norm(pts, axis=1)
        v = np.maximum(0, dists + r_target - 1.0)
        
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_pair = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists_pair, np.inf)
        w = np.maximum(0, r_target - dists_pair / 2.0)
        
        energy = np.sum(v**2) + np.sum(w**2)
        
        grad = np.zeros_like(x)
        
        mask_v = v > 0
        if np.any(mask_v):
            idx = np.where(mask_v)[0]
            norms = dists[idx]
            norms = np.where(norms == 0, 1.0, norms)
            grad_vecs = 2.0 * v[idx] * pts[idx] / norms[:, np.newaxis]
            grad[idx] += grad_vecs
            
        coeff = np.zeros_like(dists_pair)
        mask_w = w > 0
        safe_dists = np.where(dists_pair == 0, 1.0, dists_pair)
        coeff[mask_w] = w[mask_w] / safe_dists[mask_w]
        
        row_sums = np.sum(coeff, axis=1)
        
        grad_pts = -row_sums[:, np.newaxis] * pts + (coeff.T @ pts)
        grad += grad_pts.flatten()
        
        return energy, grad

    delta_r = 1e-4
    
    while time.time() < deadline:
        new_r = current_r + delta_r
        res = minimize(
            objective_and_grad, 
            points.flatten(), 
            args=(new_r,),
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        new_points = res.x.reshape(-1, 2)
        new_energy = res.fun
        
        if new_energy < 1e-8:
            points = new_points
            current_r = new_r
            delta_r *= 1.1
        else:
            delta_r *= 0.8

    return points
# EVOLVE-BLOCK-END
