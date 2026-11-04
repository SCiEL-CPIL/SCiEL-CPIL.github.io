"""
Robust packer using Energy Minimization with L-BFGS-B.
Strategy: Iteratively increase target radius and minimize violation energy.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 100


def _compute_radius(centers):
    """Compute the maximum common radius for a given set of centers."""
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0

    # Wall clearance
    max_abs = np.max(np.abs(centers), axis=1)
    wall_dists = 0.5 - max_abs
    min_wall = np.min(wall_dists)
    if min_wall < -1e-9:
        return 0.0

    # Pairwise distances
    diff = centers[:, None, :] - centers[None, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    iu = np.triu_indices(N, k=1)
    min_pair = np.sqrt(dists_sq[iu]).min()

    return min(min_pair * 0.5, min_wall)


def _energy_and_grad(x, r):
    """
    Compute energy (sum of squared violations) and gradient.
    x: (3N,) array of centers.
    r: target radius.
    """
    centers = x.reshape(-1, 3)
    n = len(centers)
    
    energy = 0.0
    grad = np.zeros_like(x)
    
    # 1. Pairwise constraints: ||c_i - c_j|| >= 2r
    iu = np.triu_indices(n, k=1)
    i_idx, j_idx = iu
    diff = centers[i_idx] - centers[j_idx]
    dists = np.linalg.norm(diff, axis=1)
    
    # Violation: max(0, 2r - dist)
    viol = np.maximum(0, 2*r - dists)
    energy += np.sum(viol**2)
    
    # Gradient of sum(viol^2)
    active = viol > 0
    if np.any(active):
        factor = np.zeros_like(dists)
        factor[active] = 2.0 * viol[active] / dists[active]
        
        active_indices = np.where(active)[0]
        for k in active_indices:
            i = i_idx[k]
            j = j_idx[k]
            f = factor[k]
            d_vec = diff[k]
            grad[i*3:(i+1)*3] -= f * d_vec
            grad[j*3:(j+1)*3] += f * d_vec

    # 2. Wall constraints: |c_ik| <= 0.5 - r
    limit = 0.5 - r
    
    # Plus walls: c_ik > limit
    viol_p = np.maximum(0, centers - limit)
    energy += np.sum(viol_p**2)
    grad += 2.0 * viol_p
    
    # Minus walls: c_ik < -limit
    viol_m = np.maximum(0, -centers - limit)
    energy += np.sum(viol_m**2)
    grad -= 2.0 * viol_m

    return energy, grad


def _optimize_for_radius(centers, r_target, max_iter=100):
    """Minimize energy for a fixed radius."""
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    def obj(x):
        e, g = _energy_and_grad(x, r_target)
        return e, g
        
    try:
        res = minimize(
            obj,
            centers.flatten(),
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'disp': False, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        return res.x.reshape(-1, 3), res.fun
    except Exception:
        return centers, np.inf


def construct():
    t0 = time.time()
    DEADLINE = t0 + 140.0
    
    best_radius = 0.0
    best_centers = None
    
    # Grid initialization
    nx, ny, nz = 5, 4, 5
    xs = np.linspace(-0.5 + 0.5/nx, 0.5 - 0.5/nx, nx)
    ys = np.linspace(-0.5 + 0.5/ny, 0.5 - 0.5/ny, ny)
    zs = np.linspace(-0.5 + 0.5/nz, 0.5 - 0.5/nz, nz)
    grid = np.array([[x, y, z] for z in zs for y in ys for x in xs])
    
    rng = np.random.default_rng(42)
    
    # Start with a small radius
    r_current = 0.05
    current_centers = grid.copy()
    
    # Main loop: Increase radius
    attempts = 0
    max_attempts = 50
    
    while time.time() < DEADLINE - 5.0 and attempts < max_attempts:
        # Try to solve for r_current
        new_centers, energy = _optimize_for_radius(current_centers, r_current, max_iter=200)
        
        # Check if valid
        r_check = _compute_radius(new_centers)
        
        if r_check >= r_current - 1e-5:
            # Success: We have a valid packing for r_current
            if r_check > best_radius:
                best_radius = r_check
                best_centers = new_centers.copy()
            
            # Increase radius for next step
            if energy < 1e-6:
                r_current *= 1.02
            else:
                r_current *= 1.005
            current_centers = new_centers.copy()
            attempts = 0
        else:
            # Failed: Reduce step or restart
            r_current *= 0.99
            attempts += 1
            if attempts > 5:
                # Restart from best known with noise
                if best_centers is not None:
                    current_centers = best_centers + rng.normal(0, 0.01, size=(N, 3))
                    current_centers = np.clip(current_centers, -0.49, 0.49)
                    r_current = max(0.05, best_radius * 0.98)
                else:
                    current_centers = grid + rng.normal(0, 0.01, size=(N, 3))
                    r_current = 0.05
                attempts = 0

    # Final Polish
    if best_centers is not None and time.time() < DEADLINE - 2.0:
        r_try = best_radius * 1.01
        polished, e = _optimize_for_radius(best_centers, r_try, max_iter=50)
        r_check = _compute_radius(polished)
        if r_check > best_radius:
            best_radius = r_check
            best_centers = polished

    if best_centers is None:
        best_centers = grid.copy()
        
    best_centers = np.clip(best_centers, -0.5 + 1e-9, 0.5 - 1e-9)
    return best_centers
