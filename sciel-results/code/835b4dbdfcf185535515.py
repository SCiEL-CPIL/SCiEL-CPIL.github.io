"""Optimize Hermite-Gaussian coefficients to minimize the uncertainty bound C4."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from math import factorial, sqrt, pi
import time

def _hermite_val(n, x):
    """Compute physicists' Hermite polynomial H_n(x) at scalar x."""
    if n == 0:
        return 1.0
    elif n == 1:
        return 2.0 * x
    H_prev = 1.0
    H_curr = 2.0 * x
    for k in range(2, n + 1):
        H_next = 2.0 * x * H_curr - 2.0 * (k - 1) * H_prev
        H_prev = H_curr
        H_curr = H_next
    return H_curr

# Basis indices: 0, 4, 8, 12
indices = [0, 4, 8, 12]

# Orthogonality: int H_n H_m exp(-x^2) dx = sqrt(pi) 2^n n! delta_nm
M_diag = np.array([sqrt(pi) * (2**n) * factorial(n) for n in indices])

# N_ij = int x^2 H_i H_j exp(-x^2) dx
# For this basis, N is diagonal: N_ii = i * M_ii
N_diag = np.array([n * m for n, m in zip(indices, M_diag)])

# c12 is determined by c0, c4, c8 to satisfy g(0)=0
H0_0 = _hermite_val(0, 0)   # 1
H4_0 = _hermite_val(4, 0)   # -12
H8_0 = _hermite_val(8, 0)   # 1680
H12_0 = _hermite_val(12, 0) # -40320

def _compute_bound(c0, c4, c8):
    """Compute the uncertainty bound for given coefficients."""
    c12 = -(c0 * H0_0 + c4 * H4_0 + c8 * H8_0) / H12_0
    coeffs = np.array([c0, c4, c8, c12])
    
    norm_sq = np.sum(coeffs**2 * M_diag)
    if norm_sq <= 1e-12:
        return 1e10
        
    x2_expectation = np.sum(coeffs**2 * N_diag)
    return x2_expectation / norm_sq

def _objective_spherical(theta, phi):
    """
    Parameterize using spherical coordinates to define a point on the unit sphere in 3D.
    This handles scale invariance naturally.
    """
    c0 = np.sin(theta) * np.cos(phi)
    c4 = np.sin(theta) * np.sin(phi)
    c8 = np.cos(theta)
    return _compute_bound(c0, c4, c8)

def _objective_vec(params):
    """Vectorized objective for differential evolution."""
    theta, phi = params
    return _objective_spherical(theta, phi)

def construct():
    """Find optimal coefficients for H0, H4, H8."""
    start_time = time.time()
    deadline = start_time + 175.0  # safety margin for 210s limit
    
    # Start with the known good point
    best_x = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    best_val = _compute_bound(*best_x)
    
    # Convert to spherical coordinates for the initial best
    norm_best = np.linalg.norm(best_x)
    if norm_best > 1e-12:
        best_unit = best_x / norm_best
        theta_best = np.arccos(np.clip(best_unit[2], -1, 1))
        phi_best = np.arctan2(best_unit[1], best_unit[0])
        if phi_best < 0:
            phi_best += 2 * np.pi
    else:
        theta_best, phi_best = np.pi/2, 0
    
    # Strategy: Multi-stage optimization
    # Phase 1: Differential evolution for global exploration
    if time.time() < deadline - 60:
        try:
            from scipy.optimize import differential_evolution
            bounds = [(0, np.pi), (0, 2*np.pi)]
            de_result = differential_evolution(
                _objective_vec,
                bounds,
                maxiter=100,
                popsize=20,
                tol=1e-8,
                seed=42,
                init='sobol',
                polish=False
            )
            if de_result.fun < best_val:
                best_val = de_result.fun
                theta_opt, phi_opt = de_result.x
                c0 = np.sin(theta_opt) * np.cos(phi_opt)
                c4 = np.sin(theta_opt) * np.sin(phi_opt)
                c8 = np.cos(theta_opt)
                best_x = np.array([c0, c4, c8])
        except Exception:
            pass
    
    # Phase 2: Grid search with multiple resolutions
    if time.time() < deadline - 30:
        # Coarse grid
        n_theta = 60
        n_phi = 120
        theta_grid = np.linspace(0, np.pi, n_theta)
        phi_grid = np.linspace(0, 2*np.pi, n_phi)
        
        grid_best_val = best_val
        grid_best_theta = theta_best
        grid_best_phi = phi_best
        
        for t in theta_grid:
            if time.time() > deadline - 25:
                break
            for p in phi_grid:
                val = _objective_spherical(t, p)
                if val < grid_best_val:
                    grid_best_val = val
                    grid_best_theta = t
                    grid_best_phi = p
        
        if grid_best_val < best_val:
            best_val = grid_best_val
            c0 = np.sin(grid_best_theta) * np.cos(grid_best_phi)
            c4 = np.sin(grid_best_theta) * np.sin(grid_best_phi)
            c8 = np.cos(grid_best_theta)
            best_x = np.array([c0, c4, c8])
    
    # Phase 3: Local refinement from multiple starting points
    start_points = []
    
    # Add current best
    start_points.append([theta_best, phi_best])
    
    # Add DE result if available
    if time.time() < deadline - 20:
        try:
            de_result
            start_points.append(list(de_result.x))
        except NameError:
            pass
    
    # Add grid best
    start_points.append([grid_best_theta, grid_best_phi])
    
    # Add random points
    rng = np.random.RandomState(42)
    for _ in range(15):
        t = rng.uniform(0, np.pi)
        p = rng.uniform(0, 2*np.pi)
        start_points.append([t, p])
    
    # Local refinement using Nelder-Mead
    for sp in start_points:
        if time.time() > deadline - 10:
            break
        try:
            res = minimize(
                _objective_spherical,
                sp,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-13, 'fatol': 1e-15}
            )
            if res.fun < best_val:
                best_val = res.fun
                theta_opt, phi_opt = res.x
                c0 = np.sin(theta_opt) * np.cos(phi_opt)
                c4 = np.sin(theta_opt) * np.sin(phi_opt)
                c8 = np.cos(theta_opt)
                best_x = np.array([c0, c4, c8])
        except Exception:
            continue
    
    # Phase 4: Additional refinement with BFGS
    if time.time() < deadline - 5:
        try:
            res_bfgs = minimize(
                _objective_spherical,
                [theta_opt, phi_opt] if 'theta_opt' in dir() else [grid_best_theta, grid_best_phi],
                method='BFGS',
                options={'maxiter': 500, 'gtol': 1e-14}
            )
            if res_bfgs.fun < best_val:
                best_val = res_bfgs.fun
                theta_opt, phi_opt = res_bfgs.x
                c0 = np.sin(theta_opt) * np.cos(phi_opt)
                c4 = np.sin(theta_opt) * np.sin(phi_opt)
                c8 = np.cos(theta_opt)
                best_x = np.array([c0, c4, c8])
        except Exception:
            pass
    
    return best_x
# EVOLVE-BLOCK-END
