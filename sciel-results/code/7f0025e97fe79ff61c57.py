"""Uncertainty bound minimization via vectorized evaluation and global stochastic search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

_DEADLINE = time.time() + 195.0

# Precompute integration constants
_SP = np.sqrt(np.pi)
_I = np.zeros(13)
_I[0] = _SP
for _n in range(1, 13):
    _I[_n] = _I[_n - 1] * (2 * _n - 1) / 2.0

# Precompute matrices M_k for k=0,1,2
# M_k[i,j] = I[i+j+k]
_M = np.zeros((3, 7, 7))
for k in range(3):
    for i in range(7):
        for j in range(7):
            idx = i + j + k
            if idx < 13:
                _M[k, i, j] = _I[idx]

# Coefficients for P(x) from a0, a4, a8, a12
# p = [p0, p1, p2, p3, p4, p5, p6]
# p0 = a0 + 3a4 + 105a8 + 945a12
# p1 = -6a4 - 420a8 - 7560a12
# p2 = a4 + 210a8 + 13860a12
# p3 = -28a8 - 6930a12
# p4 = a8 + 1155a12
# p5 = -66a12
# p6 = a12
# We can express p as A @ a where a = [a0, a4, a8, a12]
# But a12 is determined by a0, a4, a8.
# a12 = -(a0 + 3a4 + 105a8) / 945
# So p = B @ [a0, a4, a8]
# Let's compute B (7x3)
_B = np.zeros((7, 3))
# Column 0: coeff of a0
_B[0, 0] = 1.0
_B[0, 0] += 945.0 * (-1.0 / 945.0) # a12 term for p0
# Actually, let's just compute p directly in the function for clarity, it's fast enough.
# Or precompute the linear map.
# a12 = -1/945 * a0 - 3/945 * a4 - 105/945 * a8
# a12 = -1/945 * a0 - 1/315 * a4 - 1/9 * a8

# Let's just do the matrix multiplication in the vectorized function.

def _compute_bound_fast(a0, a4, a8):
    a12 = -(a0 + 3.0 * a4 + 105.0 * a8) / 945.0
    
    # p vector
    p0 = a0 + 3.0 * a4 + 105.0 * a8 + 945.0 * a12
    p1 = -6.0 * a4 - 420.0 * a8 - 7560.0 * a12
    p2 = a4 + 210.0 * a8 + 13860.0 * a12
    p3 = -28.0 * a8 - 6930.0 * a12
    p4 = a8 + 1155.0 * a12
    p5 = -66.0 * a12
    p6 = a12
    p = np.array([p0, p1, p2, p3, p4, p5, p6])
    
    # q vector for Fourier
    q = p * np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
    
    # m_k = p^T M_k p
    m0 = p @ _M[0] @ p
    m1 = p @ _M[1] @ p
    
    # n_k = 2*pi * q^T M_k q
    n0 = 2.0 * np.pi * (q @ _M[0] @ q)
    n1 = 2.0 * np.pi * (q @ _M[1] @ q)
    
    if m0 <= 1e-15 or n0 <= 1e-15:
        return 1e18
        
    vf = m1 / m0
    vF = n1 / n0
    
    if vf < 0 or vF < 0:
        return 1e18
        
    return np.sqrt(vf * vF)

def _objective_sphere(params):
    theta, phi = params
    # Convert to cartesian
    a0 = np.sin(theta) * np.cos(phi)
    a4 = np.sin(theta) * np.sin(phi)
    a8 = np.cos(theta)
    return _compute_bound_fast(a0, a4, a8)

def construct():
    # Initial seed
    seed = np.array([-1.13, 0.04, 0.000308642])
    seed_bound = _compute_bound_fast(*seed)
    
    best_bound = seed_bound
    best_params = seed.copy()
    
    # Global search using Differential Evolution on sphere
    # theta in [0, pi], phi in [0, 2*pi]
    bounds = [(0, np.pi), (0, 2 * np.pi)]
    
    try:
        # Use differential evolution
        # popsize=20, maxiter=100 is usually fast for 2D
        res_de = differential_evolution(
            _objective_sphere, 
            bounds, 
            maxiter=100, 
            popsize=20, 
            tol=1e-6, 
            seed=42,
            polish=False # We will polish manually
        )
        
        if res_de.fun < best_bound:
            best_bound = res_de.fun
            # Convert best spherical to cartesian
            theta, phi = res_de.x
            best_params = np.array([
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta)
            ])
            
        # Local refinement with Nelder-Mead
        # Start from the best point found
        # We need to optimize over theta, phi
        # Get current theta, phi from best_params
        r = np.linalg.norm(best_params)
        if r > 1e-15:
            theta0 = np.arccos(np.clip(best_params[2] / r, -1.0, 1.0))
            phi0 = np.arctan2(best_params[1], best_params[0])
            if phi0 < 0: phi0 += 2 * np.pi
        else:
            theta0, phi0 = np.pi/2, 0
            
        # Run minimize
        res_min = minimize(
            _objective_sphere, 
            x0=[theta0, phi0], 
            method='Nelder-Mead', 
            options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10}
        )
        
        if res_min.fun < best_bound:
            best_bound = res_min.fun
            theta, phi = res_min.x
            best_params = np.array([
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta)
            ])
            
        # Try a few random starts for Nelder-Mead as well, in case DE got stuck
        rng = np.random.default_rng(42)
        for _ in range(10):
            if time.time() > _DEADLINE - 10:
                break
            t_rand = rng.uniform(0, np.pi)
            p_rand = rng.uniform(0, 2 * np.pi)
            res_rand = minimize(
                _objective_sphere, 
                x0=[t_rand, p_rand], 
                method='Nelder-Mead', 
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            if res_rand.fun < best_bound:
                best_bound = res_rand.fun
                theta, phi = res_rand.x
                best_params = np.array([
                    np.sin(theta) * np.cos(phi),
                    np.sin(theta) * np.sin(phi),
                    np.cos(theta)
                ])
                
    except Exception:
        pass

    # Verify
    final_bound = _compute_bound_fast(*best_params)
    if not np.isfinite(final_bound) or final_bound > best_bound:
        return seed.copy()
        
    return best_params
# EVOLVE-BLOCK-END
