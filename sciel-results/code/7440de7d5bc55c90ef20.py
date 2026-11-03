"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import eval_hermite
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import minimize, differential_evolution

# Precompute Gauss-Hermite nodes and weights
# Need to integrate x^4 * P(x)^2 for 4th moment (degree up to 26+2=28)
# N=25 gives exact integration up to degree 49
N_QUAD = 25
_x_nodes, _w_weights = hermgauss(N_QUAD)
_x_sq = _x_nodes**2
_x_four = _x_sq**2

# Precompute Hermite polynomials at nodes
_H0 = np.ones(N_QUAD)
_H4 = eval_hermite(4, _x_nodes)
_H8 = eval_hermite(8, _x_nodes)
_H12 = eval_hermite(12, _x_nodes)

def _get_c12(c0, c4, c8):
    # H_0(0)=1, H_4(0)=-3, H_8(0)=105, H_12(0)=-945
    # c0 + c4*(-3) + c8*105 + c12*(-945) = 0
    return (c0 - 3*c4 + 105*c8) / 945.0

def _compute_bound(c0, c4, c8):
    c12 = _get_c12(c0, c4, c8)
    
    p_vals = c0*_H0 + c4*_H4 + c8*_H8 + c12*_H12
    p_sq = p_vals**2
    
    den = np.sum(_w_weights * p_sq)
    if den < 1e-15:
        return 1e10
    
    # Second moment of x
    m2_x = np.sum(_w_weights * _x_sq * p_sq) / den
    
    # For the Fourier side, we need the second moment of the Fourier transform
    # For g(x) = P(x) exp(-x^2/2), the Fourier transform involves derivatives
    # The frequency second moment can be computed from the 4th moment of x
    # via the relation: <xi^2> = <x^4> - 2<x^2> + 1 (for normalized Gaussian basis)
    
    m4_x = np.sum(_w_weights * _x_four * p_sq) / den
    
    # The variance in frequency: sigma_xi^2 = m4_x - 2*m2_x + 1
    # This comes from the fact that for exp(-x^2/2), the Fourier transform
    # shifts moments in a specific way
    m2_xi = m4_x - 2.0*m2_x + 1.0
    
    if m2_xi < 0:
        m2_xi = 1e-10
    
    product = m2_x * m2_xi
    return product

def construct():
    t_start = time.time()
    deadline = t_start + 190.0  # 20s safety margin before 210s limit
    
    def objective(c):
        return _compute_bound(c[0], c[1], c[2])
    
    # Multiple starting points for local optimization
    starting_points = [
        np.array([-1.13, 0.04, 0.000309]),  # baseline
        np.array([-1.0, 0.05, 0.0005]),
        np.array([-1.5, 0.03, 0.0002]),
        np.array([-0.8, 0.06, 0.0004]),
        np.array([-2.0, 0.02, 0.0001]),
        np.array([-1.2, 0.045, 0.00035]),
        np.array([-1.0, 0.03, 0.0002]),
        np.array([-1.3, 0.05, 0.0004]),
    ]
    
    best_c = starting_points[0].copy()
    best_val = objective(best_c)
    
    # Try local optimization from each starting point
    for i, c0 in enumerate(starting_points):
        if time.time() > deadline - 10.0:
            break
        
        try:
            res = minimize(
                objective,
                c0,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_c = res.x.copy()
        except Exception:
            pass
        
        try:
            res2 = minimize(
                objective,
                c0,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            if res2.fun < best_val:
                best_val = res2.fun
                best_c = res2.x.copy()
        except Exception:
            pass
    
    # If time allows, try differential evolution for global search
    if time.time() < deadline - 30.0:
        try:
            bounds = [(-5.0, 5.0), (-1.0, 1.0), (-0.01, 0.01)]
            de_result = differential_evolution(
                objective,
                bounds,
                maxiter=50,
                popsize=20,
                tol=1e-10,
                seed=42,
                polish=True
            )
            if de_result.fun < best_val:
                best_val = de_result.fun
                best_c = de_result.x.copy()
        except Exception:
            pass
    
    # Final polish with high-precision local optimization
    if time.time() < deadline - 5.0:
        try:
            res_final = minimize(
                objective,
                best_c,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if res_final.fun < best_val:
                best_val = res_final.fun
                best_c = res_final.x.copy()
        except Exception:
            pass
    
    return np.array(best_c)

# EVOLVE-BLOCK-END
