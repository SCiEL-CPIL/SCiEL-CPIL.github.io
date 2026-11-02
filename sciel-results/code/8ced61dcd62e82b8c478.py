"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

# Pre-compute Gauss-Hermite quadrature nodes and weights
_N_Q = 300
_x_gh, _w_gh = np.polynomial.hermite.hermgauss(_N_Q)
_w_gh *= np.sqrt(np.pi)  # Scale for exp(-x^2) weight

# Hermite polynomials evaluated at quadrature nodes (pre-compute for speed)
_x2 = _x_gh ** 2
_H0 = np.ones_like(_x_gh)
_H4 = 16*_x2**2 - 48*_x2 + 12
_H8 = 256*_x2**4 - 3072*_x2**3 + 11520*_x2**2 - 13440*_x2 + 1680
_H12 = (4096*_x2**6 - 73728*_x2**5 + 580608*_x2**4 - 2419200*_x2**3 
        + 5443200*_x2**2 - 5443200*_x2 + 2032128)

# Pre-compute derivatives of Hermite polynomials at quadrature nodes
_dH4 = 64*_x_gh**3 - 96*_x_gh
_dH8 = 2048*_x_gh**7 - 8064*_x_gh**5 + 10752*_x_gh**3 - 1344*_x_gh
_dH12 = (49152*_x_gh**11 - 322560*_x_gh**9 + 677376*_x_gh**7 
         - 653184*_x_gh**5 + 258048*_x_gh**3 - 14336*_x_gh)

# Pre-compute x * H terms for efficiency
_xH4 = _x_gh * _H4
_xH8 = _x_gh * _H8
_xH12 = _x_gh * _H12

def _compute_bound(a0, a4, a8):
    """Compute the uncertainty product (upper bound on C4) for given coefficients."""
    # Compute a12 from constraint P(0) = 0
    # H0(0)=1, H4(0)=12, H8(0)=1680, H12(0)=2032128
    a12 = -(a0 + 12.0*a4 + 1680.0*a8) / 2032128.0
    
    # Build P(x) at quadrature nodes
    P = a0*_H0 + a4*_H4 + a8*_H8 + a12*_H12
    
    # f(x) = P(x) * exp(-x^2/2)
    # |f|^2 = P^2 * exp(-x^2)
    f_sq = P**2
    
    # ||f||_2^2 = integral P^2 exp(-x^2) dx
    norm2 = np.sum(_w_gh * f_sq)
    
    if norm2 < 1e-30:
        return 1e10  # Avoid division by zero
    
    # Second moment: integral x^2 |f|^2 dx
    mom2 = np.sum(_w_gh * _x2 * f_sq)
    
    # Variance in x: <x^2> - <x>^2, but since P(0)=0 and even polynomials, <x>=0
    var_x = mom2 / norm2
    
    # f'(x) = (P'(x) - x*P(x)) * exp(-x^2/2)
    # |f'|^2 = (P' - x*P)^2 * exp(-x^2)
    dP = a4*_dH4 + a8*_dH8 + a12*_dH12  # dH0 = 0
    xP = a0*_x_gh*_H0 + a4*_xH4 + a8*_xH8 + a12*_xH12
    fp_inner = dP - xP
    fp_sq = fp_inner**2
    
    # ||f'||_2^2 = integral |f'|^2 dx
    momentum2 = np.sum(_w_gh * fp_sq)
    
    var_p = momentum2 / norm2
    
    # Uncertainty product
    if var_x < 0 or var_p < 0:
        return 1e10
    
    return np.sqrt(max(var_x, 0.0) * max(var_p, 0.0))

def _var_to_coeffs(theta):
    """Convert rotation angles to coefficients using variational ansatz.
    
    Uses a 3-level variational circuit:
    - theta[0]: rotation for a0 amplitude
    - theta[1]: rotation for a4 amplitude  
    - theta[2]: rotation for a8 amplitude
    
    The mapping uses sine/cosine to naturally bound coefficients and
    create a smooth parameterization of the coefficient space.
    """
    # Scale factors to map angles to meaningful coefficient ranges
    # Based on the seed solution scale: a0~O(1), a4~O(0.01-0.1), a8~O(0.0001-0.01)
    scale0 = 2.0   # a0 in [-2, 2]
    scale4 = 0.2   # a4 in [-0.2, 0.2]
    scale8 = 0.01  # a8 in [-0.01, 0.01]
    
    a0 = scale0 * np.sin(theta[0])
    a4 = scale4 * np.sin(theta[1])
    a8 = scale8 * np.sin(theta[2])
    
    return np.array([a0, a4, a8])

def _var_objective(theta):
    """Objective function for variational parameterization."""
    coeffs = _var_to_coeffs(theta)
    return _compute_bound(coeffs[0], coeffs[1], coeffs[2])

def _direct_objective(a):
    """Direct objective for coefficient space."""
    return _compute_bound(a[0], a[1], a[2])

def construct():
    start_time = time.time()
    deadline = 200.0  # Wall-clock limit with safety margin
    safety_margin = 10.0
    
    # Initial guess from the seed
    x0 = np.array([-1.13, 0.04, 1.0/3240.0])
    
    best_x = x0.copy()
    best_score = _compute_bound(x0[0], x0[1], x0[2])
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Differential Evolution for global search in direct coefficient space
    # This is the structural change: DE explores the full landscape more effectively
    elapsed = time.time() - start_time
    if elapsed < deadline - safety_margin - 60:
        try:
            bounds = [(-3.0, 3.0), (-0.3, 0.3), (-0.02, 0.02)]
            de_result = differential_evolution(
                _direct_objective, bounds,
                seed=42, maxiter=100, popsize=20,
                tol=1e-12, atol=1e-14,
                mutation=(0.3, 0.9), recombination=0.7,
                polish=True, init='latinhypercube',
                updating='deferred'
            )
            if de_result.fun < best_score:
                best_score = de_result.fun
                best_x = de_result.x.copy()
        except Exception:
            pass
    
    # Phase 2: Variational Quantum-Inspired Search (VQA)
    # Use rotation angles as parameters, optimize with multiple methods
    elapsed = time.time() - start_time
    if elapsed < deadline - safety_margin - 40:
        # Initialize angles from best direct solution
        # Invert the sin mapping approximately
        theta0 = np.array([
            np.arcsin(np.clip(best_x[0] / 2.0, -1, 1)),
            np.arcsin(np.clip(best_x[1] / 0.2, -1, 1)),
            np.arcsin(np.clip(best_x[2] / 0.01, -1, 1))
        ])
        
        # Multiple VQA starts
        vqa_starts = [theta0]
        for _ in range(10):
            vqa_starts.append(rng.uniform(-np.pi/2, np.pi/2, size=3))
        for _ in range(5):
            vqa_starts.append(theta0 + rng.normal(0, 0.5, size=3))
        
        for i, start in enumerate(vqa_starts):
            elapsed = time.time() - start_time
            if elapsed > deadline - safety_margin - 30:
                break
            
            try:
                max_iter = 5000 if i < 5 else 2000
                res = minimize(_var_objective, start, method='Nelder-Mead',
                              options={'maxiter': max_iter, 'xatol': 1e-14, 
                                       'fatol': 1e-16, 'adaptive': True})
                if res.fun < best_score:
                    best_score = res.fun
                    best_x = _var_to_coeffs(res.x)
            except Exception:
                pass
        
        # L-BFGS-B refinement in variational space
        if time.time() - start_time < deadline - safety_margin - 20:
            try:
                # Find angle corresponding to current best
                theta_best = np.array([
                    np.arcsin(np.clip(best_x[0] / 2.0, -1, 1)),
                    np.arcsin(np.clip(best_x[1] / 0.2, -1, 1)),
                    np.arcsin(np.clip(best_x[2] / 0.01, -1, 1))
                ])
                res_bfgs = minimize(_var_objective, theta_best, method='L-BFGS-B',
                                  options={'maxiter': 3000, 'gtol': 1e-15, 
                                           'disp': False, 'ftol': 1e-18})
                if res_bfgs.fun < best_score:
                    best_score = res_bfgs.fun
                    best_x = _var_to_coeffs(res_bfgs.x)
            except Exception:
                pass
    
    # Phase 3: Differential Evolution in variational space (second DE pass)
    elapsed = time.time() - start_time
    if elapsed < deadline - safety_margin - 30:
        try:
            bounds_theta = [(-np.pi/2, np.pi/2), (-np.pi/2, np.pi/2), (-np.pi/2, np.pi/2)]
            de_result2 = differential_evolution(
                _var_objective, bounds_theta,
                seed=123, maxiter=80, popsize=15,
                tol=1e-12, atol=1e-14,
                mutation=(0.3, 0.9), recombination=0.7,
                polish=True, init='latinhypercube',
                updating='deferred'
            )
            if de_result2.fun < best_score:
                best_score = de_result2.fun
                best_x = _var_to_coeffs(de_result2.x)
        except Exception:
            pass
    
    # Phase 4: Coarse-to-fine local refinement in direct coefficient space
    elapsed = time.time() - start_time
    if elapsed < deadline - safety_margin - 15:
        for scale in [0.05, 0.01, 0.001, 0.0001]:
            if time.time() - start_time > deadline - safety_margin - 5:
                break
            try:
                res_local = minimize(_direct_objective, best_x, method='Nelder-Mead',
                                    options={'maxiter': 8000, 'xatol': 1e-15, 
                                             'fatol': 1e-18, 'adaptive': True,
                                             'initial_simplex': np.array([
                                                 best_x + scale*np.array([1,0,0]),
                                                 best_x + scale*np.array([0,1,0]),
                                                 best_x + scale*np.array([0,0,1]),
                                                 best_x])})
                if res_local.fun < best_score:
                    best_score = res_local.fun
                    best_x = res_local.x.copy()
            except Exception:
                pass
    
    # Phase 5: Final BFGS polish
    if time.time() - start_time < deadline - safety_margin:
        try:
            res_final = minimize(_direct_objective, best_x, method='L-BFGS-B',
                               options={'maxiter': 5000, 'gtol': 1e-16, 
                                        'disp': False, 'ftol': 1e-20})
            if res_final.fun < best_score:
                best_score = res_final.fun
                best_x = res_final.x.copy()
        except Exception:
            pass
    
    # Phase 6: Ultra-fine Nelder-Mead
    if time.time() - start_time < deadline - safety_margin:
        try:
            res_fine = minimize(_direct_objective, best_x, method='Nelder-Mead',
                               options={'maxiter': 30000, 'xatol': 1e-18, 
                                        'fatol': 1e-22, 'adaptive': True,
                                        'initial_simplex': np.array([
                                            best_x + 1e-7*np.array([1,0,0]),
                                            best_x + 1e-7*np.array([0,1,0]),
                                            best_x + 1e-7*np.array([0,0,1]),
                                            best_x])})
            if res_fine.fun < best_score:
                best_score = res_fine.fun
                best_x = res_fine.x.copy()
        except Exception:
            pass
    
    return best_x

# EVOLVE-BLOCK-END
