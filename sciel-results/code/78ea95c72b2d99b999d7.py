"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

# Pre-compute Gauss-Hermite quadrature nodes and weights
_N_Q = 500
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

def _compute_bound(a0, a4, a8):
    """Compute the uncertainty product (upper bound on C4) for given coefficients."""
    a12 = -(a0 + 12.0*a4 + 1680.0*a8) / 2032128.0
    
    P = a0*_H0 + a4*_H4 + a8*_H8 + a12*_H12
    f_sq = P**2
    norm2 = np.sum(_w_gh * f_sq)
    
    if norm2 < 1e-30:
        return 1e10
    
    mom2 = np.sum(_w_gh * _x2 * f_sq)
    var_x = mom2 / norm2
    
    dP = a4*_dH4 + a8*_dH8 + a12*_dH12
    fp_inner = dP - _x_gh * P
    fp_sq = fp_inner**2
    momentum2 = np.sum(_w_gh * fp_sq)
    var_p = momentum2 / norm2
    
    if var_x < 0 or var_p < 0:
        return 1e10
    
    return np.sqrt(max(var_x, 0.0) * max(var_p, 0.0))

def _compute_bound_gradient(a0, a4, a8):
    """Compute the gradient of the bound with respect to (a0, a4, a8)."""
    a12 = -(a0 + 12.0*a4 + 1680.0*a8) / 2032128.0
    
    P = a0*_H0 + a4*_H4 + a8*_H8 + a12*_H12
    f_sq = P**2
    norm2 = np.sum(_w_gh * f_sq)
    
    if norm2 < 1e-30:
        return np.array([0.0, 0.0, 0.0])
    
    mom2 = np.sum(_w_gh * _x2 * f_sq)
    var_x = mom2 / norm2
    
    dP = a4*_dH4 + a8*_dH8 + a12*_dH12
    fp_inner = dP - _x_gh * P
    fp_sq = fp_inner**2
    momentum2 = np.sum(_w_gh * fp_sq)
    var_p = momentum2 / norm2
    
    if var_x < 0 or var_p < 0:
        return np.array([0.0, 0.0, 0.0])
    
    bound_val = np.sqrt(max(var_x, 1e-30) * max(var_p, 1e-30))
    if bound_val < 1e-30:
        return np.array([0.0, 0.0, 0.0])
    
    dP_da0 = _H0 - (1.0/2032128.0)*_H12
    dP_da4 = _H4 - (12.0/2032128.0)*_H12
    dP_da8 = _H8 - (1680.0/2032128.0)*_H12
    
    ddP_da0 = -(1.0/2032128.0)*_dH12
    ddP_da4 = _dH4 - (12.0/2032128.0)*_dH12
    ddP_da8 = _dH8 - (1680.0/2032128.0)*_dH12
    
    dnorm2_da0 = 2.0 * np.sum(_w_gh * P * dP_da0)
    dnorm2_da4 = 2.0 * np.sum(_w_gh * P * dP_da4)
    dnorm2_da8 = 2.0 * np.sum(_w_gh * P * dP_da8)
    
    dmom2_da0 = 2.0 * np.sum(_w_gh * _x2 * P * dP_da0)
    dmom2_da4 = 2.0 * np.sum(_w_gh * _x2 * P * dP_da4)
    dmom2_da8 = 2.0 * np.sum(_w_gh * _x2 * P * dP_da8)
    
    dfp_da0 = ddP_da0 - _x_gh * dP_da0
    dfp_da4 = ddP_da4 - _x_gh * dP_da4
    dfp_da8 = ddP_da8 - _x_gh * dP_da8
    
    dmom2p_da0 = 2.0 * np.sum(_w_gh * fp_inner * dfp_da0)
    dmom2p_da4 = 2.0 * np.sum(_w_gh * fp_inner * dfp_da4)
    dmom2p_da8 = 2.0 * np.sum(_w_gh * fp_inner * dfp_da8)
    
    dvar_x_da0 = (dmom2_da0 * norm2 - mom2 * dnorm2_da0) / (norm2 * norm2)
    dvar_x_da4 = (dmom2_da4 * norm2 - mom2 * dnorm2_da4) / (norm2 * norm2)
    dvar_x_da8 = (dmom2_da8 * norm2 - mom2 * dnorm2_da8) / (norm2 * norm2)
    
    dvar_p_da0 = (dmom2p_da0 * norm2 - momentum2 * dnorm2_da0) / (norm2 * norm2)
    dvar_p_da4 = (dmom2p_da4 * norm2 - momentum2 * dnorm2_da4) / (norm2 * norm2)
    dvar_p_da8 = (dmom2p_da8 * norm2 - momentum2 * dnorm2_da8) / (norm2 * norm2)
    
    grad0 = (var_p * dvar_x_da0 + var_x * dvar_p_da0) / (2.0 * bound_val)
    grad4 = (var_p * dvar_x_da4 + var_x * dvar_p_da4) / (2.0 * bound_val)
    grad8 = (var_p * dvar_x_da8 + var_x * dvar_p_da8) / (2.0 * bound_val)
    
    return np.array([grad0, grad4, grad8])

def construct():
    start_time = time.time()
    deadline = 195.0  # Wall-clock limit with safety margin before 210s kill

    def objective(a):
        return _compute_bound(a[0], a[1], a[2])

    def gradient(a):
        return _compute_bound_gradient(a[0], a[1], a[2])

    # 1. Homotopy Continuation Strategy
    # Start with a8 = 0 (simpler 2-param problem) and gradually increase a8.
    # This tracks the optimal manifold and avoids local minima.
    
    # Step 1: Optimize with a8 = 0
    # Initial guess for (a0, a4)
    a0_init = -1.0
    a4_init = 0.05
    
    current_a8 = 0.0
    current_x = np.array([a0_init, a4_init, current_a8])
    
    try:
        res = minimize(objective, current_x, jac=gradient, method='L-BFGS-B',
                       options={'maxiter': 1000, 'gtol': 1e-10, 'ftol': 1e-12})
        if res.success:
            current_x = res.x.copy()
    except Exception:
        pass

    # Step 2: Gradually increase a8 from 0 to ~0.0005 (target range)
    # We use a geometric or linear schedule. Linear is safer for small steps.
    target_a8 = 0.0005
    num_steps = 10
    a8_schedule = np.linspace(0.0, target_a8, num_steps)
    
    for a8_target in a8_schedule[1:]:
        if time.time() > deadline - 50.0:
            break
            
        # Update a8 to target, keep previous a0, a4 as initial guess
        current_x[2] = a8_target
        
        try:
            res = minimize(objective, current_x, jac=gradient, method='L-BFGS-B',
                           options={'maxiter': 500, 'gtol': 1e-10, 'ftol': 1e-12})
            if res.success:
                current_x = res.x.copy()
        except Exception:
            # If homotopy step fails, try a smaller step or break
            pass

    best_x = current_x.copy()
    best_score = objective(best_x)

    # 2. Multi-start refinement around the homotopy solution
    rng = np.random.default_rng(42)
    
    starts = [best_x]
    
    # Local perturbations
    for _ in range(20):
        starts.append(best_x + rng.normal(0, 0.005, size=3))
    
    # Wider perturbations to escape potential local minima
    for _ in range(10):
        starts.append(best_x + rng.normal(0, 0.02, size=3))
        
    # Include the original seed candidates as fallbacks
    x0_candidates = [
        np.array([-1.13, 0.04, 1.0/3240.0]),
        np.array([-1.0, 0.05, 0.0003]),
    ]
    starts.extend(x0_candidates)

    for start in starts:
        if time.time() > deadline - 20.0:
            break
            
        try:
            res = minimize(objective, start, jac=gradient, method='L-BFGS-B',
                           options={'maxiter': 2000, 'gtol': 1e-12, 'ftol': 1e-14,
                                    'maxls': 50})
            if res.fun < best_score:
                best_score = res.fun
                best_x = res.x.copy()
        except Exception:
            pass

    # 3. High-precision polish
    # Powell for robustness
    if time.time() < deadline - 10.0:
        try:
            res = minimize(objective, best_x, method='Powell',
                           options={'maxiter': 1000, 'ftol': 1e-14, 'xtol': 1e-14})
            if res.fun < best_score:
                best_score = res.fun
                best_x = res.x.copy()
        except Exception:
            pass

    # Final L-BFGS-B with very tight tolerance
    if time.time() < deadline - 5.0:
        try:
            res = minimize(objective, best_x, jac=gradient, method='L-BFGS-B',
                           options={'maxiter': 5000, 'gtol': 1e-14, 'ftol': 1e-16,
                                    'maxls': 100})
            if res.fun < best_score:
                best_score = res.fun
                best_x = res.x.copy()
        except Exception:
            pass

    return best_x

# EVOLVE-BLOCK-END
