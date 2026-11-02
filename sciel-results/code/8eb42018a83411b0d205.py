"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

# Pre-compute Gauss-Hermite quadrature nodes and weights
_N_Q = 200
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
# d/dx H4 = 64x^3 - 96x
_dH4 = 64*_x_gh**3 - 96*_x_gh
# d/dx H8 = 2048x^7 - 8064x^5 + 10752x^3 - 1344x
_dH8 = 2048*_x_gh**7 - 8064*_x_gh**5 + 10752*_x_gh**3 - 1344*_x_gh
# d/dx H12 = 49152x^11 - 322560x^9 + 677376x^7 - 653184x^5 + 258048x^3 - 14336x
_dH12 = (49152*_x_gh**11 - 322560*_x_gh**9 + 677376*_x_gh**7 
         - 653184*_x_gh**5 + 258048*_x_gh**3 - 14336*_x_gh)

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
    # Actually P is even (all terms are even powers), so <x> = 0
    var_x = mom2 / norm2
    
    # f'(x) = (P'(x) - x*P(x)) * exp(-x^2/2)
    # |f'|^2 = (P' - x*P)^2 * exp(-x^2)
    dP = a4*_dH4 + a8*_dH8 + a12*_dH12  # dH0 = 0
    fp_inner = dP - _x_gh * P
    fp_sq = fp_inner**2
    
    # ||f'||_2^2 = integral |f'|^2 dx
    momentum2 = np.sum(_w_gh * fp_sq)
    
    var_p = momentum2 / norm2
    
    # Uncertainty product
    if var_x < 0 or var_p < 0:
        return 1e10
    
    return np.sqrt(max(var_x, 0.0) * max(var_p, 0.0))

def _compute_bound_gradient(a0, a4, a8):
    """Compute the gradient of the bound with respect to (a0, a4, a8)."""
    # Compute a12 from constraint P(0) = 0
    a12 = -(a0 + 12.0*a4 + 1680.0*a8) / 2032128.0
    
    # Build P(x) at quadrature nodes
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
    
    # The bound is sqrt(var_x * var_p)
    # We need d/d(a0,a4,a8) of sqrt(var_x * var_p)
    # = (1/(2*sqrt(var_x*var_p))) * (var_p * d(var_x) + var_x * d(var_p))
    
    bound_val = np.sqrt(max(var_x, 1e-30) * max(var_p, 1e-30))
    if bound_val < 1e-30:
        return np.array([0.0, 0.0, 0.0])
    
    # Compute derivatives of var_x and var_p with respect to a0, a4, a8
    # var_x = mom2 / norm2
    # d(var_x)/da_i = (d(mom2)/da_i * norm2 - mom2 * d(norm2)/da_i) / norm2^2
    
    # d(norm2)/da_i = 2 * sum(w_gh * P * dP/da_i)
    # d(mom2)/da_i = 2 * sum(w_gh * x^2 * P * dP/da_i)
    # d(momentum2)/da_i = 2 * sum(w_gh * fp_inner * d(fp_inner)/da_i)
    
    # dP/da0 = H0 (but a12 depends on a0)
    # Actually, P = a0*H0 + a4*H4 + a8*H8 + a12*H12
    # where a12 = -(a0 + 12*a4 + 1680*a8)/2032128
    
    # dP/da0 = H0 - (1/2032128)*H12
    # dP/da4 = H4 - (12/2032128)*H12
    # dP/da8 = H8 - (1680/2032128)*H12
    
    dP_da0 = _H0 - (1.0/2032128.0)*_H12
    dP_da4 = _H4 - (12.0/2032128.0)*_H12
    dP_da8 = _H8 - (1680.0/2032128.0)*_H12
    
    # d(dP)/da0 = dH0 - (1/2032128)*dH12 = 0 - (1/2032128)*dH12
    # d(dP)/da4 = dH4 - (12/2032128)*dH12
    # d(dP)/da8 = dH8 - (1680/2032128)*dH12
    
    ddP_da0 = -(1.0/2032128.0)*_dH12
    ddP_da4 = _dH4 - (12.0/2032128.0)*_dH12
    ddP_da8 = _dH8 - (1680.0/2032128.0)*_dH12
    
    # d(norm2)/da_i = 2 * sum(w_gh * P * dP_da_i)
    dnorm2_da0 = 2.0 * np.sum(_w_gh * P * dP_da0)
    dnorm2_da4 = 2.0 * np.sum(_w_gh * P * dP_da4)
    dnorm2_da8 = 2.0 * np.sum(_w_gh * P * dP_da8)
    
    # d(mom2)/da_i = 2 * sum(w_gh * x^2 * P * dP_da_i)
    dmom2_da0 = 2.0 * np.sum(_w_gh * _x2 * P * dP_da0)
    dmom2_da4 = 2.0 * np.sum(_w_gh * _x2 * P * dP_da4)
    dmom2_da8 = 2.0 * np.sum(_w_gh * _x2 * P * dP_da8)
    
    # d(momentum2)/da_i = 2 * sum(w_gh * fp_inner * d(fp_inner)/da_i)
    # d(fp_inner)/da_i = ddP_da_i - x * dP_da_i
    dfp_da0 = ddP_da0 - _x_gh * dP_da0
    dfp_da4 = ddP_da4 - _x_gh * dP_da4
    dfp_da8 = ddP_da8 - _x_gh * dP_da8
    
    dmom2p_da0 = 2.0 * np.sum(_w_gh * fp_inner * dfp_da0)
    dmom2p_da4 = 2.0 * np.sum(_w_gh * fp_inner * dfp_da4)
    dmom2p_da8 = 2.0 * np.sum(_w_gh * fp_inner * dfp_da8)
    
    # d(var_x)/da_i = (dmom2_da_i * norm2 - mom2 * dnorm2_da_i) / norm2^2
    dvar_x_da0 = (dmom2_da0 * norm2 - mom2 * dnorm2_da0) / (norm2 * norm2)
    dvar_x_da4 = (dmom2_da4 * norm2 - mom2 * dnorm2_da4) / (norm2 * norm2)
    dvar_x_da8 = (dmom2_da8 * norm2 - mom2 * dnorm2_da8) / (norm2 * norm2)
    
    # d(var_p)/da_i = (dmom2p_da_i * norm2 - momentum2 * dnorm2_da_i) / norm2^2
    dvar_p_da0 = (dmom2p_da0 * norm2 - momentum2 * dnorm2_da0) / (norm2 * norm2)
    dvar_p_da4 = (dmom2p_da4 * norm2 - momentum2 * dnorm2_da4) / (norm2 * norm2)
    dvar_p_da8 = (dmom2p_da8 * norm2 - momentum2 * dnorm2_da8) / (norm2 * norm2)
    
    # d(bound)/da_i = (var_p * dvar_x_da_i + var_x * dvar_p_da_i) / (2 * bound_val)
    grad0 = (var_p * dvar_x_da0 + var_x * dvar_p_da0) / (2.0 * bound_val)
    grad4 = (var_p * dvar_x_da4 + var_x * dvar_p_da4) / (2.0 * bound_val)
    grad8 = (var_p * dvar_x_da8 + var_x * dvar_p_da8) / (2.0 * bound_val)
    
    return np.array([grad0, grad4, grad8])

def construct():
    start_time = time.time()
    deadline = 200.0  # Wall-clock limit with safety margin
    
    # Initial guess from the seed
    x0 = np.array([-1.13, 0.04, 1.0/3240.0])
    
    best_x = x0.copy()
    best_score = _compute_bound(x0[0], x0[1], x0[2])
    
    def objective(a):
        return _compute_bound(a[0], a[1], a[2])
    
    def gradient(a):
        return _compute_bound_gradient(a[0], a[1], a[2])
    
    # Generate multiple starting points for global search
    rng = np.random.default_rng(42)
    
    starts = [x0]
    # Local perturbations around seed
    for _ in range(10):
        starts.append(x0 + rng.normal(0, 0.05, size=3))
    # Wider exploration
    for _ in range(6):
        starts.append(x0 + rng.normal(0, 0.5, size=3))
    # Random wide starts
    for _ in range(4):
        starts.append(rng.uniform(-3, 3, size=3))
    
    # Multi-start optimization with gradient information
    for i, start in enumerate(starts):
        if time.time() > deadline - 15.0:
            break
        
        try:
            # Use L-BFGS-B with gradient for faster convergence
            res = minimize(objective, start, jac=gradient, method='L-BFGS-B',
                          options={'maxiter': 1000, 'gtol': 1e-10, 'ftol': 1e-12})
            if res.fun < best_score:
                best_score = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    # Final polish with tighter tolerance if time permits
    if time.time() < deadline - 10.0:
        try:
            res = minimize(objective, best_x, jac=gradient, method='L-BFGS-B',
                          options={'maxiter': 2000, 'gtol': 1e-12, 'ftol': 1e-14})
            if res.fun < best_score:
                best_score = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    return best_x

# EVOLVE-BLOCK-END
