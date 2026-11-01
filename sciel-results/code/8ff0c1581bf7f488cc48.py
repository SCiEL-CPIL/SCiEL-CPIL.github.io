"""
Optimized uncertainty bound minimizer using analytic moments and L-BFGS-B.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from math import lgamma, pi

_START_TIME = time.time()
_DEADLINE = _START_TIME + 190.0  # Safety margin before 210s kill

# Precompute integrals I_n = integral x^{2n} e^{-x^2} dx
# I_n = (2n)! / (4^n n!) * sqrt(pi)
def _calc_I(n):
    if n < 0:
        return 0.0
    # Use log-gamma for stability
    log_val = lgamma(2*n + 1) - n * np.log(4.0) - lgamma(n + 1) + 0.5 * np.log(pi)
    return np.exp(log_val)

# We need I_n for n up to 12 (since max degree is 12, product is 24, /2 = 12)
_I = np.array([_calc_I(n) for n in range(13)])

# Indices for Hermite polynomials used: 0, 4, 8, 12
_IDX = [0, 4, 8, 12]
_N_IDX = len(_IDX)

# Precompute K[n_idx, m_idx, k_idx] = integral x^k H_n(x) H_m(x) e^{-x^2} dx
# k=0 corresponds to power 0, k=1 corresponds to power 2.
# We need K for k=0 and k=2.
_K = np.zeros((_N_IDX, _N_IDX, 2))

# Orthogonality: integral H_n H_m e^{-x^2} dx = sqrt(pi) 2^n n! delta_{nm}
for i, n in enumerate(_IDX):
    for j, m in enumerate(_IDX):
        if n == m:
            _K[i, j, 0] = np.sqrt(pi) * (2.0 ** n) * np.exp(lgamma(n + 1))
        else:
            _K[i, j, 0] = 0.0

# Recurrence for x^2 H_n H_m
# x H_n = 0.5 H_{n+1} + n H_{n-1}
# x^2 H_n = 0.5 x H_{n+1} + n x H_{n-1}
# = 0.5 (0.5 H_{n+2} + (n+1) H_n) + n (0.5 H_n + (n-1) H_{n-2})
# = 0.25 H_{n+2} + 0.5(n+1) H_n + 0.5n H_n + n(n-1) H_{n-2}
# = 0.25 H_{n+2} + (n + 0.5) H_n + n(n-1) H_{n-2}
# So x^2 H_n(x) H_m(x) e^{-x^2} integral can be found by expanding x^2 H_n in terms of H_{n+2}, H_n, H_{n-2}
# and using orthogonality with H_m.
# Let's just compute integral x^2 H_n H_m e^{-x^2} directly using the property:
# integral x^2 H_n H_m e^{-x^2} dx
# If n=m: integral x^2 H_n^2 e^{-x^2} dx.
# We can use the recurrence on one H_n:
# integral x^2 H_n H_m e^{-x^2} = integral (0.25 H_{n+2} + (n+0.5) H_n + n(n-1) H_{n-2}) H_m e^{-x^2}
# = 0.25 * <H_{n+2}, H_m> + (n+0.5) * <H_n, H_m> + n(n-1) * <H_{n-2}, H_m>
# where <H_a, H_b> = sqrt(pi) 2^a a! delta_{ab}.

def _hermite_norm(a):
    if a < 0:
        return 0.0
    return np.sqrt(pi) * (2.0 ** a) * np.exp(lgamma(a + 1))

for i, n in enumerate(_IDX):
    for j, m in enumerate(_IDX):
        # integral x^2 H_n H_m e^{-x^2}
        term1 = 0.25 * _hermite_norm(n + 2) * (1.0 if n + 2 == m else 0.0)
        term2 = (n + 0.5) * _hermite_norm(n) * (1.0 if n == m else 0.0)
        term3 = n * (n - 1) * _hermite_norm(n - 2) * (1.0 if n - 2 == m else 0.0)
        _K[i, j, 1] = term1 + term2 + term3

# Note: The integral is symmetric in n, m. The recurrence used above assumes we expanded H_n.
# Since the integral is symmetric, this value is correct for K[n,m,1] regardless of which we expand,
# provided we are consistent. Actually, let's double check.
# If we expand H_m instead:
# integral H_n x^2 H_m = integral H_n (0.25 H_{m+2} + (m+0.5) H_m + m(m-1) H_{m-2})
# = 0.25 <H_n, H_{m+2}> + ...
# These should yield the same result.

def _compute_bound_and_grad(params):
    """
    Compute the uncertainty product sqrt(Vx * Vxi) and its gradient.
    params: [a0, a4, a8]
    """
    a0, a4, a8 = params[0], params[1], params[2]
    
    # Coefficients for H0, H4, H8, H12
    # Constraint: g(0) = 0 => a0 + 3a4 + 105a8 + 945a12 = 0
    # H0(0)=1, H4(0)=3, H8(0)=105, H12(0)=945
    a12 = -(a0 + 3.0 * a4 + 105.0 * a8) / 945.0
    
    c = np.array([a0, a4, a8, a12])
    
    # Calculate M0 = <g|g> and M2 = <x^2 g|g>
    M0 = 0.0
    M2 = 0.0
    for i in range(_N_IDX):
        for j in range(_N_IDX):
            val = c[i] * c[j]
            M0 += val * _K[i, j, 0]
            M2 += val * _K[i, j, 1]
    
    # Calculate N0 = <F(g)|F(g)> and N2 = <xi^2 F(g)|F(g)>
    # F(g)(xi) = sum c_n (-i)^n H_n(xi) e^{-xi^2/2}
    # |F(g)|^2 = sum c_n c_m (-i)^{n+m} H_n H_m e^{-xi^2}
    # Since n,m in {0,4,8,12}, n+m is even. (-i)^{n+m} = (-1)^{(n+m)/2}
    N0 = 0.0
    N2 = 0.0
    for i in range(_N_IDX):
        for j in range(_N_IDX):
            n = _IDX[i]
            m = _IDX[j]
            phase = (-1.0) ** ((n + m) // 2)
            val = c[i] * c[j] * phase
            N0 += val * _K[i, j, 0]
            N2 += val * _K[i, j, 1]
    
    if M0 <= 1e-12 or N0 <= 1e-12:
        return 1e18, np.array([1e18, 1e18, 1e18])
        
    Vx = M2 / M0
    Vxi = N2 / N0
    
    if Vx < 0 or Vxi < 0:
        return 1e18, np.array([1e18, 1e18, 1e18])
        
    bound = np.sqrt(Vx * Vxi)
    
    # Gradient calculation
    # d(bound)/da_k = 1/(2*bound) * (Vxi * d(Vx)/da_k + Vx * d(Vxi)/da_k)
    # d(Vx)/da_k = (dM2/da_k * M0 - M2 * dM0/da_k) / M0^2
    # d(Vxi)/da_k = (dN2/da_k * N0 - N2 * dN0/da_k) / N0^2
    
    # We need derivatives of c with respect to a0, a4, a8
    # c = [a0, a4, a8, a12]
    # a12 = -(a0 + 3a4 + 105a8)/945
    # dc/da0 = [1, 0, 0, -1/945]
    # dc/da4 = [0, 1, 0, -3/945]
    # dc/da8 = [0, 0, 1, -105/945]
    
    dc_da = np.array([
        [1.0, 0.0, 0.0, -1.0/945.0],
        [0.0, 1.0, 0.0, -3.0/945.0],
        [0.0, 0.0, 1.0, -105.0/945.0]
    ])
    
    grad = np.zeros(3)
    
    for k in range(3):
        dc_k = dc_da[k]
        
        # dM0/da_k = sum_{i,j} (dc_i c_j + c_i dc_j) K_{ij}^0
        # = 2 * sum_i dc_i sum_j c_j K_{ij}^0  (since K is symmetric and sum is over all i,j)
        # Actually, sum_{i,j} dc_i c_j K_{ij} + sum_{i,j} c_i dc_j K_{ij}
        # Let S0_i = sum_j c_j K_{ij}^0. Then dM0 = sum_i dc_i S0_i + sum_j dc_j S0_j = 2 sum_i dc_i S0_i.
        
        S0 = np.zeros(_N_IDX)
        S2 = np.zeros(_N_IDX)
        for i in range(_N_IDX):
            for j in range(_N_IDX):
                S0[i] += c[j] * _K[i, j, 0]
                S2[i] += c[j] * _K[i, j, 1]
                
        dM0 = 2.0 * np.dot(dc_k, S0)
        dM2 = 2.0 * np.dot(dc_k, S2)
        
        # Similarly for N
        # N0 = sum c_i c_j phase_{ij} K_{ij}^0
        # dN0 = sum (dc_i c_j + c_i dc_j) phase_{ij} K_{ij}^0
        # Let T0_i = sum_j c_j phase_{ij} K_{ij}^0.
        # dN0 = sum_i dc_i T0_i + sum_j dc_j T0_j? No, phase is symmetric?
        # phase_{ij} = (-1)^{(_IDX[i]+_IDX[j])/2}. Yes, symmetric.
        # So dN0 = 2 * sum_i dc_i T0_i.
        
        T0 = np.zeros(_N_IDX)
        T2 = np.zeros(_N_IDX)
        for i in range(_N_IDX):
            for j in range(_N_IDX):
                n = _IDX[i]
                m = _IDX[j]
                phase = (-1.0) ** ((n + m) // 2)
                T0[i] += c[j] * phase * _K[i, j, 0]
                T2[i] += c[j] * phase * _K[i, j, 1]
                
        dN0 = 2.0 * np.dot(dc_k, T0)
        dN2 = 2.0 * np.dot(dc_k, T2)
        
        dVx = (dM2 * M0 - M2 * dM0) / (M0 * M0)
        dVxi = (dN2 * N0 - N2 * dN0) / (N0 * N0)
        
        grad[k] = (Vxi * dVx + Vx * dVxi) / (2.0 * bound)
        
    return bound, grad

def _objective(params):
    val, _ = _compute_bound_and_grad(params)
    return val

def _objective_grad(params):
    val, grad = _compute_bound_and_grad(params)
    return val, grad

def construct():
    """
    Returns the three coefficients [a0, a4, a8] that minimize the uncertainty bound.
    """
    if time.time() > _DEADLINE:
        return np.array([-1.13, 0.04, 0.000308642])
        
    # Initial guesses
    # Seed: -113/100, 1/25, 1/3240
    starts = [
        np.array([-1.13, 0.04, 0.000308642]),
        np.array([0.0, 0.0, 0.0]),
        np.array([-1.0, 0.05, 0.0005]),
        np.array([-1.5, 0.08, 0.0008]),
        np.array([-0.5, 0.02, 0.0002]),
        np.array([-2.0, 0.1, 0.001]),
        np.array([1.0, -0.05, 0.0005]),
    ]
    
    # Add some random starts
    rng = np.random.default_rng(42)
    for _ in range(10):
        starts.append(rng.normal(0, 0.5, 3))
        
    best_val = np.inf
    best_params = None
    
    for start in starts:
        if time.time() > _DEADLINE - 10:
            break
        try:
            res = minimize(_objective_grad, start, method='L-BFGS-B',
                          jac=True, options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-14})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
            
    # Final polish with Nelder-Mead if we have a good candidate
    if best_params is not None and time.time() < _DEADLINE - 5:
        try:
            res = minimize(_objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
            
    if best_params is None:
        return np.array([-1.13, 0.04, 0.000308642])
        
    return best_params
# EVOLVE-BLOCK-END
