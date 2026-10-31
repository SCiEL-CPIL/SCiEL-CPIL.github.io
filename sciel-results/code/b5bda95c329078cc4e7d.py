"""Optimized uncertainty bound via log-magnitude parameterization + analytic gradient."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from math import factorial, sqrt, pi

_DEADLINE = 190.0

# Precomputed Hermite coefficients and integrals
def _hermite_coeffs(n):
    if n == 0: return {0: 1.0}
    if n == 1: return {1: 2.0}
    h_prev = {0: 1.0}
    h_curr = {1: 2.0}
    for m in range(2, n + 1):
        h_new = {}
        for p, c in h_curr.items():
            new_p = p + 1
            h_new[new_p] = h_new.get(new_p, 0.0) + 2.0 * c
        for p, c in h_prev.items():
            h_new[p] = h_new.get(p, 0.0) - 2.0 * (m - 1) * c
        h_new = {p: c for p, c in h_new.items() if abs(c) > 1e-15}
        h_prev, h_curr = h_curr, h_new
    return h_curr

def _int_x2m(m):
    if m == 0: return sqrt(pi)
    val = sqrt(pi)
    for j in range(1, m + 1):
        val *= (2 * j - 1) / 2.0
    return val

# Degrees of freedom for p(x)
DEGREES = [0, 4, 8, 12]
N_DEG = 4

# Precompute Hermite polynomials as coefficient vectors
H_POLYS = []
for d in DEGREES:
    hc = _hermite_coeffs(d)
    vec = np.zeros(d + 1)
    for p, c in hc.items():
        vec[p] = c
    H_POLYS.append(vec)

# Precompute M0, M2, L matrices
# M0_{ij} = <H_i, H_j> = 2^{n_i} n_i! delta_{ij}
M0_mat = np.zeros((N_DEG, N_DEG))
for i, d in enumerate(DEGREES):
    M0_mat[i, i] = (2 ** d) * factorial(d)

# M2_{ij} = <x^2 H_i, H_j>
# x^2 H_n = 0.5 H_{n+2} + (n+0.5) H_n + 0.5 n(n-1) H_{n-2} ? 
# Let's just compute numerically via polynomial multiplication to be safe, 
# or use the relation x^2 H_n = 0.5 H_{n+2} + (n + 0.5) H_n + 0.5 n (n-1) H_{n-2} is incorrect.
# Correct: x H_n = 0.5 H_{n+1} + n H_{n-1}
# x^2 H_n = x(0.5 H_{n+1} + n H_{n-1}) = 0.5(0.5 H_{n+2} + (n+1) H_n) + n(0.5 H_n + (n-1) H_{n-2})
# = 0.25 H_{n+2} + 0.5(n+1) H_n + 0.5 n H_n + 0.5 n(n-1) H_{n-2}
# = 0.25 H_{n+2} + (n + 0.5) H_n + 0.5 n(n-1) H_{n-2}
# We can compute the integrals <H_k, H_l> easily.
# M2_{ij} = 0.25 <H_{n_i+2}, H_{n_j}> + (n_i + 0.5) <H_{n_i}, H_{n_j}> + 0.5 n_i(n_i-1) <H_{n_i-2}, H_{n_j}>
# This requires H_{n+2} and H_{n-2} which might not be in our set.
# So we compute M2_{ij} by expanding polynomials.
M2_mat = np.zeros((N_DEG, N_DEG))
for i in range(N_DEG):
    for j in range(N_DEG):
        # Polynomial for H_i * H_j
        # H_i is degree n_i, H_j is degree n_j
        # Product is degree n_i + n_j
        # We need <x^2 H_i, H_j> = <H_i, x^2 H_j>
        # Let's just compute the integral of x^2 * P_i(x) * P_j(x) * exp(-x^2)
        # P_i * P_j is a polynomial. Multiply by x^2. Integrate.
        # We can represent polynomials as vectors.
        deg_i = DEGREES[i]
        deg_j = DEGREES[j]
        # Coeffs of H_i
        ci = H_POLYS[i]
        cj = H_POLYS[j]
        # Product coeffs
        prod = np.convolve(ci, cj)
        # Multiply by x^2: shift by 2
        prod_x2 = np.zeros(len(prod) + 2)
        prod_x2[2:] = prod
        # Integrate
        val = 0.0
        for k in range(0, len(prod_x2), 2):
            m = k // 2
            val += prod_x2[k] * _int_x2m(m)
        M2_mat[i, j] = val

# L_{ij} = <(H_i' - x H_i), (H_j' - x H_j)>
# Let Q_i = H_i' - x H_i.
# H_i' = 2 n_i H_{n_i-1}.
# x H_i = 0.5 H_{n_i+1} + n_i H_{n_i-1}.
# Q_i = 2 n_i H_{n_i-1} - 0.5 H_{n_i+1} - n_i H_{n_i-1} = n_i H_{n_i-1} - 0.5 H_{n_i+1}.
# This is a linear combination of Hermite polynomials!
# So we can compute Q_i coefficients easily.
Q_POLYS = []
for i, d in enumerate(DEGREES):
    q_vec = np.zeros(max(d + 1, 1))
    if d == 0:
        # Q_0 = H_0' - x H_0 = 0 - x = -x. H_1 = 2x. So Q_0 = -0.5 H_1.
        # Formula: 0*H_{-1} - 0.5 H_1 = -0.5 H_1. Correct.
        q_vec = np.zeros(2)
        q_vec[1] = -0.5 * 2.0 # coeff of x in -0.5 H_1 is -0.5 * 2 = -1.
        # Wait, H_1 = 2x. -0.5 H_1 = -x.
        # My q_vec is coeffs of x^k.
        q_vec[1] = -1.0
    else:
        # Q_i = n_i H_{n_i-1} - 0.5 H_{n_i+1}
        h_minus = _hermite_coeffs(d - 1)
        h_plus = _hermite_coeffs(d + 1)
        max_deg_q = d + 1
        q_vec = np.zeros(max_deg_q + 1)
        for p, c in h_minus.items():
            q_vec[p] += d * c
        for p, c in h_plus.items():
            q_vec[p] -= 0.5 * c
    Q_POLYS.append(q_vec)

L_mat = np.zeros((N_DEG, N_DEG))
for i in range(N_DEG):
    for j in range(N_DEG):
        ci = Q_POLYS[i]
        cj = Q_POLYS[j]
        prod = np.convolve(ci, cj)
        val = 0.0
        for k in range(0, len(prod), 2):
            m = k // 2
            val += prod[k] * _int_x2m(m)
        L_mat[i, j] = val

def _compute_bound_and_grad_a(a):
    """
    Compute bound and gradient w.r.t a (vector of 4 coeffs [a0, a4, a8, a12]).
    """
    M0 = a @ M0_mat @ a
    M2 = a @ M2_mat @ a
    L = a @ L_mat @ a
    
    if M0 <= 1e-12 or L <= 1e-12:
        return np.inf, np.zeros(4)
    
    bound = np.sqrt(M2 * L) / M0
    
    # Gradient
    dM0 = 2 * M0_mat @ a
    dM2 = 2 * M2_mat @ a
    dL = 2 * L_mat @ a
    
    # J = sqrt(M2 L) / M0
    # dJ/da = 1/M0 * ( 0.5 * (1/sqrt(M2 L)) * (L dM2 + M2 dL) - sqrt(M2 L) * dM0 / M0 )
    # = 1/M0 * ( (L dM2 + M2 dL) / (2 sqrt(M2 L)) - bound * dM0 / M0 )
    # = (L dM2 + M2 dL) / (2 M0 sqrt(M2 L)) - bound * dM0 / M0^2
    
    term1 = (L * dM2 + M2 * dL) / (2.0 * M0 * np.sqrt(M2 * L))
    term2 = bound * dM0 / (M0 * M0)
    
    grad_a = term1 - term2
    
    return bound, grad_a

def _objective_log(params):
    """
    params = [m0, m4, m8]
    a0 = -exp(m0), a4 = exp(m4), a8 = exp(m8)
    a12 derived from g(0)=0.
    """
    a0 = -np.exp(params[0])
    a4 = np.exp(params[1])
    a8 = np.exp(params[2])
    
    # g(0) = a0*1 + a4*12 + a8*2520 + a12*27720 = 0
    a12 = -(a0 + 12*a4 + 2520*a8) / 27720.0
    
    a = np.array([a0, a4, a8, a12])
    bound, grad_a = _compute_bound_and_grad_a(a)
    
    if np.isinf(bound):
        return bound, np.zeros(3)
    
    # Chain rule
    # da0/dm0 = a0
    # da4/dm4 = a4
    # da8/dm8 = a8
    # da12/dm0 = -1/27720 * a0
    # da12/dm4 = -12/27720 * a4
    # da12/dm8 = -2520/27720 * a8
    
    da_dm = np.array([
        [a0, 0, 0],
        [0, a4, 0],
        [0, 0, a8],
        [-a0/27720.0, -12*a4/27720.0, -2520*a8/27720.0]
    ])
    
    grad_m = da_dm.T @ grad_a
    
    return bound, grad_m

def construct():
    start_time = time.time()
    
    # Initial guess in log space
    # a0 ~ -1.13, a4 ~ 0.04, a8 ~ 0.0003
    m0_init = np.log(1.13)
    m4_init = np.log(0.04)
    m8_init = np.log(0.0003)
    
    x0 = np.array([m0_init, m4_init, m8_init])
    
    best_x = x0.copy()
    best_bound, _ = _objective_log(x0)
    
    # Optimization
    try:
        result = minimize(
            _objective_log,
            x0,
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 100}
        )
        if result.fun < best_bound:
            best_bound = result.fun
            best_x = result.x.copy()
    except Exception:
        pass
    
    # Refine
    if time.time() - start_time < _DEADLINE - 10:
        try:
            result = minimize(
                _objective_log,
                best_x,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-14}
            )
            if result.fun < best_bound:
                best_bound = result.fun
                best_x = result.x.copy()
        except Exception:
            pass
            
    # Convert back to a0, a4, a8
    a0 = -np.exp(best_x[0])
    a4 = np.exp(best_x[1])
    a8 = np.exp(best_x[2])
    
    return np.array([a0, a4, a8])

# EVOLVE-BLOCK-END
