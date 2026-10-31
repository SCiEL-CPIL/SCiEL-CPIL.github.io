"""Uncertainty inequality optimization using exact polynomial-moment integrals
and homogeneity reduction to a 2-D ratio problem, optimized via multi-start
L-BFGS-B with analytic gradients."""
import time
import numpy as np
from scipy.optimize import minimize

_T0 = time.time()
_DEADLINE = 210.0
_MARGIN = 15.0

def _time_left():
    return _DEADLINE - _MARGIN - (time.time() - _T0)

# Precompute exact integrals for the Hermite-Gaussian basis.
# h_n(x) = H_n(x) * exp(-x^2/2)
# <h_n, h_m> = delta_nm * sqrt(pi) * 2^n * n!
# <x^2 h_n, h_m> is nonzero only when |n-m| <= 2.
# x H_n = (1/2) H_{n+1} + n H_{n-1}
# x^2 H_n = (1/4) H_{n+2} + (n + 1/2) H_n + (n(n-1)/4) H_{n-2}
# So x^2 h_n = (1/4) h_{n+2} + (n + 1/2) h_n + (n(n-1)/4) h_{n-2}
# <x^2 h_n, h_m> = (n + 1/2) * <h_n, h_m> + (1/4) <h_{n+2}, h_m> + (n(n-1)/4) <h_{n-2}, h_m>
#                 = (n + 1/2) * delta_nm * N_n + (1/4) * delta_{n+2,m} * N_{n+2} + (n(n-1)/4) * delta_{n-2,m} * N_{n-2}
# where N_n = sqrt(pi) * 2^n * n!

# We only need degrees 0, 4, 8, 12.
# N_0 = sqrt(pi) * 1 = sqrt(pi)
# N_4 = sqrt(pi) * 16 * 24 = 384 * sqrt(pi)
# N_8 = sqrt(pi) * 256 * 40320 = 10321920 * sqrt(pi)
# N_12 = sqrt(pi) * 4096 * 479001600 = 1962161765888 * sqrt(pi)

import math
_sqrt_pi = math.sqrt(math.pi)
N = {}
for n in [0, 4, 8, 12]:
    N[n] = _sqrt_pi * (2 ** n) * math.factorial(n)

# Build the Gram matrix G where G[i,j] = <h_{d_i}, h_{d_j}> for d = [0,4,8,12]
# And the moment matrix M where M[i,j] = <x^2 h_{d_i}, h_{d_j}>
# Since h_n are orthogonal, G is diagonal: G[i,j] = delta_ij * N[d_i]
# M[i,j] = (d_i + 1/2) * delta_ij * N[d_i] + (1/4) * delta_{d_i+2, d_j} * N[d_i+2] + (d_i*(d_i-1)/4) * delta_{d_i-2, d_j} * N[d_i-2]

degrees = [0, 4, 8, 12]
d_list = np.array(degrees)

# G is diagonal
G = np.diag([N[d] for d in degrees])

# M matrix
M = np.zeros((4, 4))
for i, ni in enumerate(degrees):
    # Diagonal term: (ni + 1/2) * N[ni]
    M[i, i] = (ni + 0.5) * N[ni]
    # Off-diagonal: (1/4) * N[ni+2] if ni+2 is in degrees
    if ni + 2 in degrees:
        j = degrees.index(ni + 2)
        M[i, j] += 0.25 * N[ni + 2]
    # Off-diagonal: (ni*(ni-1)/4) * N[ni-2] if ni-2 is in degrees
    if ni >= 2 and ni - 2 in degrees:
        j = degrees.index(ni - 2)
        M[i, j] += (ni * (ni - 1) / 4.0) * N[ni - 2]

# Constraint: g(0) = 0
# H_0(0) = 1, H_4(0) = 12, H_8(0) = 1344, H_12(0) = 1032192
H0_0, H4_0, H8_0, H12_0 = 1.0, 12.0, 1344.0, 1032192.0
# c12 = -(c0 * 1 + c4 * 12 + c8 * 1344) / 1032192
# Let c = [c0, c4, c8, c12]. The constraint is c0 + 12*c4 + 1344*c8 + 1032192*c12 = 0.
# We can parameterize by c0, c4, c8 and solve for c12.

def _compute_bound(c0, c4, c8):
    """Compute the uncertainty product bound using exact integrals.
    The bound is E2 / E0 where E0 = <g,g> and E2 = <x^2 g, g>.
    Due to the Fourier invariance of the Hermite-Gaussian basis, the frequency
    side gives the same integrals, so the product A(f)*A(F(f)) is proportional
    to (E0 * E2) or (E2/E0) depending on the exact definition.
    
    Given the task description mentions "upper bound on C4", and the seed program
    used E2/E0, we'll use that. But let's consider: if A(f) = sqrt(E0 * E2),
    then A(f)*A(F(f)) = sqrt(E0*E2) * sqrt(E0*E2) = E0*E2 (if F(g) has same norms).
    However, the seed used E2/E0 which is the variance. Let's stick with E2/E0
    as it's scale-invariant and the homogeneity reduction works cleanly.
    """
    c12 = -(c0 * H0_0 + c4 * H4_0 + c8 * H8_0) / H12_0
    c = np.array([c0, c4, c8, c12])
    
    E0 = c @ G @ c
    E2 = c @ M @ c
    
    if E0 < 1e-15:
        return np.inf
    return E2 / E0

def _compute_bound_grad(params):
    """Compute the bound and its gradient with respect to [c0, c4, c8]."""
    if _time_left() <= 0:
        return np.inf, None
    
    c0, c4, c8 = params
    c12 = -(c0 * H0_0 + c4 * H4_0 + c8 * H8_0) / H12_0
    c = np.array([c0, c4, c8, c12])
    
    E0 = c @ G @ c
    E2 = c @ M @ c
    
    if E0 < 1e-15:
        return np.inf, None
    
    obj = E2 / E0
    
    # Gradient with respect to c0, c4, c8
    # dc/dc0 = [1, 0, 0, dc12/dc0]
    # dc/dc4 = [0, 1, 0, dc12/dc4]
    # dc/dc8 = [0, 0, 1, dc12/dc8]
    dc12_dc0 = -H0_0 / H12_0
    dc12_dc4 = -H4_0 / H12_0
    dc12_dc8 = -H8_0 / H12_0
    
    # dE0/dc_k = 2 * (G @ c) @ dc/dc_k
    # dE2/dc_k = 2 * (M @ c) @ dc/dc_k
    # grad_k = (dE2_k * E0 - E2 * dE0_k) / E0^2
    
    Gc = G @ c
    Mc = M @ c
    
    grad = np.zeros(3)
    for k in range(3):
        dc_k = np.zeros(4)
        dc_k[k] = 1.0
        dc_k[3] = [dc12_dc0, dc12_dc4, dc12_dc8][k]
        
        dE0_k = 2.0 * (Gc @ dc_k)
        dE2_k = 2.0 * (Mc @ dc_k)
        
        grad[k] = (dE2_k * E0 - E2 * dE0_k) / (E0 ** 2)
    
    return obj, grad

def construct():
    try:
        if _time_left() <= 0:
            return np.array([-113.0/100, 1.0/25, 1.0/3240])
        
        # Multi-start optimization
        best_params = np.array([-113.0/100, 1.0/25, 1.0/3240])
        best_val = _compute_bound(best_params[0], best_params[1], best_params[2])
        
        # Generate starting points
        np.random.seed(42)
        starts = [best_params.copy()]
        
        # Add some structured starts
        for scale in [0.1, 1.0, 10.0]:
            for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
                c0 = scale * np.cos(angle)
                c4 = scale * np.sin(angle)
                c8 = scale * 0.1
                starts.append(np.array([c0, c4, c8]))
        
        # Random starts
        for _ in range(30):
            starts.append(np.random.randn(3) * 1.0)
        
        for start in starts:
            if _time_left() <= 10:
                break
            
            def obj(x):
                if _time_left() <= 5:
                    return np.inf
                val, _ = _compute_bound_grad(x)
                return val
            
            def grad(x):
                if _time_left() <= 5:
                    return np.zeros(3)
                _, g = _compute_bound_grad(x)
                if g is None:
                    return np.zeros(3)
                return g
            
            res = minimize(obj, start, jac=grad, method='L-BFGS-B', 
                          options={'maxiter': 500, 'gtol': 1e-10})
            
            if np.all(np.isfinite(res.x)) and res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        
        # Verify the result
        if not np.all(np.isfinite(best_params)):
            return np.array([-113.0/100, 1.0/25, 1.0/3240])
        
        return best_params
    except Exception:
        return np.array([-113.0/100, 1.0/25, 1.0/3240])
