"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
from math import factorial, sqrt, pi

# Pre-compute analytic matrices for the objective function
# We want to minimize Delta_x^2 = <x^2> / <1> for f = g * exp(-x^2/2)
# where g = c0 H0 + c4 H4 + c8 H8 + c12 H12
# <1> = sum c_i c_j M_ij
# <x^2> = sum c_i c_j N_ij
# M_ij = int H_i H_j exp(-x^2) dx
# N_ij = int x^2 H_i H_j exp(-x^2) dx

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
# So M is diagonal.
M_diag = [sqrt(pi) * (2**n) * factorial(n) for n in indices]

# N_ij = int x^2 H_i H_j exp(-x^2) dx
# Use x^2 H_n = 0.5 H_{n+2} + n H_n
# So int x^2 H_i H_j exp(-x^2) = 0.5 int H_{i+2} H_j exp(-x^2) + i int H_i H_j exp(-x^2)
# By orthogonality:
# Term 1 is non-zero only if i+2 == j.
# Term 2 is non-zero only if i == j.
# Thus N is tridiagonal (in terms of indices difference 2).
# Specifically, N_ij = 0 unless j == i or j == i+2 or j == i-2.
# If j == i: N_ii = i * M_ii
# If j == i+2: N_i,i+2 = 0.5 * M_{i+2,i+2} (since int H_{i+2} H_{i+2} = M_{i+2,i+2})
# If j == i-2: N_i,i-2 = 0.5 * M_{i-2,i-2} (from the term where the "i" in the formula is j? No.)
# Let's re-evaluate:
# N_ij = 0.5 * int H_{i+2} H_j exp(-x^2) + i * int H_i H_j exp(-x^2)
# If j = i: N_ii = 0.5 * 0 + i * M_ii = i * M_ii
# If j = i+2: N_i,i+2 = 0.5 * M_{i+2} + i * 0 = 0.5 * M_{i+2}
# If j = i-2: N_i,i-2 = 0.5 * 0 + i * 0 = 0 ? 
# Wait, the integral is symmetric. N_ij = N_ji.
# Let's check N_4,0 (i=4, j=0).
# N_40 = 0.5 * int H_6 H_0 + 4 * int H_4 H_0 = 0.
# But N_04 (i=0, j=4) = 0.5 * int H_2 H_4 + 0 * ... = 0.
# This suggests N is diagonal?
# Let's check x^2 H_0 = 0.5 H_2. int x^2 H_0 H_2 = 0.5 int H_2 H_2 = 0.5 M_2.
# So N_02 = 0.5 M_2.
# Our indices are 0, 4, 8, 12. Differences are 4, 8, 12.
# The recurrence x^2 H_n = 0.5 H_{n+2} + n H_n connects n to n+2 and n.
# So H_0 connects to H_0 and H_2.
# H_4 connects to H_4 and H_6.
# H_8 connects to H_8 and H_10.
# H_12 connects to H_12 and H_14.
# None of the "off-diagonal" partners (2, 6, 10, 14) are in our basis {0, 4, 8, 12}.
# Therefore, N is DIAGONAL for this specific basis!
# N_ii = i * M_ii.

N_diag = [n * m for n, m in zip(indices, M_diag)]

# c12 is determined by c0, c4, c8 to satisfy g(0)=0
H0_0 = _hermite_val(0, 0)
H4_0 = _hermite_val(4, 0)
H8_0 = _hermite_val(8, 0)
H12_0 = _hermite_val(12, 0)

def _compute_bound_analytic(c0, c4, c8):
    c12 = -(c0 * H0_0 + c4 * H4_0 + c8 * H8_0) / H12_0
    coeffs = np.array([c0, c4, c8, c12])
    
    # Denominator: sum c_i^2 M_ii
    norm_sq = np.sum(coeffs**2 * np.array(M_diag))
    if norm_sq <= 1e-12:
        return 1e10
        
    # Numerator: sum c_i^2 N_ii
    x2_expectation = np.sum(coeffs**2 * np.array(N_diag))
    
    # Variance Delta_x^2 = <x^2> - <x>^2. <x>=0 due to symmetry.
    # Product Delta_x Delta_xi = Delta_x^2 because Delta_xi = Delta_x.
    return x2_expectation / norm_sq

def construct():
    """Find optimal coefficients for H0, H4, H8."""
    start_time = time.time()
    deadline = start_time + 150.0  # safety margin
    
    # Initial guess
    x0 = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    best_x = x0.copy()
    best_val = _compute_bound_analytic(*x0)
    
    # Use a fast optimizer with multiple starts
    rng = np.random.RandomState(42)
    
    # 1. Local optimization from seed
    try:
        res = minimize(
            lambda x: _compute_bound_analytic(x[0], x[1], x[2]),
            x0,
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12}
        )
        if res.fun < best_val:
            best_val = res.fun
            best_x = res.x
    except Exception:
        pass
    
    # 2. Random restarts
    for i in range(20):
        if time.time() > deadline:
            break
        x_start = x0 + rng.normal(0, 1.0, size=3)
        try:
            res = minimize(
                lambda x: _compute_bound_analytic(x[0], x[1], x[2]),
                x_start,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            continue
            
    return best_x
# EVOLVE-BLOCK-END
