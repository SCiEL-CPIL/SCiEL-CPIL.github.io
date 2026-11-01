"""Uncertainty bound minimization via vectorized evaluation, analytic gradients, and multi-stage search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

_DEADLINE = time.time() + 195.0

# Precompute integration constants
# I[n] = integral x^{2n} e^{-x^2} dx = (2n-1)!! * sqrt(pi) / 2^n
_SP = np.sqrt(np.pi)
_I = np.zeros(14)
_I[0] = _SP
for _n in range(1, 14):
    _I[_n] = _I[_n - 1] * (2 * _n - 1) / 2.0

# Precompute matrices M_k for k=0,1,2
# M_k[i,j] = I[i+j+k] where i,j are indices for powers x^{2i}, x^{2j}
# We need i,j up to 6 (since max degree is 12, so power index is 6)
# Max index in I is 6+6+2 = 14, which is within our precomputed range (0..13)
_M = np.zeros((3, 7, 7))
for k in range(3):
    for i in range(7):
        for j in range(7):
            idx = i + j + k
            if idx < 14:
                _M[k, i, j] = _I[idx]

# Precompute the linear map from [a0, a4, a8] to p = [p0..p6]
# a12 = -(a0 + 3a4 + 105a8) / 945
# p0 = a0 + 3a4 + 105a8 + 945a12
# p1 = -6a4 - 420a8 - 7560a12
# p2 = a4 + 210a8 + 13860a12
# p3 = -28a8 - 6930a12
# p4 = a8 + 1155a12
# p5 = -66a12
# p6 = a12

# Let a = [a0, a4, a8]
# a12 = -1/945 * a0 - 3/945 * a4 - 105/945 * a8
# a12 = -1/945 * a0 - 1/315 * a4 - 1/9 * a8

# p = B @ a
# B is 7x3
_B = np.zeros((7, 3))
# Coefficients for a0, a4, a8 in each p_k

# p0: a0 + 3a4 + 105a8 + 945*(-1/945 a0 - 1/315 a4 - 1/9 a8)
#    = a0(1 - 1) + a4(3 - 3) + a8(105 - 105) = 0? 
# Wait, g(0)=0 implies p0=0. Let's check.
# H0(0)=1, H4(0)=3, H8(0)=105, H12(0)=945.
# g(0) = a0*1 + a4*3 + a8*105 + a12*945 = 0 => a12 = -(a0+3a4+105a8)/945.
# p0 is the constant term of P(x).
# P(x) = a0 H0 + a4 H4 + a8 H8 + a12 H12.
# H0(x) = 1
# H4(x) = x^4 - 6x^2 + 3
# H8(x) = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
# H12(x) = x^12 - 66x^10 + 1155x^8 - 6930x^6 + 13860x^4 - 7560x^2 + 945
#
# p0 (coeff of x^0):
# H0: 1
# H4: 3
# H8: 105
# H12: 945
# p0 = a0 + 3a4 + 105a8 + 945a12
# Substituting a12:
# p0 = a0 + 3a4 + 105a8 - (a0 + 3a4 + 105a8) = 0.
# So p0 is always 0. This simplifies things significantly!

# Let's re-evaluate B.
# p0 = 0
# p1 (coeff of x^2):
# H0: 0
# H4: -6
# H8: -420
# H12: -7560
# p1 = -6a4 - 420a8 - 7560a12
#    = -6a4 - 420a8 + 8(a0 + 3a4 + 105a8)
#    = 8a0 + (-6+24)a4 + (-420+840)a8
#    = 8a0 + 18a4 + 420a8

# p2 (coeff of x^4):
# H0: 0
# H4: 1
# H8: 210
# H12: 13860
# p2 = a4 + 210a8 + 13860a12
#    = a4 + 210a8 - 14.666... (a0+3a4+105a8)
#    13860/945 = 14.666... = 44/3
#    p2 = -44/3 a0 + (1 - 44) a4 + (210 - 44*35) a8  (105*44/3 = 35*44 = 1540)
#    p2 = -44/3 a0 - 43 a4 + (210 - 1540) a8
#    p2 = -44/3 a0 - 43 a4 - 1330 a8

# p3 (coeff of x^6):
# H0: 0
# H4: 0
# H8: -28
# H12: -6930
# p3 = -28a8 - 6930a12
#    = -28a8 - 7.333... (a0+3a4+105a8)
#    6930/945 = 7.333... = 22/3
#    p3 = -22/3 a0 - 22 a4 + (-28 - 22*35) a8
#    22*35 = 770
#    p3 = -22/3 a0 - 22 a4 - 798 a8

# p4 (coeff of x^8):
# H0: 0
# H4: 0
# H8: 1
# H12: 1155
# p4 = a8 + 1155a12
#    = a8 - 1.222... (a0+3a4+105a8)
#    1155/945 = 11/9
#    p4 = -11/9 a0 - 11/3 a4 + (1 - 11*35/3) a8
#    11*35/3 = 385/3
#    1 - 385/3 = (3-385)/3 = -382/3
#    p4 = -11/9 a0 - 11/3 a4 - 382/3 a8

# p5 (coeff of x^10):
# H0: 0
# H4: 0
# H8: 0
# H12: -66
# p5 = -66a12
#    = -66/945 (a0+3a4+105a8)
#    66/945 = 22/315
#    p5 = -22/315 a0 - 22/105 a4 - 22/3 a8

# p6 (coeff of x^12):
# H0: 0
# H4: 0
# H8: 0
# H12: 1
# p6 = a12
#    = -1/945 a0 - 1/315 a4 - 1/9 a8

# Let's just compute B numerically to avoid arithmetic errors.
_B = np.zeros((7, 3))
# p0 is always 0, so row 0 is zeros.
# We can just evaluate p for basis vectors.
# Basis 1: a=[1,0,0]
a_test = np.array([1.0, 0.0, 0.0])
a12_t = -(a_test[0] + 3*a_test[1] + 105*a_test[2]) / 945.0
p_t = np.array([
    a_test[0] + 3*a_test[1] + 105*a_test[2] + 945*a12_t,
    -6*a_test[1] - 420*a_test[2] - 7560*a12_t,
    a_test[1] + 210*a_test[2] + 13860*a12_t,
    -28*a_test[2] - 6930*a12_t,
    a_test[2] + 1155*a12_t,
    -66*a12_t,
    a12_t
])
_B[:, 0] = p_t

# Basis 2: a=[0,1,0]
a_test = np.array([0.0, 1.0, 0.0])
a12_t = -(a_test[0] + 3*a_test[1] + 105*a_test[2]) / 945.0
p_t = np.array([
    a_test[0] + 3*a_test[1] + 105*a_test[2] + 945*a12_t,
    -6*a_test[1] - 420*a_test[2] - 7560*a12_t,
    a_test[1] + 210*a_test[2] + 13860*a12_t,
    -28*a_test[2] - 6930*a12_t,
    a_test[2] + 1155*a12_t,
    -66*a12_t,
    a12_t
])
_B[:, 1] = p_t

# Basis 3: a=[0,0,1]
a_test = np.array([0.0, 0.0, 1.0])
a12_t = -(a_test[0] + 3*a_test[1] + 105*a_test[2]) / 945.0
p_t = np.array([
    a_test[0] + 3*a_test[1] + 105*a_test[2] + 945*a12_t,
    -6*a_test[1] - 420*a_test[2] - 7560*a12_t,
    a_test[1] + 210*a_test[2] + 13860*a12_t,
    -28*a_test[2] - 6930*a12_t,
    a_test[2] + 1155*a12_t,
    -66*a12_t,
    a12_t
])
_B[:, 2] = p_t

# Signs for Fourier transform: q_k = p_k * (-1)^k
_SIGNS = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])

def _compute_bound_fast(a0, a4, a8):
    """Compute the upper bound on C4 for given coefficients."""
    a = np.array([a0, a4, a8])
    p = _B @ a
    
    # q = p * signs
    q = p * _SIGNS
    
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

def _compute_bound_and_grad(a0, a4, a8):
    """Compute the upper bound and its gradient."""
    a = np.array([a0, a4, a8])
    p = _B @ a
    
    # q = p * signs
    q = p * _SIGNS
    
    # m_k = p^T M_k p
    m0 = p @ _M[0] @ p
    m1 = p @ _M[1] @ p
    
    # n_k = 2*pi * q^T M_k q
    n0 = 2.0 * np.pi * (q @ _M[0] @ q)
    n1 = 2.0 * np.pi * (q @ _M[1] @ q)
    
    if m0 <= 1e-15 or n0 <= 1e-15:
        return 1e18, np.array([1e18, 1e18, 1e18])
        
    vf = m1 / m0
    vF = n1 / n0
    
    if vf < 0 or vF < 0:
        return 1e18, np.array([1e18, 1e18, 1e18])
        
    bound = np.sqrt(vf * vF)
    
    # Gradient computation
    # d(bound)/da_c = (1/(2*bound)) * (vF * d(vf)/da_c + vf * d(vF)/da_c)
    # d(vf)/da_c = (dm1/da_c * m0 - m1 * dm0/da_c) / m0^2
    # d(vF)/da_c = (dn1/da_c * n0 - n1 * dn0/da_c) / n0^2
    
    # dp/da_c = B[:, c]
    # dm_k/da_c = 2 * p^T M_k dp/da_c
    # dn_k/da_c = 4*pi * q^T M_k dq/da_c  (since n_k = 2*pi * q^T M_k q, d(n_k) = 2*pi * 2 * q^T M_k dq)
    # dq/da_c = B[:, c] * signs
    
    grad = np.zeros(3)
    for c in range(3):
        dp = _B[:, c]
        dq = dp * _SIGNS
        
        dm0 = 2.0 * (p @ _M[0] @ dp)
        dm1 = 2.0 * (p @ _M[1] @ dp)
        
        dn0 = 4.0 * np.pi * (q @ _M[0] @ dq)
        dn1 = 4.0 * np.pi * (q @ _M[1] @ dq)
        
        dvm = (dm1 * m0 - m1 * dm0) / (m0 * m0)
        dvn = (dn1 * n0 - n1 * dn0) / (n0 * n0)
        
        grad[c] = (vF * dvm + vf * dvn) / (2.0 * bound)
        
    return bound, grad

def _objective(params):
    a0, a4, a8 = params
    return _compute_bound_fast(a0, a4, a8)

def _objective_grad(params):
    a0, a4, a8 = params
    val, grad = _compute_bound_and_grad(a0, a4, a8)
    return val, grad

def construct():
    # Initial seed
    seed = np.array([-1.13, 0.04, 0.000308642])
    seed_bound = _compute_bound_fast(*seed)
    
    best_bound = seed_bound
    best_params = seed.copy()
    
    # Strategy:
    # 1. Differential Evolution for global exploration (fast, robust)
    # 2. L-BFGS-B with analytic gradients from multiple starts for local refinement
    # 3. Nelder-Mead polish from the best point
    
    try:
        # Global search using Differential Evolution
        # We optimize over the sphere to avoid singularities? 
        # Or just use bounds? The function is defined everywhere.
        # Let's use a box bound first, it's simpler and often sufficient.
        # Based on the seed, a0 is around -1, a4 around 0.04, a8 small.
        # Let's try a reasonable box.
        bounds = [(-5, 5), (-1, 1), (-0.01, 0.01)]
        
        res_de = differential_evolution(
            _objective, 
            bounds, 
            maxiter=50, 
            popsize=15, 
            tol=1e-5, 
            seed=42,
            polish=False
        )
        
        if res_de.fun < best_bound and np.isfinite(res_de.fun):
            best_bound = res_de.fun
            best_params = res_de.x.copy()
            
        # Local refinement with L-BFGS-B using analytic gradients
        # Start from the best point found by DE and the original seed
        starts = [best_params.copy(), seed.copy()]
        
        # Add some random starts
        rng = np.random.default_rng(123)
        for _ in range(10):
            if time.time() > _DEADLINE - 30:
                break
            start = np.array([
                rng.uniform(-2, 2),
                rng.uniform(-0.2, 0.2),
                rng.uniform(-0.005, 0.005)
            ])
            starts.append(start)
            
        for start in starts:
            if time.time() > _DEADLINE - 10:
                break
            try:
                res_lbfgs = minimize(
                    _objective_grad, 
                    start, 
                    method='L-BFGS-B', 
                    jac=True,
                    options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-14}
                )
                if res_lbfgs.fun < best_bound and np.isfinite(res_lbfgs.fun):
                    best_bound = res_lbfgs.fun
                    best_params = res_lbfgs.x.copy()
            except Exception:
                pass
        
        # Final polish with Nelder-Mead
        if time.time() < _DEADLINE - 5:
            try:
                res_nm = minimize(
                    _objective, 
                    best_params, 
                    method='Nelder-Mead', 
                    options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-15}
                )
                if res_nm.fun < best_bound and np.isfinite(res_nm.fun):
                    best_bound = res_nm.fun
                    best_params = res_nm.x.copy()
            except Exception:
                pass
                
    except Exception:
        pass

    # Verify
    final_bound = _compute_bound_fast(*best_params)
    if not np.isfinite(final_bound) or final_bound > best_bound:
        return seed.copy()
        
    return best_params
# EVOLVE-BLOCK-END
