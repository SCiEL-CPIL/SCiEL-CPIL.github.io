import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 200

# Precompute constants
def _precompute():
    # I_n = integral x^(2n) e^(-x^2) dx
    # I_n = (2n-1)!! / 2^n * sqrt(pi)
    # I_0 = sqrt(pi)
    # I_1 = 0.5 * sqrt(pi)
    # I_n = (2n-1)/2 * I_{n-1}
    N_MAX = 14
    I = np.zeros(N_MAX + 1)
    I[0] = np.sqrt(np.pi)
    for n in range(1, N_MAX + 1):
        I[n] = (2*n - 1) * 0.5 * I[n-1]
        
    # Matrices M_k for m_k = sum p_i p_j I_{k+i+j}
    # M_k is 7x7
    M = []
    for k in range(3):
        Mk = np.zeros((7, 7))
        for i in range(7):
            for j in range(7):
                Mk[i, j] = I[k + i + j]
        M.append(Mk)
        
    # Matrix J mapping a=[a0, a4, a8] to p
    # p0 = 1*a0 + 3*a4 + 105*a8 + 945*a12
    # a12 = -1/945 a0 - 3/945 a4 - 105/945 a8
    # p0 = 1*a0 + 3*a4 + 105*a8 + 945*(-1/945 a0 - 3/945 a4 - 105/945 a8)
    #    = 1*a0 + 3*a4 + 105*a8 - (a0 + 3*a4 + 105*a8) = 0
    # Wait, p0 is always 0?
    # g(0) = P(0) e^0 = P(0). P(0) = p0.
    # The constraint is g(0)=0 => p0=0.
    # So p0 is identically 0.
    # Let's check p1, p2 etc.
    
    # p1 = -6 a4 - 420 a8 - 7560 a12
    #    = -6 a4 - 420 a8 - 7560 (-1/945 a0 - 3/945 a4 - 105/945 a8)
    #    = -6 a4 - 420 a8 + (7560/945) a0 + (7560*3/945) a4 + (7560*105/945) a8
    # 7560/945 = 8
    # 7560*3/945 = 24
    # 7560*105/945 = 8*105 = 840
    # p1 = 8 a0 + (-6+24) a4 + (-420+840) a8 = 8 a0 + 18 a4 + 420 a8
    
    # p2 = a4 + 210 a8 + 13860 a12
    # 13860/945 = 14.666? 13860 / 945 = 14.66... 945 * 14 = 13230. 945 * 15 = 14175.
    # 13860 / 945 = 297 / 20 = 14.85?
    # 13860 / 945 = 14.857...
    # Let's use fractions or just compute numerically.
    
    J = np.zeros((7, 3))
    # Coeffs for a0, a4, a8 in a12
    c_a12 = np.array([-1.0/945.0, -3.0/945.0, -105.0/945.0])
    
    # Base coeffs for a0, a4, a8 in p_k before a12 term
    # p0: 1, 3, 105
    # p1: 0, -6, -420
    # p2: 0, 1, 210
    # p3: 0, 0, -28
    # p4: 0, 0, 1
    # p5: 0, 0, 0
    # p6: 0, 0, 0
    
    base = np.array([
        [1, 3, 105],
        [0, -6, -420],
        [0, 1, 210],
        [0, 0, -28],
        [0, 0, 1],
        [0, 0, 0],
        [0, 0, 0]
    ], dtype=float)
    
    # a12 coeff in p_k
    # p0: 945
    # p1: -7560
    # p2: 13860
    # p3: -6930
    # p4: 1155
    # p5: -66
    # p6: 1
    
    a12_coeff = np.array([945, -7560, 13860, -6930, 1155, -66, 1], dtype=float)
    
    # J[k, c] = base[k, c] + a12_coeff[k] * c_a12[c]
    for k in range(7):
        for c in range(3):
            J[k, c] = base[k, c] + a12_coeff[k] * c_a12[c]
            
    # N_k matrices for Fourier side
    # n_k = 2 pi q^T M_k q, q = D p
    # D = diag((-1)^k)
    D = np.diag([(-1.0)**k for k in range(7)])
    N = []
    for k in range(3):
        # N_k = 2 pi D M_k D
        Nk = 2.0 * np.pi * (D @ M[k] @ D)
        N.append(Nk)
        
    return I, M, N, J

_I, _M, _N, _J = _precompute()

def _bound_and_grad(params):
    a = params
    p = _J @ a
    
    # m_k = p^T M_k p
    m0 = p @ _M[0] @ p
    m2 = p @ _M[1] @ p
    
    # n_k = p^T N_k p
    n0 = p @ _N[0] @ p
    n2 = p @ _N[1] @ p
    
    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18])
        
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18, np.array([1e18, 1e18, 1e18])
        
    bound = np.sqrt(vf * vF)
    
    # Gradients
    # dm_k/da = 2 (M_k p)^T J = 2 J^T M_k p
    # dn_k/da = 2 (N_k p)^T J = 2 J^T N_k p
    
    grad = np.zeros(3)
    
    # Precompute J^T M_k p and J^T N_k p
    # J^T is 3x7
    Jt = _J.T
    
    # dm0, dm2
    dm0 = 2.0 * (Jt @ (_M[0] @ p))
    dm2 = 2.0 * (Jt @ (_M[1] @ p))
    
    # dn0, dn2
    dn0 = 2.0 * (Jt @ (_N[0] @ p))
    dn2 = 2.0 * (Jt @ (_N[1] @ p))
    
    # dvf/da = (m0 dm2 - m2 dm0) / m0^2
    dvf = (m0 * dm2 - m2 * dm0) / (m0 * m0)
    
    # dvF/da = (n0 dn2 - n2 dn0) / n0^2
    dvF = (n0 * dn2 - n2 * dn0) / (n0 * n0)
    
    grad = (vF * dvf + vf * dvF) / (2.0 * bound)
    
    return bound, grad
