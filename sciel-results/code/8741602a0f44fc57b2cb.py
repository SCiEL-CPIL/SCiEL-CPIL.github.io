# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 200

# Precompute constants
_I_TABLE = np.zeros(14)
def _init_I_table():
    sp = np.sqrt(np.pi)
    for n in range(14):
        # (2n)! / (4^n n!) * sqrt(pi)
        # Use log to avoid overflow, though n is small
        log_val = 0.5 * np.log(np.pi)
        for m in range(1, 2*n+1):
            log_val += np.log(m)
        log_val -= n * np.log(4.0)
        for m in range(1, n+1):
            log_val -= np.log(m)
        _I_TABLE[n] = np.exp(log_val)

_init_I_table()

# Precompute S_k matrices for m_k = p^T S_k p
# S_k[i,j] = I[k+i+j]
_S_M = [np.zeros((7,7)) for _ in range(3)]
_S_N = [np.zeros((7,7)) for _ in range(3)]
for k in range(3):
    for i in range(7):
        for j in range(7):
            idx = k + i + j
            if idx < 14:
                val = _I_TABLE[idx]
                _S_M[k][i,j] = val
                # For N, q_i = p_i (-1)^i, so q_i q_j = p_i p_j (-1)^{i+j}
                _S_N[k][i,j] = 2.0 * np.pi * val * ((-1.0)**(i+j))

# Precompute dp/da coefficients
# p_k depends on a0, a4, a8, a12
# a12 = -(a0 + 3a4 + 105a8)/945
# dp_k/da_c = coeff_k_c + coeff_k_12 * da12_dc
# coeff_k_c is the coefficient of a_c in the expansion of H_k
# H0 = 1
# H4 = x^4 - 6x^2 + 3
# H8 = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
# H12 = x^12 - 66x^10 + 1155x^8 - 6930x^6 + 13860x^4 - 7560x^2 + 945

# p0 = a0 + 3a4 + 105a8 + 945a12
# p1 = -6a4 - 420a8 - 7560a12
# p2 = a4 + 210a8 + 13860a12
# p3 = -28a8 - 6930a12
# p4 = a8 + 1155a12
# p5 = -66a12
# p6 = a12

# Let's store the derivative of p_k with respect to a_c in a 3x7 array?
# Or just compute dp vector inside the function, it's cheap.
# But we can precompute the matrix D where D[c, k] = dp_k / da_c
_D_P = np.zeros((3, 7))
# c=0 (a0)
_D_P[0, 0] = 1.0 + 945.0 * (-1.0/945.0) # 1 - 1 = 0
_D_P[0, 1] = 0.0 - 7560.0 * (-1.0/945.0) # 7560/945 = 8
_D_P[0, 2] = 0.0 + 13860.0 * (-1.0/945.0) # -13860/945 = -14.666? 13860/945 = 14.666. 945*14=13230, 945*15=14175.
# Let's just calculate numerically to be safe or use fractions if needed.
# 945 = 5 * 189 = 5 * 9 * 21 = 45 * 21 = 945.
# 13860 / 945: 13860 / 5 = 2772. 945 / 5 = 189. 2772 / 189 = 14.66...
# Actually, it's easier to just compute dp on the fly as it's only 7 elements.
# The bottleneck was the loops over I.

def _bound_and_grad(params):
    a0, a4, a8 = params[0], params[1], params[2]
    a12 = -(a0 + 3.0*a4 + 105.0*a8) / 945.0
    
    p = np.zeros(7)
    p[0] = a0 + 3.0*a4 + 105.0*a8 + 945.0*a12
    p[1] = -6.0*a4 - 420.0*a8 - 7560.0*a12
    p[2] = a4 + 210.0*a8 + 13860.0*a12
    p[3] = -28.0*a8 - 6930.0*a12
    p[4] = a8 + 1155.0*a12
    p[5] = -66.0*a12
    p[6] = a12
    
    q = p * np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
    
    # m_k = p^T S_M[k] p
    m0 = np.dot(p, _S_M[0].dot(p))
    m2 = np.dot(p, _S_M[1].dot(p))
    m4 = np.dot(p, _S_M[2].dot(p))
    
    n0 = np.dot(q, _S_N[0].dot(q))
    n2 = np.dot(q, _S_N[1].dot(q))
    n4 = np.dot(q, _S_N[2].dot(q))
    
    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    bound = np.sqrt(vf * vF)
    
    # Gradients
    # dm_k/da_c = 2 * (S_M[k] p) dot (dp/da_c)
    # dvn2/da_c = 2 * (S_N[k] q) dot (dq/da_c)
    
    # Precompute S_M[k] p and S_N[k] q
    Sp = [np.dot(_S_M[k], p) for k in range(3)]
    Sq = [np.dot(_S_N[k], q) for k in range(3)]
    
    grad = np.zeros(3)
    for c in range(3):
        # Compute dp/da_c
        dp = np.zeros(7)
        # da12/dc
        da12_dc = np.array([-1.0/945.0, -3.0/945.0, -105.0/945.0])[c]
        
        # Coeffs of a_c in H_k (k=0,4,8,12 mapped to p0..p6)
        # H0: [1, 0, 0, 0, 0, 0, 0]
        # H4: [3, -6, 1, 0, 0, 0, 0]
        # H8: [105, -420, 210, -28, 1, 0, 0]
        # H12: [945, -7560, 13860, -6930, 1155, -66, 1]
        
        if c == 0: # a0
            base = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        elif c == 1: # a4
            base = np.array([3.0, -6.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        else: # a8
            base = np.array([105.0, -420.0, 210.0, -28.0, 1.0, 0.0, 0.0])
            
        h12_coeff = np.array([945.0, -7560.0, 13860.0, -6930.0, 1155.0, -66.0, 1.0])
        dp = base + da12_dc * h12_coeff
        
        dq = dp * np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
        
        dm0 = 2.0 * np.dot(Sp[0], dp)
        dm2 = 2.0 * np.dot(Sp[1], dp)
        dn0 = 2.0 * np.dot(Sq[0], dq)
        dn2 = 2.0 * np.dot(Sq[1], dq)
        
        dvm2 = (dm2*m0 - m2*dm0) / (m0*m0)
        dvn2 = (dn2*n0 - n2*dn0) / (n0*n0)
        
        grad[c] = (vF * dvm2 + vf * dvn2) / (2.0 * bound)
        
    return bound, grad

def _objective(params):
    val, _ = _bound_and_grad(params)
    return val

def _objective_grad(params):
    val, grad = _bound_and_grad(params)
    return val, grad

def construct():
    starts = [
        np.array([-1.13, 0.04, 0.000308642]),
        np.array([0.0, 0.0, 0.0]),
        np.array([-1.0, 0.1, 0.001]),
        np.array([-2.0, 0.05, 0.0005]),
        np.array([-0.5, 0.02, 0.0002]),
        np.array([-1.5, 0.08, 0.0008]),
        np.array([1.0, -0.05, 0.0005]),
        np.array([-3.0, 0.15, 0.002]),
    ]
    
    rng = np.random.default_rng(123)
    for i in range(30):
        starts.append(rng.normal(0, 1.0, 3))
        
    best_val = np.inf
    best_params = None
    
    for start in starts:
        if time.time() > _DEADLINE - 20:
            break
        try:
            res = minimize(_objective_grad, start, method='L-BFGS-B',
                          jac=True, options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-14})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    if best_params is not None and time.time() < _DEADLINE - 10:
        try:
            res = minimize(_objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 50000, 'xatol': 1e-16, 'fatol': 1e-18})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    if best_params is not None and time.time() < _DEADLINE - 5:
        try:
            res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 10000, 'ftol': 1e-16, 'gtol': 1e-15})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    if best_params is None:
        return np.array([-1.13, 0.04, 0.000308642])
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
