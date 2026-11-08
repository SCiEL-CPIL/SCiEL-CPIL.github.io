import numpy as np
from scipy.optimize import minimize
import time

# Hermite polynomial coefficients (physicists' convention), stored as [x^0, x^2, x^4, ...]
H0_coeffs = np.array([1.0])
H4_coeffs = np.array([12.0, -48.0, 16.0])
H8_coeffs = np.array([3360.0, -13440.0, 11520.0, -3072.0, 256.0])
H12_coeffs = np.array([1039500.0, -6048000.0, 8064000.0, -3932160.0, 846720.0, -98304.0, 4096.0])

# Precompute integrals
def _precompute_integrals():
    I = np.zeros(13)
    for n in range(13):
        if n == 0:
            I[n] = np.sqrt(np.pi)
        else:
            result = 1.0
            for k in range(n):
                result *= (2*k + 1)
            I[n] = result * np.sqrt(np.pi) / (2.0 ** n)
    
    J = np.zeros(25)
    for n in range(25):
        if n == 0:
            J[n] = np.sqrt(np.pi / 2.0)
        else:
            result = 1.0
            for k in range(n):
                result *= (2*k + 1)
            J[n] = result * np.sqrt(np.pi) / (2.0 ** (2*n + 0.5))
    
    return I, J

I_int, J_int = _precompute_integrals()

def _compute_P(c0, c1, c2):
    c3 = -(c0 + 12*c1 + 3360*c2) / 1039500.0
    P = np.zeros(7)
    P[0] += c0 * H0_coeffs[0]
    P[0] += c1 * H4_coeffs[0]
    P[1] += c1 * H4_coeffs[1]
    P[2] += c1 * H4_coeffs[2]
    for i in range(5):
        P[i] += c2 * H8_coeffs[i]
    for i in range(7):
        P[i] += c3 * H12_coeffs[i]
    return P

def _objective(c0, c1, c2):
    P = _compute_P(c0, c1, c2)
    n = len(P)
    
    # L2^2 = P^T M P where M[i,j] = I(i+j)
    M = np.array([[I_int[i+j] for j in range(n)] for i in range(n)])
    L2_sq = float(P @ M @ P)
    
    if L2_sq <= 1e-30:
        return np.inf
    
    # L4 4th moment = (P^4)^T J (P^4) where P^4 is the 4-fold convolution
    # Q[m] = sum_{i+j=m} P[i]*P[j] (2nd moment of |P|^2)
    # Then 4th moment = sum_{a+b+c+d} P[a]P[b]P[c]P[d] * J(a+b+c+d)
    # = sum_{m,n} Q[m]*Q[n]*J[m+n]
    Q = np.zeros(13)
    for i in range(7):
        for j in range(7):
            Q[i+j] += P[i] * P[j]
    
    L4_4th = 0.0
    for m in range(13):
        for nn in range(13):
            L4_4th += Q[m] * Q[nn] * J_int[m+nn]
    
    if L4_4th <= 1e-30:
        return np.inf
    
    C4 = np.sqrt(L4_4th) / L2_sq
    return C4

def _grad(c0, c1, c2):
    """Compute gradient of C4 with respect to (c0, c1, c2) using chain rule."""
    P = _compute_P(c0, c1, c2)
    n = len(P)
    
    M = np.array([[I_int[i+j] for j in range(n)] for i in range(n)])
    L2_sq = float(P @ M @ P)
    
    Q = np.zeros(13)
    for i in range(7):
        for j in range(7):
            Q[i+j] += P[i] * P[j]
    
    L4_4th = 0.0
    for m in range(13):
        for nn in range(13):
            L4_4th += Q[m] * Q[nn] * J_int[m+nn]
    
    if L2_sq <= 1e-30 or L4_4th <= 1e-30:
        return np.array([np.inf, np.inf, np.inf])
    
    C4 = np.sqrt(L4_4th) / L2_sq
    sqrt_L4 = np.sqrt(L4_4th)
    
    # dP/dc_k for k = 0,1,2
    dP = np.zeros((3, 7))
    # dc3/dc0 = -1/1039500
    dP[0, 0] += 1.0
    for i in range(7):
        dP[0, i] += (-1.0/1039500.0) * H12_coeffs[i]
    
    # dc3/dc1 = -12/1039500
    for i in range(3):
        dP[1, i] += H4_coeffs[i]
    for i in range(7):
        dP[1, i] += (-12.0/1039500.0) * H12_coeffs[i]
    
    # dc3/dc2 = -3360/1039500
    for i in range(5):
        dP[2, i] += H8_coeffs[i]
    for i in range(7):
        dP[2, i] += (-3360.0/1039500.0) * H12_coeffs[i]
    
    # dL2_sq/dc_k = 2 * P^T M dP_k
    # dQ/dc_k[m] = sum_{i+j=m} 2 * dP_k[i] * P[j]
    # dL4_4th/dc_k = sum_{m,n} 2*dQ_k[m]*Q[n]*J[m+n] + sum_{m,n} Q[m]*2*dQ_k[n]*J[m+n]
    #               = 4 * sum_{m,n} dQ_k[m] * Q[n] * J[m+n]
    
    grad = np.zeros(3)
    for k in range(3):
        dL2 = 2.0 * float(P @ M @ dP[k])
        
        dQ = np.zeros(13)
        for i in range(7):
            for j in range(7):
                dQ[i+j] += 2.0 * dP[k, i] * P[j]
        
        dL4 = 0.0
        for m in range(13):
            for nn in range(13):
                dL4 += 4.0 * dQ[m] * Q[nn] * J_int[m+nn]
        
        grad[k] = (0.5 * dL4 / sqrt_L4 - dL2 / L2_sq) * C4
    
    return grad

def construct():
    start_time = time.time()
    deadline = start_time + 195.0
    
    # Scale invariance: fix c0 = 1, optimize (c1, c2)
    def _obj_2d(params):
        c1, c2 = params
        return _objective(1.0, c1, c2)
    
    def _grad_2d(params):
        c1, c2 = params
        g = _grad(1.0, c1, c2)
        return np.array([g[1], g[2]])
    
    # Initial best from known good point
    c0_init = -113.0/100.0
    c1_init = 1.0/25.0
    c2_init = 1.0/3240.0
    best_val = _objective(c0_init, c1_init, c2_init)
    best_2d = np.array([c1_init/c0_init, c2_init/c0_init])
    
    # Phase 1: Multi-start L-BFGS-B from diverse points
    rng = np.random.RandomState(123)
    
    # Generate diverse starting points
    starts = [best_2d.copy()]
    
    # Grid search
    grid_range = np.linspace(-8.0, 8.0, 17)
    for a in grid_range:
        for b in grid_range:
            starts.append(np.array([a, b]))
    
    # Random points in larger space
    for _ in range(200):
        starts.append(rng.uniform(-15, 15, 2))
    
    # Evaluate all starts and keep top candidates
    start_results = []
    for s in starts:
        if time.time() > deadline - 30.0:
            break
        try:
            v = _obj_2d(s)
            start_results.append((v, s))
        except Exception:
            start_results.append((np.inf, s))
    
    start_results.sort(key=lambda x: x[0])
    
    # Update best
    for v, s in start_results:
        if v < best_val:
            best_val = v
            best_2d = s.copy()
    
    # Phase 2: L-BFGS-B from top starting points
    for v, s in start_results[:50]:
        if time.time() > deadline - 10.0:
            break
        try:
            result = minimize(
                _obj_2d,
                s,
                jac=_grad_2d,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'gtol': 1e-15, 'ftol': 1e-15}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_2d = result.x.copy()
        except Exception:
            pass
    
    # Phase 3: Newton-CG from best point
    if time.time() < deadline - 15.0:
        try:
            # Need Hessian for Newton-CG; use finite difference
            def _hess_2d(params):
                c1, c2 = params
                eps = 1e-6
                hess = np.zeros((2, 2))
                for i in range(2):
                    for j in range(2):
                        pp = params.copy()
                        pm = params.copy()
                        pp[i] += eps
                        pm[i] -= eps
                        if j != i:
                            pp2 = params.copy()
                            pm2 = params.copy()
                            pp2[i] += eps
                            pp2[j] += eps
                            pm2[i] -= eps
                            pm2[j] -= eps
                            hess[i, j] = (_obj_2d(pp2) - _obj_2d(pp) - _obj_2d(pm) + _obj_2d(params) / 1.0) / (4*eps*eps) if False else (
                                (_obj_2d(pp2) - _obj_2d(pp) - _obj_2d(pm) + _obj_2d(params)) / (4*eps*eps)
                            )
                        else:
                            hess[i, j] = (_obj_2d(pp) - 2*_obj_2d(params) + _obj_2d(pm)) / (eps*eps)
                return hess
            
            result = minimize(
                _obj_2d,
                best_2d,
                jac=_grad_2d,
                hess=_hess_2d,
                method='Newton-CG',
                options={'maxiter': 500, 'gtol': 1e-15}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_2d = result.x.copy()
        except Exception:
            pass
    
    # Final polish with very tight tolerance
    if time.time() < deadline - 5.0:
        try:
            result = minimize(
                _obj_2d,
                best_2d,
                jac=_grad_2d,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'gtol': 1e-18, 'ftol': 1e-18}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_2d = result.x.copy()
        except Exception:
            pass
    
    c0 = 1.0
    c1 = best_2d[0]
    c2 = best_2d[1]
    
    return np.array([c0, c1, c2])
