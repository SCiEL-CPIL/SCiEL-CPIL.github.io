"""Analytic uncertainty bound with multi-start L-BFGS-B + stationary-point root-finding + Newton polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, root

_DEADLINE = time.time() + 200
_START_TIME = time.time()

# Precompute I_n = integral x^{2n} e^{-x^2} dx = (2n)! / (4^n n!) * sqrt(pi)
def _precompute_I():
    I = np.zeros(14)
    sp = np.sqrt(np.pi)
    I[0] = sp
    for n in range(1, 14):
        I[n] = (2*n - 1) * I[n-1] / (4.0 * n)
    return I

_I = _precompute_I()

# Hermite polynomial coefficients in power basis x^{2k}
# H0 = 1
# H4 = x^4 - 6x^2 + 3
# H8 = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
# H12 = x^12 - 66x^10 + 1155x^8 - 6930x^6 + 13860x^4 - 7560x^2 + 945
H0_c = np.array([1, 0, 0, 0, 0, 0, 0])
H4_c = np.array([3, -6, 1, 0, 0, 0, 0])
H8_c = np.array([105, -420, 210, -28, 1, 0, 0])
H12_c = np.array([945, -7560, 13860, -6930, 1155, -66, 1])

# a12 = -(a0 + 3*a4 + 105*a8) / 945
c12 = np.array([-1.0/945.0, -3.0/945.0, -105.0/945.0])

# p = a0*(H0_c + c12_0*H12_c) + a4*(H4_c + c12_1*H12_c) + a8*(H8_c + c12_2*H12_c)
col0 = H0_c + c12[0]*H12_c
col1 = H4_c + c12[1]*H12_c
col2 = H8_c + c12[2]*H12_c
A_mat = np.column_stack((col0, col1, col2))  # Shape (7, 3)

def _get_p(a):
    return A_mat @ a

def _bound_and_grad(params):
    """
    Compute uncertainty product and its analytic gradient.
    g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
    a12 = -(a0 + 3*a4 + 105*a8)/945  (enforces g(0)=0)
    """
    a0, a4, a8 = params[0], params[1], params[2]
    a12 = -(a0 + 3.0*a4 + 105.0*a8) / 945.0
    
    p = np.empty(7)
    p[0] = 0.0
    p[1] = -6.0*a4 - 420.0*a8 - 7560.0*a12
    p[2] = a4 + 210.0*a8 + 13860.0*a12
    p[3] = -28.0*a8 - 6930.0*a12
    p[4] = a8 + 1155.0*a12
    p[5] = -66.0*a12
    p[6] = a12
    
    I = _I
    pp = np.outer(p, p)
    m = np.zeros(3)
    for k in range(3):
        idx_mat = k + np.add.outer(np.arange(7), np.arange(7))
        m[k] = np.sum(pp * I[idx_mat])
    
    signs = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
    q = p * signs
    qq = np.outer(q, q)
    n = np.zeros(3)
    for k in range(3):
        n[k] = 2.0 * np.pi * np.sum(qq * I[k + np.add.outer(np.arange(7), np.arange(7))])
    
    m0, m2 = m[0], m[1]
    n0, n2 = n[0], n[1]
    
    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    bound = np.sqrt(vf * vF)
    
    S_p = np.zeros((3, 7))
    S_q = np.zeros((3, 7))
    for k in range(3):
        for i in range(7):
            s_p = 0.0
            s_q = 0.0
            for j in range(7):
                idx = k + i + j
                if idx < 14:
                    s_p += p[j] * I[idx]
                    s_q += q[j] * I[idx]
            S_p[k, i] = s_p
            S_q[k, i] = s_q
    
    dpda = np.zeros((3, 7))
    dpda[0, 1] = 8.0
    dpda[1, 1] = -30.0
    dpda[2, 1] = -1260.0
    dpda[0, 2] = -13860.0/945.0
    dpda[1, 2] = 45.0
    dpda[2, 2] = 1750.0
    dpda[0, 3] = 6930.0/945.0
    dpda[1, 3] = -22.0
    dpda[2, 3] = -798.0
    dpda[0, 4] = -1155.0/945.0
    dpda[1, 4] = 3465.0/945.0
    dpda[2, 4] = 1 + 1155.0*105.0/945.0
    dpda[0, 5] = 66.0/945.0
    dpda[1, 5] = -198.0/945.0
    dpda[2, 5] = -6930.0/945.0
    dpda[0, 6] = -1.0/945.0
    dpda[1, 6] = -3.0/945.0
    dpda[2, 6] = -105.0/945.0
    
    grad = np.zeros(3)
    for c in range(3):
        dp = dpda[c]
        dm = 2.0 * (dp @ S_p)
        dq = dp * signs
        dn = 4.0 * np.pi * (dq @ S_q)
        dvm2 = (dm[1]*m0 - m2*dm[0]) / (m0*m0)
        dvn2 = (dn[1]*n0 - n2*dn[0]) / (n0*n0)
        grad[c] = (vF * dvm2 + vf * dvn2) / (2.0 * bound)
    
    return bound, grad

def _objective(params):
    val, _ = _bound_and_grad(params)
    return val

def _objective_grad(params):
    val, grad = _bound_and_grad(params)
    return val, grad

def _stationary_equations(a):
    """
    Returns the gradient of F = Vx * Vxi (equivalent to grad of bound up to scale).
    We want to find roots of this system.
    """
    p = _get_p(a)
    q = p * np.array([1, -1, 1, -1, 1, -1, 1])
    
    Mx = np.zeros(2)
    Mxi = np.zeros(2)
    
    for i in range(7):
        pi = p[i]
        if pi == 0: continue
        for j in range(7):
            Mx[0] += pi * p[j] * _I[i+j]
            Mx[1] += pi * p[j] * _I[i+j+1]
            Mxi[0] += q[i] * q[j] * _I[i+j]
            Mxi[1] += q[i] * q[j] * _I[i+j+1]
            
    if Mx[0] <= 1e-15 or Mxi[0] <= 1e-15:
        return np.array([1e18, 1e18, 1e18])
        
    vx = Mx[1] / Mx[0]
    vxi = Mxi[1] / Mxi[0]
    
    if vx <= 1e-15 or vxi <= 1e-15:
        return np.array([1e18, 1e18, 1e18])
        
    Sx = np.zeros((2, 7))
    Sxi = np.zeros((2, 7))
    
    for n in range(2):
        for i in range(7):
            Sx[n, i] = np.dot(p, _I[i+n:i+n+7])
            Sxi[n, i] = np.dot(q, _I[i+n:i+n+7])
            
    dMx = np.zeros((2, 3))
    dMxi = np.zeros((2, 3))
    
    for m in range(3):
        for n in range(2):
            dMx[n, m] = 2.0 * np.dot(A_mat[:, m], Sx[n, :])
            dMxi[n, m] = 2.0 * np.dot(A_mat[:, m], Sxi[n, :])
            
    dvx = np.zeros(3)
    dvxi = np.zeros(3)
    
    for m in range(3):
        dvx[m] = (dMx[1, m]*Mx[0] - Mx[1]*dMx[0, m]) / (Mx[0]*Mx[0])
        dvxi[m] = (dMxi[1, m]*Mxi[0] - Mxi[1]*dMxi[0, m]) / (Mxi[0]*Mxi[0])
        
    grad_F = vxi * dvx + vx * dvxi
    return grad_F

def _bound_hessian(params):
    """
    Compute the Hessian of the bound w.r.t. (a0, a4, a8) via finite differences of the gradient.
    Returns 3x3 Hessian matrix.
    """
    h = 1e-7
    H = np.zeros((3, 3))
    for i in range(3):
        for j in range(i, 3):
            ei = np.zeros(3)
            ej = np.zeros(3)
            ei[i] = h
            ej[j] = h
            if i == j:
                gp, _ = _bound_and_grad(params + ei)
                gm, _ = _bound_and_grad(params - ei)
                H[i, i] = (gp[i] - gm[i]) / (2*h)
            else:
                gp, _ = _bound_and_grad(params + ei + ej)
                gm1, _ = _bound_and_grad(params + ei - ej)
                gm2, _ = _bound_and_grad(params - ei + ej)
                gm3, _ = _bound_and_grad(params - ei - ej)
                H[i, j] = (gp[j] - gm1[j] - gm2[j] + gm3[j]) / (4*h)
                H[j, i] = H[i, j]
    return H

def _is_local_min(a):
    """
    Check if a is a local minimum by checking Hessian positive definiteness.
    """
    grad0, _ = _bound_and_grad(a)
    if np.any(np.abs(grad0) > 1e-4):
        return False
        
    hess = _bound_hessian(a)
    try:
        eigvals = np.linalg.eigvalsh(hess)
        return np.all(eigvals > -1e-6)
    except:
        return False

def _newton_polish(x, max_iter=50):
    """Newton-style polish using analytic Hessian."""
    best_val, _ = _bound_and_grad(x)
    for _ in range(max_iter):
        if time.time() > _DEADLINE - 5:
            break
        f, g = _bound_and_grad(x)
        if not np.isfinite(f) or np.linalg.norm(g) < 1e-14:
            break
        H = _bound_hessian(x)
        H_reg = H + 1e-10 * np.eye(3)
        try:
            step = np.linalg.solve(H_reg, -g)
        except np.linalg.LinAlgError:
            step = -g / max(np.linalg.norm(g)**2, 1e-12)
        alpha = 1.0
        f_new = f
        for _ in range(50):
            x_new = x + alpha * step
            f_new, _ = _bound_and_grad(x_new)
            if f_new < f:
                break
            alpha *= 0.5
        x = x + alpha * step
        if f_new < best_val and np.isfinite(f_new):
            best_val = f_new
    return x, best_val

def construct():
    seed_coeffs = np.array([-1.13, 0.04, 0.000308642])
    
    best_val = np.inf
    best_params = None
    
    # Generate diverse starting points
    rng = np.random.default_rng(42)
    starts = [seed_coeffs.copy()]
    
    # Grid-based starts in linear coordinates
    for a0 in np.linspace(-2, 1, 5):
        for a4 in np.linspace(-1, 1, 5):
            for a8 in np.linspace(-0.5, 0.5, 3):
                starts.append(np.array([a0, a4, a8]))
    
    # Random starts
    for _ in range(80):
        starts.append(rng.normal(0, 1.0, size=3))
    
    # Structured starts
    for scale in [0.01, 0.1, 1.0, 10.0]:
        for sign in [-1, 1]:
            starts.append(np.array([sign*scale, sign*scale*0.1, sign*scale*0.01]))
    
    # Phase 1: Multi-start L-BFGS-B
    for start in starts:
        if time.time() > _DEADLINE - 80:
            break
        try:
            res = minimize(_objective_grad, start, method='L-BFGS-B',
                          jac=True, options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-13, 'maxls': 30})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 2: Root-finding on stationary equations from best L-BFGS result and diverse starts
    if time.time() < _DEADLINE - 40:
        root_starts = [seed_coeffs.copy()]
        if best_params is not None:
            root_starts.append(best_params.copy())
        
        # Add diverse root-finding starts
        for _ in range(30):
            root_starts.append(rng.normal(0, 1.0, size=3))
        
        for start in root_starts:
            if time.time() > _DEADLINE - 20:
                break
            try:
                res = root(_stationary_equations, start, method='hybr', tol=1e-14, maxfev=500)
                if res.success and np.isfinite(res.fun):
                    candidate = res.x
                    # Verify it's a local minimum
                    if _is_local_min(candidate):
                        val, _ = _bound_and_grad(candidate)
                        if val < best_val and np.isfinite(val):
                            best_val = val
                            best_params = candidate.copy()
            except Exception:
                pass
    
    # Phase 3: Newton polish on best result
    if best_params is not None and time.time() < _DEADLINE - 10:
        try:
            polished, polished_val = _newton_polish(best_params)
            if polished_val < best_val and np.isfinite(polished_val):
                best_val = polished_val
                best_params = polished
        except Exception:
            pass
    
    # Phase 4: Final high-precision L-BFGS-B
    if best_params is not None and time.time() < _DEADLINE - 3:
        try:
            res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 30000, 'ftol': 1e-19, 'gtol': 1e-17, 'maxls': 50})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 5: Final Newton polish
    if best_params is not None and time.time() < _DEADLINE - 1:
        try:
            polished, polished_val = _newton_polish(best_params, max_iter=30)
            if polished_val < best_val and np.isfinite(polished_val):
                best_val = polished_val
                best_params = polished
        except Exception:
            pass
    
    if best_params is None:
        return np.array([-1.13, 0.04, 0.000308642])
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
