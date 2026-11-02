import numpy as np
import math
import time

def _fact(n):
    if n < 0:
        return 0.0
    return float(math.factorial(n))

def _build_moments(degrees):
    """Precompute M matrix for <H_k x^2 H_l> and D vector for Ideriv."""
    n = len(degrees)
    M = np.zeros((n, n))
    D = np.zeros(n)
    for i, k in enumerate(degrees):
        D[i] = 4.0 * k * k * _fact(k - 1) if k > 0 else 0.0
        for j, l in enumerate(degrees):
            v = 0.0
            if k + 2 == l:
                v += 0.5 * _fact(k + 2)
            if k == l:
                v += k * _fact(k)
            if k >= 2 and k - 2 == l:
                v += 0.5 * k * (k - 1) * _fact(k - 2)
            M[i, j] = v
    return M, D

def _eval_vec(c_vec, M, D, F):
    """Given full coefficient vector (len 4), compute (I0, Ix2, Ideriv)."""
    I0 = float(c_vec @ F @ c_vec)
    Ix2 = float(c_vec @ M @ c_vec)
    Ideriv = float(np.sum(c_vec * c_vec * D))
    return I0, Ix2, Ideriv

def _bound_from(I0, Ix2, Ideriv):
    if I0 <= 0 or Ix2 <= 0 or Ideriv <= 0:
        return 1e10
    Af2 = Ix2 / I0
    Ag2 = Ideriv / (4.0 * math.pi * math.pi * I0)
    return math.sqrt(Af2 * Ag2)

def _grad_bound(I0, Ix2, Ideriv, dI0, dIx2, dId):
    val = _bound_from(I0, Ix2, Ideriv)
    if val >= 1e9:
        return val, np.zeros(3)
    s = math.sqrt(Ix2 * Ideriv)
    # d(sqrt(Ix2*Id))/dc = (Id*dIx2 + Ix2*dId)/(2*s)
    ds = (Id * dIx2 + Ix2 * dId) / (2.0 * s)
    # f = sqrt(Ix2*Id)/(2*pi*I0)
    # df = (1/(2*pi)) * (ds/I0 - s*dI0/I0^2)
    g = (1.0 / (2.0 * math.pi)) * (ds / I0 - s * dI0 / (I0 * I0))
    return val, g

def _objective_and_grad(c, M, D, F, A, F12, D12, M12row, c12base, c12coeff):
    """c is length-3 vector (c0,c4,c8). Returns (val, grad)."""
    c12 = c12base + float(c @ c12coeff)
    c_full = np.empty(4)
    c_full[:3] = c
    c_full[3] = c12
    
    I0, Ix2, Ideriv = _eval_vec(c_full, M, D, F)
    if I0 <= 0 or Ix2 <= 0 or Ideriv <= 0:
        return 1e10, np.zeros(3)
    
    # Gradients with respect to c0,c4,c8
    # dI0/dc_i = 2*(F[i,:] @ c_full) + 2*F12[i]*dc12/dc_i
    # dIx2/dc_i = 2*(M[i,:] @ c_full) + 2*M12row[i]*dc12/dc_i  (M symmetric)
    # dId/dc_i = 2*D[i]*c[i] + 2*D12*c12*dc12/dc_i
    
    dI0_base = 2.0 * (F[:3, :] @ c_full)
    dIx2_base = 2.0 * (M[:3, :] @ c_full)
    dId_base = 2.0 * (D[:3] * c_full[:3])
    
    dI0 = dI0_base + 2.0 * F12 * c12coeff
    dIx2 = dIx2_base + 2.0 * M12row * c12coeff
    dId = dId_base + 2.0 * D12 * c12 * c12coeff
    
    val, grad = _grad_bound(I0, Ix2, Ideriv, dI0, dIx2, dId)
    return val, grad

def construct():
    t0 = time.time()
    DL = 195.0  # deadline
    
    degrees = [0, 4, 8, 12]
    M, D = _build_moments(degrees)
    F = np.diag([_fact(d) for d in degrees])
    
    # c12 = c12base + c12coeff @ c  where c = (c0,c4,c8)
    # Constraint: g(0) = 0 => c0 + 3*c4 + 105*c8 + 10395*c12 = 0
    # c12 = -(c0 + 3*c4 + 105*c8)/10395
    c12base = 0.0
    c12coeff = np.array([-1.0/10395.0, -3.0/10395.0, -105.0/10395.0])
    
    F12 = np.array([F[i, 3] for i in range(3)])
    D12 = D[3]
    M12row = np.array([M[i, 3] for i in range(3)])
    
    def obj(c):
        v, _ = _objective_and_grad(c, M, D, F, None, F12, D12, M12row, c12base, c12coeff)
        return v
    
    def objg(c):
        return _objective_and_grad(c, M, D, F, None, F12, D12, M12row, c12base, c12coeff)
    
    # Seed point from Goncalves et al.
    x0 = np.array([-1.13, 0.04, 1.0/3240.0])
    best_val, _ = objg(x0)
    best_c = x0.copy()
    
    rng = np.random.RandomState(2024)
    
    # Phase 1: Dense grid search in 2D (fix c8, vary c0,c4) + 1D sweep of c8
    if time.time() - t0 < DL - 60:
        # Sweep c8 over a range
        for c8 in np.linspace(-0.5, 0.5, 40):
            for c0 in np.linspace(-3.0, 3.0, 60):
                for c4 in np.linspace(-0.5, 0.5, 40):
                    c = np.array([c0, c4, c8])
                    v, _ = objg(c)
                    if v < best_val:
                        best_val = v
                        best_c = c
            if time.time() - t0 > DL - 60:
                break
    
    # Phase 2: L-BFGS-B from best and several random starts
    starts = [best_c, x0]
    for _ in range(20):
        starts.append(best_c + rng.normal(0, 0.2, 3))
    for _ in range(10):
        starts.append(x0 + rng.normal(0, 0.3, 3))
    
    import scipy.optimize as opt
    for c0 in starts:
        if time.time() - t0 > DL - 20:
            break
        try:
            res = opt.minimize(obj, c0, jac=lambda c: objg(c)[1], method='L-BFGS-B',
                               options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12})
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val = res.fun
                best_c = res.x.copy()
        except:
            pass
    
    # Phase 3: Nelder-Mead polish
    if time.time() - t0 < DL - 5:
        try:
            res = opt.minimize(obj, best_c, method='Nelder-Mead',
                               options={'xatol': 1e-13, 'fatol': 1e-15, 'maxiter': 1000, 'adaptive': True})
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val = res.fun
                best_c = res.x.copy()
        except:
            pass
    
    # Phase 4: Final L-BFGS-B polish
    if time.time() - t0 < DL - 2:
        try:
            res = opt.minimize(obj, best_c, jac=lambda c: objg(c)[1], method='L-BFGS-B',
                               options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-14})
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val = res.fun
                best_c = res.x.copy()
        except:
            pass
    
    return best_c
