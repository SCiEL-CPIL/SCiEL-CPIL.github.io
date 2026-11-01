import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar, differential_evolution

START_TIME = time.time()
TIME_LIMIT = 205.0

# Hermite polynomial coefficients in power basis [x^0, x^2, x^4, x^6, x^8, x^10, x^12]
_H0 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
_H4 = np.array([3.0, -6.0, 1.0, 0.0, 0.0, 0.0, 0.0])
_H8 = np.array([105.0, -420.0, 210.0, 0.0, 1.0, 0.0, 0.0])
_H12 = np.array([945.0, -226800.0, 151200.0, -30240.0, 2520.0, -84.0, 1.0])

# Precompute integral table: I[n] = integral x^{2n} exp(-x^2) dx from -inf to inf
_I_table = np.zeros(30)
_I_table[0] = np.sqrt(np.pi)
for _n in range(1, 30):
    _I_table[_n] = _I_table[_n-1] * (2*_n - 1) / 2.0

# Precompute dP/dc vectors (constant)
_dP_dc0 = _H0 - (1.0/945.0)*_H12
_dP_dc4 = _H4 - (3.0/945.0)*_H12
_dP_dc8 = _H8 - (105.0/945.0)*_H12
_dP_dc = np.stack([_dP_dc0, _dP_dc4, _dP_dc8])  # shape (3, 7)

# Precompute dR/dP structure
_dR_dP = np.zeros((7, 7))
for i in range(6):
    _dR_dP[i, i] = -1.0
    _dR_dP[i, i+1] = 2.0*(i+1)
_dR_dP[6, 6] = -1.0

def _get_P(c0, c4, c8):
    c12 = -(c0 + 3.0*c4 + 105.0*c8) / 945.0
    P = c0*_H0 + c4*_H4 + c8*_H8 + c12*_H12
    return P

def _compute_bound(c0, c4, c8):
    P = _get_P(c0, c4, c8)
    Q = np.zeros(13)
    for i in range(7):
        for j in range(7):
            Q[i+j] += P[i] * P[j]
    norm_sq = np.dot(Q, _I_table[:13])
    x_norm_sq = np.dot(Q, _I_table[1:14])
    R = np.zeros(7)
    for k in range(6):
        R[k] = 2*(k+1)*P[k+1] - P[k]
    R[6] = -P[6]
    S = np.zeros(14)
    for i in range(7):
        for j in range(7):
            S[i+j] += R[i] * R[j]
    deriv_norm_sq = np.dot(S, _I_table[:14])
    if norm_sq <= 0:
        return 0.0
    bound = np.sqrt(x_norm_sq * deriv_norm_sq) / norm_sq
    return bound

def _compute_bound_grad(c0, c4, c8):
    P = _get_P(c0, c4, c8)
    Q = np.zeros(13)
    for i in range(7):
        Q[i:i+7] += P[i] * P[:7-i]
    norm_sq = np.dot(Q, _I_table[:13])
    x_norm_sq = np.dot(Q, _I_table[1:14])
    R = np.zeros(7)
    for k in range(6):
        R[k] = 2*(k+1)*P[k+1] - P[k]
    R[6] = -P[6]
    S = np.zeros(14)
    for i in range(7):
        S[i:i+7] += R[i] * R[:7-i]
    deriv_norm_sq = np.dot(S, _I_table[:14])
    if norm_sq <= 0:
        return 0.0, np.zeros(3)
    bound = np.sqrt(x_norm_sq * deriv_norm_sq) / norm_sq
    if bound <= 0:
        return bound, np.zeros(3)
    d_norm_dP = np.array([2.0 * np.dot(P, _I_table[j:j+7]) for j in range(7)])
    d_xnorm_dP = np.array([2.0 * np.dot(P, _I_table[j+1:j+8]) for j in range(7)])
    d_deriv_dP = np.zeros(7)
    for j in range(7):
        s = 0.0
        for i in range(7):
            s += R[i] * _dR_dP[i, j] * _I_table[i+j]
        d_deriv_dP[j] = 2.0 * s
    dA_dc = np.array([np.dot(d_xnorm_dP, _dP_dc[k]) for k in range(3)])
    dB_dc = np.array([np.dot(d_deriv_dP, _dP_dc[k]) for k in range(3)])
    dC_dc = np.array([np.dot(d_norm_dP, _dP_dc[k]) for k in range(3)])
    A = x_norm_sq
    B = deriv_norm_sq
    C = norm_sq
    bound_sq = A * B / (C * C)
    if bound_sq <= 0:
        return bound, np.zeros(3)
    factor = 0.5 / bound
    AB_over_C2 = A * B / (C * C)
    d_bound_sq_dc = (B*dA_dc + A*dB_dc - 2.0*AB_over_C2*dC_dc) / C
    grad = factor * d_bound_sq_dc
    return bound, grad

def _objective(coeffs):
    return _compute_bound(coeffs[0], coeffs[1], coeffs[2])

def _objective_grad(coeffs):
    return _compute_bound_grad(coeffs[0], coeffs[1], coeffs[2])

def _newton_step(x, max_iter=50):
    """Newton's method with exact Hessian via central differences of gradient."""
    x = x.copy()
    for it in range(max_iter):
        if time.time() > TIME_LIMIT - 10:
            break
        f, g = _objective_grad(x)
        if np.linalg.norm(g) < 1e-14:
            break
        # Compute Hessian via central differences
        h = 1e-7
        H = np.zeros((3, 3))
        for i in range(3):
            xp = x.copy(); xp[i] += h
            xm = x.copy(); xm[i] -= h
            fp, _ = _objective_grad(xp)
            fm, _ = _objective_grad(xm)
            for j in range(3):
                H[i, j] = (fp[j] - fm[j]) / (2*h)
        # Damped Newton step
        try:
            step = np.linalg.solve(H, -g)
        except np.linalg.LinAlgError:
            step = -g / np.linalg.norm(g)
        # Line search
        alpha = 1.0
        for _ in range(50):
            x_new = x + alpha * step
            f_new = _objective(x_new)
            if f_new < f:
                break
            alpha *= 0.5
        x = x_new
    return x, _objective(x)

def construct():
    seed = np.array([-1.13, 0.04, 1.0/3240.0])
    best_score = _objective(seed)
    best_coeffs = seed.copy()
    
    np.random.seed(42)
    bounds = [(-10.0, 10.0), (-5.0, 5.0), (-2.0, 2.0)]
    
    # Phase 1: Differential evolution for global exploration
    if time.time() < TIME_LIMIT - 60:
        try:
            de_res = differential_evolution(
                _objective, bounds, seed=42, maxiter=200, popsize=15,
                tol=1e-12, polish=True, init='sobol'
            )
            if de_res.fun < best_score:
                best_score = de_res.fun
                best_coeffs = de_res.x.copy()
        except Exception:
            pass
    
    # Phase 2: Multi-start L-BFGS-B with diverse initializations
    num_starts = 80
    for i in range(num_starts):
        if time.time() > TIME_LIMIT - 40:
            break
        
        if i == 0:
            x0 = seed.copy()
        elif i == 1:
            x0 = np.array([0.0, 0.0, 0.0])
        elif i == 2:
            x0 = np.array([1.0, -0.1, 0.001])
        elif i == 3:
            x0 = np.array([-1.0, 0.1, -0.001])
        elif i == 4:
            x0 = np.array([0.5, 0.0, 0.0])
        elif i == 5:
            x0 = np.array([-0.5, 0.0, 0.0])
        elif i == 6:
            x0 = np.array([0.0, 0.5, 0.0])
        elif i == 7:
            x0 = np.array([0.0, -0.5, 0.0])
        elif i == 8:
            x0 = np.array([2.0, -0.2, 0.002])
        elif i == 9:
            x0 = np.array([-2.0, 0.2, -0.002])
        else:
            if i % 3 == 0:
                x0 = np.array([
                    np.random.uniform(bounds[0][0], bounds[0][1]),
                    np.random.uniform(bounds[1][0], bounds[1][1]),
                    np.random.uniform(bounds[2][0], bounds[2][1])
                ])
            elif i % 3 == 1:
                x0 = seed + np.random.normal(0, 0.5, size=3)
                x0 = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])
            else:
                x0 = seed + np.random.normal(0, 1.0, size=3)
                x0 = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])
        
        try:
            res = minimize(_objective_grad, x0, method='L-BFGS-B',
                           jac=True, bounds=bounds,
                           options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12})
            if res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    # Phase 3: Newton's method refinement (superlinear convergence)
    if time.time() < TIME_LIMIT - 20:
        x_newton, f_newton = _newton_step(best_coeffs, max_iter=30)
        if f_newton < best_score:
            best_score = f_newton
            best_coeffs = x_newton.copy()
    
    # Phase 4: Local refinement around best solution
    for i in range(30):
        if time.time() > TIME_LIMIT - 15:
            break
        sigma = 0.1 * (1.0 - i/30.0)**2 + 0.0005
        x0 = best_coeffs + np.random.normal(0, sigma, size=3)
        x0 = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])
        try:
            res = minimize(_objective_grad, x0, method='L-BFGS-B',
                           jac=True, bounds=bounds,
                           options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-13})
            if res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    # Phase 5: Newton polish
    if time.time() < TIME_LIMIT - 10:
        x_newton, f_newton = _newton_step(best_coeffs, max_iter=30)
        if f_newton < best_score:
            best_score = f_newton
            best_coeffs = x_newton.copy()
    
    # Phase 6: 1D line searches for final polish
    if time.time() < TIME_LIMIT - 5:
        for coord in range(3):
            if time.time() > TIME_LIMIT - 3:
                break
            def line_obj(t, coord=coord):
                x = best_coeffs.copy()
                x[coord] = t
                return _objective(x)
            try:
                lo = max(bounds[coord][0], best_coeffs[coord] - 1.0)
                hi = min(bounds[coord][1], best_coeffs[coord] + 1.0)
                res = minimize_scalar(line_obj, bounds=(lo, hi),
                                      method='bounded', options={'xatol': 1e-14})
                if res.fun < best_score:
                    new_x = best_coeffs.copy()
                    new_x[coord] = res.x
                    best_coeffs = new_x
                    best_score = _objective(best_coeffs)
            except Exception:
                pass
    
    # Phase 7: Final high-precision L-BFGS-B
    if time.time() < TIME_LIMIT - 3:
        try:
            res = minimize(_objective_grad, best_coeffs, method='L-BFGS-B',
                           jac=True, bounds=bounds,
                           options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-15})
            if res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    # Phase 8: Final Newton polish
    if time.time() < TIME_LIMIT - 1:
        x_newton, f_newton = _newton_step(best_coeffs, max_iter=20)
        if f_newton < best_score:
            best_score = f_newton
            best_coeffs = x_newton.copy()
    
    return best_coeffs
