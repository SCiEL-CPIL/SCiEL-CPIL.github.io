import numpy as np
from scipy.optimize import minimize, differential_evolution
import time
import math

# Precompute matrices once at module load (they are constant)
_DEGREES = np.array([0, 4, 8, 12], dtype=np.int64)
_N = 4

def _build_matrices():
    idx = _DEGREES
    M_x = np.zeros((_N, _N))
    M_x2 = np.zeros((_N, _N))
    M_p = np.zeros((_N, _N), dtype=complex)
    M_p2 = np.zeros((_N, _N))

    for i in range(_N):
        for j in range(_N):
            n = idx[i]
            m = idx[j]
            if n == m + 1:
                M_x[i, j] = np.sqrt(m + 1) / np.sqrt(2)
            elif n == m - 1:
                M_x[i, j] = np.sqrt(m) / np.sqrt(2)

    for i in range(_N):
        for j in range(_N):
            n = idx[i]
            m = idx[j]
            if n == m + 2:
                M_x2[i, j] = 0.5 * np.sqrt((m + 1) * (m + 2))
            elif n == m:
                M_x2[i, j] = m + 0.5
            elif n == m - 2:
                M_x2[i, j] = 0.5 * np.sqrt(m * (m - 1))

    for i in range(_N):
        for j in range(_N):
            n = idx[i]
            m = idx[j]
            if n == m - 1:
                M_p[i, j] = 1j * np.sqrt(m) / np.sqrt(2)
            elif n == m + 1:
                M_p[i, j] = -1j * np.sqrt(m + 1) / np.sqrt(2)

    for i in range(_N):
        for j in range(_N):
            if i == j:
                M_p2[i, j] = 2.0 * idx[j] + 1.0
            M_p2[i, j] -= M_x2[i, j]

    return M_x, M_x2, M_p, M_p2

_M_x, _M_x2, _M_p, _M_p2 = _build_matrices()

# Precompute the weight scaling: w_k = c_k * sqrt(2^{d_k} d_k!)
_wscale = np.zeros(_N)
for k in range(_N):
    fact = math.factorial(int(_DEGREES[k]))
    _wscale[k] = np.sqrt((2.0 ** int(_DEGREES[k])) * fact)

# Constraint: g(0) = sum c_n H_n(0) e^0 = 0
# H_0(0)=1, H_4(0)=-12, H_8(0)=1680, H_12(0)=-3991680
# So: c0 - 12 c4 + 1680 c8 - 3991680 c12 = 0  =>  c12 = (c0 - 12 c4 + 1680 c8)/3991680
# Wait, check: H_4(0) = -12? Let's verify.
# Physicists' Hermite: H_0=1, H_1=2x, H_2=4x^2-2, H_3=8x^3-12x, H_4=16x^4-48x^2+12
# So H_4(0) = 12. H_8(0): H_8 = 256x^8 - 4480x^6 + 26880x^4 - 6720x^2 + 1680 => H_8(0)=1680
# H_12(0) = -3991680 (from standard tables)
# So g(0) = c0*1 + c4*12 + c8*1680 + c12*(-3991680) = 0
# => c12 = (c0 + 12*c4 + 1680*c8) / 3991680
_H0_0 = 1.0
_H4_0 = 12.0
_H8_0 = 1680.0
_H12_0 = -3991680.0

def _c12_from(c4, c8, c0=1.0):
    return (c0 + _H4_0 * c4 + _H8_0 * c8) / (-_H12_0)

def _weights_from_params(c4, c8, c0=1.0):
    c12 = _c12_from(c4, c8, c0)
    c = np.array([c0, c4, c8, c12])
    return c * _wscale

def _uncertainty_product(w):
    """Compute sqrt(var_x * var_p) given orthonormal-basis weights w."""
    norm_sq = float(np.dot(w, w))
    if norm_sq < 1e-15:
        return np.inf
    exp_x = float(np.dot(w, _M_x @ w))
    exp_x2 = float(np.dot(w, _M_x2 @ w))
    exp_p = float(np.dot(w, _M_p @ w).real)
    exp_p2 = float(np.dot(w, _M_p2 @ w))
    var_x = exp_x2 - exp_x * exp_x
    var_p = exp_p2 - exp_p * exp_p
    if var_x < 0: var_x = 0.0
    if var_p < 0: var_p = 0.0
    return math.sqrt(var_x * var_p)

def _objective_and_grad(params):
    """Objective and analytic gradient via finite-difference-free chain rule.
    Since w is linear in params, and the objective is a smooth function of w,
    we compute the gradient using the chain rule with precomputed matrix products."""
    c4, c8 = params
    w = _weights_from_params(c4, c8)
    
    exp_x = float(w @ _M_x @ w)
    exp_x2 = float(w @ _M_x2 @ w)
    exp_p = float((w @ _M_p @ w).real)
    exp_p2 = float(w @ _M_p2 @ w)
    
    var_x = max(exp_x2 - exp_x**2, 0.0)
    var_p = max(exp_p2 - exp_p**2, 0.0)
    
    if var_x < 1e-30 or var_p < 1e-30:
        return 1e10, np.zeros(2)
    
    obj = math.sqrt(var_x * var_p)
    
    # d(obj)/dw = (1/(2*sqrt(var_x*var_p))) * (d(var_x)/dw * var_p + var_x * d(var_p)/dw)
    # d(var_x)/dw = 2*M_x2*w - 2*exp_x*M_x*w  (since var_x = w^T M_x2 w - (w^T M_x w)^2)
    # d(var_p)/dw = 2*M_p2*w - 2*exp_p*M_p*w  (real part)
    # Actually d(var_p)/dw: var_p = w^T M_p2 w - (Re(w^T M_p w))^2
    # d(w^T M_p w)/dw = (M_p + M_p^H) w; since M_p is Hermitian, = 2 Re(M_p) w ... 
    # Let's be careful. M_p is complex Hermitian. w^T M_p w is complex.
    # exp_p = Re(w^T M_p w). d(exp_p)/dw = Re((M_p + M_p^T) w) = Re(2 Re(M_p) w) since M_p^H = M_p.
    # Actually M_p^H = M_p, so M_p^T = M_p^*. For real w, w^T M_p w = sum_{ij} w_i M_p[i,j] w_j.
    # d/dw_k = sum_j M_p[k,j] w_j + sum_i w_i M_p[i,k] = (M_p w)_k + (M_p^T w)_k = (M_p w)_k + (M_p^* w)_k = 2 Re((M_p w)_k)
    # So d(exp_p)/dw = 2 Re(M_p @ w)
    
    dvarx_dw = 2.0 * (_M_x2 @ w) - 2.0 * exp_x * (_M_x @ w)
    dvarp_dw = 2.0 * (_M_p2 @ w) - 2.0 * exp_p * (2.0 * np.real(_M_p @ w))
    
    # Wait, that's not right. Let me redo.
    # var_p = w^T M_p2 w - (Re(w^T M_p w))^2
    # d(var_p)/dw = 2 M_p2 w - 2 * Re(w^T M_p w) * d(Re(w^T M_p w))/dw
    # d(Re(w^T M_p w))/dw = 2 Re(M_p w)  [for real w]
    # So dvarp_dw = 2 M_p2 w - 2 exp_p * 2 Re(M_p w) = 2 M_p2 w - 4 exp_p Re(M_p w)
    dvarp_dw = 2.0 * (_M_p2 @ w) - 4.0 * exp_p * np.real(_M_p @ w)
    
    dobj_dw = (dvarx_dw * var_p + var_x * dvarp_dw) / (2.0 * obj)
    
    # dw/dparams: w is linear in (c4, c8)
    # w_k = _wscale[k] * c_k, where c = [1, c4, c8, c12]
    # c12 = (1 + 12*c4 + 1680*c8)/3991680
    # dw/dc4 = _wscale * [0, 1, 0, 12/3991680]
    # dw/dc8 = _wscale * [0, 0, 1, 1680/3991680]
    dw_dc4 = _wscale * np.array([0.0, 1.0, 0.0, _H4_0 / (-_H12_0)])
    dw_dc8 = _wscale * np.array([0.0, 0.0, 1.0, _H8_0 / (-_H12_0)])
    
    grad_c4 = float(dobj_dw @ dw_dc4)
    grad_c8 = float(dobj_dw @ dw_dc8)
    
    return obj, np.array([grad_c4, grad_c8])

def construct():
    start_time = time.time()
    deadline = 205.0  # safety margin before 210s kill
    
    def elapsed():
        return time.time() - start_time
    
    def time_left():
        return deadline - elapsed()
    
    best_score = np.inf
    best_params = np.array([0.0, 0.0])
    
    def try_update(params, score):
        nonlocal best_score, best_params
        if score < best_score and np.isfinite(score):
            best_score = score
            best_params = params.copy()
    
    # Phase 1: Multi-start Newton from a deterministic grid
    if time_left() > 5.0:
        # Coarse grid
        n_coarse = 25
        c4_grid = np.linspace(-0.5, 0.5, n_coarse)
        c8_grid = np.linspace(-0.05, 0.05, n_coarse)
        
        for c4_val in c4_grid:
            if time_left() < 3.0:
                break
            for c8_val in c8_grid:
                if time_left() < 3.0:
                    break
                params = np.array([c4_val, c8_val])
                try:
                    obj, grad = _objective_and_grad(params)
                    if np.isfinite(obj):
                        # Newton step
                        # Use BFGS-like or simple gradient descent with line search
                        # Actually, let's just do a local minimize from this start
                        if time_left() < 2.0:
                            try_update(params, obj)
                            continue
                        res = minimize(
                            lambda p: _objective_and_grad(p)[0],
                            params,
                            jac=lambda p: _objective_and_grad(p)[1],
                            method='BFGS',
                            options={'maxiter': 100, 'gtol': 1e-10}
                        )
                        if res.success and res.fun < best_score:
                            try_update(res.x, res.fun)
                except Exception:
                    pass
    
    # Phase 2: Differential evolution for global exploration if time remains
    if time_left() > 15.0:
        try:
            bounds = [(-0.5, 0.5), (-0.05, 0.05)]
            de_result = differential_evolution(
                lambda p: _objective_and_grad(p)[0],
                bounds,
                seed=42,
                maxiter=100,
                popsize=15,
                tol=1e-8,
                polish=False,
                init='sobol'
            )
            if de_result.fun < best_score:
                try_update(de_result.x, de_result.fun)
        except Exception:
            pass
    
    # Phase 3: Final polish with tight tolerance
    if time_left() > 3.0 and np.isfinite(best_score):
        try:
            res = minimize(
                lambda p: _objective_and_grad(p)[0],
                best_params,
                jac=lambda p: _objective_and_grad(p)[1],
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-14, 'fatol': 1e-15}
            )
            if res.fun < best_score:
                try_update(res.x, res.fun)
        except Exception:
            pass
    
    return np.array([1.0, best_params[0], best_params[1]])
