"""Analytic uncertainty bound with exact gradient + L-BFGS-B optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, brentq

_DEADLINE = time.time() + 195

# Precompute I_n = integral x^{2n} e^{-x^2} dx = (2n)! / (4^n n!) * sqrt(pi)
def _precompute_I():
    """Precompute I_n for n=0..13"""
    I = np.zeros(14)
    sp = np.sqrt(np.pi)
    I[0] = sp
    for n in range(1, 14):
        I[n] = (2*n - 1) * I[n-1] / (4.0 * n)
    return I

_I = _precompute_I()


def _bound_and_grad(params):
    """
    Compute uncertainty product and its analytic gradient.
    g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
    a12 = -(a0 + 3*a4 + 105*a8)/945  (enforces g(0)=0)
    """
    a0, a4, a8 = params[0], params[1], params[2]
    a12 = -(a0 + 3.0*a4 + 105.0*a8) / 945.0
    
    # Coefficients of P(x) = sum_{k=0}^{6} p_k x^{2k}
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
    dpda[0, 0] = 0.0
    dpda[1, 0] = 0.0
    dpda[2, 0] = 0.0
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


def _newton_raphson(start, max_iter=100, tol=1e-14):
    """
    Custom Newton-Raphson solver using analytic gradient and numerical Hessian.
    """
    x = start.copy()
    for i in range(max_iter):
        if time.time() > _DEADLINE - 5:
            break
        f, g = _bound_and_grad(x)
        if not np.isfinite(f):
            break
        if np.linalg.norm(g, ord=np.inf) < tol:
            return x, f, True
        
        h = 1e-7
        n = len(x)
        H = np.zeros((n, n))
        for i in range(n):
            x_plus = x.copy()
            x_plus[i] += h
            x_minus = x.copy()
            x_minus[i] -= h
            _, g_plus = _bound_and_grad(x_plus)
            _, g_minus = _bound_and_grad(x_minus)
            for j in range(n):
                H[i, j] = (g_plus[j] - g_minus[j]) / (2 * h)
        
        try:
            dx = np.linalg.solve(H, -g)
        except np.linalg.LinAlgError:
            try:
                dx = np.linalg.solve(H + 1e-12 * np.eye(n), -g)
            except np.linalg.LinAlgError:
                break
        
        alpha = 1.0
        f_new = f
        for _ in range(20):
            x_new = x + alpha * dx
            f_new, _ = _bound_and_grad(x_new)
            if f_new < f:
                break
            alpha *= 0.5
            if alpha < 1e-16:
                break
        
        if not np.isfinite(f_new):
            break
            
        if np.linalg.norm(dx * alpha, ord=np.inf) < tol:
            x = x_new
            break
            
        x = x_new
        
    f_final, _ = _bound_and_grad(x)
    return x, f_final, False


def _stationary_equations(params):
    """Return the gradient (stationary point equations)."""
    _, grad = _bound_and_grad(params)
    return grad


def _ratio_1d_search(base_params, t_range=(-2.0, 2.0), n_grid=200):
    """
    1D global optimization over a ratio parameter t.
    We parameterize as:
      a4 = base_a4 * exp(t)
      a8 = base_a8 * exp(t)
      a0 = base_a0  (fixed)
    This explores a 1D manifold in the 3D space.
    """
    a0 = base_params[0]
    base_a4 = base_params[1]
    base_a8 = base_params[2]
    
    if base_a4 == 0 or base_a8 == 0:
        return base_params, _objective(base_params)
    
    def f_of_t(t):
        a4 = base_a4 * np.exp(t)
        a8 = base_a8 * np.exp(t)
        params = np.array([a0, a4, a8])
        val, _ = _bound_and_grad(params)
        return val, params
    
    # Grid sampling
    t_vals = np.linspace(t_range[0], t_range[1], n_grid)
    best_val = np.inf
    best_t = None
    best_params = base_params.copy()
    
    for t in t_vals:
        val, params = f_of_t(t)
        if val < best_val:
            best_val = val
            best_t = t
            best_params = params
    
    # Refine with Brent's method on sub-intervals
    if best_t is not None and time.time() < _DEADLINE - 10:
        t_idx = np.argmin([f_of_t(t)[0] for t in t_vals])
        # Check neighbors for sign change in derivative
        for i in range(max(0, t_idx-2), min(n_grid-1, t_idx+3)):
            t_left = t_vals[i]
            t_right = t_vals[i+1]
            try:
                val_left, _ = f_of_t(t_left)
                val_right, _ = f_of_t(t_right)
                if val_left > best_val and val_right > best_val:
                    # Both above best, skip
                    continue
                # Try Brent on this interval
                def f_brent(t):
                    val, _ = f_of_t(t)
                    return val
                t_opt = brentq(lambda t: f_brent(t) - best_val, t_left, t_right, 
                              xtol=1e-12, maxiter=100)
                val_opt, params_opt = f_of_t(t_opt)
                if val_opt < best_val:
                    best_val = val_opt
                    best_params = params_opt
            except (ValueError, RuntimeError):
                pass
    
    return best_params, best_val


def construct():
    # Multiple diverse starting points
    starts = [
        np.array([-1.13, 0.04, 0.000308642]),  # seed
        np.array([0.0, 0.0, 0.0]),
        np.array([-1.0, 0.1, 0.001]),
        np.array([-2.0, 0.05, 0.0005]),
        np.array([-0.5, 0.02, 0.0002]),
        np.array([-1.5, 0.08, 0.0008]),
        np.array([1.0, -0.05, 0.0005]),
        np.array([-3.0, 0.15, 0.002]),
        np.array([-0.8, 0.03, 0.0002]),
        np.array([-1.3, 0.05, 0.0004]),
        np.array([-1.7, 0.06, 0.0005]),
        np.array([-2.5, 0.12, 0.0015]),
        np.array([0.5, -0.02, 0.0001]),
        np.array([-0.3, 0.01, 0.0001]),
        np.array([-1.2, 0.045, 0.00035]),
        np.array([-1.4, 0.035, 0.00025]),
        np.array([-1.6, 0.055, 0.00045]),
        np.array([-1.0, 0.05, 0.0004]),
        np.array([-2.2, 0.07, 0.0006]),
    ]
    
    rng = np.random.default_rng(42)
    for i in range(20):
        starts.append(rng.normal(0, 0.8, 3))
    
    best_val = np.inf
    best_params = None
    all_candidates = []
    
    # Phase 1: L-BFGS-B from multiple starts
    for start in starts:
        if time.time() > _DEADLINE - 60:
            break
        try:
            res = minimize(_objective_grad, start, method='L-BFGS-B',
                          jac=True, options={'maxiter': 10000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 50})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
            if np.isfinite(res.fun):
                all_candidates.append(res.x.copy())
        except Exception:
            pass
    
    # Phase 1b: 1D ratio search from best candidate
    if best_params is not None and time.time() < _DEADLINE - 50:
        try:
            rp, rv = _ratio_1d_search(best_params, t_range=(-2.0, 2.0), n_grid=300)
            if rv < best_val and np.isfinite(rv):
                best_val = rv
                best_params = rp.copy()
                all_candidates.append(rp.copy())
        except Exception:
            pass
    
    # Phase 2: High-precision Newton-Raphson polish from top candidates
    if time.time() < _DEADLINE - 40:
        top_candidates = sorted(all_candidates, key=lambda x: _objective(x))[:10] if all_candidates else [best_params]
        
        for tc in top_candidates:
            if time.time() > _DEADLINE - 40:
                break
            if tc is None:
                continue
            
            local_best = tc.copy()
            local_best_val = _objective(tc)
            
            step = 0.01
            for _ in range(20):
                if time.time() > _DEADLINE - 40:
                    break
                improved = False
                for i in range(3):
                    for sign in [-1, 1]:
                        trial = local_best.copy()
                        trial[i] += sign * step
                        val = _objective(trial)
                        if val < local_best_val:
                            local_best = trial
                            local_best_val = val
                            improved = True
                if not improved:
                    step *= 0.5
                    if step < 1e-6:
                        break
            
            if local_best_val < best_val:
                best_val = local_best_val
                best_params = local_best.copy()
            
            try:
                x_new, f_new, converged = _newton_raphson(local_best)
                if converged and np.isfinite(f_new) and f_new < best_val:
                    best_val = f_new
                    best_params = x_new.copy()
                    all_candidates.append(x_new.copy())
            except Exception:
                pass
    
    # Phase 3: Polish with Nelder-Mead from best
    if best_params is not None and time.time() < _DEADLINE - 20:
        try:
            res = minimize(_objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 100000, 'xatol': 1e-16, 'fatol': 1e-18, 'adaptive': True})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 4: Final high-precision polish with L-BFGS-B
    if best_params is not None and time.time() < _DEADLINE - 10:
        try:
            res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 20000, 'ftol': 1e-18, 'gtol': 1e-16, 'maxls': 50})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 5: Final high-precision Newton polish
    if best_params is not None and time.time() < _DEADLINE - 5:
        try:
            x_new, f_new, converged = _newton_raphson(best_params, max_iter=50, tol=1e-15)
            if np.isfinite(f_new) and f_new < best_val:
                best_val = f_new
                best_params = x_new.copy()
        except Exception:
            pass
    
    if best_params is None:
        return np.array([-1.13, 0.04, 0.000308642])
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
