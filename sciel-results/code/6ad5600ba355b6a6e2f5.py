# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.special import lpmv, factorial

_DEADLINE = time.time() + 615.0
_K = 10
_NTH = 70
_NPHI = 130


def _build_harmonics():
    """Return Y_flat (NTH*NPHI, 2k+1) and w_flat (NTH*NPHI,) for quadrature."""
    k = _K
    mu, w = np.polynomial.legendre.leggauss(_NTH)
    phi = np.linspace(0.0, 2.0 * np.pi, _NPHI, endpoint=False)
    dphi = 2.0 * np.pi / _NPHI

    Y = np.zeros((_NTH, _NPHI, 2 * k + 1), dtype=np.float64)
    for idx, m in enumerate(range(-k, k + 1)):
        am = abs(m)
        N = np.sqrt((2 * k + 1) / (4.0 * np.pi)
                    * float(factorial(k - am, exact=True))
                    / float(factorial(k + am, exact=True)))
        P = lpmv(am, k, mu)
        if m == 0:
            Y[:, :, idx] = N * P[:, None]
        elif m > 0:
            Y[:, :, idx] = np.sqrt(2.0) * N * P[:, None] * np.cos(m * phi)[None, :]
        else:
            Y[:, :, idx] = np.sqrt(2.0) * N * P[:, None] * np.sin(am * phi)[None, :]

    Y_flat = Y.reshape(_NTH * _NPHI, 2 * k + 1)
    w_flat = np.outer(w, np.full(_NPHI, dphi)).ravel()

    return Y_flat, w_flat


def _objective_and_grad(c, Y, w):
    """
    Compute R^4 = S4/S2^2 and its gradient. Minimizing R^4 is equivalent to minimizing R.
    """
    f = Y @ c
    f2 = f * f
    f3 = f2 * f

    S2 = np.dot(w, f2)
    S4 = np.dot(w, f2 * f2)

    if S2 < 1e-15:
        return 1e12, np.zeros_like(c)

    val = S4 / (S2 * S2)

    Yt_wf3 = Y.T @ (w * f3)
    Yt_wf = Y.T @ (w * f)

    grad = 4.0 * (S2 * Yt_wf3 - S4 * Yt_wf) / (S2 * S2 * S2)

    return val, grad


def _hessian_vec(c, v, Y, w):
    """
    Compute Hessian-vector product H @ v for the objective J(c) = S4 / S2^2.
    """
    f = Y @ c
    f2 = f * f
    f3 = f2 * f

    S2 = np.dot(w, f2)
    S4 = np.dot(w, f2 * f2)

    if S2 < 1e-15:
        return np.zeros_like(v)

    # Gradients of S2 and S4
    dS2 = 2.0 * Y.T @ (w * f)
    dS4 = 4.0 * Y.T @ (w * f3)

    # Hessian of S2: H2 v = 2 Y^T (w * (Y v))
    Yv = Y @ v
    H2_v = 2.0 * Y.T @ (w * Yv)

    # Hessian of S4: H4 v = 12 Y^T (w * f^2 * (Y v))
    H4_v = 12.0 * Y.T @ (w * f2 * Yv)

    # J = S4 * S2^-2
    # dJ = dS4 * S2^-2 - 2 S4 * S2^-3 * dS2
    # d2J(v) = H4_v * S2^-2 - 2 S4 * S2^-3 * H2_v 
    #          + dS4 * (-2 S2^-3 dS2(v)) - 2 S4 * (-3 S2^-4 (dS2)^2(v)) 
    #          - 2 S4 * S2^-3 * (dS2 * dS2(v) + dS2(v) * dS2)
    # Simplified:
    term1 = S2**-2 * H4_v
    term2 = -2.0 * S2**-3 * (dS4 @ v) * dS2
    term3 = -2.0 * S4 * S2**-3 * H2_v
    term4 = 6.0 * S4 * S2**-4 * (dS2 @ v) * dS2

    return term1 + term2 + term3 + term4


def _ratio_val(c, Y, w):
    """Compute R = ||f||_4 / ||f||_2."""
    f = Y @ c
    f2 = f * f
    S2 = np.dot(w, f2)
    S4 = np.dot(w, f2 * f2)
    if S2 < 1e-15:
        return np.inf
    return (S4 ** 0.25) / np.sqrt(S2)


def _make_inits(rng, k):
    """Diverse initial directions on the unit sphere."""
    inits = []
    n = 2 * k + 1
    # Random
    for _ in range(20):
        v = rng.standard_normal(n)
        inits.append(v / np.linalg.norm(v))
    # Single dominant m
    for m in range(-k, k + 1):
        v = np.zeros(n)
        v[m + k] = 1.0
        inits.append(v)
    # Paired cos/sin
    for m in range(1, k + 1):
        v = np.zeros(n)
        v[m + k] = 1.0
        v[-m + k] = 1.0
        inits.append(v / np.linalg.norm(v))
    # Alternating signs
    v = np.ones(n)
    v[::2] = -1.0
    inits.append(v / np.linalg.norm(v))
    # Random sparse
    for _ in range(8):
        v = np.zeros(n)
        idx = rng.choice(n, size=rng.integers(2, 9), replace=False)
        v[idx] = rng.standard_normal(len(idx))
        if np.linalg.norm(v) > 1e-12:
            inits.append(v / np.linalg.norm(v))
    # High-m dominance (m=10, 9)
    for m in [10, 9, 8]:
        v = np.zeros(n)
        v[m + k] = 1.0
        v[-m + k] = 0.5
        inits.append(v / np.linalg.norm(v))
    return inits


def _trust_region_step(c, g, Y, w, delta):
    """
    Solve the trust-region subproblem in the tangent space.
    Minimize 0.5 v^T H_t v + g_t^T v subject to ||v|| <= delta and c^T v = 0.
    H_t = P_c H P_c, g_t = P_c g.
    """
    n = len(c)
    P_c = np.eye(n) - np.outer(c, c)
    g_t = P_c @ g
    
    # Form H_t explicitly
    H_t = np.zeros((n, n))
    for i in range(n):
        e_i = np.zeros(n)
        e_i[i] = 1.0
        H_t[:, i] = P_c @ _hessian_vec(c, P_c @ e_i, Y, w)
    
    # Symmetrize H_t
    H_t = 0.5 * (H_t + H_t.T)
    
    # Eigen decomposition
    eigvals, eigvecs = np.linalg.eigh(H_t)
    
    a = eigvecs.T @ g_t
    
    def norm_sq(sigma):
        if sigma == 0:
            mask = np.abs(eigvals) > 1e-12
            return np.sum((eigvals[mask] * a[mask]) ** 2 / (eigvals[mask] + sigma) ** 2)
        return np.sum((eigvals * a) ** 2 / (eigvals + sigma) ** 2)
    
    # Check sigma = 0
    if norm_sq(0) <= delta ** 2:
        sigma = 0.0
    else:
        # Bisection for sigma
        lo = 0.0
        hi = np.max(-eigvals) + 1.0
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            if norm_sq(mid) > delta ** 2:
                lo = mid
            else:
                hi = mid
        sigma = 0.5 * (lo + hi)
        
    # Compute v
    coeffs = np.zeros(n)
    mask = np.abs(eigvals + sigma) > 1e-12
    coeffs[mask] = -eigvals[mask] * a[mask] / (eigvals[mask] + sigma)
    v = eigvecs @ coeffs
    
    # Ensure v is in tangent space
    v = P_c @ v
    
    return v


def _optimize_trust_region(c0, Y, w, maxiter=100, delta0=1.0):
    """Run Trust-Region method on the sphere."""
    c = c0.copy()
    delta = delta0
    n = len(c)
    
    for i in range(maxiter):
        if time.time() > _DEADLINE - 30.0:
            break
            
        val, g = _objective_and_grad(c, Y, w)
        if not np.isfinite(val):
            break
            
        # Check convergence
        g_t = g - np.dot(c, g) * c
        g_norm = np.linalg.norm(g_t)
        if g_norm < 1e-10:
            break
            
        # Compute step
        v = _trust_region_step(c, g, Y, w, delta)
        step_norm = np.linalg.norm(v)
        
        if step_norm < 1e-12:
            break
            
        # New point
        c_new = c + v
        c_new = c_new / np.linalg.norm(c_new)
        
        val_new, _ = _objective_and_grad(c_new, Y, w)
        
        # Predicted reduction
        Hv = _hessian_vec(c, v, Y, w)
        pred_red = -np.dot(g, v) - 0.5 * np.dot(v, Hv)
        
        act_red = val - val_new
        
        if pred_red < 1e-12:
            delta *= 0.5
            continue
            
        rho = act_red / pred_red
        
        if rho > 0.75:
            delta = min(2.0 * delta, 10.0)
        elif rho < 0.25:
            delta *= 0.5
        
        c = c_new
        
    return c, _ratio_val(c, Y, w)


def _fixed_point_iteration(c, Y, w, maxiter=500, tol=1e-12):
    """
    Fixed-point iteration for the quartic eigenvalue problem.
    
    The critical points of R = ||f||_4 / ||f||_2 satisfy:
    Y^T (Yc)^3 = lambda * Y^T (Yc)
    
    where lambda = S4 / (2 * S2^2).
    
    We iterate: c_new = Y^T (Yc)^3 - lambda * Y^T (Yc), then normalize.
    This exploits the quartic structure for fast convergence.
    """
    c = c.copy()
    n = len(c)
    
    for i in range(maxiter):
        if time.time() > _DEADLINE - 30.0:
            break
        
        f = Y @ c
        f2 = f * f
        f3 = f2 * f
        
        S2 = np.dot(w, f2)
        S4 = np.dot(w, f2 * f2)
        
        if S2 < 1e-15:
            break
        
        # Compute lambda = S4 / (2 * S2^2)
        lam = S4 / (2.0 * S2 * S2)
        
        # Compute the fixed-point map
        Yt_wf3 = Y.T @ (w * f3)
        Yt_wf = Y.T @ (w * f)
        
        c_new = Yt_wf3 - lam * Yt_wf
        
        # Normalize
        c_new_norm = np.linalg.norm(c_new)
        if c_new_norm < 1e-12:
            break
        c_new = c_new / c_new_norm
        
        # Check convergence
        diff = np.linalg.norm(c_new - c)
        c = c_new
        
        if diff < tol:
            break
    
    return c, _ratio_val(c, Y, w)


def _polish_lbfgs(c, Y, w, maxiter=200):
    """
    Final polish using L-BFGS-B with analytic gradient.
    Uses a penalty method for the sphere constraint.
    """
    from scipy.optimize import minimize
    
    def fun_pen(c):
        c_norm = np.linalg.norm(c)
        if c_norm < 1e-12:
            return 1e12, np.zeros_like(c)
        c_u = c / c_norm
        val, grad = _objective_and_grad(c_u, Y, w)
        # Penalty for deviation from unit sphere
        penalty = 1e6 * (c_norm - 1.0) ** 2
        # Gradient of penalty: 2e6 (c_norm - 1) * c / c_norm
        pen_grad = 2e6 * (c_norm - 1.0) * c / c_norm
        # Chain rule for c_u = c / c_norm
        # d(c_u)/dc = (I - c c^T / c_norm^2) / c_norm
        # So grad_c = (I - c c^T / c_norm^2) / c_norm @ grad_u + pen_grad
        P = np.eye(len(c)) - np.outer(c, c) / (c_norm * c_norm)
        grad_c = P @ grad / c_norm + pen_grad
        return val + penalty, grad_c
    
    res = minimize(fun_pen, c, jac=True, method='L-BFGS-B',
                   options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-12})
    
    if res.success or np.isfinite(res.fun):
        c_opt = res.x / np.linalg.norm(res.x)
        return c_opt, _ratio_val(c_opt, Y, w)
    
    return c, _ratio_val(c, Y, w)


def rudin_problem10() -> np.ndarray:
    Y, w = _build_harmonics()
    k = _K
    n = 2 * k + 1
    rng = np.random.default_rng(42)
    inits = _make_inits(rng, k)

    best_c = inits[0].copy()
    best_val = np.inf

    # Phase 1: Fixed-point iteration from diverse starts (fast convergence)
    scored = []
    for i, x0 in enumerate(inits):
        if time.time() > _DEADLINE - 180.0:
            break
        c_opt, val_opt = _fixed_point_iteration(x0, Y, w, maxiter=300, tol=1e-10)
        if np.isfinite(val_opt):
            scored.append((val_opt, c_opt))
            if val_opt < best_val:
                best_val = val_opt
                best_c = c_opt.copy()

    scored.sort(key=lambda t: t[0])
    top = [c for _, c in scored[:15]]
    if not top:
        top = [best_c]

    # Phase 2: Refine top candidates with trust-region (higher precision)
    for x0 in top:
        if time.time() > _DEADLINE - 90.0:
            break
        c_opt, val_opt = _optimize_trust_region(x0, Y, w, maxiter=100, delta0=0.5)
        if np.isfinite(val_opt) and val_opt < best_val:
            best_val = val_opt
            best_c = c_opt.copy()

    # Phase 3: More random restarts with fixed-point (cheap and fast)
    while time.time() < _DEADLINE - 60.0:
        x0 = best_c + 0.3 * rng.standard_normal(n)
        x0 = x0 / np.linalg.norm(x0)
        c_opt, val_opt = _fixed_point_iteration(x0, Y, w, maxiter=200, tol=1e-10)
        if np.isfinite(val_opt) and val_opt < best_val:
            best_val = val_opt
            best_c = c_opt.copy()

    # Phase 4: Final high-precision polish with fixed-point + L-BFGS
    if time.time() < _DEADLINE - 30.0:
        c_opt, val_opt = _fixed_point_iteration(best_c, Y, w, maxiter=500, tol=1e-13)
        if np.isfinite(val_opt) and val_opt < best_val:
            best_val = val_opt
            best_c = c_opt.copy()
    
    if time.time() < _DEADLINE - 20.0:
        c_opt, val_opt = _polish_lbfgs(best_c, Y, w, maxiter=200)
        if np.isfinite(val_opt) and val_opt < best_val:
            best_val = val_opt
            best_c = c_opt.copy()

    # Final normalization
    out = best_c.astype(np.float64)
    nrm = np.linalg.norm(out)
    if nrm < 1e-12:
        out = np.ones(n, dtype=np.float64)
    else:
        out = out / nrm

    # Validation
    if out.shape != (21,):
        out = np.ones(21, dtype=np.float64)
    if np.any(np.isnan(out)) or np.any(np.isinf(out)):
        out = np.ones(21, dtype=np.float64)

    max_abs = np.max(np.abs(out))
    if max_abs < 1e-3:
        out = out / max_abs * 1.0
    elif max_abs > 1e5:
        out = out / max_abs * 1.0

    return out


# EVOLVE-BLOCK-END
