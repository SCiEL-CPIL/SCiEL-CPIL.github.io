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
    # dS2 = 2 Y^T (w * f)
    # dS4 = 4 Y^T (w * f^3)
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
    # Let's simplify:
    # J = S4 S2^-2
    # dJ = (dS4) S2^-2 - 2 S4 S2^-3 (dS2)
    # d2J(v) = (H4_v) S2^-2 - 2 S4 S2^-3 (H2_v) 
    #          + (dS4) (-2 S2^-3 dS2(v)) - 2 S4 (-3 S2^-4 (dS2)^2(v)) 
    #          - 2 S4 S2^-3 (dS2 dS2(v) + dS2(v) dS2)
    # Note: dS2(v) is a scalar: dS2 @ v.
    
    term1 = S2**-2 * H4_v
    term2 = -2.0 * S4 * S2**-3 * H2_v
    term3 = -2.0 * S2**-3 * (dS4 @ v) * dS2
    term4 = 6.0 * S4 * S2**-4 * (dS2 @ v) * dS2
    term5 = -2.0 * S4 * S2**-3 * ((dS2 @ v) * dS2 + (dS2 @ v) * dS2) # dS2 is a vector, dS2(v) is scalar.
    # Wait, the product rule for d(S4 S2^-2) is:
    # d(dS4 S2^-2) = H4_v S2^-2 + dS4 d(S2^-2)
    # d(S2^-2) = -2 S2^-3 dS2
    # d(dS2) = H2_v
    # So d(dS4 S2^-2) = H4_v S2^-2 + dS4 (-2 S2^-3 H2_v)  <-- This is wrong, dS4 is a vector, H2_v is a vector.
    # Let's do it carefully.
    # J(c) = S4(c) S2(c)^-2
    # dJ(c; v) = dS4(c; v) S2^-2 + S4 (-2 S2^-3 dS2(c; v))
    # d2J(c; v) = d(dS4(c; v) S2^-2) + d(S4 (-2 S2^-3 dS2(c; v)))
    # = H4_v S2^-2 + dS4 d(S2^-2; v) + dS4 (-2 S2^-3 dS2(c; v)) + S4 (-2 d(S2^-3; v) dS2(c; v) + S2^-3 d(H2_v))
    # d(S2^-2; v) = -2 S2^-3 dS2(c; v)
    # d(S2^-3; v) = -3 S2^-4 dS2(c; v)
    # dS4 is a vector, dS2(c; v) is a scalar.
    # So:
    # Term A: H4_v S2^-2
    # Term B: dS4 (-2 S2^-3 dS2(c; v))
    # Term C: dS4 (-2 S2^-3 dS2(c; v))  <-- Wait, d(S4) is dS4, but we are differentiating the product.
    # Let's use the formula: d2 (u v) = du dv + u dv + v du.
    # Let u = S4, v = S2^-2.
    # d2 (u v) = du dv + u dv + v du.
    # du = dS4, dv = -2 S2^-3 dS2.
    # du dv = dS4 (-2 S2^-3 dS2(c; v))
    # u dv = S4 (-2 S2^-3 dS2(c; v))  <-- No, dv is a 1-form.
    # Let's stick to the components.
    # J = S4 S2^-2
    # dJ = dS4 S2^-2 - 2 S4 S2^-3 dS2
    # d2J(v) = H4_v S2^-2 + dS4 (-2 S2^-3 dS2(v)) - 2 S4 S2^-3 H2_v - 2 S4 (-3 S2^-4 (dS2)^2(v))
    #        = S2^-2 H4_v - 2 S2^-3 (dS4 @ v) dS2 - 2 S4 S2^-3 H2_v + 6 S4 S2^-4 (dS2 @ v) dS2
    
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


def _trust_region_step(c, g, H, Y, w, delta):
    """
    Solve the trust-region subproblem in the tangent space.
    Minimize 0.5 v^T H_t v + g_t^T v subject to ||v|| <= delta and c^T v = 0.
    H_t = P_c H P_c, g_t = P_c g.
    """
    n = len(c)
    P_c = np.eye(n) - np.outer(c, c)
    g_t = P_c @ g
    # We need H_t v = P_c H (P_c v)
    # We can solve the eigenvalue problem for H_t.
    # Since n=21, we can just form H_t explicitly.
    H_t = np.zeros((n, n))
    for i in range(n):
        e_i = np.zeros(n)
        e_i[i] = 1.0
        H_t[:, i] = P_c @ _hessian_vec(c, P_c @ e_i, Y, w)
    
    # Symmetrize H_t
    H_t = 0.5 * (H_t + H_t.T)
    
    # Eigen decomposition
    eigvals, eigvecs = np.linalg.eigh(H_t)
    
    # Solve for sigma such that ||v|| = delta if unconstrained min is outside.
    # v = sum (-lambda_i / (lambda_i + sigma)) (q_i^T g_t) q_i
    # Let a_i = q_i^T g_t.
    # v = sum (-lambda_i a_i / (lambda_i + sigma)) q_i
    # ||v||^2 = sum (lambda_i^2 a_i^2 / (lambda_i + sigma)^2)
    # We want to find sigma >= 0 such that ||v|| = delta.
    # If sigma=0 gives ||v|| <= delta, use sigma=0.
    
    a = eigvecs.T @ g_t
    
    def norm_sq(sigma):
        if sigma == 0:
            # Avoid division by zero if lambda_i is 0
            # If lambda_i is 0, the term is 0 unless a_i is inf, which it isn't.
            # Actually if lambda_i = 0, the term is 0.
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
    # v = sum (-lambda_i a_i / (lambda_i + sigma)) q_i
    coeffs = np.zeros(n)
    mask = np.abs(eigvals + sigma) > 1e-12
    coeffs[mask] = -eigvals[mask] * a[mask] / (eigvals[mask] + sigma)
    v = eigvecs @ coeffs
    
    # Ensure v is in tangent space (it should be, but numerical errors)
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
        # Project gradient to tangent space
        g_t = g - np.dot(c, g) * c
        g_norm = np.linalg.norm(g_t)
        if g_norm < 1e-10:
            break
            
        # Compute step
        v = _trust_region_step(c, g, None, Y, w, delta)
        step_norm = np.linalg.norm(v)
        
        if step_norm < 1e-12:
            break
            
        # New point
        c_new = c + v
        # Project to sphere
        c_new = c_new / np.linalg.norm(c_new)
        
        val_new, _ = _objective_and_grad(c_new, Y, w)
        
        # Predicted reduction
        # Model: m(v) = f(c) + g^T v + 0.5 v^T H v
        # Since we solved the subproblem, the reduction in the model is:
        # rho = (f(c) - f(c_new)) / (f(c) - m(v))
        # m(v) = f(c) + g^T v + 0.5 v^T H v
        # We can approximate the model reduction by using the step we took.
        # Actually, we can just use the ratio of actual to predicted.
        # Predicted reduction is f(c) - (f(c) + g^T v + 0.5 v^T H v) = -g^T v - 0.5 v^T H v.
        # We need H v.
        Hv = _hessian_vec(c, v, Y, w)
        pred_red = -np.dot(g, v) - 0.5 * np.dot(v, Hv)
        
        act_red = val - val_new
        
        if pred_red < 1e-12:
            # No predicted reduction, shrink trust region
            delta *= 0.5
            continue
            
        rho = act_red / pred_red
        
        if rho > 0.75:
            delta = min(2.0 * delta, 10.0)
        elif rho < 0.25:
            delta *= 0.5
        else:
            # Keep delta
            pass
            
        c = c_new
        
    return c, _ratio_val(c, Y, w)


def rudin_problem10() -> np.ndarray:
    Y, w = _build_harmonics()
    k = _K
    n = 2 * k + 1
    rng = np.random.default_rng(42)
    inits = _make_inits(rng, k)

    best_c = inits[0].copy()
    best_val = np.inf

    # Phase 1: Trust Region from diverse starts
    scored = []
    for i, x0 in enumerate(inits):
        if time.time() > _DEADLINE - 180.0:
            break
        c_opt, val_opt = _optimize_trust_region(x0, Y, w, maxiter=100, delta0=1.0)
        if np.isfinite(val_opt):
            scored.append((val_opt, c_opt))
            if val_opt < best_val:
                best_val = val_opt
                best_c = c_opt.copy()

    scored.sort(key=lambda t: t[0])
    top = [c for _, c in scored[:10]]
    if not top:
        top = [best_c]

    # Phase 2: Refine top candidates with more iterations
    for x0 in top:
        if time.time() > _DEADLINE - 90.0:
            break
        c_opt, val_opt = _optimize_trust_region(x0, Y, w, maxiter=200, delta0=0.1)
        if np.isfinite(val_opt) and val_opt < best_val:
            best_val = val_opt
            best_c = c_opt.copy()

    # Phase 3: Random restarts from perturbed best
    while time.time() < _DEADLINE - 30.0:
        x0 = best_c + 0.2 * rng.standard_normal(n)
        x0 = x0 / np.linalg.norm(x0)
        c_opt, val_opt = _optimize_trust_region(x0, Y, w, maxiter=100, delta0=1.0)
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
