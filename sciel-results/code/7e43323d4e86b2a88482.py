# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import lpmv
from scipy.optimize import minimize

_K = 10
_NPHI = 130
_NTH = 70
_NPTS = _NTH * _NPHI
_NC = 2 * _K + 1


def _build_quadrature():
    x, w = np.polynomial.legendre.leggauss(_NTH)
    phi = np.linspace(0.0, 2.0 * np.pi, _NPHI, endpoint=False)
    dphi = 2.0 * np.pi / _NPHI
    W = np.outer(w, np.full(_NPHI, dphi))

    Y = np.zeros((_NTH, _NPHI, _NC))
    for idx, m in enumerate(range(-_K, _K + 1)):
        am = abs(m)
        # N_m = sqrt((2K+1)/(4pi) * (K-am)!/(K+am)!)
        denom = 1.0
        for n in range(_K - am + 1, _K + am + 1):
            denom *= n
        fact_ratio = 1.0 / denom
        Nm = np.sqrt((2 * _K + 1) / (4.0 * np.pi) * fact_ratio)
        P = lpmv(am, _K, x)
        if m == 0:
            Y[:, :, idx] = Nm * P[:, None]
        elif m > 0:
            Y[:, :, idx] = np.sqrt(2.0) * Nm * P[:, None] * np.cos(m * phi)[None, :]
        else:
            Y[:, :, idx] = np.sqrt(2.0) * Nm * P[:, None] * np.sin(am * phi)[None, :]
    return Y, W


_Y, _W = _build_quadrature()
_Wflat = _W.ravel()
_Yflat = _Y.reshape(_NPTS, _NC)


def _objective_L4(c):
    f = _Yflat @ c
    return float(np.dot(_Wflat, f * f * f * f))


def _gradient_L4(c):
    f = _Yflat @ c
    return 4.0 * (_Yflat.T @ (_Wflat * f * f * f))


def _objective_unconstrained(c):
    norm_sq = float(np.dot(c, c))
    if norm_sq < 1e-30:
        return 1e30
    f = _Yflat @ c
    H = float(np.dot(_Wflat, f * f * f * f))
    return H / (norm_sq * norm_sq)


def _gradient_unconstrained(c):
    norm_sq = float(np.dot(c, c))
    if norm_sq < 1e-30:
        return np.zeros(_NC)
    f = _Yflat @ c
    H = float(np.dot(_Wflat, f * f * f * f))
    dH = 4.0 * (_Yflat.T @ (_Wflat * f * f * f))
    grad = (dH - H * c / norm_sq) / (norm_sq * norm_sq)
    return grad


def _conjugate_gradient_sphere(c, max_iter=5000, tol=1e-17, deadline=None):
    """
    Conjugate gradient on the unit sphere with Fletcher-Reeves restart.
    Uses backtracking line search with Armijo condition.
    """
    c = np.asarray(c, dtype=np.float64)
    norm = np.linalg.norm(c)
    if norm < 1e-12:
        c = np.ones(_NC)
        c /= np.linalg.norm(c)
    else:
        c = c / norm

    best_val = _objective_L4(c)
    best_c = c.copy()

    # Compute initial gradient and project to tangent space
    f = _Yflat @ c
    grad = 4.0 * (_Yflat.T @ (_Wflat * f * f * f))
    grad_tan = grad - np.dot(grad, c) * c

    d = -grad_tan.copy()
    prev_grad_tan = grad_tan.copy()
    prev_grad_norm_sq = float(np.dot(prev_grad_tan, prev_grad_tan))

    for it in range(max_iter):
        if deadline is not None and time.time() > deadline - 2.0:
            break

        # Backtracking line search with Armijo condition
        step = 1.0
        f_curr = best_val
        grad_dot_d = float(np.dot(grad_tan, d))
        success = False
        for _ in range(50):
            c_trial = c + step * d
            c_norm = np.linalg.norm(c_trial)
            if c_norm < 1e-12:
                step *= 0.5
                continue
            c_trial = c_trial / c_norm
            f_trial = _objective_L4(c_trial)
            if f_trial < f_curr - 1e-12 * step * abs(grad_dot_d):
                success = True
                break
            step *= 0.5

        if not success:
            break

        c = c_trial
        if f_trial < best_val:
            best_val = f_trial
            best_c = c.copy()

        # Compute new gradient
        f = _Yflat @ c
        grad = 4.0 * (_Yflat.T @ (_Wflat * f * f * f))
        grad_tan = grad - np.dot(grad, c) * c

        gnorm = np.linalg.norm(grad_tan)
        if gnorm < tol:
            break

        # Fletcher-Reeves update with periodic restart
        if it % 10 == 9:
            d = -grad_tan.copy()
        else:
            curr_grad_norm_sq = float(np.dot(grad_tan, grad_tan))
            beta = curr_grad_norm_sq / max(prev_grad_norm_sq, 1e-30)
            if beta < 0 or beta > 10.0:
                d = -grad_tan.copy()
            else:
                d = -grad_tan + beta * d

        prev_grad_tan = grad_tan.copy()
        prev_grad_norm_sq = curr_grad_norm_sq

    return best_c, best_val


def _complex_sphere_cg(c, max_iter=3000, tol=1e-16, deadline=None):
    """
    Optimize using complex parameterization of (cos, sin) pairs for each |m|.
    For m > 0: c[m] + i*c[-m] represents the pair.
    We optimize on the complex sphere and map back to real coefficients.
    """
    c = np.asarray(c, dtype=np.float64)
    c = c / np.linalg.norm(c)

    # Build complex representation: z[0] = c[0], z[k] = c[K+k] + i*c[K-k] for k=1..K
    # c index: m=-10..10, so c[0]=m=-10, c[10]=m=0, c[20]=m=10
    # For m>0: c[K+m] is cos coeff, c[K-m] is sin coeff
    # Complex: z[m] = c[K+m] + 1j*c[K-m] for m=1..K, z[0] = c[K]
    z = np.zeros(_K + 1, dtype=np.complex128)
    z[0] = c[_K]
    for m in range(1, _K + 1):
        z[m] = c[_K + m] + 1j * c[_K - m]

    # Normalize z on complex sphere
    z_norm = np.linalg.norm(z)
    if z_norm < 1e-12:
        z = np.ones(_K + 1, dtype=np.complex128)
        z /= np.linalg.norm(z)
    else:
        z = z / z_norm

    def _z_to_c(zv):
        """Map complex vector to real coefficients."""
        cv = np.zeros(_NC, dtype=np.float64)
        cv[_K] = zv[0].real
        for m in range(1, _K + 1):
            cv[_K + m] = zv[m].real
            cv[_K - m] = zv[m].imag
        return cv

    def _objective_z(zv):
        cv = _z_to_c(zv)
        return _objective_L4(cv)

    def _gradient_z(zv):
        """Compute gradient with respect to complex parameters."""
        cv = _z_to_c(zv)
        f = _Yflat @ cv
        grad_real = 4.0 * (_Yflat.T @ (_Wflat * f * f * f))
        
        gz = np.zeros(_K + 1, dtype=np.complex128)
        gz[0] = grad_real[_K]
        for m in range(1, _K + 1):
            gz[m] = grad_real[_K + m] + 1j * grad_real[_K - m]
        return gz

    def _project_tangent(zv, grad_z):
        """Project gradient to tangent space of complex sphere."""
        proj = np.dot(grad_z, zv)  # inner product on complex sphere
        return grad_z - proj * zv

    best_val = _objective_z(z)
    best_z = z.copy()

    # Initial direction
    grad_z = _gradient_z(z)
    grad_tan = _project_tangent(z, grad_z)
    d = -grad_tan.copy()
    prev_grad_tan = grad_tan.copy()
    prev_gnorm_sq = float(np.dot(prev_grad_tan, prev_grad_tan).real)

    for it in range(max_iter):
        if deadline is not None and time.time() > deadline - 2.0:
            break

        # Backtracking line search
        step = 1.0
        f_curr = best_val
        grad_dot_d = float(np.dot(grad_tan, d).real)
        success = False
        for _ in range(40):
            z_trial = z + step * d
            z_norm_trial = np.linalg.norm(z_trial)
            if z_norm_trial < 1e-12:
                step *= 0.5
                continue
            z_trial = z_trial / z_norm_trial
            f_trial = _objective_z(z_trial)
            if f_trial < f_curr - 1e-12 * step * abs(grad_dot_d):
                success = True
                break
            step *= 0.5

        if not success:
            break

        z = z_trial
        if f_trial < best_val:
            best_val = f_trial
            best_z = z.copy()

        # New gradient
        grad_z = _gradient_z(z)
        grad_tan = _project_tangent(z, grad_z)
        gnorm_sq = float(np.dot(grad_tan, grad_tan).real)
        if gnorm_sq < tol * tol:
            break

        # Fletcher-Reeves with restart
        if it % 8 == 7:
            d = -grad_tan.copy()
        else:
            beta = gnorm_sq / max(prev_gnorm_sq, 1e-30)
            if beta < 0 or beta > 8.0:
                d = -grad_tan.copy()
            else:
                d = -grad_tan + beta * d

        prev_grad_tan = grad_tan.copy()
        prev_gnorm_sq = gnorm_sq

    best_c = _z_to_c(best_z)
    best_c = best_c / np.linalg.norm(best_c)
    return best_c, _objective_L4(best_c)


def _polish_gradient_descent(c, max_iter=10000, tol=1e-18, deadline=None):
    """
    High-precision polish using projected gradient descent on the sphere.
    Uses adaptive step size and backtracking line search.
    """
    c = np.asarray(c, dtype=np.float64)
    c = c / np.linalg.norm(c)
    best_val = _objective_L4(c)
    best_c = c.copy()
    step_size = 1.0

    for it in range(max_iter):
        if deadline is not None and time.time() > deadline - 1.0:
            break

        f = _Yflat @ c
        grad = 4.0 * (_Yflat.T @ (_Wflat * f * f * f))
        grad_tan = grad - np.dot(grad, c) * c
        gnorm = np.linalg.norm(grad_tan)
        if gnorm < tol:
            break

        # Backtracking line search
        step = step_size
        f_curr = best_val
        success = False
        for _ in range(40):
            c_trial = c - step * grad_tan
            c_norm = np.linalg.norm(c_trial)
            if c_norm < 1e-12:
                step *= 0.5
                continue
            c_trial = c_trial / c_norm
            f_trial = _objective_L4(c_trial)
            if f_trial < f_curr - 1e-14 * step * gnorm * gnorm:
                success = True
                break
            step *= 0.5

        if not success:
            step_size *= 0.5
            if step_size < 1e-20:
                break
            continue

        c = c_trial
        if f_trial < best_val:
            best_val = f_trial
            best_c = c.copy()
            step_size = min(step * 1.5, 1.0)
        else:
            step_size *= 0.5

    return best_c, best_val


def _hessian_polish(c, max_iter=50, tol=1e-16, deadline=None):
    """
    Final polish using approximate Newton steps on the sphere.
    Computes Hessian-vector products and solves the KKT system.
    """
    c = np.asarray(c, dtype=np.float64)
    c = c / np.linalg.norm(c)
    best_val = _objective_L4(c)
    best_c = c.copy()

    for it in range(max_iter):
        if deadline is not None and time.time() > deadline - 1.0:
            break

        f = _Yflat @ c
        grad = 4.0 * (_Yflat.T @ (_Wflat * f * f * f))
        grad_tan = grad - np.dot(grad, c) * c
        gnorm = np.linalg.norm(grad_tan)
        if gnorm < tol:
            break

        # Approximate Hessian-vector product using finite differences
        # H[v] = 4 * Y^T (W * 3*f^2 * Y[v]) + 4 * Y^T (W * f^3 * 0) ... 
        # Actually d2H/dc^2 = 4 * Y^T diag(W * 3*f^2) Y
        # For tangent direction v: H_tan[v] = H[v] - (H[v]·c + grad·v)*c projected
        # Simplified: use a few FD steps to estimate curvature
        
        # Use a simple trust-region-like approach: try Newton direction
        # H_approx = 4 * Y^T diag(3*W*f^2) Y
        # This is expensive (NPTS x NC matrix), so use CG on the Hessian
        
        # Instead, use a damped Newton step with line search
        # Estimate Hessian diagonal for scaling
        h_diag = np.zeros(_NC)
        for j in range(_NC):
            ej = np.zeros(_NC)
            ej[j] = 1.0
            ej_tan = ej - np.dot(ej, c) * c
            en = np.linalg.norm(ej_tan)
            if en < 1e-12:
                h_diag[j] = 1.0
                continue
            ej_tan /= en
            # FD Hessian
            eps = 1e-5
            f_plus = _Yflat @ (c + eps * ej_tan)
            f_plus = f_plus / np.linalg.norm(_Yflat @ (c + eps * ej_tan))  # approximate
            f_minus = _Yflat @ (c - eps * ej_tan)
            H_pp = _objective_L4(c + eps * ej_tan / np.linalg.norm(c + eps * ej_tan))
            H_mm = _objective_L4(c - eps * ej_tan / np.linalg.norm(c - eps * ej_tan))
            H_00 = best_val
            h_diag[j] = max((H_pp + H_mm - 2.0 * H_00) / (eps * eps), 1e-10)

        # Damped Newton direction: solve (H + mu*I) d = -grad_tan in tangent space
        mu = 1e-3
        # Simple iterative solve: d = -(grad_tan + mu * 0) / (diag(H) + mu)
        # Project to tangent
        d = -grad_tan / (h_diag + mu)
        d_tan = d - np.dot(d, c) * c
        dnorm = np.linalg.norm(d_tan)
        if dnorm < 1e-15:
            break
        d_tan /= dnorm

        # Line search
        step = 1.0
        success = False
        for _ in range(30):
            c_trial = c + step * d_tan
            c_norm = np.linalg.norm(c_trial)
            if c_norm < 1e-12:
                step *= 0.5
                continue
            c_trial = c_trial / c_norm
            f_trial = _objective_L4(c_trial)
            if f_trial < best_val - 1e-14 * step * gnorm:
                success = True
                break
            step *= 0.5

        if not success:
            break

        c = c_trial
        if f_trial < best_val:
            best_val = f_trial
            best_c = c.copy()

    return best_c, best_val


def _optimize_sphere(c0, deadline):
    c0 = np.asarray(c0, dtype=np.float64)
    norm = np.linalg.norm(c0)
    if norm < 1e-12:
        c0 = np.ones(_NC)
        c0 /= np.linalg.norm(c0)
    else:
        c0 = c0 / norm

    def fun(c):
        return _objective_L4(c)

    def jac(c):
        return _gradient_L4(c)

    def cons_fun(c):
        return np.sum(c * c) - 1.0

    def cons_jac(c):
        return 2.0 * c

    constraints = [{'type': 'eq', 'fun': cons_fun, 'jac': cons_jac}]

    best_c = c0.copy()
    best_val = fun(c0)

    # Stage 1: SLSQP for initial constraint satisfaction
    if time.time() < deadline - 10.0:
        try:
            res = minimize(fun, c0, jac=jac, method='SLSQP', constraints=constraints,
                           options={'maxiter': 500, 'ftol': 1e-12, 'disp': False})
            if res.fun < best_val:
                best_val = res.fun
                best_c = res.x.copy()
                best_c /= np.linalg.norm(best_c)
        except Exception:
            pass

    # Stage 2: Conjugate gradient on sphere (primary optimizer)
    if time.time() < deadline - 15.0:
        try:
            c_cg, val_cg = _conjugate_gradient_sphere(best_c, max_iter=5000, tol=1e-17, deadline=deadline)
            if val_cg < best_val:
                best_val = val_cg
                best_c = c_cg.copy()
        except Exception:
            pass

    # Stage 2b: Complex-sphere conjugate gradient
    if time.time() < deadline - 15.0:
        try:
            c_cxcg, val_cxcg = _complex_sphere_cg(best_c, max_iter=3000, tol=1e-16, deadline=deadline)
            if val_cxcg < best_val:
                best_val = val_cxcg
                best_c = c_cxcg.copy()
        except Exception:
            pass

    # Stage 3: L-BFGS-B on unconstrained scale-invariant objective
    if time.time() < deadline - 10.0:
        try:
            res2 = minimize(_objective_unconstrained, best_c, jac=_gradient_unconstrained,
                            method='L-BFGS-B', options={'maxiter': 3000, 'ftol': 1e-16, 'disp': False})
            if res2.fun < best_val:
                nc = res2.x / np.linalg.norm(res2.x)
                val_check = _objective_L4(nc)
                if val_check < best_val:
                    best_val = val_check
                    best_c = nc
        except Exception:
            pass

    # Stage 4: Second conjugate gradient pass from improved point
    if time.time() < deadline - 10.0:
        try:
            c_cg2, val_cg2 = _conjugate_gradient_sphere(best_c, max_iter=5000, tol=1e-18, deadline=deadline)
            if val_cg2 < best_val:
                best_val = val_cg2
                best_c = c_cg2.copy()
        except Exception:
            pass

    # Stage 4b: Second complex-sphere CG pass
    if time.time() < deadline - 10.0:
        try:
            c_cxcg2, val_cxcg2 = _complex_sphere_cg(best_c, max_iter=3000, tol=1e-17, deadline=deadline)
            if val_cxcg2 < best_val:
                best_val = val_cxcg2
                best_c = c_cxcg2.copy()
        except Exception:
            pass

    # Stage 5: Hessian-based polish
    if time.time() < deadline - 5.0:
        try:
            c_newton, val_newton = _hessian_polish(best_c, max_iter=50, tol=1e-16, deadline=deadline)
            if val_newton < best_val:
                best_val = val_newton
                best_c = c_newton.copy()
        except Exception:
            pass

    # Stage 6: Final high-precision gradient descent polish
    if time.time() < deadline - 3.0:
        try:
            c_polished, val_polished = _polish_gradient_descent(best_c, max_iter=10000, tol=1e-18, deadline=deadline)
            if val_polished < best_val:
                best_val = val_polished
                best_c = c_polished.copy()
        except Exception:
            pass

    return best_c, best_val


def rudin_problem10() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 600.0  # Safety margin before 630s kill

    # Generate diverse starting points
    rng = np.random.default_rng(2024)

    starts = []

    # All ones (symmetric)
    c1 = np.ones(_NC)
    c1 /= np.linalg.norm(c1)
    starts.append(c1)

    # Even m only
    c2 = np.array([1.0 if (i - _K) % 2 == 0 else 0.0 for i in range(_NC)])
    n2 = np.linalg.norm(c2)
    if n2 > 1e-12:
        c2 /= n2
    starts.append(c2)

    # Odd m only
    c3 = np.array([1.0 if (i - _K) % 2 != 0 else 0.0 for i in range(_NC)])
    n3 = np.linalg.norm(c3)
    if n3 > 1e-12:
        c3 /= n3
    starts.append(c3)

    # m >= 0 positive, m < 0 negative
    c4 = np.array([1.0 if m >= 0 else -1.0 for m in range(-_K, _K + 1)])
    c4 /= np.linalg.norm(c4)
    starts.append(c4)

    # m >= 0 negative, m < 0 positive
    c5 = np.array([-1.0 if m >= 0 else 1.0 for m in range(-_K, _K + 1)])
    c5 /= np.linalg.norm(c5)
    starts.append(c5)

    # Random starts with different seeds
    for seed in [42, 123, 456, 789, 101, 202, 303, 404, 505, 606]:
        cr = rng.standard_normal(_NC)
        cr /= np.linalg.norm(cr)
        starts.append(cr)

    # Structured: cosine-weighted
    c6 = np.array([np.cos(np.pi * (m + _K) / (2 * _K)) for m in range(-_K, _K + 1)])
    n6 = np.linalg.norm(c6)
    if n6 > 1e-12:
        c6 /= n6
    starts.append(c6)

    # Structured: m^2 weighted
    c7 = np.array([float(m * m) for m in range(-_K, _K + 1)])
    n7 = np.linalg.norm(c7)
    if n7 > 1e-12:
        c7 /= n7
    starts.append(c7)

    # Structured: alternating with decay
    c8 = np.array([(-1) ** m * np.exp(-0.1 * abs(m)) for m in range(-_K, _K + 1)])
    n8 = np.linalg.norm(c8)
    if n8 > 1e-12:
        c8 /= n8
    starts.append(c8)

    # Structured: only m=0 and m=±10 (extremal degrees)
    c9 = np.zeros(_NC)
    c9[_K] = 1.0  # m=0
    c9[0] = 1.0   # m=-10
    c9[_NC - 1] = 1.0  # m=10
    n9 = np.linalg.norm(c9)
    if n9 > 1e-12:
        c9 /= n9
    starts.append(c9)

    # Structured: m=0, ±5, ±10
    c10 = np.zeros(_NC)
    c10[_K] = 1.0
    c10[_K - 5] = 1.0
    c10[_K + 5] = 1.0
    c10[0] = 1.0
    c10[_NC - 1] = 1.0
    n10 = np.linalg.norm(c10)
    if n10 > 1e-12:
        c10 /= n10
    starts.append(c10)

    # Structured: sin-weighted (emphasize sine components)
    c11 = np.array([np.sin(np.pi * (m + _K) / (2 * _K + 1)) for m in range(-_K, _K + 1)])
    n11 = np.linalg.norm(c11)
    if n11 > 1e-12:
        c11 /= n11
    starts.append(c11)

    # Structured: m=0 dominant
    c12 = np.zeros(_NC)
    c12[_K] = 1.0
    for m in range(1, _K + 1):
        c12[_K + m] = 0.1
        c12[_K - m] = 0.1
    n12 = np.linalg.norm(c12)
    if n12 > 1e-12:
        c12 /= n12
    starts.append(c12)

    best_c = starts[0].copy()
    best_val = _objective_L4(best_c)

    # Optimize from each starting point
    for i, c0 in enumerate(starts):
        if time.time() > deadline - 20.0:
            break
        try:
            c_opt, val_opt = _optimize_sphere(c0, deadline)
            if val_opt < best_val:
                best_val = val_opt
                best_c = c_opt.copy()
        except Exception:
            continue

    # Final global polish from the best found
    if time.time() < deadline - 5.0:
        try:
            c_final, val_final = _conjugate_gradient_sphere(best_c, max_iter=10000, tol=1e-18, deadline=deadline)
            if val_final < best_val:
                best_val = val_final
                best_c = c_final.copy()
        except Exception:
            pass

    if time.time() < deadline - 5.0:
        try:
            c_cxfinal, val_cxfinal = _complex_sphere_cg(best_c, max_iter=5000, tol=1e-17, deadline=deadline)
            if val_cxfinal < best_val:
                best_val = val_cxfinal
                best_c = c_cxfinal.copy()
        except Exception:
            pass

    if time.time() < deadline - 3.0:
        try:
            c_hess, val_hess = _hessian_polish(best_c, max_iter=30, tol=1e-17, deadline=deadline)
            if val_hess < best_val:
                best_val = val_hess
                best_c = c_hess.copy()
        except Exception:
            pass

    if time.time() < deadline - 2.0:
        try:
            c_polished, val_polished = _polish_gradient_descent(best_c, max_iter=20000, tol=1e-19, deadline=deadline)
            if val_polished < best_val:
                best_val = val_polished
                best_c = c_polished.copy()
        except Exception:
            pass

    # Final safety check
    best_c = np.asarray(best_c, dtype=np.float64)
    if not np.all(np.isfinite(best_c)):
        best_c = starts[0]
    mx = float(np.max(np.abs(best_c)))
    if mx < 1e-3:
        best_c = best_c * (1.0 / mx)
    if mx > 1e5:
        best_c = best_c * (1e5 / mx)

    return best_c

# EVOLVE-BLOCK-END
