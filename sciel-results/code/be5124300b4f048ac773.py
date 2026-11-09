# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import lpmv
from scipy.optimize import minimize, line_search

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


def _objective(c):
    f = _Yflat @ c
    f2 = f * f
    l2sq = float(np.dot(_Wflat, f2))
    if l2sq <= 1e-30:
        return 1e30
    f4 = f2 * f2
    l4sq = float(np.dot(_Wflat, f4))
    return l4sq / (l2sq * l2sq)


def _gradient(c):
    f = _Yflat @ c
    f2 = f * f
    f3 = f2 * f
    l2sq = float(np.dot(_Wflat, f2))
    if l2sq <= 1e-30:
        return np.zeros_like(c)
    l4sq = float(np.dot(_Wflat, f2 * f2))
    wf3 = _Wflat * f3
    wf = _Wflat * f
    grad_f3 = _Yflat.T @ wf3
    grad_f = _Yflat.T @ wf
    B = l2sq
    A = l4sq
    grad = (4.0 / (B * B)) * grad_f3 - (4.0 * A / (B * B * B)) * grad_f
    return grad


def _project_to_sphere(c):
    nrm = np.linalg.norm(c)
    if nrm < 1e-14:
        return None
    return c / nrm


def _conjugate_gradient_sphere(fun, c0, maxiter=300, tol=1e-12, deadline=None):
    """
    Conjugate gradient on the unit sphere (Frisch's method / projected CG).
    """
    x = np.asarray(c0, dtype=np.float64).copy()
    f_curr = fun(x)
    g_curr = _gradient(x)

    # Remove radial component of gradient to get tangent
    g_tan = g_curr - np.dot(g_curr, x) * x
    s = -g_tan
    s_norm = np.linalg.norm(s)
    if s_norm < 1e-15:
        return x, f_curr

    s = s / s_norm  # Unit tangent step
    y_curr = x.copy()
    best_x = x.copy()
    best_f = f_curr
    best_g_norm = s_norm

    for i in range(maxiter):
        if deadline is not None and time.time() > deadline - 10.0:
            break

        # Line search along geodesic: x(t) = cos(t)*x + sin(t)*s
        # t in [0, pi]
        f_start = f_curr

        def f_of_t(t):
            ct = np.cos(t)
            st = np.sin(t)
            x_new = ct * y_curr + st * s
            return fun(x_new)

        ls_res = line_search(f_of_t, 0.0, f_start, 0.1, c0=1.0, amax=6.283185)
        if ls_res[0]:
            t_new = ls_res[2]
        else:
            t_new = 0.01
            if t_new < 0 or t_new > np.pi:
                t_new = 0.01

        x_new = np.cos(t_new) * y_curr + np.sin(t_new) * s
        f_new = fun(x_new)

        if f_new >= f_curr:
            t_new = 0.001
            x_new = np.cos(t_new) * y_curr + np.sin(t_new) * s
            f_new = fun(x_new)
            if f_new >= f_curr:
                break

        g_new = _gradient(x_new)
        g_tan_new = g_new - np.dot(g_new, x_new) * x_new
        g_norm = np.linalg.norm(g_tan_new)

        if g_norm < best_g_norm:
            best_g_norm = g_norm
            best_x = x_new.copy()
            best_f = f_new

        if g_norm < tol:
            break

        if best_f > f_new:
            best_f = f_new
            best_x = x_new.copy()

        dot_g = np.dot(g_tan_new, g_tan)
        norm_g_new_sq = np.dot(g_tan_new, g_tan_new)
        if dot_g < 1e-15:
            beta = 0.0
        else:
            beta = -norm_g_new_sq / dot_g

        if beta < 0:
            beta = 0.0

        s_new = -g_tan_new + beta * s
        s_new = s_new - np.dot(s_new, x_new) * x_new
        s_new_norm = np.linalg.norm(s_new)

        if s_new_norm < 1e-15:
            s_new = -g_tan_new
            s_new_norm = np.linalg.norm(s_new)
            if s_new_norm < 1e-15:
                break
            s_new = s_new / s_new_norm
            y_curr = x_new
            f_curr = f_new
            g_curr = g_new
            g_tan = g_tan_new
            continue

        s = s_new / s_new_norm
        y_curr = x_new
        f_curr = f_new
        g_curr = g_new
        g_tan = g_tan_new

    return best_x, best_f


def _slsqp_on_complex_sphere(z0, deadline, maxiter=200, tol=1e-12):
    """SLSQP optimization on the complex sphere using real parameterization."""
    z = np.asarray(z0, dtype=np.complex128).copy()
    z_norm = np.linalg.norm(z)
    if z_norm < 1e-14:
        return None, np.inf
    z = z / z_norm

    def fun(x):
        # x is real vector of length 2*(K+1): [Re(z0), Im(z0), Re(z1), Im(z1), ...]
        zv = np.zeros(_K + 1, dtype=np.complex128)
        for i in range(_K + 1):
            zv[i] = x[2*i] + 1j * x[2*i+1]
        zv_norm = np.linalg.norm(zv)
        if zv_norm < 1e-14:
            return 1e30
        zv = zv / zv_norm
        return _complex_objective(zv)

    def jac(x):
        zv = np.zeros(_K + 1, dtype=np.complex128)
        for i in range(_K + 1):
            zv[i] = x[2*i] + 1j * x[2*i+1]
        zv_norm = np.linalg.norm(zv)
        if zv_norm < 1e-14:
            return np.zeros(2*(_K+1))
        zv = zv / zv_norm
        g = _complex_gradient(zv)
        # Project gradient to tangent space
        inner = np.vdot(zv, g)
        g_tan = g - inner * zv
        # Convert to real Jacobian
        j = np.zeros(2*(_K+1))
        for i in range(_K + 1):
            j[2*i] = g_tan[i].real
            j[2*i+1] = g_tan[i].imag
        return j

    # Initial point
    x0 = np.zeros(2*(_K+1))
    for i in range(_K + 1):
        x0[2*i] = z[i].real
        x0[2*i+1] = z[i].imag

    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x*x) - 1.0, 'jac': lambda x: 2.0 * x}
    ]

    best_x = x0.copy()
    best_val = fun(x0)

    try:
        res = minimize(fun, x0, jac=jac, method='SLSQP',
                       constraints=constraints,
                       options={'maxiter': maxiter, 'ftol': tol, 'disp': False})
        if res.success and res.fun < best_val:
            best_val = res.fun
            best_x = res.x.copy()
    except Exception:
        pass

    # Convert back
    zv = np.zeros(_K + 1, dtype=np.complex128)
    for i in range(_K + 1):
        zv[i] = best_x[2*i] + 1j * best_x[2*i+1]
    zv_norm = np.linalg.norm(zv)
    if zv_norm < 1e-14:
        return None, np.inf
    zv = zv / zv_norm
    c = _complex_to_real(zv)
    c = _project_to_sphere(c)
    if c is None:
        return None, np.inf
    return c, float(_objective(c))


def _deterministic_grid_search(deadline):
    """
    Grid search over the 10-dimensional complex amplitude subspace.
    We parameterize z_m for m=0..10 with magnitudes and phases.
    Due to rotational invariance, we can fix one phase and search the rest.
    """
    best_c = None
    best_val = np.inf

    def eval_complex(z):
        c = _complex_to_real(z)
        return _objective(c)

    # Single amplitude: just z_m = 1 for each m
    for m in range(_K + 1):
        z = np.zeros(_K + 1, dtype=np.complex128)
        z[m] = 1.0
        val = eval_complex(z)
        if val < best_val:
            best_val = val
            best_c = _complex_to_real(z).copy()

    # Two amplitudes: z_a, z_b with |z_a|^2 + |z_b|^2 = 1
    for a in range(_K + 1):
        for b in range(a + 1, _K + 1):
            for r in np.linspace(0.0, 1.0, 9):
                if r < 1e-6 or r > 1 - 1e-6:
                    continue
                rb = np.sqrt(1.0 - r * r)
                for phi in np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False):
                    z = np.zeros(_K + 1, dtype=np.complex128)
                    z[a] = r
                    z[b] = rb * np.exp(1j * phi)
                    val = eval_complex(z)
                    if val < best_val:
                        best_val = val
                        best_c = _complex_to_real(z).copy()

    # Three amplitudes
    for a in range(_K + 1):
        for b in range(a + 1, _K + 1):
            for c_idx in range(b + 1, _K + 1):
                for r1 in [0.5, 0.7, 0.8]:
                    for r2 in [0.3, 0.5, 0.7]:
                        r3 = np.sqrt(max(0.0, 1.0 - r1 * r1 - r2 * r2))
                        if r3 < 1e-6:
                            continue
                        for phi1 in [0.0, np.pi / 2, np.pi, 3 * np.pi / 2]:
                            for phi2 in [0.0, np.pi / 2, np.pi, 3 * np.pi / 2]:
                                z = np.zeros(_K + 1, dtype=np.complex128)
                                z[a] = r1
                                z[b] = r2 * np.exp(1j * phi1)
                                z[c_idx] = r3 * np.exp(1j * phi2)
                                val = eval_complex(z)
                                if val < best_val:
                                    best_val = val
                                    best_c = _complex_to_real(z).copy()

    # Polish with SLSQP
    if best_c is not None and time.time() < deadline - 15.0:
        z_best = _real_to_complex(best_c)
        c_polished, val_polished = _slsqp_on_complex_sphere(z_best, deadline, maxiter=150, tol=1e-12)
        if c_polished is not None and val_polished < best_val:
            best_val = val_polished
            best_c = c_polished

    return best_c, best_val


def _random_grid_search(rng, deadline, n_iter=200):
    """Random search in the complex amplitude space."""
    best_c = None
    best_val = np.inf

    def eval_complex(z):
        c = _complex_to_real(z)
        return _objective(c)

    for _ in range(n_iter):
        if time.time() > deadline - 10.0:
            break

        z = (rng.standard_normal(_K + 1) + 1j * rng.standard_normal(_K + 1)).astype(np.complex128)
        z_norm = np.linalg.norm(z)
        if z_norm < 1e-14:
            continue
        z = z / z_norm

        val = eval_complex(z)
        if val < best_val:
            best_val = val
            best_c = _complex_to_real(z).copy()

        if val < best_val + 0.05 and time.time() < deadline - 20.0:
            c_polished, val_polished = _slsqp_on_complex_sphere(z, deadline, maxiter=100, tol=1e-12)
            if c_polished is not None and val_polished < best_val:
                best_val = val_polished
                best_c = c_polished

    return best_c, best_val


def _optimize_from(c0, deadline):
    c = np.asarray(c0, dtype=np.float64).copy()
    c = _project_to_sphere(c)
    if c is None:
        return None

    def fun(x):
        return _objective(x)

    def jac(x):
        return _gradient(x)

    best_c = c.copy()
    best_val = fun(c)

    # 1. L-BFGS-B
    try:
        res = minimize(fun, c, jac=jac, method="L-BFGS-B",
                       options={"maxiter": 300, "maxfun": 2000,
                                "ftol": 1e-16, "gtol": 1e-10})
        if res.fun < best_val:
            best_val = res.fun
            best_c = res.x.copy()
    except Exception:
        pass

    if time.time() > deadline - 10.0:
        proj = _project_to_sphere(best_c)
        if proj is not None:
            return proj, float(_objective(proj))
        return None

    # 2. SLSQP on complex sphere
    if time.time() < deadline - 15.0:
        try:
            z_init = _real_to_complex(best_c)
            c_slq, val_slq = _slsqp_on_complex_sphere(z_init, deadline, maxiter=200, tol=1e-12)
            if c_slq is not None and val_slq < best_val:
                best_val = val_slq
                best_c = c_slq.copy()
        except Exception:
            pass

    if time.time() > deadline - 10.0:
        proj = _project_to_sphere(best_c)
        if proj is not None:
            return proj, float(_objective(proj))
        return None

    # 3. Conjugate Gradient on Sphere
    c_cg_init = _project_to_sphere(best_c)
    if c_cg_init is not None:
        try:
            cg_c, cg_val = _conjugate_gradient_sphere(fun, c_cg_init, maxiter=200, tol=1e-12, deadline=deadline)
            if cg_val < best_val:
                best_val = cg_val
                best_c = cg_c.copy()
        except Exception:
            pass

    if time.time() > deadline - 10.0:
        proj = _project_to_sphere(best_c)
        if proj is not None:
            return proj, float(_objective(proj))
        return None

    # 4. Nelder-Mead polish
    if time.time() < deadline - 10.0:
        try:
            res2 = minimize(fun, best_c, method="Nelder-Mead",
                            options={"maxiter": 1500, "xatol": 1e-12,
                                     "fatol": 1e-14, "adaptive": True})
            if res2.fun < best_val:
                best_val = res2.fun
                best_c = res2.x.copy()
        except Exception:
            pass

    # 5. Second L-BFGS-B pass
    if time.time() < deadline - 5.0:
        try:
            res3 = minimize(fun, best_c, jac=jac, method="L-BFGS-B",
                            options={"maxiter": 200, "maxfun": 1000,
                                     "ftol": 1e-16, "gtol": 1e-12})
            if res3.fun < best_val:
                best_val = res3.fun
                best_c = res3.x.copy()
        except Exception:
            pass

    # 6. Final CG polish with tighter tolerance
    if time.time() < deadline - 2.0:
        c_cg_final_init = _project_to_sphere(best_c)
        if c_cg_final_init is not None:
            try:
                cg_c_final, cg_val_final = _conjugate_gradient_sphere(fun, c_cg_final_init, maxiter=150, tol=1e-14, deadline=deadline)
                if cg_val_final < best_val:
                    best_val = cg_val_final
                    best_c = cg_c_final.copy()
            except Exception:
                pass

    # 7. Final SLSQP polish
    if time.time() < deadline - 1.0:
        try:
            z_final = _real_to_complex(best_c)
            c_final, val_final = _slsqp_on_complex_sphere(z_final, deadline, maxiter=100, tol=1e-14)
            if c_final is not None and val_final < best_val:
                best_val = val_final
                best_c = c_final.copy()
        except Exception:
            pass

    final_c = _project_to_sphere(best_c)
    if final_c is None:
        return None
    return final_c, float(_objective(final_c))


def _make_seeds(rng, n_seeds):
    seeds = []
    for i in range(n_seeds):
        c = np.zeros(_NC)
        if i == 0:
            c[_K] = 1.0
        elif i == 1:
            c[_K + 10] = 1.0
        elif i == 2:
            c[_K - 10] = 1.0
        elif i == 3:
            for m in range(-_K, _K + 1):
                c[m + _K] = np.cos(0.5 * m)
        elif i == 4:
            for m in range(-_K, _K + 1):
                c[m + _K] = (-1) ** m
        elif i == 5:
            c[:] = rng.standard_normal(_NC)
        elif i == 6:
            for m in range(-_K, _K + 1, 2):
                c[m + _K] = rng.standard_normal()
        elif i == 7:
            for m in range(-_K + 1, _K + 1, 2):
                c[m + _K] = rng.standard_normal()
        elif i == 8:
            c[_K] = 1.0
            c[_K + 10] = rng.standard_normal()
            c[_K - 10] = rng.standard_normal()
        elif i == 9:
            c[_K + 5] = rng.standard_normal()
            c[_K - 5] = rng.standard_normal()
        elif i == 10:
            for m in range(0, _K + 1):
                v = rng.standard_normal()
                c[m + _K] = v
                c[-m + _K] = v
        elif i == 11:
            for m in range(1, _K + 1):
                v = rng.standard_normal()
                c[m + _K] = v
                c[-m + _K] = -v
        elif i == 12:
            c[_K] = 0.5
            c[_K + 10] = 1.0
            c[_K - 10] = 1.0
        elif i == 13:
            for m in range(-_K, _K + 1):
                if abs(m) >= 8:
                    c[m + _K] = rng.standard_normal()
        elif i == 14:
            for m in range(-_K, _K + 1):
                if abs(m) >= 6:
                    c[m + _K] = rng.standard_normal()
        elif i == 15:
            for m in [0, 1, -1, 10, -10]:
                c[m + _K] = rng.standard_normal()
        elif i == 16:
            for m in [0, 2, -2, 10, -10]:
                c[m + _K] = rng.standard_normal()
        elif i == 17:
            for m in [0, 3, -3, 10, -10]:
                c[m + _K] = rng.standard_normal()
        elif i == 18:
            c[_K] = 0.3
            c[_K + 10] = 1.0
            c[_K - 10] = 1.0
        elif i == 19:
            c[_K] = 0.7
            c[_K + 10] = 1.0
            c[_K - 10] = 1.0
        elif i == 20:
            c[:] = rng.standard_normal(_NC)
            for m in range(-_K, _K + 1):
                c[m + _K] *= (1.0 + abs(m) / 10.0)
        elif i == 21:
            c[:] = rng.standard_normal(_NC)
            for m in range(-_K, _K + 1):
                c[m + _K] *= (1.0 + (10 - abs(m)) / 10.0)
        elif i == 22:
            c[_K + 10] = 1.0
            c[_K - 10] = 1.0
        else:
            c[:] = rng.standard_normal(_NC)
            c[_K] += 0.5
            c[_K + 10] += 0.3
            c[_K - 10] += 0.3
        nrm = np.linalg.norm(c)
        if nrm > 1e-14:
            c /= nrm
        seeds.append(c)
    return seeds


def rudin_problem10() -> np.ndarray:
    start = time.time()
    deadline = start + 610.0
    rng = np.random.default_rng(42)

    seeds = _make_seeds(rng, 24)
    best_c = None
    best_val = np.inf

    for i, c0 in enumerate(seeds):
        if time.time() > deadline - 20.0:
            break
        try:
            out = _optimize_from(c0, deadline)
        except Exception:
            continue
        if out is None:
            continue
        c, val = out
        if val < best_val:
            best_val = val
            best_c = c.copy()

    # Random restarts
    n_extra = 0
    while time.time() < deadline - 30.0:
        c0 = rng.standard_normal(_NC)
        nrm = np.linalg.norm(c0)
        if nrm < 1e-14:
            continue
        c0 /= nrm
        try:
            out = _optimize_from(c0, deadline)
        except Exception:
            continue
        if out is None:
            continue
        c, val = out
        if val < best_val:
            best_val = val
            best_c = c.copy()
        n_extra += 1

    if best_c is None:
        best_c = np.ones(_NC, dtype=np.float64)
        best_c /= np.linalg.norm(best_c)

    best_c = np.asarray(best_c, dtype=np.float64)
    if not np.all(np.isfinite(best_c)):
        best_c = np.ones(_NC, dtype=np.float64)
        best_c /= np.linalg.norm(best_c)
    mx = float(np.max(np.abs(best_c)))
    if mx < 1e-3:
        best_c = best_c * (1.0 / mx)
    if mx > 1e5:
        best_c = best_c * (1e5 / mx)
    return best_c


# EVOLVE-BLOCK-END
