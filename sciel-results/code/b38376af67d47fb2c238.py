"""Spherical-harmonic critical-point method for minimizing log-energy of 150 points."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import lpmv

N = 150
MAX_DEG = 6


def _add_l_m(pts, l, m):
    """Add a band-limited (degree <= MAX_DEG) harmonic term with random coefficients."""
    for ell in range(1, MAX_DEG + 1):
        for mm in range(-ell, ell + 1):
            if np.random.random() < 0.4:
                pts["a_re"][ell, mm] = np.random.randn()
                pts["a_im"][ell, mm] = np.random.randn()
    return pts


def _spherical_harmonic(pts, l, m, theta, phi):
    """Evaluate Y_l^m(theta, phi) for m possibly negative. theta=polar, phi=azimuth."""
    if m < 0:
        Y = (-1) ** m * np.sqrt((2 * l + 1) / (4 * np.pi) *
                                np.math.factorial(l + m) / np.math.factorial(l - m))
        Y *= lpmv(abs(m), l, np.cos(theta))
        Y *= np.exp(1j * m * phi)
    else:
        Y = np.sqrt((2 * l + 1) / (4 * np.pi) *
                    np.math.factorial(l - m) / np.math.factorial(l + m))
        Y *= lpmv(m, l, np.cos(theta))
        Y *= np.exp(1j * m * phi)
    return Y


def _eval_harmonic(pts, theta, phi):
    """Real-valued band-limited function value at (theta, phi) arrays."""
    val = np.zeros_like(theta, dtype=float)
    for ell in range(1, MAX_DEG + 1):
        for mm in range(-ell, ell + 1):
            are = pts["a_re"][ell, mm]
            aim = pts["a_im"][ell, mm]
            if are == 0.0 and aim == 0.0:
                continue
            Y = _spherical_harmonic(pts, ell, mm, theta, phi)
            val += are * Y.real + aim * Y.imag
    return val


def _grad_harmonic(pts, theta, phi):
    """Gradient (dtheta, dphi) of the band-limited function at (theta, phi) arrays."""
    gtheta = np.zeros_like(theta, dtype=float)
    gphi = np.zeros_like(theta, dtype=float)
    for ell in range(1, MAX_DEG + 1):
        for mm in range(-ell, ell + 1):
            are = pts["a_re"][ell, mm]
            aim = pts["a_im"][ell, mm]
            if are == 0.0 and aim == 0.0:
                continue
            Y = _spherical_harmonic(pts, ell, mm, theta, phi)
            # dY/dtheta = -lpmv(m,l,sin(theta))*sin(theta) + lpmv(m,l,cos(theta))*cos(theta)
            Y_t = (-lpmv(abs(mm), ell, np.sin(theta)) * np.sin(theta)
                   + lpmv(abs(mm), ell, np.cos(theta)) * np.cos(theta))
            Y_t = Y_t * np.exp(1j * mm * phi)
            if mm < 0:
                Y_t *= (-1) ** mm * np.sqrt((2 * ell + 1) / (4 * np.pi) *
                                            np.math.factorial(ell + mm) / np.math.factorial(ell - mm))
            else:
                Y_t *= np.sqrt((2 * ell + 1) / (4 * np.pi) *
                               np.math.factorial(ell - mm) / np.math.factorial(ell + mm))
            # dY/dphi = i*m*Y
            Y_p = 1j * mm * Y
            gtheta += are * Y_t.real + aim * Y_t.imag
            gphi += are * Y_p.real + aim * Y_p.imag
    return gtheta, gphi


def _fibonacci_grid(n, offset=0.5):
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return np.column_stack([x, y, z])


def _to_sph(xyz):
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    r = np.sqrt(x * x + y * y + z * z)
    r = np.maximum(r, 1e-12)
    theta = np.arccos(np.clip(z / r, -1.0, 1.0))
    phi = np.arctan2(y, x)
    return theta, phi


def _log_energy(pts):
    d2 = np.sum(pts * pts, axis=1)[:, None] + np.sum(pts * pts, axis=1)[None, :] - 2 * pts @ pts.T
    d2 = np.maximum(d2, 1e-18)
    d = np.sqrt(d2)
    iu = np.triu_indices(N, k=1)
    return np.sum(np.log(1.0 / d[iu]))


def _find_critical_points(pts, n_init=4000):
    """Find local minima of the band-limited potential via Newton refinement."""
    theta, phi = _to_sph(_fibonacci_grid(n_init, offset=0.3))
    cands = []
    for t0, p0 in zip(theta, phi):
        try:
            res = minimize(
                lambda v: _eval_harmonic(pts, v[0:1], v[1:1])[0],
                x0=[t0, p0],
                method="L-BFGS-B",
                bounds=[(1e-4, np.pi - 1e-4), (-np.pi, np.pi)],
                options={"maxiter": 30, "ftol": 1e-10, "gtol": 1e-8},
            )
            if res.success or res.fun < 0:
                cands.append(res.x)
        except Exception:
            continue
    if len(cands) < N:
        return None
    cands = np.array(cands)
    # deduplicate close candidates
    cands = cands[np.argsort(cands[:, 0])]
    keep = [0]
    for i in range(1, len(cands)):
        if np.hypot(cands[i, 0] - cands[keep[-1], 0],
                    np.arctan2(np.sin(cands[i, 1] - cands[keep[-1], 1]),
                               np.cos(cands[i, 1] - cands[keep[-1], 1]))) > 0.05:
            keep.append(i)
    cands = cands[keep]
    if len(cands) < N:
        return None
    # pick N most negative-potential candidates
    vals = np.array([_eval_harmonic(pts, c[0:1], c[1:1])[0] for c in cands])
    idx = np.argsort(vals)[:N]
    cands = cands[idx]
    th = cands[:, 0]
    ph = cands[:, 1]
    xyz = np.column_stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
    return xyz


def _refine_log_energy(xyz):
    """Local refinement of log-energy via gradient-free Nelder-Mead in spherical coords."""
    theta, phi = _to_sph(xyz)
    params = np.column_stack([theta, phi]).flatten()

    def energy(p):
        th = np.clip(p[0::2], 1e-6, np.pi - 1e-6)
        ph = p[1::2]
        xyz = np.column_stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
        return _log_energy(xyz)

    def grad(p):
        th = np.clip(p[0::2], 1e-6, np.pi - 1e-6)
        ph = p[1::2]
        xyz = np.column_stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
        g = np.zeros_like(p)
        for i in range(N):
            # gradient of log(1/d) = -d/dx_i log d = ... use numerical per-point
            pass
        # numerical gradient
        eps = 1e-6
        g = np.zeros_like(p)
        f0 = energy(p)
        for k in range(len(p)):
            pp = p.copy()
            pp[k] += eps
            g[k] = (energy(pp) - f0) / eps
        return g

    res = minimize(energy, params, method="L-BFGS-B", jac=grad,
                   options={"maxiter": 600, "ftol": 1e-12, "gtol": 1e-8, "maxls": 40})
    th = np.clip(res.x[0::2], 1e-6, np.pi - 1e-6)
    ph = res.x[1::2]
    xyz = np.column_stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
    return xyz


def construct():
    np.random.seed(12345)
    # Build a random band-limited potential
    pts = {"a_re": np.zeros((MAX_DEG + 1, 2 * MAX_DEG + 1)),
           "a_im": np.zeros((MAX_DEG + 1, 2 * MAX_DEG + 1))}
    _add_l_m(pts, 0, 0)

    # Find critical points
    best = None
    for attempt in range(3):
        xyz = _find_critical_points(pts)
        if xyz is None:
            pts = {"a_re": np.zeros((MAX_DEG + 1, 2 * MAX_DEG + 1)),
                   "a_im": np.zeros((MAX_DEG + 1, 2 * MAX_DEG + 1))}
            _add_l_m(pts, 0, 0)
            continue
        # Refine
        xyz = _refine_log_energy(xyz)
        e = _log_energy(xyz)
        if best is None or e < best[0]:
            best = (e, xyz)

    if best is None:
        # Fallback to fibonacci
        return _fibonacci_grid(N, offset=0.5)

    xyz = best[1]
    # Normalize
    xyz = xyz / np.linalg.norm(xyz, axis=1, keepdims=True)
    # Validate: no points closer than 1e-9
    d2 = np.sum(xyz * xyz, axis=1)[:, None] + np.sum(xyz * xyz, axis=1)[None, :] - 2 * xyz @ xyz.T
    d = np.sqrt(np.maximum(d2, 0.0))
    iu = np.triu_indices(N, k=1)
    if d[iu].min() < 1e-9:
        # nudge apart
        for _ in range(1000):
            i, j = iu
            if d[i, j] < 1e-9:
                break
        # just return fibonacci if still bad
        return _fibonacci_grid(N, offset=0.5)
    return xyz
