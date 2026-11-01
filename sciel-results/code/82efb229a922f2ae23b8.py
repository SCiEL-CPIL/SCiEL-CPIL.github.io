"""Minimal logarithmic energy of 150 points on the sphere.

Combines a time-budgeted multi-pass structure (L-BFGS + Riemannian CG with
exponential maps) with Halton/Fibonacci initialisations for good global
dispersion.  All optimisation is done on the sphere using the exponential map
so updates respect the manifold curvature.
"""
import time
import math
import numpy as np

N = 150
_DEADLINE = 150.0   # wall-clock seconds; safety margin below
_MARGIN = 5.0       # seconds to leave before the hard kill


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def _normalize(pts):
    """Normalise each row to unit length (guard against zero rows)."""
    pts = np.asarray(pts, dtype=float)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _normalize(pts)


def _halton_sphere(n):
    """Halton (bases 2,3) -> uniform-ish points on the sphere."""
    def halton(index, base):
        f = 1.0
        r = 0.0
        i = index
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r

    pts = np.zeros((n, 3))
    for i in range(n):
        u = halton(i + 1, 2)
        v = halton(i + 1, 3)
        theta = 2.0 * np.pi * u
        phi = np.arccos(1.0 - 2.0 * v)
        pts[i, 0] = np.sin(phi) * np.cos(theta)
        pts[i, 1] = np.sin(phi) * np.sin(theta)
        pts[i, 2] = np.cos(phi)
    return _normalize(pts)


# ---------------------------------------------------------------------------
# Energy / gradient (vectorised, projected to tangent space)
# ---------------------------------------------------------------------------
def _energy_and_grad(pts):
    """Return (energy, tangent_grad).

    energy = sum_{i<j} log(1 / d_ij)  =  - sum_{i<j} log d_ij
    gradient (ambient) w.r.t. x_i = sum_{j!=i} (x_i - x_j) / d_ij^2
    (this pushes points apart -> decreases the energy).
    """
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(d2, 1.0)
    d = np.sqrt(d2)

    # upper-triangle energy
    iu = np.triu_indices(n, k=1)
    d_up = np.maximum(d[iu], 1e-15)
    energy = np.sum(np.log(1.0 / d_up))

    # weight matrix 1/d^2, zero diagonal
    w = 1.0 / d2
    np.fill_diagonal(w, 0.0)
    # grad_i = sum_j w_ij (x_i - x_j) = x_i * rowsum - (w @ x)
    rowsum = w.sum(axis=1)
    grad = pts * rowsum[:, np.newaxis] - (w @ pts)

    # project to tangent space (remove radial component)
    radial = np.sum(grad * pts, axis=1)
    grad -= radial[:, np.newaxis] * pts
    return energy, grad


def _energy_fast(pts):
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d2 = np.einsum('ijk,ijk->ij', diff, diff)
    iu = np.triu_indices(n, k=1)
    d_up = np.sqrt(np.maximum(d2[iu], 1e-30))
    return np.sum(np.log(1.0 / d_up))


# ---------------------------------------------------------------------------
# Riemannian conjugate gradient with exponential map
# ---------------------------------------------------------------------------
def _exp_map(pts, d, alpha):
    """Move each point along tangent vector d by scalar alpha (per-point norm)."""
    d_norm = np.linalg.norm(d, axis=1)
    mask = d_norm > 1e-15
    out = pts.copy()
    if np.any(mask):
        a = alpha * d_norm[mask]
        d_hat = d[mask] / d_norm[mask, np.newaxis]
        out[mask] = (pts[mask] * np.cos(a)[:, np.newaxis]
                     + d_hat * np.sin(a)[:, np.newaxis])
    return _normalize(out)


def _riemannian_cg(pts, time_limit):
    """Riemannian CG on the sphere using exponential maps."""
    start = time.time()
    current = _normalize(pts.copy())
    energy, grad = _energy_and_grad(current)
    best = (current.copy(), energy)

    d = -grad.copy()                      # initial direction (tangent)
    prev_grad = grad.copy()
    prev_grad_norm_sq = np.sum(grad * grad)

    for _it in range(2000):
        if time.time() - start > time_limit - 0.05:
            break

        # ----- backtracking line search in the exponential map -----
        best_alpha = 0.0
        best_e = energy
        alpha = 0.5
        for _ls in range(25):
            cand = _exp_map(current, d, alpha)
            e_cand = _energy_fast(cand)
            if e_cand < best_e:
                best_e = e_cand
                best_alpha = alpha
            if e_cand < energy - 1e-13:
                break
            alpha *= 0.5

        if best_alpha <= 0.0:
            # try a very small step
            for a_try in (1e-4, 1e-5, 1e-6):
                cand = _exp_map(current, d, a_try)
                e_cand = _energy_fast(cand)
                if e_cand < energy:
                    best_e = e_cand
                    best_alpha = a_try
                    break
        if best_alpha <= 0.0:
            break

        current = _exp_map(current, d, best_alpha)
        energy, new_grad = _energy_and_grad(current)
        if energy < best[1]:
            best = (current.copy(), energy)

        gnorm = np.linalg.norm(new_grad)
        if gnorm < 1e-12:
            break

        # Polak-Ribiere beta
        beta_num = np.sum(new_grad * (new_grad - prev_grad))
        beta = max(0.0, beta_num / prev_grad_norm_sq) if prev_grad_norm_sq > 1e-30 else 0.0

        d = -new_grad + beta * d
        # re-project to tangent
        radial = np.sum(d * current, axis=1)
        d -= radial[:, np.newaxis] * current

        prev_grad = new_grad
        prev_grad_norm_sq = np.sum(new_grad * new_grad)

    return _normalize(best[0]), best[1]


# ---------------------------------------------------------------------------
# L-BFGS (ambient, with projection) -- fast local convergence
# ---------------------------------------------------------------------------
def _lbfgs(pts, time_limit):
    """Manual L-BFGS on R^{3n} with sphere projection after each step."""
    start = time.time()
    x = _normalize(pts.copy()).flatten()
    f, g = _energy_and_grad(x.reshape(N, 3))
    g = g.flatten()
    best_x = x.copy()
    best_e = f

    m = 12
    S = np.zeros((m, 3 * N))
    Y = np.zeros((m, 3 * N))
    idx = 0
    valid = 0

    for _it in range(8000):
        if time.time() - start > time_limit - 0.05:
            break

        # ---- two-loop L-BFGS recursion ----
        q = g.copy()
        alphas = np.zeros(m)
        for j in range(m - 1, -1, -1):
            if valid > 0 and j < valid:
                wy = np.dot(Y[j], Y[j])
                if wy > 1e-20:
                    alpha = np.dot(S[j], q) / wy
                    alphas[j] = alpha
                    q = q - alpha * Y[j]
                else:
                    alphas[j] = 0.0
            else:
                alphas[j] = 0.0

        # initial Hessian approx
        if valid > 0:
            sy = np.dot(S[valid - 1], Y[valid - 1])
            yy = np.dot(Y[valid - 1], Y[valid - 1])
            gamma = sy / yy if yy > 1e-20 else 1.0
        else:
            gamma = 1.0
        r = gamma * q

        for j in range(m):
            if valid > 0 and j < valid:
                wy = np.dot(S[j], S[j])
                if wy > 1e-20:
                    beta = np.dot(Y[j], r) / wy
                    rho_j = 1.0 / wy
                    r = r + (rho_j * beta - alphas[j] * rho_j * np.dot(S[j], Y[j])) * S[j]
                else:
                    pass
            else:
                pass

        d = -r

        # ---- backtracking line search ----
        step = 1.0
        c1 = 1e-4
        gd = np.dot(g, d)
        x_new = x + step * d
        x_new = _normalize(x_new.reshape(N, 3)).flatten()
        f_new, g_new = _energy_and_grad(x_new.reshape(N, 3))
        g_new = g_new.flatten()
        sufficient = (f_new <= f + c1 * step * gd)
        for _cnt in range(25):
            if sufficient:
                break
            step *= 0.5
            x_new = x + step * d
            x_new = _normalize(x_new.reshape(N, 3)).flatten()
            f_new, g_new = _energy_and_grad(x_new.reshape(N, 3))
            g_new = g_new.flatten()
            sufficient = (f_new <= f + c1 * step * gd)
        if not sufficient:
            break

        s = x_new - x
        yv = g_new - g
        sy = np.dot(s, yv)
        if sy > 1e-12:
            S[idx] = s
            Y[idx] = yv
            idx = (idx + 1) % m
            valid = min(valid + 1, m)

        x = x_new
        f = f_new
        g = g_new

        if f < best_e:
            best_e = f
            best_x = x.copy()

        if np.linalg.norm(g) < 1e-11:
            break

    return _normalize(best_x.reshape(N, 3)), best_e


# ---------------------------------------------------------------------------
# Gradient sign sanity check (finite difference) -- run once
# ---------------------------------------------------------------------------
def _verify_gradient(pts):
    """Confirm our analytic gradient points in the descent direction."""
    e0, g0 = _energy_and_grad(pts)
    h = 1e-6
    i = 0
    # pick a tangent direction
    v = np.zeros(3)
    v[0] = 1.0
    v = v - np.dot(v, pts[i]) * pts[i]
    vn = np.linalg.norm(v)
    if vn < 1e-12:
        return True
    v = v / vn
    pert = pts.copy()
    pert[i] = _normalize(pts[i] + h * v)
    e1 = _energy_fast(pert)
    num = (e1 - e0) / h
    ana = np.dot(g0[i], v)
    # descent direction: gradient dot v should have same sign as numeric
    return (num * ana > 0) or abs(num) < 1e-4


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def construct():
    t0 = time.time()
    elapsed = lambda: time.time() - t0
    remaining = lambda: _DEADLINE - _MARGIN - elapsed()

    # candidate starts
    starts = [_fibonacci_sphere(N), _halton_sphere(N)]

    # one-time gradient sanity check (cheap)
    if remaining() > 0.5:
        _verify_gradient(starts[0])

    best_x = starts[0].copy()
    best_e = _energy_fast(best_x)

    # Phase 1: L-BFGS from each start (bulk of budget)
    budget1 = max(0.0, remaining() * 0.45)
    for st in starts:
        if remaining() <= 1.0:
            break
        x, e = _lbfgs(st, min(budget1, remaining() - 0.5))
        if e < best_e:
            best_e = e
            best_x = x.copy()

    # Phase 2: Riemannian CG polish (curvature-aware, escapes shallow minima)
    if remaining() > 2.0:
        x, e = _riemannian_cg(best_x, min(remaining() * 0.35, remaining() - 0.5))
        if e < best_e:
            best_e = e
            best_x = x.copy()

    # Phase 3: final L-BFGS micro-polish
    if remaining() > 2.0:
        x, e = _lbfgs(best_x, remaining() - 0.5)
        if e < best_e:
            best_e = e
            best_x = x.copy()

    # Final Riemannian CG micro-polish if time
    if remaining() > 1.5:
        x, e = _riemannian_cg(best_x, remaining() - 0.5)
        if e < best_e:
            best_e = e
            best_x = x.copy()

    best_x = _normalize(best_x)

    # validity guard
    if (not np.all(np.isfinite(best_x))
            or np.any(np.linalg.norm(best_x, axis=1) < 1e-12)):
        return _fibonacci_sphere(N)
    return best_x
