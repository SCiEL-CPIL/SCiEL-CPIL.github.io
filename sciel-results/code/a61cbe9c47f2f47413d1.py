import numpy as np
from scipy.special import lpmv, factorial
from scipy.optimize import minimize
import time


def _build_basis():
    """Build flattened quadrature grid and (weighted) basis matrix for degree-10 real harmonics."""
    k = 10
    n_m = 2 * k + 1
    mu, w = np.polynomial.legendre.leggauss(70)
    n_phi = 130
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    d_phi = 2 * np.pi / n_phi

    mu_flat, phi_flat = np.meshgrid(mu, phi, indexing='ij')
    mu_flat = mu_flat.reshape(-1)
    phi_flat = phi_flat.reshape(-1)
    w_flat = np.repeat(w, n_phi)          # (N,)  surface weights per grid point
    n_points = mu_flat.size

    def get_norm(m_abs):
        val = (21.0 / (4.0 * np.pi)) * (
            factorial(10 - m_abs, exact=False) / factorial(10 + m_abs, exact=False))
        return np.sqrt(val)

    A = np.zeros((n_points, n_m))
    # m = 0
    A[:, 10] = get_norm(0) * lpmv(0, 10, mu_flat)
    # m > 0: cos (index 10+m) and sin (index 10-m)
    for m in range(1, k + 1):
        P = lpmv(m, 10, mu_flat)
        N = np.sqrt(2.0) * get_norm(m)
        A[:, 10 + m] = N * P * np.cos(m * phi_flat)
        A[:, 10 - m] = N * P * np.sin(m * phi_flat)

    B = A * w_flat[:, None]              # pre-weighted basis for fast gradient sums
    return A, B, w_flat, d_phi, n_points


def _make_objective(A, B, w_flat, d_phi, n_m, deadline):
    """Return (obj, grad) for R(u)=||f||_4/||f||_2 with u on the unit sphere (scale-invariant)."""
    def compute_R(c):
        norm_c = np.linalg.norm(c)
        if norm_c < 1e-14:
            return 1e10, np.zeros(n_m)
        u = c / norm_c
        f = A @ u
        f2 = f * f
        f4 = f2 * f2
        S2 = d_phi * np.dot(w_flat, f2)
        S4 = d_phi * np.dot(w_flat, f4)
        if S2 < 1e-14:
            return 1e10, np.zeros(n_m)
        R = S4 ** 0.25 / np.sqrt(S2)
        w_f3 = d_phi * (B.T @ (f ** 3))
        w_f = d_phi * (B.T @ f)
        coef1 = S4 ** (-0.75) * S2 ** (-0.5)
        coef2 = S4 ** 0.25 * S2 ** (-1.5)
        grad_u = coef1 * w_f3 - coef2 * w_f
        grad_tangent = grad_u - np.dot(u, grad_u) * u
        grad_c = grad_tangent / norm_c
        return R, grad_c

    def obj(c):
        R, g = compute_R(c)
        return R, g

    return obj


def _sphere_polish(c, obj, max_iter, initial_lr, deadline):
    """Gradient descent with backtracking on the unit sphere (constraint removed by re-normalization)."""
    c = np.asarray(c, dtype=np.float64).copy()
    n_m = c.size
    norm_c = np.linalg.norm(c)
    if norm_c < 1e-14:
        return c
    u = c / norm_c
    lr = initial_lr
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        R_curr, grad_u = obj(u)
        grad_tangent = grad_u - np.dot(u, grad_u) * u
        if np.linalg.norm(grad_tangent) < 1e-13:
            break
        best_R = R_curr
        best_u = u
        step = lr
        for _ in range(25):
            if step < 1e-15:
                break
            u_new = u - step * grad_tangent
            n_new = np.linalg.norm(u_new)
            if n_new < 1e-14:
                step *= 0.5
                continue
            u_new = u_new / n_new
            R_new, _ = obj(u_new)
            if R_new < best_R:
                best_R = R_new
                best_u = u_new
            step *= 0.7
        if best_R < R_curr:
            u = best_u
            lr = min(step / 0.7, initial_lr * 2.0)
        else:
            lr *= 0.5
        if lr < 1e-10:
            break
    return u


def rudin_problem10() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580.0   # safety margin well before the 630 s kill

    k = 10
    n_m = 2 * k + 1
    A, B, w_flat, d_phi, n_points = _build_basis()
    obj = _make_objective(A, B, w_flat, d_phi, n_m, deadline)

    def val(c):
        return obj(c)[0]

    rng = np.random.default_rng(123)
    candidates = []

    # Pure modes
    for m in range(-k, k + 1):
        c = np.zeros(n_m)
        c[10 + m] = 1.0
        candidates.append(c)

    # Even-m cos pairs
    for m1 in range(0, k + 1, 2):
        for m2 in range(m1 + 2, k + 1, 2):
            for amp in [0.3, 0.7, 1.0]:
                c = np.zeros(n_m)
                c[10 + m1] = 1.0
                c[10 + m2] = amp
                c /= np.linalg.norm(c)
                candidates.append(c)

    # Symmetric / antisymmetric structured patterns
    for s in range(60):
        c = np.zeros(n_m)
        n_nonzero = rng.integers(3, 15)
        idx = rng.choice(n_m, size=n_nonzero, replace=False)
        for i in idx:
            c[i] = rng.standard_normal()
        if s % 3 == 0:
            c = (c + c[::-1]) / 2.0
        elif s % 3 == 1:
            c = (c - c[::-1]) / 2.0
        nrm = np.linalg.norm(c)
        if nrm > 1e-12:
            candidates.append(c / nrm)

    # Low-m emphasis
    probs = np.array([1.0 / (1 + abs(m)) for m in range(-k, k + 1)])
    probs /= probs.sum()
    for _ in range(80):
        c = np.zeros(n_m)
        n_nonzero = rng.integers(2, 10)
        idx = rng.choice(n_m, size=n_nonzero, replace=False, p=probs)
        for i in idx:
            c[i] = rng.standard_normal() * (1.0 / (1 + abs(i - 10)))
        nrm = np.linalg.norm(c)
        if nrm > 1e-12:
            candidates.append(c / nrm)

    # Fully random
    for _ in range(150):
        c = rng.standard_normal(n_m)
        candidates.append(c / np.linalg.norm(c))

    # Evaluate candidates (fast matmul objective)
    scored = []
    for c in candidates:
        if time.time() > deadline - 200:
            break
        scored.append((val(c), c))
    scored.sort(key=lambda x: x[0])

    best_c = None
    best_val = np.inf

    # Optimize the best initial candidates with L-BFGS-B (analytic gradient)
    for _, c_init in scored[:40]:
        if time.time() > deadline - 40:
            break
        try:
            res = minimize(obj, c_init, jac=True, method='L-BFGS-B',
                           options={'maxiter': 1200, 'ftol': 1e-18, 'gtol': 1e-12, 'maxls': 80})
            if res.fun < best_val and np.all(np.isfinite(res.x)):
                best_val = res.fun
                best_c = res.x.copy()
        except Exception:
            pass

    # Sphere-projected backtracking polish (constraint removed, exact local solve on manifold)
    if best_c is not None:
        polished = _sphere_polish(best_c, obj, 1500, 0.05, deadline - 10)
        v = val(polished)
        if v < best_val:
            best_val = v
            best_c = polished.copy()

    # Multi-start perturbations around the best, polished on the sphere
    if best_c is not None and time.time() < deadline - 20:
        for _ in range(8):
            c_pert = best_c + rng.standard_normal(n_m) * 0.1
            c_pert /= np.linalg.norm(c_pert)
            polished = _sphere_polish(c_pert, obj, 900, 0.03, deadline - 10)
            v = val(polished)
            if v < best_val:
                best_val = v
                best_c = polished.copy()
            if time.time() > deadline - 10:
                break

    # Final high-precision sphere polish
    if best_c is not None and time.time() < deadline - 5:
        polished = _sphere_polish(best_c, obj, 2000, 0.02, deadline - 3)
        v = val(polished)
        if v < best_val:
            best_val = v
            best_c = polished.copy()

    # Validate / scale into the allowed range
    best_c = np.asarray(best_c, dtype=np.float64)
    if best_c.shape != (n_m,) or not np.all(np.isfinite(best_c)):
        return np.ones(n_m)
    max_abs = np.max(np.abs(best_c))
    if max_abs < 1e-3 or max_abs > 1e5:
        if max_abs < 1e-3:
            best_c = best_c * (1.0 / max_abs) if max_abs > 0 else np.ones(n_m)
        else:
            best_c = best_c * (1e5 / max_abs)
        if not np.all(np.isfinite(best_c)):
            return np.ones(n_m)
    return best_c
