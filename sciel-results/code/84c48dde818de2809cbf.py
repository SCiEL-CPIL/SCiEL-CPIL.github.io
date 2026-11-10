# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.special import lpmv, factorial

_DEADLINE = time.time() + 615.0
_K = 10
_NTH = 70
_NPHI = 130


def _build_harmonics():
    """Return Y (nphi, nth, 2k+1), weights (nth,), and dphi."""
    k = _K
    mu, w = np.polynomial.legendre.leggauss(_NTH)
    theta = np.arccos(mu)
    phi = np.linspace(0.0, 2.0 * np.pi, _NPHI, endpoint=False)
    dphi = 2.0 * np.pi / _NPHI

    Y = np.zeros((_NPHI, _NTH, 2 * k + 1), dtype=np.float64)
    for idx, m in enumerate(range(-k, k + 1)):
        am = abs(m)
        N = np.sqrt((2 * k + 1) / (4.0 * np.pi)
                    * float(factorial(k - am, exact=True))
                    / float(factorial(k + am, exact=True)))
        P = lpmv(am, k, mu)
        if m == 0:
            Y[:, :, idx] = N * P
        elif m > 0:
            Y[:, :, idx] = np.sqrt(2.0) * N * P * np.cos(m * phi)[:, None]
        else:
            Y[:, :, idx] = np.sqrt(2.0) * N * P * np.sin(am * phi)[:, None]
    return Y, w, dphi


def _objective_and_grad(c, Y, w, dphi):
    """
    Compute R(c) = ||f||_4 / ||f||_2 and its gradient w.r.t. c.
    f = Y @ c, where Y has shape (nphi, nth, 2k+1).
    """
    f = Y @ c
    f2 = f * f
    f3 = f * f2
    f4 = f2 * f2

    # ||f||_2^2 = sum over grid of f^2 * w * dphi
    n2_sq = np.sum(f2 * w[None, :], axis=1)
    n2_sq = np.sum(n2_sq) * dphi

    # ||f||_4^4 = sum over grid of f^4 * w * dphi
    n4_4 = np.sum(f4 * w[None, :], axis=1)
    n4_4 = np.sum(n4_4) * dphi

    if n2_sq < 1e-12:
        return 1e12, np.zeros_like(c)

    n2 = np.sqrt(n2_sq)
    n4 = n4_4 ** 0.25

    R = n4 / n2

    # Gradient of R = ||f||_4 / ||f||_2
    # dA/dc where A = ||f||_4
    # dA/dc = A^{-3} * dphi * sum_{i,j} f[i,j]^3 * w[j] * Y[i,j,m]
    grad_n4 = dphi * np.einsum('ij,ijm,j->m', f3, Y, w)
    dA_dc = (n4 ** (-3)) * grad_n4

    # dB/dc where B = ||f||_2
    # dB/dc = (1/n2) * 2 * dphi * sum_{i,j} f[i,j] * w[j] * Y[i,j,m]
    grad_n2 = 2.0 * dphi * np.einsum('ij,ijm,j->m', f, Y, w)
    dB_dc = (1.0 / n2) * grad_n2

    grad_R = (1.0 / n2) * dA_dc - (n4 / (n2 ** 2)) * dB_dc

    return R, grad_R


def _sphere_minimize(x0, Y, w, dphi, maxiter=2000, tol=1e-14):
    """
    Minimize R(c) on the unit sphere using gradient descent with projection.
    Uses adaptive step size.
    """
    c = x0.copy()
    norm_c = np.linalg.norm(c)
    if norm_c < 1e-12:
        c = np.random.randn(len(c))
        c = c / np.linalg.norm(c)
    else:
        c = c / norm_c
        
    R, grad = _objective_and_grad(c, Y, w, dphi)
    best_c = c.copy()
    best_R = R

    # Project gradient onto tangent space
    grad_tan = grad - np.dot(grad, c) * c

    step = 0.1
    for it in range(maxiter):
        if time.time() > _DEADLINE - 30.0:
            break
        if np.linalg.norm(grad_tan) < tol:
            break

        # Try step
        c_new = c + step * grad_tan
        norm_new = np.linalg.norm(c_new)
        if norm_new < 1e-12:
            step *= 0.5
            continue
        c_new = c_new / norm_new
        R_new, grad_new = _objective_and_grad(c_new, Y, w, dphi)

        if R_new < R:
            c = c_new
            R = R_new
            grad = grad_new
            grad_tan = grad - np.dot(grad, c) * c
            step *= 1.5
            if R < best_R:
                best_R = R
                best_c = c.copy()
        else:
            step *= 0.5
            if step < 1e-16:
                break

    return best_c, best_R


def _make_inits(rng, k):
    """Diverse initial directions."""
    inits = []
    dim = 2 * k + 1

    # Highest order m = ±k
    for m in [k, -k]:
        v = np.zeros(dim)
        v[m + k] = 1.0
        inits.append(v)

    # Pair of m = ±k
    v = np.zeros(dim)
    v[k + k] = 1.0
    v[-k + k] = 1.0
    inits.append(v / np.linalg.norm(v))

    # Next highest m = ±(k-1)
    for m in [k-1, -(k-1)]:
        v = np.zeros(dim)
        v[m + k] = 1.0
        inits.append(v)
    v = np.zeros(dim)
    v[k-1 + k] = 1.0
    v[-(k-1) + k] = 1.0
    inits.append(v / np.linalg.norm(v))

    # m = ±(k-2)
    for m in [k-2, -(k-2)]:
        v = np.zeros(dim)
        v[m + k] = 1.0
        inits.append(v)
    v = np.zeros(dim)
    v[k-2 + k] = 1.0
    v[-(k-2) + k] = 1.0
    inits.append(v / np.linalg.norm(v))

    # Random
    for _ in range(30):
        v = rng.standard_normal(dim)
        inits.append(v / np.linalg.norm(v))

    # Single dominant m
    for m in range(-k, k + 1):
        v = np.zeros(dim)
        v[m + k] = 1.0
        inits.append(v)

    # Paired cos/sin
    for m in range(1, k + 1):
        v = np.zeros(dim)
        v[m + k] = 1.0
        v[-m + k] = 1.0
        inits.append(v / np.linalg.norm(v))

    # Alternating signs
    v = np.ones(dim)
    v[::2] = -1.0
    inits.append(v / np.linalg.norm(v))

    # Random sparse
    for _ in range(10):
        v = np.zeros(dim)
        idx = rng.choice(dim, size=rng.integers(2, 10), replace=False)
        v[idx] = rng.standard_normal(len(idx))
        if np.linalg.norm(v) > 0:
            inits.append(v / np.linalg.norm(v))

    return inits


def rudin_problem10() -> np.ndarray:
    Y, w, dphi = _build_harmonics()
    k = _K
    rng = np.random.default_rng(12345)
    inits = _make_inits(rng, k)

    best_c = inits[0].copy()
    best_val = np.inf

    # Phase 1: Run gradient descent from many diverse starts
    scored = []
    for i, x0 in enumerate(inits):
        if time.time() > _DEADLINE - 120.0:
            break
        c, val = _sphere_minimize(x0, Y, w, dphi, maxiter=500)
        scored.append((val, c))

    scored.sort(key=lambda t: t[0])
    top = [c for _, c in scored[:10]]
    if not top:
        top = [best_c]

    # Phase 2: Refine top candidates with more iterations
    for x0 in top:
        if time.time() > _DEADLINE - 60.0:
            break
        c, val = _sphere_minimize(x0, Y, w, dphi, maxiter=2000, tol=1e-15)
        if val < best_val:
            best_val = val
            best_c = c

    # Phase 3: Random restarts from perturbed best
    while time.time() < _DEADLINE - 30.0:
        x0 = best_c + 0.3 * rng.standard_normal(2 * k + 1)
        x0 = x0 / np.linalg.norm(x0)
        c, val = _sphere_minimize(x0, Y, w, dphi, maxiter=500)
        if val < best_val:
            best_val = val
            best_c = c

    out = best_c.astype(np.float64)
    out = out / np.linalg.norm(out)
    return out


# EVOLVE-BLOCK-END
