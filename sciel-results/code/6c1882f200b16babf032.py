# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = 620.0  # seconds wall-clock budget (safety margin under 630)
_START = time.monotonic()


def _elapsed():
    return time.monotonic() - _START


def _compute_R(f):
    """R = ||f||_4 / ||f||_2 via 70x130 Gauss-Legendre quadrature."""
    x, w = np.polynomial.legendre.leggauss(70)
    phi = 2.0 * np.pi * np.arange(130) / 130.0
    theta = np.arccos(x)
    F = f[:, None] * np.exp(1j * phi[None, :])
    F = F.reshape(70, 130)
    W = w[:, None]
    l2 = np.sqrt(np.sum((F.real**2 + F.imag**2) * W))
    l4 = np.sqrt(np.sum((F.real**2 + F.imag**2)**2 * W))
    return l4 / l2


def _precompute_basis():
    """Return real orthonormal degree-10 harmonics sampled on the 70x130 grid.
    Shape (21, 70, 130)."""
    from scipy.special import lpmv
    x, w = np.polynomial.legendre.leggauss(70)
    theta = np.arccos(x)
    phi = 2.0 * np.pi * np.arange(130) / 130.0
    k = 10
    N = np.zeros(2 * k + 1)
    for m in range(0, k + 1):
        N[m] = np.sqrt(21.0 / (4.0 * np.pi) *
                       np.math.factorial(k - m) / np.math.factorial(k + m))
    basis = np.zeros((2 * k + 1, 70, 130))
    for m in range(-k, k + 1):
        idx = m + k
        am = abs(m)
        P = lpmv(am, k, x)
        if m == 0:
            basis[idx] = N[am] * P[:, None] * np.ones((1, 130))
        elif m > 0:
            basis[idx] = np.sqrt(2.0) * N[am] * P[:, None] * np.cos(m * phi)[None, :]
        else:
            basis[idx] = np.sqrt(2.0) * N[am] * P[:, None] * np.sin(-m * phi)[None, :]
    return basis, x, w


def _compute_R_from_precomputed(c, basis, w):
    f = np.tensordot(c, basis, axes=([0], [0]))  # (70,130)
    W = w[:, None]
    f2 = f * f
    l2 = np.sqrt(np.sum(f2 * W))
    l4 = np.sqrt(np.sum(f2 * f2 * W))
    return l4 / l2


def _compute_grad_R(c, basis, w):
    """Gradient of R = l4/l2 w.r.t. c. Returns (grad, R)."""
    f = np.tensordot(c, basis, axes=([0], [0]))
    W = w[:, None]
    f2 = f * f
    l2sq = np.sum(f2 * W)
    l4sq = np.sum(f2 * f2 * W)
    l2 = np.sqrt(l2sq)
    l4 = np.sqrt(l4sq)
    R = l4 / l2
    # dR/dc_i = (1/l2) dL4/dc_i - (l4/l2^2) dL2/dc_i
    # dL2/dc_i = (1/l2) sum(f * W * basis_i)
    # dL4/dc_i = (1/l4) sum(f^3 * W * basis_i)
    f3 = f * f2
    dL2 = np.tensordot(f * W, basis, axes=([0, 1], [1, 2]))  # (21,)
    dL4 = np.tensordot(f3 * W, basis, axes=([0, 1], [1, 2]))
    grad = (dL4 / l4) / l2 - (l4 / l2**2) * (dL2 / l2)
    return grad, R


def _local_optimize(c0, basis, w, max_iter=4000, deadline=_DEADLINE):
    """L-BFGS-B style gradient descent with line search to minimize R."""
    c = c0.copy()
    grad, R = _compute_grad_R(c, basis, w)
    best_c = c.copy()
    best_R = R
    step = 1e-3
    for it in range(max_iter):
        if _elapsed() > deadline:
            break
        if np.all(np.isnan(grad)) or np.all(np.isinf(grad)):
            break
        gnorm = np.linalg.norm(grad)
        if gnorm < 1e-12:
            break
        direction = -grad / gnorm
        # Backtracking line search
        accepted = False
        for ls in range(30):
            trial = c + step * direction
            _, R_try = _compute_grad_R(trial, basis, w)
            if R_try < R - 1e-12 * R:
                c = trial
                R = R_try
                grad = _compute_grad_R(c, basis, w)[0]
                if R < best_R:
                    best_R = R
                    best_c = c.copy()
                step = min(step * 1.05, 0.1)
                accepted = True
                break
            step *= 0.5
        if not accepted:
            step *= 0.5
            if step < 1e-12:
                break
    return best_c, best_R


def rudin_problem10() -> np.ndarray:
    k = 10
    basis, x, w = _precompute_basis()

    # Seed: all ones (matches baseline behavior)
    seeds = [np.ones(2 * k + 1, dtype=np.float64)]

    # Add a few deterministic diverse seeds
    rng = np.random.default_rng(12345)
    for _ in range(15):
        seeds.append(rng.standard_normal(2 * k + 1))

    best_c = seeds[0].copy()
    best_R = _compute_R_from_precomputed(best_c, basis, w)

    for s in seeds:
        if _elapsed() > deadline - 30:
            break
        c_opt, R_opt = _local_optimize(s, basis, w, max_iter=3000,
                                       deadline=deadline - 30)
        if R_opt < best_R:
            best_R = R_opt
            best_c = c_opt

    # Final high-precision polish on the best
    c_polish, R_polish = _local_optimize(best_c, basis, w, max_iter=6000,
                                         deadline=deadline)
    if R_polish < best_R:
        best_R = R_polish
        best_c = c_polish

    # Validate
    c = np.asarray(best_c, dtype=np.float64).ravel()
    if c.size != 21 or np.any(~np.isfinite(c)):
        c = np.ones(21, dtype=np.float64)
    amax = np.max(np.abs(c))
    if amax < 1e-3:
        c = c / amax * 1e-2
    elif amax > 1e5:
        c = c / amax * 1e4
    return c


# EVOLVE-BLOCK-END
