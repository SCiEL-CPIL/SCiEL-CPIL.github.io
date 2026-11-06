"""Design a symmetric 32x32 TM photonic-crystal unit cell with a large 4th-5th band gap.

Strategy: parameterized centrally-symmetric disk (even angular harmonics) +
FFT-based plane-wave surrogate band solver (M matrix is k-independent, so one
eigendecomposition yields all bands) + coarse-to-fine search, with a hard
wall-clock deadline guard.
"""
import math
import time
import numpy as np
from scipy.optimize import minimize

_DEADLINE = 205.0  # wall-clock safety margin (evaluator kills at ~230s)


def _build_eps_grid(R0, a4, a8, a12, a16, size=32):
    """Return a size x size grid of 1 (dielectric) / 0 (air)."""
    theta = (np.arange(size) + 0.5) / size * 2.0 * math.pi - math.pi
    c, s = np.cos(theta), np.sin(theta)
    r = (R0
         + a4 * (c**4 - 6 * c**2 * s**2 + s**4)
         + a8 * (c**8 - 28 * c**6 * s**2 + 70 * c**4 * s**4 - 28 * c**2 * s**6 + s**8)
         + a12 * (c**12 - 66 * c**10 * s**2 + 495 * c**8 * s**4 - 924 * c**6 * s**6 + 495 * c**4 * s**8 - 66 * c**2 * s**10 + s**12)
         + a16 * (c**16 - 80 * c**14 * s**2 + 1001 * c**12 * s**4 - 3432 * c**10 * s**6 + 3432 * c**8 * s**8 - 1001 * c**6 * s**10 + 3432 * c**4 * s**12 - 80 * c**2 * s**14 + s**16))
    x = (np.arange(size) + 0.5) / size - 0.5
    yy, xx = np.meshgrid(x, x, indexing="ij")
    rr = np.sqrt(xx * xx + yy * yy)
    return (rr < r).astype(int)


def _build_M(eps_grid, Np):
    """Build the k-independent matrix M for the plane-wave Galerkin method.
    Eigenvalues lambda of M give omega = 2*pi*sqrt(lambda) for all k simultaneously.
    """
    n = eps_grid.shape[0]
    # eps^{-1} Fourier coefficients
    inv_eps = np.zeros((n, n), dtype=float)
    inv_eps[eps_grid > 0] = 1.0 / 12.0
    inv_eps[eps_grid == 0] = 1.0
    inv_eps_f = np.fft.fft2(inv_eps) / (n * n)

    idx = np.arange(-Np, Np + 1)
    N = 2 * Np + 1
    M = np.zeros((N * N, N * N))
    # M_{mn} = inv_eps_f[(m-n)]  (in 2D index space)
    for im in range(N):
        for jm in range(N):
            for in_ in range(N):
                for jn_ in range(N):
                    dm = im - in_
                    dn = jm - jn_
                    if -N < dm < N and -N < dn < N:
                        M[im * N + jm, in_ * N + jn_] = inv_eps_f[Np + dm, Np + dn]
    # Subtract identity
    M[np.arange(N * N), np.arange(N * N)] -= 1.0
    return M


def _surrogate_score(params, Np=4):
    """Compute gap-midgap ratio using the surrogate solver."""
    R0, a4, a8, a12, a16 = params
    if R0 <= 0.05 or R0 >= 0.5:
        return -1.0
    grid = _build_eps_grid(R0, a4, a8, a12, a16, 32)
    if grid.sum() < 5 or grid.sum() > 32 * 32 - 5:
        return -1.0
    M = _build_M(grid, Np)
    eigvals = np.linalg.eigvalsh(M)
    pos = eigvals[eigvals > 1e-10]
    if len(pos) < 5:
        return -1.0
    omegas = 2.0 * math.pi * np.sqrt(pos)
    omegas.sort()
    o4, o5 = omegas[3], omegas[4]
    denom = o4 + o5
    if denom <= 0:
        return -1.0
    return 2.0 * (o5 - o4) / denom


def design(problem):
    """Return a 32x32 symmetric unit cell (1=dielectric, 0=air)."""
    size = problem["pixels"]
    t_start = time.time()

    def _fallback_disk(R0=0.2):
        return _build_eps_grid(R0, 0.0, 0.0, 0.0, 0.0, size)

    best_cell = _fallback_disk(0.2)
    best_score = _surrogate_score([0.2, 0, 0, 0, 0], Np=4)
    best_params = [0.2, 0, 0, 0, 0]

    rng = np.random.default_rng(42)

    # Coarse random search
    for _ in range(50):
        if time.time() > _DEADLINE - 20:
            break
        R0 = rng.uniform(0.10, 0.42)
        a4 = rng.uniform(-0.10, 0.10)
        a8 = rng.uniform(-0.06, 0.06)
        a12 = rng.uniform(-0.04, 0.04)
        a16 = rng.uniform(-0.03, 0.03)
        p = [R0, a4, a8, a12, a16]
        sc = _surrogate_score(p, Np=4)
        if sc > best_score:
            best_score = sc
            best_params = p
            best_cell = _build_eps_grid(*p, size)

    # Local refinement with Nelder-Mead
    if time.time() < _DEADLINE - 15 and best_score > -0.5:
        def _neg_score(p):
            return -_surrogate_score(p, Np=4)
        try:
            res = minimize(_neg_score, best_params, method="Nelder-Mead",
                           options={"maxiter": 200, "xatol": 1e-4, "fatol": 1e-4})
            if time.time() < _DEADLINE - 5:
                sc = -res.fun
                if sc > best_score:
                    best_score = sc
                    best_params = res.x.tolist()
                    best_cell = _build_eps_grid(*best_params, size)
        except Exception:
            pass

    # Second local search from a different starting point if time permits
    if time.time() < _DEADLINE - 10:
        p2 = [rng.uniform(0.18, 0.32), rng.uniform(-0.06, 0.06),
              rng.uniform(-0.04, 0.04), rng.uniform(-0.02, 0.02),
              rng.uniform(-0.02, 0.02)]
        sc2 = _surrogate_score(p2, Np=4)
        if sc2 > best_score:
            best_score = sc2
            best_params = p2
            best_cell = _build_eps_grid(*p2, size)

    best_cell = [list(map(int, row)) for row in best_cell]
    return best_cell
