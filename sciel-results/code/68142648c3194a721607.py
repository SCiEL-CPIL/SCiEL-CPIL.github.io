"""Uncertainty-inequality test-function tuning via scale-invariant fixed-point search."""
# EVOLVE-BLOCK-START
import time
import numpy as np

DEGREES = [0, 4, 8, 12]
N_BASIS = len(DEGREES)


def _hermite_prob(x, n):
    """Probabilists' Hermite polynomial Hn(x)."""
    if n == 0:
        return np.ones_like(x)
    if n == 4:
        return x**4 - 6*x**2 + 3
    if n == 8:
        return x**8 - 28*x**6 + 210*x**4 - 420*x**2 + 105
    if n == 12:
        return (x**12 - 66*x**10 + 1540*x**8 - 15400*x**6
                + 69300*x**4 - 118800*x**2 + 945)
    raise ValueError("Unsupported Hermite degree")


def _precompute_pos_matrices():
    nq = 200
    u, w = np.polynomial.hermite.hermgauss(nq)
    x = u * np.sqrt(2.0)
    wx = w * np.sqrt(2.0)
    H = np.zeros((N_BASIS, nq))
    for i, deg in enumerate(DEGREES):
        H[i] = _hermite_prob(x, deg)
    WH = w[:, None] * H
    A = WH @ H.T
    x2 = x**2
    B = (WH * x2[None, :]) @ H.T
    return A, B


def _precompute_freq_matrices():
    nq = 200
    u, w = np.polynomial.hermite.hermgauss(nq)
    xi = u / np.sqrt(2.0)
    wxi = w / np.sqrt(2.0)
    V = np.zeros((N_BASIS, nq))
    for i, deg in enumerate(DEGREES):
        V[i] = _hermite_prob(u / 2.0, deg)
    WV = w[:, None] * V
    M = WV @ V.T
    xi2 = xi**2
    N = (WV * xi2[None, :]) @ V.T
    return M, N


A_POS, B_POS = _precompute_pos_matrices()
M_FREQ, N_FREQ = _precompute_freq_matrices()

# Frequency coefficient scaling c = a * S, S_k = (-i)^k (sqrt(2))^k
_SCALES = np.array([([1, -1j, -1, 1j][d % 4]) * (np.sqrt(2.0) ** d) for d in DEGREES])
_S = _SCALES

# a12 = (a0 + 3 a4 + 105 a8)/945  ->  a = C @ p,  p = (a0, a4, a8)
_C = np.array([[1.0, 0.0, 0.0],
               [0.0, 1.0, 0.0],
               [0.0, 0.0, 1.0],
               [1.0/945.0, 3.0/945.0, 105.0/945.0]])
_Ap = _C.T @ A_POS @ _C
_Bp = _C.T @ B_POS @ _C
_Sp = _S @ _C
_Mp = _Sp @ M_FREQ @ _Sp.T
_Np = _Sp @ N_FREQ @ _Sp.T

# Precompute Cholesky for normalization
try:
    _LA = np.linalg.cholesky(_Ap)
    _LM = np.linalg.cholesky(_Mp)
except np.linalg.LinAlgError:
    _LA = np.linalg.cholesky(_Ap + 1e-12*np.eye(3))
    _LM = np.linalg.cholesky(_Mp + 1e-12*np.eye(3))


def _norm_a(p):
    n = np.linalg.norm(_LA.T @ p)
    return p / n if n > 1e-15 else None


def _norm_c(p):
    n = np.linalg.norm(_LM.T @ p)
    return p / n if n > 1e-15 else None


def _score_p(p):
    """Exact score (product) for p=(a0,a4,a8)."""
    a = _C @ p
    c = a * _S
    I0 = float(np.dot(a, A_POS @ a))
    I2 = float(np.dot(a, B_POS @ a))
    J0 = float(np.vdot(c, M_FREQ @ c).real)
    J2 = float(np.vdot(c, N_FREQ @ c).real)
    if min(I0, I2, J0, J2) < 1e-14:
        return 1e30
    return float(np.sqrt((I2 * J2) / (I0 * J0)))


def _fixed_point(p0, max_iter=4000, tol=1e-13):
    """Scale-invariant fixed-point iteration on the stationary system.
    Returns (p, R) or (None, None) on failure."""
    p = _norm_a(np.asarray(p0, dtype=float))
    if p is None:
        return None, None
    c = _norm_c(_Sp @ p)
    if c is None:
        return None, None
    R = None
    for _ in range(max_iter):
        a = _C @ p
        B_a = B_POS @ a
        N_c = N_FREQ @ c
        p_new = _norm_a(_Ap @ p + 0.5 * R * (_C.T @ B_a))
        c_new = _norm_c(_Mp @ c + 0.5 * R * (_Sp.T @ N_c))
        if p_new is None or c_new is None:
            return None, None
        p, c = p_new, c_new
        a = _C @ p
        I2 = float(np.dot(a, B_POS @ a))
        J2 = float(np.vdot(c, N_FREQ @ c).real)
        R_new = I2 * J2
        if R is not None and abs(R_new - R) < tol * max(1.0, abs(R)):
            return p, R_new
        R = R_new
    return p, R


def construct():
    deadline = time.time() + 185.0
    seed = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    best_p, best_score = seed.copy(), _score_p(seed)

    def try_start(p0):
        nonlocal best_p, best_score
        if time.time() > deadline:
            return
        p, _ = _fixed_point(p0)
        if p is None:
            return
        s = _score_p(p)
        if s < best_score:
            best_score, best_p = s, p

    # Deterministic structured starts around the seed
    for scale in (0.05, 0.1, 0.2, 0.4, 0.8):
        for d in ([-1, 0, 0], [0, -1, 0], [0, 0, -1],
                  [1, 0, 0], [0, 1, 0], [0, 0, 1],
                  [-1, 1, 0], [1, -1, 0], [-1, 0, 1], [1, 0, -1],
                  [1, 1, 1], [-1, -1, -1]):
            try_start(seed + scale * np.array(d, dtype=float))

    # Random restarts (fixed seed for reproducibility)
    rng = np.random.default_rng(12345)
    for k in range(400):
        if time.time() > deadline:
            break
        p0 = seed + rng.standard_normal(3) * (0.1 + 0.9 * (k % 10) / 9.0)
        try_start(p0)

    return best_p


# EVOLVE-BLOCK-END
