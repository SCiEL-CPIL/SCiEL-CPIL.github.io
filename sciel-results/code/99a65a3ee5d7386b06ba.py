"""Optimize 14 points in 3D to minimize (max_pairwise/min_pairwise)^2.

Strategy: smooth surrogate objective (log-sum-exp max, soft-min) with a
deadzone to avoid the min->0 singularity, refined by BFGS from several
randomly rotated Fibonacci-sphere initializations.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 14
D = 3

# Pair indices for the n(n-1)/2 distinct pairs
_IDX = np.array(
    [(i, j) for i in range(N) for j in range(i + 1, N)], dtype=int
)
_NPAIRS = _IDX.shape[0]


def _fibonacci_sphere(n):
    """Evenly distributed points on the unit sphere."""
    idx = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * idx / n)
    theta = np.pi * (1.0 + 5.0) * idx
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.column_stack((x, y, z))


def _random_rotation_3d(rng):
    """Random orthogonal 3x3 matrix (QR of a Gaussian matrix)."""
    g = rng.standard_normal((3, 3))
    q, r = np.linalg.qr(g)
    d = np.sign(np.diag(r))
    q = q * d
    return q


def _objective_flat(flat, beta, deadzone):
    """Smooth objective: softmax(log d) - log(softmin(d)), squared.

    Encourages all pairwise distances to be equal (minimizes max/min ratio).
    The deadzone prevents the min from collapsing to zero.
    """
    pts = flat.reshape(N, D)
    diff = pts[_IDX[:, 0]] - pts[_IDX[:, 1]]
    d = np.sqrt(np.sum(diff * diff, axis=1))

    logd = np.log(d + 1e-12)
    max_log = np.logaddexp.reduce(logd)
    softmin = -np.logaddexp.reduce(-logd)

    # Push the minimum distance up to the deadzone
    min_d = np.exp(softmin)
    penalty = np.maximum(0.0, deadzone - min_d) ** 2

    ratio = max_log - softmin
    return ratio * ratio + penalty


def _objective_grad_flat(flat, beta, deadzone):
    pts = flat.reshape(N, D)
    diff = pts[_IDX[:, 0]] - pts[_IDX[:, 1]]
    d = np.sqrt(np.sum(diff * diff, axis=1))
    logd = np.log(d + 1e-12)

    max_log = np.logaddexp.reduce(logd)
    softmin = -np.logaddexp.reduce(-logd)

    # weights for softmax (over log d)
    w_max = np.exp(logd - max_log)
    w_max /= w_max.sum()
    # weights for softmin
    w_min = np.exp(-logd - softmin)
    w_min /= w_min.sum()

    # d(log d)/d(pts) = (1/d) * (d(pts)/d(pts))
    # derivative of ratio^2 = 2*ratio * (d max_log - d softmin)
    # d max_log = sum w_max * (1/d_i) * dd_i
    # d softmin = sum w_min * (1/d_i) * dd_i
    # where dd_i = (diff_i)/d_i  (gradient wrt the difference)

    ratio = max_log - softmin
    coeff = 2.0 * ratio

    # accumulate gradient
    grad = np.zeros_like(pts)
    # contribution from max term: +coeff * sum w_max_i * (1/d_i) * (diff_i/d_i)
    # = coeff * sum w_max_i * diff_i / d_i^2
    invd2 = 1.0 / (d * d)
    wmax_eff = coeff * w_max * invd2
    for k in range(_NPAIRS):
        a, b = _IDX[k]
        grad[a] += wmax_eff[k] * diff[k]
        grad[b] -= wmax_eff[k] * diff[k]

    # contribution from softmin term: -coeff * sum w_min_i * (1/d_i) * (diff_i/d_i)
    wmin_eff = -coeff * w_min * invd2
    for k in range(_NPAIRS):
        a, b = _IDX[k]
        grad[a] += wmin_eff[k] * diff[k]
        grad[b] -= wmin_eff[k] * diff[k]

    # deadzone penalty gradient: 2*(deadzone - min_d) * (-d min_d / d pts)
    min_d = np.exp(softmin)
    if min_d < deadzone:
        pcoef = 2.0 * (deadzone - min_d)
        # d(min_d)/d(pts) = min_d * sum w_min_i * (1/d_i) * (diff_i/d_i)
        # = min_d * sum w_min_i * diff_i / d_i^2
        wmin_dz = pcoef * min_d * w_min * invd2
        for k in range(_NPAIRS):
            a, b = _IDX[k]
            grad[a] += wmin_dz[k] * diff[k]
            grad[b] -= wmin_dz[k] * diff[k]

    return grad


def _score(pts):
    diff = pts[_IDX[:, 0]] - pts[_IDX[:, 1]]
    d = np.sqrt(np.sum(diff * diff, axis=1))
    d = d[d > 1e-9]
    if d.size == 0:
        return np.inf
    return (d.max() / d.min()) ** 2


def construct():
    deadline = time.time() + 110.0  # safety margin under 120s
    rng = np.random.default_rng(12345)

    base = _fibonacci_sphere(N)
    # scale so typical distances ~ 1
    base = base * 1.0

    beta = 8.0
    deadzone = 0.2

    best_flat = base.ravel().copy()
    best_score = _score(base)

    n_restarts = 12
    for r in range(n_restarts):
        if time.time() > deadline:
            break
        R = _random_rotation_3d(rng)
        rot = base @ R.T
        # small radial jitter for diversity
        rad = np.linalg.norm(rot, axis=1, keepdims=True)
        jitter = 1.0 + 0.15 * rng.standard_normal((N, 1))
        x0 = rot * jitter
        x0 = x0.ravel()

        try:
            res = minimize(
                _objective_flat,
                x0,
                jac=_objective_grad_flat,
                args=(beta, deadzone),
                method="BFGS",
                options={"maxiter": 800, "gtol": 1e-9, "disp": False},
            )
        except Exception:
            continue
        if res.x is None:
            continue
        cand = res.x.reshape(N, D)
        s = _score(cand)
        if np.isfinite(s) and s < best_score:
            best_score = s
            best_flat = res.x.copy()
            # quick polish with tighter tolerance
            try:
                res2 = minimize(
                    _objective_flat,
                    res.x,
                    jac=_objective_grad_flat,
                    args=(beta, deadzone),
                    method="BFGS",
                    options={"maxiter": 400, "gtol": 1e-12, "disp": False},
                )
                if res2.x is not None:
                    cand2 = res2.x.reshape(N, D)
                    s2 = _score(cand2)
                    if np.isfinite(s2) and s2 < best_score:
                        best_score = s2
                        best_flat = res2.x.copy()
            except Exception:
                pass

    pts = best_flat.reshape(N, D)
    # center the point cloud (translation invariance)
    pts = pts - pts.mean(axis=0)
    return pts
