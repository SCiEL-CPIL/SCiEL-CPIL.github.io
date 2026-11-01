"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50
_DEADLINE = time.time() + 140.0   # safety margin before the 150 s wall-clock kill


def _norm(x, axis=1):
    n = np.linalg.norm(x, axis=axis)
    n = np.where(n < 1e-15, 1.0, n)
    return x / n[:, None] if x.ndim == 2 else x / n


def _true_radius(P):
    # Exact covering radius: largest circumradius of convex-hull facets
    # (equivalently max over nearest-neighbour angular distance).
    # Compute via spherical Voronoi inradii using the convex hull.
    from scipy.spatial import ConvexHull
    P = np.asarray(P, dtype=float)
    P = _norm(P)
    try:
        hull = ConvexHull(P)
        # inradius of each facet == radius of empty cap == angular distance from facet
        # centre to the nearest vertex (in radians), since vertices are on unit sphere.
        radii = hull.simplices
        # Use scipy's inradius: distance from hull to each facet plane through its centroid
        # For points on the unit sphere, the inradius of a facet equals the angular radius.
        inradii = hull.inradius  # distance from centroid of hull to each facet
        # For a unit-sphere point set, facet circumradius (angular) = arccos( -? )
        # Simpler robust route: brute-force nearest angular distance on a dense grid.
    except Exception:
        pass
    # Brute force dense sampling of the sphere for the true max nearest distance.
    # Use a Fibonacci grid of many points as a fine approximation.
    M = 40000
    i = np.arange(M) + 0.5
    z = 1.0 - 2.0 * i / M
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, 1.0))
    Y = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    # nearest distance to each Y
    # chunk to control memory
    best = 0.0
    chunk = 4000
    for s in range(0, M, chunk):
        Yc = Y[s:s + chunk]
        D = 1.0 - Yc @ P.T          # 1 - cos(angle)
        nearest = D.min(axis=1)
        ang = np.arccos(np.clip(1.0 - nearest, -1.0, 1.0))
        best = max(best, float(ang.max()))
    return best


def _softmax_obj(P, beta):
    """Differentiable surrogate of the covering radius (radians)."""
    P = _norm(P)
    M = 2000
    i = np.arange(M) + 0.5
    z = 1.0 - 2.0 * i / M
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, 1.0))
    Y = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    # cos of angular distance from each sample to each point
    C = Y @ P.T                     # (M, N) cos values
    # angular distance d = arccos(C); we want max over i of d for each sample
    # use log-sum-exp on -C (since larger distance <=> smaller cos)
    # -C large when C small (far). logsumexp(-beta*(-C)) = logsumexp(beta*C)
    # Actually maximize d ~ -C. Use logsumexp(-beta * C) to emphasise far points.
    XC = -beta * C
    XC = XC - XC.max(axis=1, keepdims=True)
    LSE = np.log(np.exp(XC).sum(axis=1)) / beta   # approximates max_i(-C_i) = -min C
    # convert -min C (a value in [-1,1]) to angle: angle = arccos(min C)
    minC = -np.tanh(LSE)  # map to (-1,1) smoothly
    minC = np.clip(minC, -1.0 + 1e-9, 1.0 - 1e-9)
    return float(np.arccos(minC).max())


def _softmax_obj_and_grad(x, beta):
    P = _norm(x)
    M = 1500
    i = np.arange(M) + 0.5
    z = 1.0 - 2.0 * i / M
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, 1.0))
    Y = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    C = Y @ P.T                      # (M, N)
    XC = -beta * C
    XC = XC - XC.max(axis=1, keepdims=True)
    E = np.exp(XC)
    S = E.sum(axis=1, keepdims=True)
    W = E / S                        # softmax weights over points, (M, N)
    # objective f = max_m arccos( minC_m ) where minC_m = -sum W C  (soft min of C)
    minC = -(W * C).sum(axis=1)      # (M,)
    minC = np.clip(minC, -1.0 + 1e-9, 1.0 - 1e-9)
    ang = np.arccos(minC)            # (M,)
    m = int(np.argmax(ang))
    # d f / d minC_m = -1/sqrt(1-minC^2)
    dminC = -1.0 / np.sqrt(np.clip(1.0 - minC * minC, 1e-12, None))
    # only the maximising sample contributes to the max (subgradient)
    gminC = np.zeros(M)
    gminC[m] = dminC[m]
    # d minC / d C = -W
    # d C / d P_j = Y @ e_j  => for point j, dC_j = Y[:, j]
    # grad w.r.t P_j = sum_m gminC_m * (-W[m, j]) * Y[:, j]
    gP = -Y.T @ (gminC[:, None] * W)      # (N, 3)
    # chain rule through normalisation
    Pn = P
    gP = gP - (gP * Pn).sum(axis=1, keepdims=True) * Pn
    return float(ang.max()), gP.flatten()


def _fibonacci(seed=0.0, jitter=0.0):
    rng = np.random.default_rng(seed)
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i + seed
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, 1.0))
    P = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    if jitter > 0:
        P = P + rng.normal(0, jitter, P.shape)
    return _norm(P)


def _antipodal(seed=0.0, jitter=0.0):
    rng = np.random.default_rng(seed)
    P = rng.normal(size=(N, 3))
    P = _norm(P)
    if jitter > 0:
        P = P + rng.normal(0, jitter, P.shape)
    return _norm(P)


def _refine(P0, betas, max_time):
    best = P0.copy()
    best_f = _true_radius(P0)
    for beta in betas:
        if time.time() > max_time - 5:
            break
        try:
            res = minimize(_softmax_obj_and_grad, best.flatten(),
                           args=(beta,), method='L-BFGS-B',
                           jac=True,
                           options={'maxiter': 200, 'gtol': 1e-7,
                                     'maxfun': 400, 'disp': False})
        except Exception:
            continue
        Pn = _norm(res.x.reshape(N, 3))
        f = _true_radius(Pn)
        if f < best_f - 1e-12:
            best = Pn
            best_f = f
    return best, best_f


def construct():
    global _DEADLINE
    start = time.time()
    max_time = min(_DEADLINE, start + 140.0)

    # Build a diverse pool of starting configurations
    starts = []
    starts.append(_fibonacci(seed=0.0, jitter=0.0))
    starts.append(_fibonacci(seed=1.7, jitter=0.0))
    starts.append(_fibonacci(seed=0.0, jitter=0.15))
    starts.append(_fibonacci(seed=3.1, jitter=0.15))
    starts.append(_antipodal(seed=7, jitter=0.0))
    starts.append(_antipodal(seed=42, jitter=0.0))
    starts.append(_antipodal(seed=123, jitter=0.0))

    betas = [8.0, 20.0, 60.0, 150.0, 400.0]

    best = starts[0].copy()
    best_f = _true_radius(best)

    for k, P0 in enumerate(starts):
        if time.time() > max_time - 20:
            break
        P, f = _refine(P0, betas, max_time)
        if f < best_f - 1e-12:
            best = P
            best_f = f

    # Final global polish on the best with high beta
    if time.time() < max_time - 10:
        P, f = _refine(best, [300.0, 800.0], max_time)
        if f < best_f - 1e-12:
            best = P
            best_f = f

    return _norm(best)
