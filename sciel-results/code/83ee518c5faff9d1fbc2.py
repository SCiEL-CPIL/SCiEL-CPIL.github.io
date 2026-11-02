"""Logarithmic energy minimization for 150 points on the unit sphere."""
import time
import math
import numpy as np

N = 150
_DEADLINE = 120.0  # seconds; leave margin before the 150 s kill.


def _energy_and_grad(P):
    """Return (energy, grad) for P of shape (N,3) on the sphere.

    energy = sum_{i<j} log(1 / |x_i - x_j|)
    grad_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    """
    P = np.asarray(P, dtype=float)
    n = P.shape[0]
    D = P[:, None, :] - P[None, :, :]          # (n, n, 3)
    D2 = np.einsum('ijk,ijk->ij', D, D)
    np.fill_diagonal(D2, 1.0)
    D2 = np.maximum(D2, 1e-30)
    dist = np.sqrt(D2)
    np.fill_diagonal(dist, 1.0)

    M = np.zeros((n, n), dtype=float)
    np.fill_diagonal(M, 1.0)
    E = 0.5 * np.sum(M * (-np.log(dist)))
    G = np.einsum('ij,ijk->ik', 1.0 / D2, D)
    return E, G


def _project(P):
    P = np.asarray(P, float)
    nrm = np.linalg.norm(P, axis=1, keepdims=True)
    nrm = np.maximum(nrm, 1e-30)
    return P / nrm


def _fibonacci(N):
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _sobol(N):
    try:
        from scipy.stats import qmc
        sampler = qmc.Sobol(d=2, scramble=True)
        u = sampler.random(N)
        z = 2.0 * u[:, 0] - 1.0
        phi = 2.0 * np.pi * u[:, 1]
        r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    except Exception:
        return _fibonacci(N)


def _random(N, seed):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((N, 3))
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.maximum(n, 1e-30)


def _grad_step(P, G, step):
    Pn = P + step * G
    return _project(Pn)


def _line_search(P, G, E, max_step, min_step, tol, deadline):
    """Backtracking line search on the sphere."""
    step = max_step
    for _ in range(12):
        if time.time() > deadline:
            return P, E, step
        Pn = _grad_step(P, G, step)
        En, _ = _energy_and_grad(Pn)
        if En < E - 1e-12 * step * np.linalg.norm(G):
            return Pn, En, step
        step *= 0.5
    return P, E, step


def _lbfgs(P, tol, deadline, max_iter=2000, m=10):
    """Simple L-BFGS with line search on the sphere."""
    P = _project(P)
    E, G = _energy_and_grad(P)
    s_list = []
    y_list = []
    step = 1.0 / N
    for it in range(max_iter):
        if time.time() > deadline:
            break
        # build L-BFGS two-loop
        W = G.copy()
        for s, y in zip(reversed(s_list), reversed(y_list)):
            alpha = np.dot(s, y)
            if abs(alpha) < 1e-30:
                continue
            beta = np.dot(y, W) / alpha
            W = W + (beta - alpha) * s / alpha
        # initial step scaling
        rG = np.linalg.norm(G)
        if rG < 1e-30:
            break
        W = W / rG
        # two-loop to get direction
        Q = -W
        for s, y in zip(s_list, y_list):
            alpha = np.dot(s, y)
            if abs(alpha) < 1e-30:
                continue
            beta = np.dot(y, Q) / alpha
            Q = Q + (np.dot(s, Q) / alpha - beta) * s
        # line search
        Pn, En, used = _line_search(P, G, E, step, 1e-12, tol, deadline)
        if En >= E:
            break
        s = Pn - P
        y = _energy_and_grad(Pn)[1] - G
        sy = np.dot(s, y)
        if sy > 1e-20:
            s_list.append(s)
            y_list.append(y)
            if len(s_list) > m:
                s_list.pop(0)
                y_list.pop(0)
        P = Pn
        E = En
        step = used
    return P, E


def _gradient_descent(P, tol, deadline, max_iter=3000, step0=None):
    P = _project(P)
    E, G = _energy_and_grad(P)
    step = step0 if step0 is not None else 1.0 / N
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        rG = np.linalg.norm(G)
        if rG < 1e-12:
            break
        Pn, En, used = _line_search(P, G, E, step, 1e-12, tol, deadline)
        if En >= E:
            break
        P = Pn
        E = En
        step = used
    return P, E


def _polish(P, tol, deadline):
    """High-precision Newton-ish polish using exact local solve."""
    P = _project(P)
    E, G = _energy_and_grad(P)
    for _ in range(200):
        if time.time() > deadline:
            break
        rG = np.linalg.norm(G)
        if rG < 1e-14:
            break
        # small damped step
        Pn = _grad_step(P, G, 1.0 / (N * (1.0 + rG)))
        En, _ = _energy_and_grad(Pn)
        if En < E:
            P, E = Pn, En
        else:
            break
    return P, E


def construct():
    t0 = time.time()
    deadline = t0 + _DEADLINE - 5.0  # safety margin

    starts = [_fibonacci(N), _sobol(N)]
    for s in range(1, 1 + 8):
        if time.time() > deadline:
            break
        starts.append(_random(N, seed=1000 + s))

    best_P = None
    best_E = np.inf

    for idx, P0 in enumerate(starts):
        if time.time() > deadline:
            break
        P0 = _project(P0)
        E0, _ = _energy_and_grad(P0)
        if E0 < best_E:
            best_E = E0
            best_P = P0.copy()

        # L-BFGS phase
        P1, E1 = _lbfgs(P0, tol=1e-10, deadline=deadline, max_iter=1500, m=8)
        if E1 < best_E:
            best_E = E1
            best_P = P1.copy()

        # Gradient descent continuation from L-BFGS result
        if time.time() < deadline:
            P2, E2 = _gradient_descent(P1, tol=1e-12, deadline=deadline, max_iter=2000,
                                       step0=1.0 / (N * 4))
            if E2 < best_E:
                best_E = E2
                best_P = P2.copy()

    # Final high-precision polish on the best candidate
    if best_P is not None and time.time() < deadline:
        best_P, best_E = _polish(best_P, tol=1e-14, deadline=deadline)

    # Validate: ensure non-zero, finite, unit length, no near-duplicates
    best_P = _project(best_P)
    if not np.all(np.isfinite(best_P)):
        best_P = _fibonacci(N)
    # check min pairwise distance
    D = best_P[:, None, :] - best_P[None, :, :]
    D2 = np.einsum('ijk,ijk->ij', D, D)
    np.fill_diagonal(D2, 1.0)
    min_d2 = np.min(D2)
    if min_d2 < (1e-9) ** 2:
        # fall back to fibonacci if points collapsed
        best_P = _fibonacci(N)

    return best_P
