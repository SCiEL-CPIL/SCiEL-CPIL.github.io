"""
Minimize logarithmic energy of 150 points on the unit sphere.
E = sum_{i<j} log(1/|x_i - x_j|)  (natural log, chordal distances)

Strategy:
  1. Fast initial optimization from a Fibonacci-spiral start using L-BFGS on the sphere.
  2. Additional starts from structured and random configurations.
  3. High-precision polish phase with tighter tolerances.
  4. Local perturbations to escape shallow local minima.
"""
import time
import math
import numpy as np

N = 150
_DEADLINE = 150.0
_START = time.time()
_SAFETY = 5.0
_TIME_LIMIT = _DEADLINE - _SAFETY


def _time_left():
    return _TIME_LIMIT - (time.time() - _START)


def _retract(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.where(n < 1e-15, 1.0, n)
    return X / n


def _energy_and_grad(X, eps=1e-14):
    """Vectorized energy and gradient. Returns (E, G) where G is the Euclidean gradient."""
    diff = X[:, None, :] - X[None, :, :]
    d2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(d2, 1.0)
    d2 = np.maximum(d2, eps * eps)
    d = np.sqrt(d2)
    # E = -sum_{i<j} log(d_ij) = -0.5 * sum_{i!=j} log(d_ij)
    log_d = np.log(d)
    np.fill_diagonal(log_d, 0.0)
    E = -0.5 * float(np.sum(log_d))
    # grad_i = sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    inv = 1.0 / d2
    np.fill_diagonal(inv, 0.0)
    S1 = inv @ X
    S2 = (inv.sum(axis=1, keepdims=True)) * X
    G = S2 - S1
    return E, G


def _project_tangent(G, X):
    a = np.einsum('ij,ij->i', G, X)
    return G - a[:, None] * X


def _bfgs_two_loop(v, s_list, y_list, rho_list, m):
    n = len(v)
    q = v.copy()
    alphas = np.zeros(m)
    for k in range(m - 1, -1, -1):
        s = s_list[k]
        y = y_list[k]
        rho = rho_list[k]
        alphas[k] = rho * float(np.dot(s, q))
        q = q - alphas[k] * y
    r = np.zeros(n)
    if y_list:
        yy = float(np.dot(y_list[-1], y_list[-1]))
        if yy > 1e-20:
            gamma = float(np.dot(s_list[-1], y_list[-1])) / yy
            r = gamma * q
    for k in range(m):
        s = s_list[k]
        y = y_list[k]
        rho = rho_list[k]
        beta = rho * float(np.dot(y, r))
        r = r + (alphas[k] - beta) * s
    return q + r


def _lbfgs_sphere(X0, time_budget, m=15, max_iter=5000):
    """L-BFGS on the sphere with retraction. Returns (best_X, best_E)."""
    X = _retract(X0.copy())
    E, G = _energy_and_grad(X)
    g = _project_tangent(G, X)
    flat = g.ravel()

    s_list, y_list, rho_list = [], [], []
    it = 0
    best_E = E
    best_X = X.copy()

    while _time_left() > 2.0 and it < max_iter:
        # Search direction
        p_flat = -_bfgs_two_loop(flat, s_list, y_list, rho_list, min(len(s_list), m))
        p = p_flat.reshape(N, 3)
        p = _project_tangent(p, X)
        pnorm = float(np.linalg.norm(p))
        if pnorm < 1e-13:
            break

        gnorm = float(np.linalg.norm(flat))
        if gnorm < 1e-10:
            break

        # Armijo line search
        c1 = 1e-4
        c2 = 0.9
        slope = float(np.dot(flat, p_flat))
        if slope >= 0:
            # Direction is not descent; try steepest descent
            p = -g
            p = _project_tangent(p, X)
            p_flat = p.ravel()
            slope = float(np.dot(flat, p_flat))
            if slope >= 0:
                break

        alpha = 1.0
        Xn = X
        En = E
        Gn = G
        gn = g
        for _ in range(40):
            if _time_left() < 3.0:
                break
            Xn = _retract(X + alpha * p)
            En, Gn = _energy_and_grad(Xn)
            gn = _project_tangent(Gn, Xn)
            if En <= E + c1 * alpha * slope:
                if float(np.dot(gn.ravel(), p_flat)) <= c2 * alpha * slope:
                    break
            alpha *= 0.5

        if En >= E:
            # Try a smaller step
            alpha = max(alpha * 0.1, 1e-8)
            Xn = _retract(X + alpha * p)
            En, Gn = _energy_and_grad(Xn)
            gn = _project_tangent(Gn, Xn)
            if En >= E:
                break

        # Update history
        dX = (Xn - X).ravel()
        dy = gn.ravel() - flat
        sy = float(np.dot(dX, dy))
        if sy > 1e-14:
            s_list.append(dX)
            y_list.append(dy)
            rho_list.append(1.0 / sy)
            if len(s_list) > m:
                s_list.pop(0)
                y_list.pop(0)
                rho_list.pop(0)

        X = Xn
        E = En
        G = Gn
        g = gn
        flat = g.ravel()
        it += 1

        if E < best_E:
            best_E = E
            best_X = X.copy()

    return best_X, best_E


def _fibonacci_spiral():
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_points(seed):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(N, 3))
    return _retract(v)


def _structured_grid():
    """Generate points on a structured grid on the sphere (latitude/longitude with jitter)."""
    n_lat = 12
    n_lon = 13
    points = []
    for i in range(n_lat):
        theta = (i + 0.5) * math.pi / n_lat
        for j in range(n_lon):
            phi = (j + 0.5) * 2 * math.pi / n_lon
            x = math.sin(theta) * math.cos(phi)
            y = math.sin(theta) * math.sin(phi)
            z = math.cos(theta)
            points.append([x, y, z])
    # Add extra points to reach N
    while len(points) < N:
        rng = np.random.default_rng(len(points) * 7919)
        v = rng.normal(size=3)
        v /= np.linalg.norm(v)
        points.append(v)
    return _retract(np.array(points[:N]))


def _perturb(X, seed, sigma):
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=X.shape) * sigma
    return _retract(X + noise)


def _validate(X):
    """Check validity: finite, unit norm, no near-duplicates."""
    if not np.all(np.isfinite(X)):
        return False
    norms = np.linalg.norm(X, axis=1)
    if not np.all(np.abs(norms - 1.0) < 1e-6):
        return False
    # Check minimum pairwise distance
    diff = X[:, None, :] - X[None, :, :]
    dd = np.sqrt(np.einsum('ijk,ijk->ij', diff, diff))
    np.fill_diagonal(dd, 2.0)
    if float(dd.min()) < 1e-9:
        return False
    return True


def _log_energy_precise(X):
    """Compute energy using math.fsum for precision."""
    total = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            d = math.sqrt(np.dot(X[i] - X[j], X[i] - X[j]))
            total += math.log(1.0 / d)
    return total


def construct():
    best_X = _fibonacci_spiral()
    best_E = _energy_and_grad(best_X)[0]

    # Phase 1: Optimize from Fibonacci spiral
    if _time_left() > 20.0:
        X, E = _lbfgs_sphere(_fibonacci_spiral(), 40.0)
        if E < best_E:
            best_E, best_X = E, X

    # Phase 2: Additional starts
    starts = []
    starts.append(_structured_grid())
    for s in [42, 123, 456, 789, 1024, 2048, 3000, 4096]:
        starts.append(_random_points(s))

    for start in starts:
        if _time_left() > 15.0:
            X, E = _lbfgs_sphere(start, 20.0)
            if E < best_E:
                best_E, best_X = E, X

    # Phase 3: Local perturbations around best to escape local minima
    for sigma in [0.2, 0.1, 0.05, 0.02]:
        if _time_left() > 10.0:
            for trial in range(3):
                if _time_left() < 5.0:
                    break
                start = _perturb(best_X, trial * 1000 + int(sigma * 10000), sigma)
                X, E = _lbfgs_sphere(start, 10.0)
                if E < best_E:
                    best_E, best_X = E, X

    # Phase 4: High-precision polish
    if _time_left() > 15.0:
        X, E = _lbfgs_sphere(best_X, 15.0, m=20, max_iter=2000)
        if E < best_E:
            best_E, best_X = E, X

    # Final validation
    best_X = _retract(best_X)
    if not _validate(best_X):
        # Fallback to Fibonacci spiral
        best_X = _fibonacci_spiral()

    return best_X
