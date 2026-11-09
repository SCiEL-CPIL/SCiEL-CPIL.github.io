import time
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist

_DEADLINE_SECONDS = 630.0
_SAFETY_MARGIN = 60.0


def _raw_score(points):
    """
    Calculate the score: (d_min / d_max)^2.
    Returns 0 if invalid.
    """
    if points.shape != (14, 3):
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0

    dists = pdist(points)
    if len(dists) == 0:
        return 0.0
    dmin = np.min(dists)
    dmax = np.max(dists)
    if not (np.isfinite(dmin) and np.isfinite(dmax)):
        return 0.0
    if dmax <= 1e-15 or dmin <= 1e-15:
        return 0.0
    r = dmin / dmax
    return float(r * r)


def _objective_de(x):
    """
    Objective for differential evolution.
    We want to maximize (d_min/d_max)^2, so minimize its negative.
    x is a flat array of shape (42,).
    """
    X = x.reshape(14, 3)
    dists = pdist(X)
    dmin = np.min(dists)
    dmax = np.max(dists)
    if dmax <= 1e-15 or dmin <= 1e-15:
        return 1.0
    return -(dmin / dmax) ** 2


def _objective_grad(x):
    """
    Objective for local minimization (L-BFGS-B).
    We want to maximize (d_min/d_max)^2.
    Minimize f = -log(d_min/d_max) = log(d_max) - log(d_min)
    This is smoother than the squared ratio.
    """
    X = x.reshape(14, 3)
    D = X[:, None, :] - X[None, :, :]
    D2 = np.sum(D * D, axis=2)
    D2 = np.maximum(D2, 1e-30)
    dists = np.sqrt(D2)
    G = D / dists[:, :, None]

    iu = np.triu_indices(14, k=1)
    pd_d = dists[iu]
    pd_G = G[iu]

    dmin = np.min(pd_d)
    dmax = np.max(pd_d)

    if dmin <= 1e-15 or dmax <= 1e-15:
        return 1.0, np.zeros(42)

    # Gradient of log(dmax) - log(dmin)
    # d(log dmax)/dX[k] = (X[k] - X[argmax])/dmax for the max pair
    # d(log dmin)/dX[k] = (X[k] - X[argmin])/dmin for the min pair
    # We want to minimize f = log(dmax) - log(dmin)
    # grad_f = grad_log_dmax - grad_log_dmin

    max_idx = np.argmax(pd_d)
    min_idx = np.argmin(pd_d)

    k_max = iu[0][max_idx]
    l_max = iu[1][max_idx]
    k_min = iu[0][min_idx]
    l_min = iu[1][min_idx]

    g = np.zeros((14, 3))

    # Gradient of log(dmax)
    g[k_max] += pd_G[max_idx] / dmax
    g[l_max] -= pd_G[max_idx] / dmax

    # Gradient of log(dmin)
    g[k_min] -= pd_G[min_idx] / dmin
    g[l_min] += pd_G[min_idx] / dmin

    value = np.log(dmax) - np.log(dmin)
    return value, g.reshape(-1)


def _polish(x0, deadline):
    """
    Polish a starting point using L-BFGS-B.
    """
    def obj_and_grad(x):
        if time.time() > deadline:
            raise RuntimeError("deadline")
        return _objective_grad(x)

    try:
        res = minimize(
            lambda x: _objective_grad(x)[0],
            x0,
            jac=lambda x: _objective_grad(x)[1],
            method="L-BFGS-B",
            options={
                "maxiter": 5000,
                "maxfun": 100000,
                "ftol": 1e-15,
                "gtol": 1e-14,
                "maxls": 200,
            },
        )
        return res.x
    except Exception:
        return x0


def min_max_dist_dim3_14():
    start_time = time.time()
    deadline = start_time + _DEADLINE_SECONDS - _SAFETY_MARGIN

    # Bounds for each coordinate
    # We expect points to be in a reasonable range, say [-2, 2]
    bounds = [(-2.0, 2.0)] * 42

    best_pts = None
    best_score = 0.0

    # Phase 1: Global search with Differential Evolution
    # Use a small population to stay within time limits
    # We will run multiple DE iterations if time permits

    # Initial guess for DE
    rng = np.random.default_rng(42)
    x_init = rng.uniform(-1.0, 1.0, 42)

    # Run DE
    # popsize=10 means 10*42=420 initial points, which is a lot.
    # Let's try popsize=5 or 6 to be faster.
    # maxiter=100
    # tol=1e-10

    de_kwargs = {
        "bounds": bounds,
        "x0": x_init,
        "maxiter": 200,
        "popsize": 6,
        "tol": 1e-10,
        "mutation": (0.4, 1.0),
        "recombination": 0.7,
        "seed": 42,
        "polish": False,  # We'll do our own polish
        "init": "latinhypercube",
    }

    # We need to check time during DE. DE doesn't support callbacks easily for early stopping in scipy.
    # We can run it and hope it finishes, or wrap it.
    # Given the 630s limit, and DE with popsize 6, 200 iters might be slow.
    # Let's reduce maxiter to 50 and popsize to 4.
    de_kwargs["maxiter"] = 50
    de_kwargs["popsize"] = 4

    try:
        res = differential_evolution(_objective_de, **de_kwargs)
        x_de = res.x
        s_de = _raw_score(x_de.reshape(14, 3))
        if s_de > best_score:
            best_score = s_de
            best_pts = x_de.copy()
    except Exception:
        pass

    # Phase 2: Polish the best result from DE
    if best_pts is not None and time.time() < deadline:
        x_polished = _polish(best_pts, deadline)
        s_polished = _raw_score(x_polished.reshape(14, 3))
        if s_polished > best_score:
            best_score = s_polished
            best_pts = x_polished.copy()

    # Phase 3: Local restarts from random points if time permits
    # This helps if DE got stuck in a local minimum.
    rng = np.random.default_rng(123)
    for i in range(5):
        if time.time() > deadline - 60:
            break
        x_rand = rng.uniform(-1.0, 1.0, 42)
        x_polished = _polish(x_rand, deadline)
        s_polished = _raw_score(x_polished.reshape(14, 3))
        if s_polished > best_score:
            best_score = s_polished
            best_pts = x_polished.copy()

    # If we still have time, do a final high-precision polish on the best point
    if best_pts is not None and time.time() < deadline - 30:
        x_final = _polish(best_pts, deadline)
        s_final = _raw_score(x_final.reshape(14, 3))
        if s_final > best_score:
            best_score = s_final
            best_pts = x_final.copy()

    if best_pts is None:
        # Fallback
        best_pts = np.zeros(42)

    best_pts = np.asarray(best_pts, dtype=np.float64).reshape(14, 3)
    if not np.all(np.isfinite(best_pts)):
        best_pts = np.zeros((14, 3))

    return best_pts
