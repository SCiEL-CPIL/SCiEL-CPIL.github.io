"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100

# EVOLVE-BLOCK-START


def _energy_and_grad(X):
    """Return (energy, gradient) for points on the unit sphere.

    X: (N, 3) array of unit vectors.
    Energy E = sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij), chordal d_ij.
    Gradient (Cartesian): for point i, sum_{j != i} (x_i - x_j) / d_ij^2.
    """
    # pairwise squared distances
    sq = np.einsum('ij,ij->i', X, X)[:, None] + np.einsum('ij,ij->i', X, X)[None, :] \
         - 2.0 * (X @ X.T)
    np.fill_diagonal(sq, 0.0)
    sq = np.maximum(sq, 1e-16)
    d = np.sqrt(sq)
    np.fill_diagonal(d, 1.0)

    # energy via math.fsum over upper triangle
    iu = np.triu_indices(N, k=1)
    dvals = d[iu]
    terms = -np.log(dvals)
    E = math.fsum(terms.tolist())

    # gradient: grad_i = sum_{j != i} (x_i - x_j) / d_ij^2
    # 1/d^2 matrix (zero diagonal)
    inv_d2 = 1.0 / sq
    np.fill_diagonal(inv_d2, 0.0)
    # (x_i - x_j) * inv_d2[i,j] summed over j for each i
    # diff[i,j] = x_i - x_j
    diff = X[:, None, :] - X[None, :, :]
    grad = np.einsum('ij,ijk->ik', inv_d2, diff)
    return E, grad


def _project(X):
    """Project rows back to the unit sphere (numerical safety)."""
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    nrm = np.maximum(nrm, 1e-12)
    return X / nrm


def construct():
    t0 = time.time()
    DEADLINE = 150.0
    MARGIN = 8.0

    # --- Initialisation: jittered Fibonacci spiral ---
    def make_init(seed):
        rng = np.random.default_rng(seed)
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
        X = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        # small jitter in Cartesian space, then re-project
        X = X + rng.normal(0.0, 0.02, size=X.shape)
        return _project(X)

    X = make_init(0)
    E, _ = _energy_and_grad(X)
    best_X, best_E = X.copy(), E

    def time_left():
        return DEADLINE - MARGIN - (time.time() - t0)

    # --- Main L-BFGS-B optimisation ---
    def fun_g(x):
        X = _project(x.reshape(N, 3))
        e, g = _energy_and_grad(X)
        return e, g.reshape(-1)

    try:
        res = minimize(fun_g, X.reshape(-1), method='L-BFGS-B',
                       options={'maxiter': 600, 'maxfun': 20000,
                                'gtol': 1e-10, 'ftol': 1e-14})
        X = _project(res.x.reshape(N, 3))
        E, _ = _energy_and_grad(X)
        if E < best_E:
            best_E, best_X = E, X.copy()
    except Exception:
        pass

    # --- High-precision polish pass (tighter tolerances) ---
    if time_left() > 20.0:
        try:
            res2 = minimize(fun_g, best_X.reshape(-1), method='L-BFGS-B',
                            options={'maxiter': 400, 'maxfun': 20000,
                                     'gtol': 1e-13, 'ftol': 1e-16})
            X = _project(res2.x.reshape(N, 3))
            E, _ = _energy_and_grad(X)
            if E < best_E:
                best_E, best_X = E, X.copy()
        except Exception:
            pass

    # --- Final normalisation and validity guard ---
    X = _project(best_X)
    if time.time() > DEADLINE - 1.0:
        return X

    return X
# EVOLVE-BLOCK-END
