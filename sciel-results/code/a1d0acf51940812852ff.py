import numpy as np
import scipy.optimize as opt
import time

# Physicists' Hermite polynomials
def H0(x):
    return np.ones_like(x)

def H4(x):
    return 16.0 * x**4 - 48.0 * x**2 + 12.0

def H8(x):
    return (
        256.0 * x**8
        - 3840.0 * x**6
        + 13440.0 * x**4
        - 13440.0 * x**2
        + 1680.0
    )

def H12(x):
    return (
        4096.0 * x**12
        - 122880.0 * x**10
        + 1290240.0 * x**8
        - 6635520.0 * x**6
        + 14515200.0 * x**4
        - 14515200.0 * x**2
        + 4790016.0
    )

# Precompute a moderately dense grid.
# The objective is smooth and the Gaussian decays fast, so this is enough.
X = np.linspace(-15.0, 15.0, 1200)
DX = X[1] - X[0]
GAUSS = np.exp(-0.5 * X**2)

H0_G = H0(X) * GAUSS
H4_G = H4(X) * GAUSS
H8_G = H8(X) * GAUSS
H12_G = H12(X) * GAUSS

# Values at the origin for the g(0)=0 constraint
H0_0 = 1.0
H4_0 = 12.0
H8_0 = 1680.0
H12_0 = 665280.0

def _d_from_abc(a, b, c):
    return -(a * H0_0 + b * H4_0 + c * H8_0) / H12_0

def _objective_single(abc):
    a, b, c = abc
    d = _d_from_abc(a, b, c)
    g = a * H0_G + b * H4_G + c * H8_G + d * H12_G
    g2 = g * g
    n2 = np.sum(g2) * DX
    if n2 <= 0.0:
        return 1e30
    n4 = np.sum(g2 * g2) * DX
    return n4 / n2

def _objective_batch(abc_array):
    # abc_array: shape (N, 3)
    a = abc_array[:, 0]
    b = abc_array[:, 1]
    c = abc_array[:, 2]
    d = -(a * H0_0 + b * H4_0 + c * H8_0) / H12_0

    # Build g for all candidates at once: shape (N, len(X))
    g = (
        a[:, None] * H0_G[None, :]
        + b[:, None] * H4_G[None, :]
        + c[:, None] * H8_G[None, :]
        + d[:, None] * H12_G[None, :]
    )

    g2 = g * g
    n2 = np.sum(g2, axis=1) * DX
    n4 = np.sum(g2 * g2, axis=1) * DX

    with np.errstate(divide="ignore", invalid="ignore"):
        vals = np.where(n2 > 0.0, n4 / n2, 1e30)

    return vals

def _candidate_pool():
    # A compact, structured pool of coefficient triples.
    # The seed is included, plus a small set of manually tuned points
    # and a low-resolution lattice around the seed and around the origin.
    pool = []

    # Seed
    pool.append((-1.13, 0.04, 1.0 / 3240.0))

    # A few hand-picked points that often help in this kind of smooth problem
    pool.extend([
        (-1.0, 0.03, 0.0003),
        (-1.2, 0.05, 0.0002),
        (-0.9, 0.02, 0.0004),
        (-1.3, 0.06, 0.0001),
        (-1.1, 0.035, 0.00035),
        (-1.15, 0.045, 0.00025),
        (-1.05, 0.035, 0.00045),
        (-1.25, 0.055, 0.00015),
    ])

    # Small lattice around the seed
    a0, b0, c0 = -1.13, 0.04, 1.0 / 3240.0
    da = np.linspace(-0.35, 0.35, 7)
    db = np.linspace(-0.035, 0.035, 7)
    dc = np.linspace(-0.00035, 0.00035, 5)
    for a in da:
        for b in db:
            for c in dc:
                pool.append((a0 + a, b0 + b, c0 + c))

    # Small lattice around the origin
    for a in np.linspace(-0.4, 0.4, 5):
        for b in np.linspace(-0.02, 0.02, 5):
            for c in np.linspace(-0.0002, 0.0002, 3):
                pool.append((a, b, c))

    # Convert to array and remove duplicates
    arr = np.array(pool, dtype=float)
    if arr.size == 0:
        return np.array([[-1.13, 0.04, 1.0 / 3240.0]], dtype=float)

    # Deduplicate exactly (fast)
    _, idx = np.unique(arr, axis=0, return_index=True)
    arr = arr[np.sort(idx)]
    return arr

def construct():
    start = time.time()
    deadline = start + 175.0  # explicit safety margin

    # Start with the seed
    best_params = np.array([-1.13, 0.04, 1.0 / 3240.0], dtype=float)
    best_val = _objective_single(best_params)

    # Phase 1: fast global scan over a compact candidate pool
    try:
        pool = _candidate_pool()
        # Evaluate in chunks to keep memory reasonable
        chunk = 256
        for i in range(0, len(pool), chunk):
            if time.time() > deadline - 30.0:
                break
            block = pool[i:i + chunk]
            vals = _objective_batch(block)
            j = int(np.argmin(vals))
            if vals[j] < best_val:
                best_val = float(vals[j])
                best_params = block[j].copy()
    except Exception:
        pass

    # Phase 2: bounded local refinement from the best point
    try:
        if time.time() < deadline - 15.0:
            bounds = [
                (best_params[0] - 0.35, best_params[0] + 0.35),
                (best_params[1] - 0.035, best_params[1] + 0.035),
                (best_params[2] - 0.00035, best_params[2] + 0.00035),
            ]
            res = opt.minimize(
                _objective_single,
                best_params,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 300, "ftol": 1e-14, "gtol": 1e-14},
            )
            if res.fun < best_val:
                best_val = float(res.fun)
                best_params = np.asarray(res.x, dtype=float)
    except Exception:
        pass

    # Phase 3: very short final polish if time remains
    try:
        if time.time() < deadline - 5.0:
            res = opt.minimize(
                _objective_single,
                best_params,
                method="Nelder-Mead",
                options={"xatol": 1e-12, "fatol": 1e-14, "maxiter": 500},
            )
            if res.fun < best_val:
                best_val = float(res.fun)
                best_params = np.asarray(res.x, dtype=float)
    except Exception:
        pass

    return np.asarray(best_params, dtype=float)
