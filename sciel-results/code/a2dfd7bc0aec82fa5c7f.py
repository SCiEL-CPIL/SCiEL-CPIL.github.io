# EVOLVE-BLOCK-START
import time
import numpy as np

# Precompute constants
_N = 64
_A = (np.arange(_N) + 1) / _N          # a_i
_W = 1.0 / _N                            # base width
_M = 20000
_Y = (np.arange(_M) + 0.5) / _M          # quadrature heights (M,)
_WY = (1.0 - _Y) / _N                    # width at each height (M,)
# L_i = x_i + a_i * y  ->  shape (M, N) if x is (N,)
# We precompute a_i * y as (M, N)
_AY = np.outer(_Y, _A)                   # (M, N)


def _area(x):
    """Compute the union-area metric (lower is better). Fast vectorized."""
    # L = x_i + a_i * y_m  ->  (M, N)
    L = _AY + x[None, :]                 # broadcast x (N,) to (M, N)
    # Sort L along axis=1
    Ls = np.sort(L, axis=1)
    # Consecutive gaps
    gaps = np.diff(Ls, axis=1)           # (M, N-1)
    # union length = w + sum(min(gap, w))
    # w varies with row: _WY (M,)
    w = _WY[:, None]                     # (M, 1)
    union_len = w + np.minimum(gaps, w).sum(axis=1)  # (M,)
    area = union_len.mean()
    return area


def _keich_positions():
    n = 64
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    positions = []
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        positions.append(np.sum((1 - j) * eps * (2.0 ** -j) / k))
    return np.array(positions, dtype=np.float64)


def _aligned_positions():
    # x_i = -a_i + 0.5  (apex at 0.5 for all, bases shifted left)
    return 0.5 - _A.copy()


def _random_positions(scale=0.5, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    return rng.uniform(-scale, scale, size=_N)


def _local_search(x0, max_iter=3000, tol=1e-12, step=0.01, time_limit=None, start_time=None):
    """
    Coordinate-descent local search with adaptive step size.
    Tries each coordinate, accepts if area decreases.
    """
    x = x0.copy()
    f = _area(x)
    best_x = x.copy()
    best_f = f
    step_size = step
    stall = 0

    for it in range(max_iter):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        improved = False
        for i in range(_N):
            # Try +step
            x[i] += step_size
            f_new = _area(x)
            if f_new < f:
                f = f_new
                improved = True
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
            else:
                x[i] -= 2 * step_size  # try -step (net -step from original)
                f_new = _area(x)
                if f_new < f + 2 * step_size * 0:  # just check
                    pass
                # Actually recompute properly:
                x[i] += step_size  # back to original
                x[i] -= step_size
                f_new = _area(x)
                if f_new < f:
                    f = f_new
                    improved = True
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()
                else:
                    x[i] += step_size  # back to original
        if not improved:
            step_size *= 0.7
            stall += 1
            if step_size < 1e-10 or stall > 20:
                break
        else:
            stall = 0
    return best_x, best_f


def kakeya_needle_2d_positions64() -> np.ndarray:
    start_time = time.time()
    deadline = 580.0  # safety margin before 630s kill

    # Generate diverse initializations
    inits = []
    inits.append(_keich_positions())
    inits.append(_aligned_positions())

    rng = np.random.default_rng(42)
    # Random perturbations of keich
    for _ in range(8):
        x = _keich_positions() + rng.normal(0, 0.05, size=_N)
        inits.append(x)
    # Random perturbations of aligned
    for _ in range(8):
        x = _aligned_positions() + rng.normal(0, 0.05, size=_N)
        inits.append(x)
    # Pure random
    for _ in range(4):
        x = _random_positions(scale=1.0, rng=rng)
        inits.append(x)

    best_x = None
    best_f = np.inf

    for x0 in inits:
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining < 5:
            break
        # Give each local search a time budget
        ts = max(2.0, remaining / max(1, len(inits) + 1))
        x_opt, f_opt = _local_search(x0, max_iter=5000, step=0.02,
                                     time_limit=ts, start_time=start_time)
        if f_opt < best_f:
            best_f = f_opt
            best_x = x_opt.copy()

    # Final check
    if best_x is None:
        best_x = _keich_positions()

    # Ensure finite
    if not np.all(np.isfinite(best_x)):
        best_x = _keich_positions()

    return best_x.astype(np.float64)


# EVOLVE-BLOCK-END
