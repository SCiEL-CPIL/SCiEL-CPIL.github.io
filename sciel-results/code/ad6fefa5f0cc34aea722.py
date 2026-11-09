# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution

_START_TIME = time.time()
_DEADLINE = 630.0
_MARGIN = 30.0


def _remaining_time():
    return _DEADLINE - _MARGIN - (time.time() - _START_TIME)


def _compute_volume(positions):
    """Compute the fixed-sample Monte Carlo union volume exactly as specified."""
    if positions is None or positions.shape != (64, 2):
        return np.inf
    if not np.all(np.isfinite(positions)):
        return np.inf

    N = 8
    i = positions[:, 0] * 0  # placeholder
    # Build i, j arrays
    i_idx = np.repeat(np.arange(N), N)
    j_idx = np.tile(np.arange(N), N)

    xk = positions[:, 0]
    yk = positions[:, 1]

    # At height c, tube (i,j) occupies:
    #   x in [xk + c*i/8, xk + c*i/8 + 1/8]
    #   y in [yk + c*j/8, yk + c*j/8 + 1/8]
    # So the bounding box:
    #   X0 = min_k xk
    #   X1 = max_k (xk + (i_k+1)/8)
    #   Y0 = min_k yk
    #   Y1 = max_k (yk + (j_k+1)/8)

    X0 = np.min(xk)
    X1 = np.max(xk + (i_idx + 1) / 8.0)
    Y0 = np.min(yk)
    Y1 = np.max(yk + (j_idx + 1) / 8.0)

    area = (X1 - X0) * (Y1 - Y0)
    if area <= 1e-9:
        return np.inf

    rng = np.random.default_rng(12345)
    a = rng.uniform(X0, X1, 500000)
    b = rng.uniform(Y0, Y1, 500000)
    c = rng.uniform(0, 1, 500000)

    # Check if point (a, b, c) is inside at least one tube.
    # Tube k: x in [xk + c*ik/8, xk + c*ik/8 + 1/8], y in [yk + c*jk/8, yk + c*jk/8 + 1/8]
    # Vectorized: for each tube, compute the x-interval and y-interval at height c.
    # We need: exists k s.t. (a - c*ik/8 - xk) in [0, 1/8] and (b - c*jk/8 - yk) in [0, 1/8]

    # a shape (500000,), xk shape (64,), c shape (500000,)
    # For each k: a - c * ik/8 - xk should be in [0, 1/8]
    # This is a (500000, 64) check.

    # To save memory, do it in chunks or use broadcasting carefully.
    # 500000 * 64 * 8 bytes = ~256 MB per array. We have a few such arrays. Should be OK.

    # x_shift = c[:, None] * (i_idx[None, :] / 8.0) + xk[None, :]
    # Check: (a[:, None] - x_shift) >= 0 and (a[:, None] - x_shift) <= 1/8
    # Similarly for y.

    # But 500000*64 = 32M elements per array. 3-4 such arrays = ~1GB. Might be tight.
    # Let's do it in chunks of 100000 points.

    in_any = np.zeros(500000, dtype=bool)
    chunk = 100000
    for start in range(0, 500000, chunk):
        end = min(start + chunk, 500000)
        a_c = a[start:end]
        b_c = b[start:end]
        c_c = c[start:end]

        # x intervals: lower_k = xk + c_c * ik/8, upper_k = lower_k + 1/8
        # We want: exists k s.t. a_c >= lower_k and a_c <= upper_k AND b_c >= lower_y_k and b_c <= upper_y_k

        x_lower = c_c[:, None] * (i_idx[None, :] / 8.0) + xk[None, :]
        x_upper = x_lower + 1.0 / 8.0
        y_lower = c_c[:, None] * (j_idx[None, :] / 8.0) + yk[None, :]
        y_upper = y_lower + 1.0 / 8.0

        x_ok = (a_c[:, None] >= x_lower) & (a_c[:, None] <= x_upper)
        y_ok = (b_c[:, None] >= y_lower) & (b_c[:, None] <= y_upper)

        in_any[start:end] = np.any(x_ok & y_ok, axis=1)

    fraction = np.mean(in_any)
    V = fraction * area
    if V <= 0:
        return np.inf
    return V


def _objective(params):
    """Objective: union volume. params is a flat array of 128 values (64 x, 64 y)."""
    if _remaining_time() < 5.0:
        return 1e10
    positions = params.reshape(64, 2)
    V = _compute_volume(positions)
    if not np.isfinite(V) or V <= 0:
        return 1e10
    return V


def _keich_init(N):
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    return keich


def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    # Start with Keich separable init
    keich = _keich_init(N)
    i_idx = np.repeat(np.arange(N), N)
    j_idx = np.tile(np.arange(N), N)
    base_x = keich[i_idx]
    base_y = keich[j_idx]
    positions0 = np.stack([base_x, base_y], axis=1)
    params0 = positions0.flatten()

    V0 = _compute_volume(positions0)
    best_params = params0.copy()
    best_V = V0

    # Bounds for differential evolution: positions can range roughly in [-1, 2]
    # The box is determined by the positions, so we want to keep them bounded.
    # Let's use bounds = [(-1, 2)] * 128
    bounds = [(-1.0, 2.0)] * 128

    # Use differential evolution with a time-limited approach.
    # We'll run DE with a callback to check time.
    def callback(xk, convergence):
        return _remaining_time() < 10.0

    try:
        result = differential_evolution(
            _objective,
            bounds,
            x0=params0,
            maxiter=200,
            popsize=15,
            tol=1e-8,
            mutation=(0.3, 1.0),
            recombination=0.8,
            seed=42,
            callback=callback,
            polish=True,
            init='sobol',
            update_interval=1,
        )
        if result.success and result.fun < best_V:
            best_params = result.x
            best_V = result.fun
    except Exception:
        pass

    # If time remains, do a local coordinate-descent refinement
    if _remaining_time() > 30.0:
        positions = best_params.reshape(64, 2)
        lr = 0.05
        for iteration in range(500):
            if _remaining_time() < 10.0:
                break
            improved = False
            for k in range(64):
                for dim in range(2):
                    old_val = positions[k, dim]
                    for delta in [lr, -lr, lr * 0.5, -lr * 0.5]:
                        positions[k, dim] = old_val + delta
                        V = _compute_volume(positions)
                        if V < best_V - 1e-10:
                            best_V = V
                            best_params = positions.copy().flatten()
                            improved = True
                            break
                    positions[k, dim] = old_val
            if not improved:
                lr *= 0.7
                if lr < 1e-4:
                    break

    final_positions = best_params.reshape(64, 2)
    if not np.all(np.isfinite(final_positions)):
        final_positions = positions0
    return final_positions


# EVOLVE-BLOCK-END
