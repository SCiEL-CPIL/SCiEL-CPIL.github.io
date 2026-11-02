"""Fresh design: simulated annealing over sign space with block-flip moves."""
import time
import numpy as np

N = 400
_SEED = 42


def _autoconv_peak(x):
    """Upper-bound proxy: max |autoconvolution| of the signed step function.

    The autoconvolution of the step function on [-1/4,1/4] with N equal intervals
    is the discrete convolution of the sign sequence with itself. Its peak
    magnitude drives the C3 upper bound, so minimizing max|conv| minimizes the bound.
    """
    conv = np.convolve(x, x, mode="full")
    return float(np.max(np.abs(conv)))


def _energy(x):
    return _autoconv_peak(x)


def _rand_block_flip(x, rng, min_b, max_b):
    """Return (new_x, changed_indices) applying one random block sign flip."""
    b = int(rng.integers(min_b, max_b + 1))
    start = int(rng.integers(0, N - b + 1))
    new_x = x.copy()
    new_x[start:start + b] *= -1.0
    return new_x, (start, b)


def construct():
    deadline = time.time() + 110.0  # safety margin vs 120s wall-clock kill
    rng = np.random.default_rng(_SEED)

    # Start from a low-concentration initial guess (alternating) rather than all +1.
    best = np.where(np.arange(N) % 2 == 0, 1.0, -1.0)
    best_e = _energy(best)

    T = 1.0
    T_min = 1e-4
    cooling = 0.9995
    min_b, max_b = 4, 32

    while time.time() < deadline:
        x, (start, b) = _rand_block_flip(best, rng, min_b, max_b)
        e = _energy(x)
        if e < best_e or (T > T_min and rng.random() < np.exp(-(e - best_e) / T)):
            best, best_e = x, e
            T = max(T_min, T * cooling)
        else:
            T = max(T_min, T * cooling)
        # Periodic restart to escape deep basins
        if best_e < 1e-3 and rng.random() < 0.01:
            T = 1.0

    return best


# EVOLVE-BLOCK-END
