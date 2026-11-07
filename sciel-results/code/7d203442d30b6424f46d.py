"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _objective(h):
    one_minus_h = 1.0 - h
    corr = np.correlate(h, one_minus_h, mode="full")
    return np.max(corr)


def construct():
    start = time.time()
    deadline = start + 130  # 20s safety margin before 150s hard limit

    bounds = [(0.0, 1.0)] * N
    constraints = [{'type': 'eq', 'fun': lambda h: np.sum(h) - N / 2.0}]

    def objective(h):
        return _objective(h)

    best_h = np.full(N, 0.5)
    best_val = objective(best_h)

    np.random.seed(42)

    # Multi-frequency sinusoidal initializations to spread correlation energy
    def make_sinusoidal_init(frequencies, amplitudes, phase=0.0):
        h = np.zeros(N)
        for f, a in zip(frequencies, amplitudes):
            h += a * np.cos(2 * np.pi * f * np.arange(N) / N + phase)
        h = 0.5 + h
        h = np.clip(h, 0, 1)
        # Enforce sum constraint
        h = h + (N / 2.0 - np.sum(h)) / N
        h = np.clip(h, 0, 1)
        return h

    inits = []
    # Single frequency
    inits.append(make_sinusoidal_init([1], [0.4]))
    inits.append(make_sinusoidal_init([2], [0.4]))
    inits.append(make_sinusoidal_init([3], [0.4]))
    # Multi-frequency
    inits.append(make_sinusoidal_init([1, 3], [0.3, 0.2]))
    inits.append(make_sinusoidal_init([1, 2, 4], [0.25, 0.2, 0.15]))
    inits.append(make_sinusoidal_init([2, 3, 5], [0.25, 0.2, 0.15]))
    # Alternating with smoothing
    alt = np.where(np.arange(N) % 2 == 0, 0.8, 0.2)
    alt = alt + (N / 2.0 - np.sum(alt)) / N
    alt = np.clip(alt, 0, 1)
    inits.append(alt)
    # Block pattern: first half 1s, second half 0s
    h_block = np.zeros(N)
    h_block[:N // 2] = 1.0
    h_block = h_block + (N / 2.0 - np.sum(h_block)) / N
    h_block = np.clip(h_block, 0, 1)
    inits.append(h_block)
    # Three-block pattern
    h_three = np.zeros(N)
    h_three[:N // 3] = 1.0
    h_three[2 * N // 3:] = 1.0
    h_three = h_three + (N / 2.0 - np.sum(h_three)) / N
    h_three = np.clip(h_three, 0, 1)
    inits.append(h_three)
    # Random
    for _ in range(5):
        r = np.random.uniform(0, 1, N)
        r = r + (N / 2.0 - np.sum(r)) / N
        r = np.clip(r, 0, 1)
        inits.append(r)

    for h_init in inits:
        if time.time() > deadline:
            break
        try:
            result = minimize(
                objective, h_init,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-12}
            )
            if result.fun < best_val:
                best_h = result.x
                best_val = result.fun
        except Exception:
            continue

    # Final polish with tighter tolerance
    if time.time() < deadline - 10:
        try:
            result = minimize(
                objective, best_h,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-14}
            )
            if result.fun < best_val:
                best_h = result.x
                best_val = result.fun
        except Exception:
            pass

    # Ensure constraints are satisfied
    best_h = np.clip(best_h, 0, 1)
    best_h = best_h + (N / 2.0 - np.sum(best_h)) / N
    best_h = np.clip(best_h, 0, 1)

    return best_h
# EVOLVE-BLOCK-END
