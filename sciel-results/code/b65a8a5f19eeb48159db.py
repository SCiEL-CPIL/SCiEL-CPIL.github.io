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
    deadline = start + 140  # 20s safety margin before 160s hard limit

    def objective(h):
        return _objective(h)

    bounds = [(0.0, 1.0)] * N
    constraints = [{'type': 'eq', 'fun': lambda h: np.sum(h) - N / 2.0}]

    best_h = np.full(N, 0.5)
    best_val = objective(best_h)

    np.random.seed(42)

    # Try several restarts with different initializations
    for restart in range(15):
        if time.time() > deadline:
            break

        if restart == 0:
            h_init = best_h.copy()
        elif restart == 1:
            # Alternating pattern
            h_init = np.where(np.arange(N) % 2 == 0, 0.8, 0.2)
            h_init = h_init + (N / 2.0 - np.sum(h_init)) / N
            h_init = np.clip(h_init, 0, 1)
        elif restart == 2:
            # Block pattern: first half 1s, second half 0s
            h_init = np.zeros(N)
            h_init[:N // 2] = 1.0
            h_init = h_init + (N / 2.0 - np.sum(h_init)) / N
            h_init = np.clip(h_init, 0, 1)
        elif restart == 3:
            # Three-block pattern
            h_init = np.zeros(N)
            h_init[:N // 3] = 1.0
            h_init[2 * N // 3:] = 1.0
            h_init = h_init + (N / 2.0 - np.sum(h_init)) / N
            h_init = np.clip(h_init, 0, 1)
        elif restart == 4:
            # Sinusoidal
            h_init = 0.5 + 0.4 * np.cos(2 * np.pi * np.arange(N) / N)
            h_init = np.clip(h_init, 0, 1)
            h_init = h_init + (N / 2.0 - np.sum(h_init)) / N
            h_init = np.clip(h_init, 0, 1)
        elif restart == 5:
            # Higher frequency sinusoid
            h_init = 0.5 + 0.4 * np.cos(4 * np.pi * np.arange(N) / N)
            h_init = np.clip(h_init, 0, 1)
            h_init = h_init + (N / 2.0 - np.sum(h_init)) / N
            h_init = np.clip(h_init, 0, 1)
        else:
            # Random
            h_init = np.random.uniform(0, 1, N)
            h_init = h_init + (N / 2.0 - np.sum(h_init)) / N
            h_init = np.clip(h_init, 0, 1)

        try:
            result = minimize(
                objective, h_init,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 300, 'ftol': 1e-10}
            )
            if result.success and result.fun < best_val:
                best_h = result.x
                best_val = result.fun
        except Exception:
            continue

    # Final polish
    if time.time() < deadline - 10:
        try:
            result = minimize(
                objective, best_h,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-14}
            )
            if result.success and result.fun < best_val:
                best_h = result.x
                best_val = result.fun
        except Exception:
            pass

    # Ensure constraints
    best_h = np.clip(best_h, 0, 1)
    best_h = best_h + (N / 2.0 - np.sum(best_h)) / N
    best_h = np.clip(best_h, 0, 1)

    return best_h
# EVOLVE-BLOCK-END
