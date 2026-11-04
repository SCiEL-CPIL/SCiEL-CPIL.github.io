"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    rng = np.random.RandomState(42)

    # Initialize with random signs
    best_h = rng.choice([-1.0, 1.0], size=N)
    best_H = np.fft.fft(best_h)
    best_score = np.sum(np.abs(best_H) ** 4)

    # Precompute phase factors for fast DFT update
    # H[k] = sum_i h[i] * exp(-i * 2*pi*k*i/N)
    # Flipping h[i]: H_new[k] = H[k] - 2*h[i]*exp(-i*2*pi*k*i/N)
    idx = np.arange(N)
    k_idx = np.arange(N)
    # phases[i, k] = exp(-i * 2*pi*k*i/N)
    phases = np.exp(-1j * 2 * np.pi * np.outer(idx, k_idx) / N)

    # Simulated annealing
    current_h = best_h.copy()
    current_H = best_H.copy()
    current_score = best_score

    T = 1.0  # Initial temperature
    cooling = 0.9995

    while time.time() < deadline:
        T *= cooling
        if T < 1e-8:
            T = 1e-8

        # Try a random flip
        i = rng.randint(N)
        old_val = current_h[i]
        # New H after flip
        delta = -2.0 * old_val * phases[i, :]
        new_H = current_H + delta
        new_score = np.sum(np.abs(new_H) ** 4)

        delta_score = new_score - current_score
        if delta_score < 0 or rng.rand() < np.exp(-delta_score / T):
            # Accept
            current_h[i] = -old_val
            current_H = new_H
            current_score = new_score

            if current_score < best_score:
                best_score = current_score
                best_h = current_h.copy()
                best_H = current_H.copy()
        else:
            # Reject, no change needed
            pass

    return best_h
# EVOLVE-BLOCK-END
