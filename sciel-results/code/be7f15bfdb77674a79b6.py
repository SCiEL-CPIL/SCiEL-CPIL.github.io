"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import math

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s kill

    N = 24
    sqrt_N = math.sqrt(N)

    # Initial candidate: Golay-Rudin-Shapiro truncated to 24
    p, q = [1], [1]
    while len(p) < N:
        p, q = p + q, p + [-c for c in q]
    best_coeffs = np.array(p[:N], dtype=np.int8)

    # Pre-compute a dense grid for fast sup-norm estimation
    n_grid = 4096
    angles = 2 * np.pi * np.arange(n_grid) / n_grid
    zs = np.exp(1j * angles)
    # Power matrix: (n_grid, N)
    powers = np.ones((n_grid, N), dtype=complex)
    for k in range(1, N):
        powers[:, k] = powers[:, k - 1] * zs

    def fast_sup_norm(coeffs):
        """Fast sup-norm estimate on the fixed grid."""
        vals = powers @ coeffs
        return np.max(np.abs(vals))

    def precise_sup_norm(coeffs):
        """
        High-accuracy sup-norm: find top peaks on dense grid, then refine
        each peak via golden-section search for max |P(z)|.
        """
        vals = powers @ coeffs
        abs_vals = np.abs(vals)
        # Find top 5 peaks
        top_idx = np.argsort(abs_vals)[-5:]
        best_val = 0.0
        for idx in top_idx:
            # Golden-section refinement around angles[idx]
            a = angles[idx] - 2 * np.pi / n_grid
            b = angles[idx] + 2 * np.pi / n_grid
            # Make sure a,b are in [0, 2*pi)
            a = a % (2 * np.pi)
            b = b % (2 * np.pi)
            if a > b:
                a, b = b, a
            # Golden-section search for max |P(z)|
            gr = (5 ** 0.5 - 1) / 2  # golden ratio conjugate
            c = b - gr * (b - a)
            d = a + gr * (b - a)
            fc = abs(np.sum(coeffs * np.exp(1j * c * np.arange(N))))
            fd = abs(np.sum(coeffs * np.exp(1j * d * np.arange(N))))
            for _ in range(40):
                if fc > fd:
                    b = d
                    d = c
                    fd = fc
                    c = b - gr * (b - a)
                    fc = abs(np.sum(coeffs * np.exp(1j * c * np.arange(N))))
                else:
                    a = c
                    c = d
                    fc = fd
                    d = a + gr * (b - a)
                    fd = abs(np.sum(coeffs * np.exp(1j * d * np.arange(N))))
            mid = (a + b) / 2
            fmid = abs(np.sum(coeffs * np.exp(1j * mid * np.arange(N))))
            best_val = max(best_val, fmid)
        return best_val

    best_score = fast_sup_norm(best_coeffs) / sqrt_N

    # Peak-driven coordinate descent
    # At each step, try flipping each coefficient; keep the flip that most reduces
    # the fast sup-norm. This is deterministic and exploits structure.

    current = best_coeffs.copy()
    current_val = fast_sup_norm(current)

    no_improve_count = 0
    max_no_improve = 50

    while time.time() < deadline:
        improved = False
        # Try each coefficient flip
        best_flip_idx = -1
        best_flip_val = current_val
        for i in range(N):
            trial = current.copy()
            trial[i] = -trial[i]
            trial_val = fast_sup_norm(trial)
            if trial_val < best_flip_val:
                best_flip_val = trial_val
                best_flip_idx = i
                best_trial = trial.copy()

        if best_flip_idx >= 0:
            current = best_trial
            current_val = best_flip_val
            improved = True
            no_improve_count = 0

            if current_val / sqrt_N < best_score:
                best_score = current_val / sqrt_N
                best_coeffs = current.copy()
        else:
            no_improve_count += 1

        # If stuck, try a small random perturbation to escape local min
        if no_improve_count >= max_no_improve:
            # Flip 2-3 random coefficients
            n_flips = 2 + (no_improve_count % 2)
            for _ in range(n_flips):
                idx = np.random.randint(0, N)
                current[idx] = -current[idx]
            current_val = fast_sup_norm(current)
            if current_val / sqrt_N < best_score:
                best_score = current_val / sqrt_N
                best_coeffs = current.copy()
            no_improve_count = 0

        # Periodically refine with precise sup-norm to verify
        if time.time() < deadline - 5.0:
            refined = precise_sup_norm(best_coeffs) / sqrt_N
            if refined < best_score:
                best_score = refined

    # Final precise evaluation
    final_score = precise_sup_norm(best_coeffs) / sqrt_N
    return best_coeffs.astype(int).tolist()
# EVOLVE-BLOCK-END
