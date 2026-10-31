"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.signal import fftconvolve


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s kill

    def autoconv_peak(x):
        """Compute max absolute autoconvolution value."""
        if np.all(x == 0):
            return 0.0
        ac = fftconvolve(x, x, mode='full')
        return np.max(np.abs(ac))

    def score(x):
        """max|autoconv| / (sum)^2 - lower is better."""
        s = np.sum(x)
        if abs(s) < 1e-12:
            return float('inf')
        return autoconv_peak(x) / (s * s)

    # Start with all ones (constant) as baseline
    best_x = np.ones(N)
    best_val = score(best_x)

    rng = np.random.default_rng(42)

    # Van der Corput sequence (base-2) for low-discrepancy initialization
    def van_der_corput(n, base=2):
        """Generate first n values of Van der Corput sequence in [0,1)."""
        seq = np.zeros(n)
        for i in range(n):
            f = 1.0 / base
            num = i
            val = 0.0
            while num > 0:
                val += f * (num % base)
                num //= base
                f /= base
            seq[i] = val
        return seq

    # Kronecker sequence
    def kronecker(n, alpha):
        """Generate Kronecker sequence {n*alpha mod 1} for n=0..N-1."""
        return np.mod(np.arange(n) * alpha, 1.0)

    # Try diverse initialization strategies with local search
    n_restarts = 60
    for r in range(n_restarts):
        if time.time() > deadline:
            break

        # Different initialization strategies
        if r % 6 == 0:
            # Random ±1
            x = rng.choice([-1.0, 1.0], size=N)
        elif r % 6 == 1:
            # Van der Corput based: sign from fractional part
            vdc = van_der_corput(N, base=2)
            x = np.where(vdc < 0.5, 1.0, -1.0)
        elif r % 6 == 2:
            # Kronecker sequence with golden ratio
            alpha = (1 + np.sqrt(5)) / 2  # golden ratio
            kron = kronecker(N, alpha)
            x = np.where(kron < 0.5, 1.0, -1.0)
        elif r % 6 == 3:
            # Chirp-like pattern
            t = np.arange(N)
            x = np.sign(np.cos(2 * np.pi * t * (t + 1) / (2 * N)))
        elif r % 6 == 4:
            # Block pattern with random block sizes
            x = np.ones(N)
            pos = 0
            while pos < N:
                block_len = rng.integers(1, max(2, N // 10 + 1))
                sign = rng.choice([-1.0, 1.0])
                x[pos:pos + min(pos + block_len, N)] = sign
                pos += block_len
        else:
            # Mostly +1 with some random -1
            x = np.ones(N)
            n_neg = rng.integers(N // 4, N // 2)
            neg_idx = rng.choice(N, n_neg, replace=False)
            x[neg_idx] = -1.0

        # Ensure sum is not too small; flip if needed
        s = np.sum(x)
        if abs(s) < N * 0.1:
            if s < 0:
                x = -x
            # Add a slight bias toward +1
            x = np.sign(x + 0.3)
            x[x == 0] = 1.0

        val = score(x)
        if val < best_val:
            best_val = val
            best_x = x.copy()

        # Local search: try flipping individual elements
        improved = True
        max_iters = 100
        for it in range(max_iters):
            if time.time() > deadline:
                break
            if not improved:
                break
            improved = False

            # Try random subset of positions
            trial_indices = rng.choice(N, size=min(30, N), replace=False)
            for idx in trial_indices:
                if time.time() > deadline:
                    break
                orig = x[idx]
                # Try flipping sign
                x[idx] = -orig
                new_val = score(x)
                if new_val < best_val - 1e-12:
                    best_val = new_val
                    best_x = x.copy()
                    improved = True
                else:
                    x[idx] = orig

    # Final polish: try small perturbations on the best solution
    if time.time() < deadline - 5:
        x = best_x.copy()
        for trial in range(300):
            if time.time() > deadline - 2:
                break
            idx = rng.integers(N)
            orig = x[idx]
            x[idx] = -orig
            new_val = score(x)
            if new_val < best_val - 1e-12:
                best_val = new_val
                best_x = x.copy()
            else:
                x[idx] = orig

    return best_x
# EVOLVE-BLOCK-END
