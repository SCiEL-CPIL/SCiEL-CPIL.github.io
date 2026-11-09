"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def construct():
    """Construct a signed step function that minimizes the autoconvolution upper bound.
    
    The score is seed/value where value is the raw upper bound (lower is better).
    We minimize the peak sidelobe magnitude of the aperiodic autocorrelation.
    Uses greedy local search with FFT-based evaluation.
    """
    t_start = time.time()
    DEADLINE = 115.0  # Safety margin before 120s kill

    # Precompute FFT size (next power of 2 >= 2N)
    n_fft = 1 << (2 * N).bit_length()

    def compute_peak_sidelobe(h):
        """Compute peak sidelobe magnitude of aperiodic autocorrelation.
        
        Uses FFT for O(N log N) computation.
        Returns max |r[k]| for k = 1, ..., N-1.
        """
        F = np.fft.rfft(h, n=n_fft)
        power = F * np.conj(F)
        r = np.fft.irfft(power, n=n_fft)
        # r[0] is lag 0, r[1:N] are positive lags 1..N-1
        # For real signals, |r[k]| = |r[-k]|, so we only need positive lags
        sidelobes = r[1:N]
        return np.max(np.abs(sidelobes))

    # Initialize with random signs
    rng = np.random.default_rng(42)
    h = rng.choice([-1.0, 1.0], size=N)

    best_h = h.copy()
    best_peak = compute_peak_sidelobe(h)

    no_improve_count = 0
    max_no_improve = 30

    while time.time() - t_start < DEADLINE:
        improved = False

        # Try flipping each position in a full pass
        for i in range(N):
            if time.time() - t_start > DEADLINE - 5:
                break

            # Try flipping x[i]
            old_val = h[i]
            h[i] = -old_val

            new_peak = compute_peak_sidelobe(h)

            if new_peak < best_peak - 1e-12:
                best_peak = new_peak
                best_h = h.copy()
                improved = True
            else:
                h[i] = old_val

        if not improved:
            no_improve_count += 1
            if no_improve_count >= max_no_improve:
                # Restart with new random initialization
                h = rng.choice([-1.0, 1.0], size=N)
                current_peak = compute_peak_sidelobe(h)
                no_improve_count = 0

                # If this new random is better, update best
                if current_peak < best_peak:
                    best_peak = current_peak
                    best_h = h.copy()
        else:
            no_improve_count = 0

    return best_h
# EVOLVE-BLOCK-END
