"""Improved program for the third autocorrelation inequality task."""
import numpy as np
import time

N = 400


def _fft_autocorr_max_abs(s):
    """Compute max |autocorr| over non-zero lags for a length-N integer sequence.

    Returns the maximum of |r[k]| for k=1..N-1 where r = ifft(|fft(s, n_fft)|^2).real
    and r[k] is the circular autocorrelation at lag k (with n_fft large enough to
    capture the linear autocorrelation).
    """
    n_fft = 1
    while n_fft < 2 * N:
        n_fft *= 2
    s_f = s.astype(np.float64)
    F = np.fft.fft(s_f, n=n_fft)
    acf = np.fft.ifft(F * np.conj(F)).real[:N]
    # Normalize by N so that lag-0 is ~sum(s^2)/N
    acf /= N
    return float(np.max(np.abs(acf[1:])))


def _score(s):
    """Score = upper bound on C3; lower is better."""
    return _fft_autocorr_max_abs(s)


def _random_walk_sequence():
    """Generate a random +/-1 sequence."""
    return np.random.choice([-1, 1], size=N)


def _construct_seed():
    """Construct the all-ones seed (score 1.0 baseline)."""
    return np.ones(N, dtype=np.int8)


def construct():
    """
    Search for a signed step function (400 intervals of +/-1) that minimizes
    the max absolute off-peak autocorrelation, which upper-bounds C3.

    Strategy:
    - Use Simulated Annealing with multi-bit flips to escape local minima.
    - Follow up with a strict hill-climbing polish.
    - Keep the best sequence found within the time budget.
    """
    deadline = time.time() + 110  # safety margin before 120s kill
    rng = np.random.default_rng(42)

    # Start with a random sequence, as all-ones is a poor local minimum
    seq = rng.choice([-1, 1], size=N).astype(np.int8)
    current_score = _score(seq)
    best_seq = seq.copy()
    best_score = current_score

    # Simulated Annealing Parameters
    T = 0.1  # Initial temperature
    T_min = 1e-6
    alpha = 0.995  # Cooling rate
    steps = 0

    while time.time() < deadline - 10:  # Leave 10s for polish
        # Choose number of bits to flip (1 to 5)
        n_flips = rng.integers(1, 6)
        idxs = rng.choice(N, size=n_flips, replace=False)
        
        # Apply flip
        seq[idxs] *= -1
        new_score = _score(seq)
        delta = new_score - current_score
        
        # Accept if better, or with probability exp(-delta/T)
        if delta < 0 or rng.random() < np.exp(-delta / T):
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_seq = seq.copy()
        else:
            # Revert
            seq[idxs] *= -1
            
        T *= alpha
        if T < T_min:
            T = T_min
        steps += 1

    # Polish phase: strict hill climbing with single-bit flips
    seq = best_seq.copy()
    current_score = best_score
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in range(N):
            old_val = seq[idx]
            seq[idx] = -old_val
            new_score = _score(seq)
            if new_score < current_score:
                current_score = new_score
                best_score = current_score
                best_seq = seq.copy()
                improved = True
            else:
                seq[idx] = old_val

    return best_seq
