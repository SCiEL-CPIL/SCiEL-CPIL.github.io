"""Program that searches for step heights minimizing the autocorrelation inequality value."""
import numpy as np
import time

N = 400


def _eval_value(h):
    """Compute the objective value (lower is better)."""
    h = np.asarray(h, dtype=np.float64)
    if np.any(~np.isfinite(h)):
        return np.inf
    sh = float(np.sum(h))
    if sh == 0.0:
        return np.inf
    cc = np.convolve(h, h, mode='full')
    return abs(2.0 * N * float(np.max(cc)) / (sh * sh))


def construct():
    """
    Search for step heights h of length N that minimize:
        value = abs(2*N*max(convolve(h,h)) / sum(h)^2)
    
    Strategy: Start from a random vector, perform a limited number of
    coordinate-descent / random-perturbation steps, always keeping the best.
    Use a deadline guard so we finish well under 120s.
    """
    deadline = time.time() + 110.0  # safety margin before 120s wall-clock
    rng = np.random.default_rng(42)

    # --- Initial candidate: random vector with zero mean to make sum(h)=0?
    # No, sum(h) must be nonzero. Start with a random vector.
    # Actually, a good starting point might be a random vector with small mean.
    # Let's try a few different initializations and pick the best after a short warm-up.
    
    best_h = None
    best_val = np.inf

    # Try several random initializations
    for init in range(8):
        if time.time() > deadline:
            break
        if init == 0:
            h = np.ones(N)
        else:
            h = rng.standard_normal(N)
            # Ensure sum is not zero
            if abs(np.sum(h)) < 1e-10:
                h[0] += 1.0

        val = _eval_value(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()

    # Now do a limited number of perturbation steps from the best so far
    if best_h is None:
        best_h = np.ones(N)
        best_val = _eval_value(best_h)

    current_h = best_h.copy()
    current_val = best_val

    # Coordinate perturbation: pick a random index, add a small random delta
    n_steps = 50000
    for step in range(n_steps):
        if time.time() > deadline:
            break

        idx = rng.integers(0, N)
        delta = rng.standard_normal() * 0.5
        old = current_h[idx]
        current_h[idx] = old + delta

        val = _eval_value(current_h)
        if val < best_val:
            best_val = val
            best_h = current_h.copy()
        else:
            # revert if worse
            current_h[idx] = old
            # occasionally re-sync current to best
            if step % 100 == 0 and val > best_val * 1.5:
                current_h = best_h.copy()
                current_val = best_val

    return best_h
