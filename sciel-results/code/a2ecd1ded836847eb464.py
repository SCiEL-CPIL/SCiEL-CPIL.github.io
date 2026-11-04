"""Spectral approach to Erdos minimum overlap: minimize max overlap of h with shifted (1-h)."""

import time
import numpy as np
from scipy.optimize import minimize

N = 100

# EVOLVE-BLOCK-START

def _score(h):
    """Compute the maximum overlap of h with shifted (1-h)."""
    h = np.asarray(h, dtype=float)
    n = len(h)
    # Circular cross-correlation: overlap_k = sum_i h_i * (1 - h_{(i-k) mod n})
    # Using FFT for efficiency
    H = np.fft.fft(h)
    # cross-correlation of h with h
    auto = np.real(np.fft.ifft(H * np.conj(H)))
    # overlap_k = sum_i h_i - sum_i h_i h_{i-k} = sum(h) - auto[k]
    # sum(h) = n/2 = 50
    s = h.sum()
    overlap = s - auto
    return float(np.max(overlap))


def _project(h):
    """Project to feasible set: [0,1] with sum = N/2."""
    h = np.clip(h, 0.0, 1.0)
    # Adjust to make sum exactly N/2
    target = N / 2.0
    diff = target - h.sum()
    if diff > 0:
        # Need to add: add to elements below 1
        room = 1.0 - h
        total_room = room.sum()
        if total_room > 0:
            h += room * (diff / total_room)
    elif diff < 0:
        # Need to subtract: subtract from elements above 0
        room = h
        total_room = room.sum()
        if total_room > 0:
            h -= room * (-diff / total_room)
    # Final clip and exact adjustment
    h = np.clip(h, 0.0, 1.0)
    # Exact sum correction via iterative water-filling
    for _ in range(20):
        diff = target - h.sum()
        if abs(diff) < 1e-15:
            break
        if diff > 0:
            mask = h < 1.0
        else:
            mask = h > 0.0
        if mask.any():
            h[mask] += diff / mask.sum()
            h = np.clip(h, 0.0, 1.0)
    return h


def _objective(params):
    """Objective: max overlap, with penalty for constraint violation."""
    h = _project(params)
    return _score(h)


def construct():
    start = time.time()
    deadline = start + 140.0  # safety margin before 150s kill

    # Start with a structured initial guess
    # Alternate high/low to reduce overlap
    h0 = np.zeros(N)
    h0[::2] = 1.0
    h0[1::2] = 0.0
    h0 = _project(h0)

    best_h = h0.copy()
    best_score = _score(best_h)

    # Try several random initializations with gradient descent
    rng = np.random.default_rng(42)

    n_trials = 30
    for trial in range(n_trials):
        if time.time() > deadline:
            break

        # Generate a random feasible starting point
        if trial == 0:
            h_init = h0.copy()
        elif trial == 1:
            # Smooth random
            h_init = 0.5 + 0.3 * np.sin(np.linspace(0, 4*np.pi, N))
            h_init = _project(h_init)
        elif trial == 2:
            # Another smooth pattern
            h_init = 0.5 + 0.3 * np.cos(np.linspace(0, 2*np.pi, N))
            h_init = _project(h_init)
        else:
            # Random with some structure
            freq = rng.integers(1, N // 2)
            phase = rng.uniform(0, 2*np.pi)
            h_init = 0.5 + 0.25 * np.sin(2*np.pi*freq*np.arange(N)/N + phase)
            h_init = _project(h_init)

        # Use scipy minimize with L-BFGS-B
        try:
            res = minimize(
                _objective,
                h_init,
                method='L-BFGS-B',
                bounds=[(0.0, 1.0)] * N,
                options={'maxiter': 500, 'maxfun': 2000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            h_res = _project(res.x)
            s = _score(h_res)
            if s < best_score:
                best_score = s
                best_h = h_res.copy()
        except Exception:
            pass

    # Local refinement: coordinate descent on the best solution
    if time.time() < deadline - 10:
        h = best_h.copy()
        improved = True
        max_rounds = 50
        for round_idx in range(max_rounds):
            if time.time() > deadline - 5:
                break
            improved = False
            for i in range(N):
                if time.time() > deadline - 5:
                    break
                # Try adjusting h[i] while keeping sum constant
                # Adjust h[i] and redistribute to neighbor
                for j in range(N):
                    if j == i:
                        continue
                    # Move delta from i to j
                    for delta in [0.01, 0.05, 0.1, 0.2]:
                        h_new = h.copy()
                        h_new[i] -= delta
                        h_new[j] += delta
                        if h_new[i] < 0 or h_new[j] > 1:
                            continue
                        h_new = _project(h_new)
                        s = _score(h_new)
                        if s < best_score - 1e-12:
                            best_score = s
                            best_h = h_new.copy()
                            h = h_new.copy()
                            improved = True
                            break
                    if improved:
                        break
            if not improved:
                break

    # Final exact sum correction
    best_h = _project(best_h)
    return best_h

# EVOLVE-BLOCK-END
