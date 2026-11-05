"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
import numpy as np


def _value(h):
    """Compute the task objective for a given height array."""
    s = np.sum(h)
    if s == 0 or not np.all(np.isfinite(h)):
        return np.inf
    conv = np.convolve(h, h)
    return abs(2 * N * np.max(conv) / (s ** 2))


def _evaluate_theta(theta, t0, deadline):
    """Evaluate the objective for a chirp signal with given theta."""
    if time.time() > deadline:
        return np.inf, None
    k = np.arange(N)
    h = np.cos(theta * k)
    s = np.sum(h)
    if abs(s) < 1e-12:
        return np.inf, None
    conv = np.convolve(h, h)
    val = abs(2 * N * np.max(conv) / (s ** 2))
    return val, h


def construct():
    t0 = time.time()
    deadline = t0 + 115.0  # safety margin before 120s limit
    
    # Try a grid of theta values
    best_val = np.inf
    best_h = None
    
    # First, try some structured theta values
    thetas_to_try = []
    
    # Uniform grid
    n_grid = 2000
    thetas_to_try.extend(np.linspace(0, np.pi, n_grid, endpoint=False))
    
    # Also try some specific values
    for m in range(1, 50):
        thetas_to_try.append(2 * np.pi * m / N)
    
    for theta in thetas_to_try:
        if time.time() > deadline:
            break
        val, h = _evaluate_theta(theta, t0, deadline)
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    # Refine around best theta using local search
    if best_h is not None and time.time() < deadline:
        # Find the best theta from the grid
        best_theta = None
        best_val_refine = best_val
        for theta in thetas_to_try:
            if time.time() > deadline:
                break
            val, _ = _evaluate_theta(theta, t0, deadline)
            if val < best_val_refine:
                best_val_refine = val
                best_theta = theta
        
        if best_theta is not None:
            # Local refinement
            step = 2 * np.pi / N
            for _ in range(100):
                if time.time() > deadline:
                    break
                improved = False
                for delta in [step / 10, -step / 10, step / 100, -step / 100]:
                    new_theta = best_theta + delta
                    val, h = _evaluate_theta(new_theta, t0, deadline)
                    if val < best_val_refine:
                        best_val_refine = val
                        best_h = h.copy()
                        best_theta = new_theta
                        improved = True
                if not improved:
                    break
    
    if best_h is None:
        return np.ones(N)
    
    return best_h
# EVOLVE-BLOCK-END
