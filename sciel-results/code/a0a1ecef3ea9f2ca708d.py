<<<<<<< SEARCH
N = 100

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    return np.full(N, 0.5)
# EVOLVE-BLOCK-END
=======
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_start_time = time.time()
_DEADLINE = 140  # seconds, safety margin before 150s kill


def _compute_max_overlap(h):
    """Compute max overlap of h with shifted 1-h over all cyclic shifts."""
    n = len(h)
    overlaps = np.zeros(n)
    for k in range(n):
        shifted = np.roll(h, k)
        overlaps[k] = np.dot(h, 1.0 - shifted)
    return np.max(overs)


def _objective(params):
    """Objective: max overlap, with constraint handled via penalty."""
    h = np.clip(params, 0.0, 1.0)
    # Penalty for constraint violation: sum(h) should be N/2
    sum_penalty = (np.sum(h) - N / 2.0) ** 2 * 1000.0
    return _compute_max_overlap(h) + sum_penalty


def _try_gradient_descent(h_init, max_iter=2000, tol=1e-8):
    """Use scipy minimize with L-BFGS-B."""
    result = minimize(
        _objective,
        h_init,
        method='L-BFGS-B',
        bounds=[(0, 1)] * N,
        options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-10}
    )
    h = np.clip(result.x, 0.0, 1.0)
    # Project back to satisfy constraint exactly
    diff = N / 2.0 - np.sum(h)
    h = h + diff / N
    h = np.clip(h, 0.0, 1.0)
    # Final projection to ensure exact sum
    diff = N / 2.0 - np.sum(h)
    if abs(diff) > 1e-12:
        # Adjust one element
        idx = np.argmin(h) if diff > 0 else np.argmax(h)
        h[idx] += diff
        h = np.clip(h, 0.0, 1.0)
    return h, _compute_max_overlap(h)


def construct():
    global _start_time
    best_h = np.full(N, 0.5)
    best_score = _compute_max_overlap(best_h)
    
    # Try multiple random initializations
    np.random.seed(42)
    n_tries = 0
    max_tries = 50
    
    while time.time() - _start_time < _DEADLINE:
        n_tries += 1
        # Generate random initial point
        h_init = np.random.uniform(0, 1, N)
        # Scale to satisfy constraint
        h_init = h_init / np.sum(h_init) * (N / 2.0)
        h_init = np.clip(h_init, 0.0, 1.0)
        
        h, score = _try_gradient_descent(h_init, max_iter=1000)
        
        if score < best_score:
            best_score = score
            best_h = h.copy()
        
        if n_tries >= max_tries:
            break
    
    # Final projection to ensure exact sum
    diff = N / 2.0 - np.sum(best_h)
    if abs(diff) > 1e-12:
        # Distribute adjustment evenly, then fix any out-of-bounds
        best_h = best_h + diff / N
        best_h = np.clip(best_h, 0.0, 1.0)
        # If still not exact, adjust one element
        diff2 = N / 2.0 - np.sum(best_h)
        if abs(diff2) > 1e-12:
            idx = np.argmin(best_h) if diff2 > 0 else np.argmax(best_h)
            best_h[idx] += diff2
    
    return best_h
# EVOLVE-BLOCK-END
>>>>>>> REPLACE

Wait, I have a typo in `_compute_max_overlap` - I wrote `overs` instead of `overlaps`. Let me fix that.

Also, let me reconsider the approach. The gradient-based optimization might be slow. Let me think about whether there's a smarter structure.

Actually, let me reconsider. The problem is to maximize min_k C(k) where C(k) is the autocorrelation. This is a well-studied problem in signal processing. Sequences with low peak sidelobe or good autocorrelation properties might help.

Let me try a different approach: use a smooth function like a raised cosine or other band-limited function, which tends to have smoother autocorrelation.

Let me also fix the typo and make the code more robust:

