"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution


def _objective(h):
    """Compute the value to minimize."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10
    ac = np.convolve(h, h, mode='full')
    return abs(2.0 * N * np.max(ac) / (s * s))


def _evaluate(h):
    return _objective(h)


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    best_h = np.ones(N)
    best_val = _evaluate(best_h)
    
    # Try several random initializations with mixed signs
    rng = np.random.default_rng(42)
    
    candidates = []
    for trial in range(50):
        if time.time() > deadline - 5.0:
            break
        # Random initialization: mix of +1 and -1 with some structure
        h = rng.choice([-1.0, 1.0], size=N)
        # Ensure sum is not too small
        s = np.sum(h)
        if abs(s) < 10:
            # Adjust: flip some to increase sum
            for i in rng.permutation(N):
                if abs(np.sum(h)) > 50:
                    break
                h[i] = 1.0 if s < 0 else -1.0
                s = np.sum(h)
        candidates.append(h.copy())
    
    # Also try structured patterns
    # Pattern: alternating blocks
    for block_size in [10, 20, 50, 100]:
        if time.time() > deadline - 5.0:
            break
        h = np.ones(N)
        for i in range(0, N, block_size * 2):
            h[i:i+block_size] = -1.0
        s = np.sum(h)
        if abs(s) > 20:
            candidates.append(h.copy())
    
    # Pattern: low-frequency cosine with sign
    for freq in [1, 2, 3, 5]:
        if time.time() > deadline - 5.0:
            break
        x = np.linspace(0, freq * 2 * np.pi, N)
        h = np.sign(np.cos(x))
        s = np.sum(h)
        if abs(s) > 20:
            candidates.append(h.copy())
    
    # Evaluate all candidates
    for h in candidates:
        if time.time() > deadline - 5.0:
            break
        val = _evaluate(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    # Refine best candidate using local optimization
    if time.time() < deadline - 30.0:
        def obj_func(x):
            return _evaluate(x)
        
        try:
            result = minimize(
                obj_func,
                best_h,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            if time.time() < deadline - 5.0:
                val = _evaluate(result.x)
                if val < best_val:
                    best_val = val
                    best_h = result.x.copy()
        except Exception:
            pass
    
    # Try a few more local optimizations from different starts
    for h in candidates[:5]:
        if time.time() > deadline - 10.0:
            break
        try:
            result = minimize(
                obj_func,
                h,
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            if time.time() > deadline - 5.0:
                val = _evaluate(result.x)
                if val < best_val:
                    best_val = val
                    best_h = result.x.copy()
        except Exception:
            pass
    
    return best_h
# EVOLVE-BLOCK-END
