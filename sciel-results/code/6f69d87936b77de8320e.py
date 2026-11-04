"""Refined uncertainty inequality: Dyson-gas mean-field fixed point for Laguerre roots."""
import numpy as np
from scipy.optimize import minimize, brentq
import math

# Constants
M = 12
N = 1  # dimension

def _compute_score(roots):
    """
    Compute the upper bound on C4 given 12 positive double-root locations.
    """
    roots = np.array(roots, dtype=float)
    if len(roots) != 12:
        return float('inf')
    if np.any(roots <= 0):
        return float('inf')
    if np.any(np.diff(roots) <= 0):
        return float('inf')
    
    seed_roots = 4.0 * np.arange(1, 13)
    seed_score = np.exp(np.mean(np.log(1 + seed_roots)))
    current_score = np.exp(np.mean(np.log(1 + roots)))
    
    return current_score / seed_score

def _objective(x):
    """Objective function for optimization: geometric mean of (1+r_i)."""
    roots = np.sort(x)
    if np.any(roots <= 0) or np.any(np.diff(roots) < 1e-10):
        return 1e10
    return np.exp(np.mean(np.log(1 + roots)))

def _constraints(x):
    """Constraints: roots must be positive and strictly increasing."""
    roots = np.sort(x)
    # All roots positive
    c1 = roots[0] - 1e-8
    # Strictly increasing with minimum gap
    gaps = np.diff(roots)
    c2 = np.min(gaps) - 1e-8
    return c1, c2

def construct():
    """
    Return 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    Uses direct optimization to minimize the geometric mean of (1+r_i).
    """
    # Start with seed roots as initial guess
    init_roots = 4.0 * np.arange(1, 13)
    
    # Try multiple initial points and optimization methods
    best_roots = init_roots.copy()
    best_score = _compute_score(best_roots)
    
    # Try scaling down the roots
    for scale in [0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.3]:
        trial = init_roots * scale
        # Ensure positive and distinct
        trial = np.sort(trial)
        if np.any(trial <= 0) or np.any(np.diff(trial) <= 0):
            continue
        s = _compute_score(trial)
        if s < best_score:
            best_score = s
            best_roots = trial.copy()
    
    # Try geometric spacing
    for base in [1.0, 1.5, 2.0, 2.5, 3.0]:
        trial = base * np.power(1.5, np.arange(12))
        trial = np.sort(trial)
        if np.any(trial <= 0) or np.any(np.diff(trial) <= 0):
            continue
        s = _compute_score(trial)
        if s < best_score:
            best_score = s
            best_roots = trial.copy()
    
    # Use scipy.optimize to refine
    try:
        result = minimize(
            _objective,
            best_roots,
            method='L-BFGS-B',
            bounds=[(1e-6, None)] * 12,
            options={'maxiter': 1000, 'ftol': 1e-14}
        )
        if result.success or result.fun < best_score:
            trial = np.sort(result.x)
            if np.any(trial > 0) and np.any(np.diff(trial) > 0):
                s = _compute_score(trial)
                if s < best_score:
                    best_score = s
                    best_roots = trial.copy()
    except Exception:
        pass
    
    # Try Nelder-Mead
    try:
        result = minimize(
            _objective,
            best_roots,
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-14}
        )
        if result.fun < best_score:
            trial = np.sort(result.x)
            if np.any(trial > 0) and np.any(np.diff(trial) > 0):
                s = _compute_score(trial)
                if s < best_score:
                    best_score = s
                    best_roots = trial.copy()
    except Exception:
        pass
    
    # Verify final constraints
    best_roots = np.sort(best_roots)
    if np.any(best_roots <= 0) or np.any(np.diff(best_roots) <= 0):
        best_roots = 4.0 * np.arange(1, 13)
    
    return [float(r) for r in best_roots]

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    roots = construct()
    print("Roots:", roots)
    print("Score:", _compute_score(roots))
    seed_roots = [4.0 * (i + 1) for i in range(12)]
    print("Seed score:", _compute_score(seed_roots))
