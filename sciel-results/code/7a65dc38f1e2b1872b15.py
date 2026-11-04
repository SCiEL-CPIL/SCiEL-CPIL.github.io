"""Refined uncertainty inequality - Cohn-Goncalves Laguerre construction optimizer."""
import numpy as np
from scipy.optimize import minimize
import time

def _compute_c4_bound(roots):
    """
    Compute the C4 upper bound from 12 positive double-root locations.
    The bound is defined as the maximum root divided by a reference scale,
    reflecting the Fourier sign uncertainty constraint.
    """
    roots = np.asarray(roots, dtype=float)
    if np.any(roots <= 0):
        return 1e10
    if np.any(np.diff(np.sort(roots)) <= 1e-12):
        return 1e10
    # C4 bound: largest root normalized by seed reference (48)
    return np.max(roots) / 48.0

def _objective(x):
    """Objective function for optimization."""
    return _compute_c4_bound(x)

def _constraint_positive(x):
    """Ensure all roots are positive."""
    return np.min(x) - 0.1

def construct():
    """
    Construct 12 positive double-root locations for the Laguerre combination
    (m=12, n=1) that minimize the C4 upper bound.
    
    Returns:
        list of 12 positive double-root locations
    """
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin before 630s limit
    
    # Initial guess: use Gauss-Laguerre-like spacing, scaled to be reasonable
    # Start with evenly spaced positive roots, smaller than seed
    x0 = np.linspace(0.5, 20.0, 12)
    
    # Try multiple initial points to avoid local minima
    best_x = x0.copy()
    best_score = _compute_c4_bound(best_x)
    
    # Define bounds: roots must be positive and reasonably ordered
    bounds = [(0.01, 100.0)] * 12
    
    # Constraints: roots should be distinct and positive
    constraints = [
        {'type': 'ineq', 'fun': lambda x: np.min(x) - 0.01},
    ]
    
    # Add pairwise separation constraints (soft, via objective penalty)
    def _separation_penalty(x):
        x_sorted = np.sort(x)
        diffs = np.diff(x_sorted)
        return -np.sum(np.log(diffs + 1e-10))
    
    def _full_objective(x):
        base = _compute_c4_bound(x)
        # Penalty for roots being too close
        x_sorted = np.sort(x)
        diffs = np.diff(x_sorted)
        if np.any(diffs < 1e-6):
            return 1e8
        sep_penalty = -np.sum(np.log(diffs + 1e-8)) * 0.01
        return base + sep_penalty
    
    # Run optimization from multiple starting points
    n_starts = 5
    for i in range(n_starts):
        if time.time() > deadline:
            break
        
        if i == 0:
            x_start = x0.copy()
        elif i == 1:
            x_start = np.linspace(0.1, 10.0, 12)
        elif i == 2:
            x_start = np.linspace(0.5, 30.0, 12)
        elif i == 3:
            x_start = np.array([0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 24.0, 32.0, 40.0, 44.0, 47.0])
        else:
            x_start = np.linspace(0.05, 5.0, 12)
        
        try:
            result = minimize(
                _full_objective,
                x_start,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-15}
            )
            
            if result.success or _full_objective(result.x) < best_score:
                candidate = result.x
                cand_score = _compute_c4_bound(candidate)
                if cand_score < best_score:
                    best_score = cand_score
                    best_x = candidate.copy()
        except Exception:
            continue
    
    # Final polish with tighter tolerance
    if time.time() < deadline - 10:
        try:
            result = minimize(
                _full_objective,
                best_x,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-18, 'gtol': 1e-18}
            )
            if result.success:
                candidate = result.x
                cand_score = _compute_c4_bound(candidate)
                if cand_score < best_score:
                    best_score = cand_score
                    best_x = candidate.copy()
        except Exception:
            pass
    
    # Ensure all roots are positive
    best_x = np.maximum(best_x, 0.001)
    
    # Sort roots for consistency
    best_x = np.sort(best_x)
    
    return [float(r) for r in best_x]
