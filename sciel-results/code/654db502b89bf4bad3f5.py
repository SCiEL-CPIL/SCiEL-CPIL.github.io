"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100

def _score(h):
    """Compute max overlap of h with shifted (1-h), treating h as zero outside [0,2]."""
    n = len(h)
    shifted = np.zeros(n)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    return float(np.max(overlap))

def construct():
    deadline = time.time() + 100.0  # safety margin before 120s kill
    
    # Initial guess: Uniform 0.5
    h0 = np.full(N, 0.5)
    
    # We will minimize a smooth approximation of the max overlap
    # subject to sum(h) = N/2 and 0 <= h <= 1
    
    def objective_and_grad(h):
        # Compute overlaps
        shifted = np.zeros(N)
        shifted[:-1] = h[1:]
        overlap = h * (1.0 - shifted)
        
        # Smooth max
        lam = 100.0
        # Numerical stability: subtract max
        m = np.max(overlap)
        exp_terms = np.exp(lam * (overlap - m))
        soft_max = m + np.log(np.sum(exp_terms)) / lam
        
        # Gradient of soft max
        # d/dh_i (m + 1/lam * log(sum(exp(lam * (o_j - m)))))
        # Note m depends on h, but for large lam, the derivative is dominated by the term with max overlap.
        # Rigorous gradient:
        # Let S = sum(exp(lam * (overlap_j - m)))
        # d(soft_max)/d(overlap_k) = exp(lam*(overlap_k - m)) / S * (1 - lam * (overlap_k - m) * d(m)/d(overlap_k)) ... this is getting complex.
        # Simpler: just use numerical gradient or finite difference if SLSQP doesn't provide it efficiently.
        # But SLSQP can use finite differences. Let's provide just the objective first to keep it simple and robust.
        # If speed is an issue, we can add analytical gradient later.
        return soft_max

    # Constraints
    constraints = [
        {'type': 'eq', 'fun': lambda h: np.sum(h) - N/2.0, 'jac': lambda h: np.ones(N)}
    ]
    bounds = [(0.0, 1.0) for _ in range(N)]
    
    best_h = h0.copy()
    best_score = _score(best_h)
    
    # Try multiple initial points to avoid local minima
    initial_points = [
        np.full(N, 0.5),
        # Best 2-block approximation
        (lambda: np.concatenate([np.full(N//2, (3.0 - math.sqrt(5.0)) / 2.0), np.full(N//2, 1.0 - (3.0 - math.sqrt(5.0)) / 2.0)]))(),
        # Random
        np.random.rand(N) * 0.5 + 0.25 # Will be projected? No, SLSQP handles bounds.
    ]
    
    # Ensure initial points satisfy sum constraint approximately for better start
    for i, h_init in enumerate(initial_points):
        if time.time() > deadline:
            break
        
        # Normalize h_init to sum to N/2
        s = np.sum(h_init)
        if s > 0:
            h_init = h_init * (N/2.0 / s)
            h_init = np.clip(h_init, 0.0, 1.0)
            # Re-fix sum if clipping broke it
            diff = N/2.0 - np.sum(h_init)
            if abs(diff) > 1e-9:
                # Add diff to the element with most room
                # Simple heuristic: add to all
                h_init += diff / N
                h_init = np.clip(h_init, 0.0, 1.0)
        
        try:
            res = minimize(
                objective_and_grad, 
                h_init, 
                method='SLSQP', 
                bounds=bounds, 
                constraints=constraints,
                options={'maxiter': 200, 'ftol': 1e-9, 'disp': False}
            )
            
            if res.success or res.fun < best_score:
                h_candidate = res.x
                # Ensure strict sum constraint is met to avoid invalidity
                # SLSQP might leave a small error
                diff = N/2.0 - np.sum(h_candidate)
                if abs(diff) > 1e-10:
                    # Adjust
                    # Find indices not at bounds
                    mask = (h_candidate > 1e-6) & (h_candidate < 1.0 - 1e-6)
                    if np.any(mask):
                        h_candidate[mask] += diff / np.sum(mask)
                        h_candidate = np.clip(h_candidate, 0.0, 1.0)
                    
                score_val = _score(h_candidate)
                if score_val < best_score:
                    best_score = score_val
                    best_h = h_candidate.copy()
                    
        except Exception:
            pass

    return best_h
