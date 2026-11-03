"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _compute_overlap(h):
    """Compute maximum overlap of h with shifted 1-h."""
    N = len(h)
    max_overlap = 0.0
    for k in range(1, N):
        shifted_h = np.roll(h, k)
        overlap = np.sum(h * (1 - shifted_h))
        max_overlap = max(max_overlap, overlap)
    return max_overlap


def _project_to_constraints(h, target_sum=N/2):
    """Project h onto [0,1]^N with sum constraint using bisection."""
    h = np.clip(h, 0, 1)
    current_sum = np.sum(h)
    if abs(current_sum - target_sum) < 1e-10:
        return h
    
    # Use bisection to find alpha such that sum(clip(h + alpha, 0, 1)) = target_sum
    lo, hi = -1.0, 1.0
    for _ in range(50):
        mid = (lo + hi) / 2
        h_adj = np.clip(h + mid, 0, 1)
        s = np.sum(h_adj)
        if s < target_sum:
            lo = mid
        else:
            hi = mid
    h_adj = np.clip(h + (lo + hi) / 2, 0, 1)
    return h_adj


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Start with uniform solution as baseline
    best_result = np.full(N, 0.5)
    best_score = _compute_overlap(best_result)
    
    # Try gradient-based optimization with different initializations
    np.random.seed(42)
    
    num_attempts = 5
    for attempt in range(num_attempts):
        if time.time() > deadline - 30:  # Leave 30s safety margin
            break
        
        # Initialize with random perturbation around uniform
        if attempt == 0:
            h_init = np.full(N, 0.5)
        else:
            # Create structured initializations
            if attempt == 1:
                # Alternating pattern
                h_init = np.where(np.arange(N) % 2 == 0, 0.7, 0.3)
            elif attempt == 2:
                # Block pattern
                h_init = np.where(np.arange(N) % 4 < 2, 0.8, 0.2)
            elif attempt == 3:
                # Sinusoidal perturbation
                h_init = 0.5 + 0.3 * np.sin(2 * np.pi * np.arange(N) / N)
            else:
                # Random
                h_init = np.random.uniform(0, 1, N)
        
        h_init = _project_to_constraints(h_init)
        
        # Define objective function for optimization
        def objective(x):
            h = _project_to_constraints(x)
            return _compute_overlap(h)
        
        # Use Nelder-Mead for derivative-free optimization
        try:
            result = minimize(
                objective,
                h_init,
                method='Nelder-Mead',
                options={
                    'maxiter': 200,
                    'xatol': 1e-6,
                    'fatol': 1e-6,
                    'adaptive': True
                }
            )
            
            if result.fun < best_score:
                best_result = _project_to_constraints(result.x)
                best_score = result.fun
                print(f"Attempt {attempt}: improved to {best_score:.6f}")
        except Exception as e:
            print(f"Attempt {attempt} failed: {e}")
        
        if time.time() > deadline - 60:  # Leave 60s safety margin
            break
    
    # Final verification
    best_result = np.clip(best_result, 0, 1)
    # Ensure exact sum
    diff = N/2 - np.sum(best_result)
    if abs(diff) > 1e-10:
        # Distribute the difference evenly
        best_result += diff / N
        best_result = np.clip(best_result, 0, 1)
        # Re-project if needed
        best_result = _project_to_constraints(best_result)
    
    assert np.all(best_result >= 0) and np.all(best_result <= 1)
    assert abs(np.sum(best_result) - N / 2) < 1e-6
    
    return best_result
# EVOLVE-BLOCK-END
