"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 100

# EVOLVE-BLOCK-START

def _objective(h):
    # h is a numpy array of size N
    one_minus_h = 1.0 - h
    corr = np.correlate(h, one_minus_h, mode="full")
    return np.max(corr)

def _smooth_objective(h, beta=50.0):
    """Smooth approximation of max(corr) using log-sum-exp."""
    one_minus_h = 1.0 - h
    corr = np.correlate(h, one_minus_h, mode="full")
    # Use log-sum-exp for numerical stability
    max_corr = np.max(corr)
    shifted = corr - max_corr
    return max_corr + np.log(np.sum(np.exp(beta * shifted))) / beta

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    # Initial best is the seed (all 0.5)
    best_h = np.full(N, 0.5)
    best_val = _objective(best_h)
    
    # Helper to check validity and update best
    def try_update(h_candidate):
        nonlocal best_h, best_val
        if not np.all(np.isfinite(h_candidate)):
            return
        # Enforce bounds [0,1] and sum constraint approximately
        h_candidate = np.clip(h_candidate, 0.0, 1.0)
        # Adjust sum to be exactly N/2 if slightly off
        diff = N / 2.0 - np.sum(h_candidate)
        if abs(diff) > 1e-9:
            h_candidate += diff / N
            h_candidate = np.clip(h_candidate, 0.0, 1.0)
            
        val = _objective(h_candidate)
        if val < best_val:
            best_val = val
            best_h = h_candidate.copy()

    # 1. Try structured initializations
    np.random.seed(42)
    
    # List of initializations to try
    inits = [
        best_h.copy(),
        np.linspace(0.0, 1.0, N, endpoint=False),
        np.linspace(1.0, 0.0, N, endpoint=False),
        np.random.uniform(0, 1, N),
        np.where(np.arange(N) % 2 == 0, 0.8, 0.2),
        np.where(np.arange(N) % 2 == 1, 0.8, 0.2),
        # Periodic 5
        np.tile([0.0, 0.0, 1.0, 1.0, 0.5], 20),
        # Periodic 4
        np.tile([0.0, 0.5, 1.0, 0.5], 25),
    ]
    
    # Add sinusoidal inits from inspiration
    def make_sinusoidal_init(frequencies, amplitudes, phase=0.0):
        h = np.zeros(N)
        for f, a in zip(frequencies, amplitudes):
            h += a * np.cos(2 * np.pi * f * np.arange(N) / N + phase)
        h = 0.5 + h
        h = np.clip(h, 0, 1)
        h = h + (N / 2.0 - np.sum(h)) / N
        h = np.clip(h, 0, 1)
        return h

    inits.append(make_sinusoidal_init([1], [0.4]))
    inits.append(make_sinusoidal_init([2], [0.4]))
    inits.append(make_sinusoidal_init([3], [0.4]))
    inits.append(make_sinusoidal_init([1, 3], [0.3, 0.2]))
    inits.append(make_sinusoidal_init([1, 2, 4], [0.25, 0.2, 0.15]))
    inits.append(make_sinusoidal_init([2, 3, 5], [0.25, 0.2, 0.15]))
    
    # Block patterns from inspiration 2
    h_block = np.zeros(N)
    h_block[:N // 2] = 1.0
    h_block = h_block + (N / 2.0 - np.sum(h_block)) / N
    h_block = np.clip(h_block, 0, 1)
    inits.append(h_block)
    
    h_three = np.zeros(N)
    h_three[:N // 3] = 1.0
    h_three[2 * N // 3:] = 1.0
    h_three = h_three + (N / 2.0 - np.sum(h_three)) / N
    h_three = np.clip(h_three, 0, 1)
    inits.append(h_three)

    # Normalize sums for inits to start closer to feasible region
    processed_inits = []
    for h_init in inits:
        h_init = np.clip(h_init, 0, 1)
        diff = N/2.0 - np.sum(h_init)
        h_init += diff/N
        h_init = np.clip(h_init, 0, 1)
        processed_inits.append(h_init)
        
    # Add random restarts
    num_random_restarts = 10
    for i in range(num_random_restarts):
        if time.time() > deadline - 20: # Leave time for final checks
            break
        h_rand = np.random.uniform(0, 1, N)
        diff = N/2.0 - np.sum(h_rand)
        h_rand += diff/N
        h_rand = np.clip(h_rand, 0, 1)
        processed_inits.append(h_rand)

    bounds = [(0.0, 1.0)] * N
    constraints = [{'type': 'eq', 'fun': lambda h: np.sum(h) - N / 2.0}]

    # Phase 1: Progressive smooth optimization with increasing beta
    # For each init, do a multi-pass refinement: low beta → high beta
    # Each pass starts from the result of the previous pass
    for i, h_init in enumerate(processed_inits):
        if time.time() > deadline - 40:
            break
        
        current_h = h_init.copy()
        # Progressive beta schedule: start smooth, get sharper
        beta_schedule = [5.0, 10.0, 20.0, 40.0, 80.0, 150.0]
        
        for beta in beta_schedule:
            if time.time() > deadline - 30:
                break
            try:
                def smooth_obj(h, b=beta):
                    return _smooth_objective(h, b)
                
                res = minimize(
                    smooth_obj, 
                    current_h, 
                    method='SLSQP', 
                    bounds=bounds, 
                    constraints=constraints, 
                    options={'maxiter': 200, 'ftol': 1e-12}
                )
                if np.all(np.isfinite(res.x)):
                    try_update(res.x)
                    # Use the updated best as starting point for next beta
                    current_h = best_h.copy()
            except Exception:
                continue

    # Phase 2: Perturbation restarts from best solution
    if time.time() < deadline - 25:
        np.random.seed(123)
        num_perturb = 8
        for p in range(num_perturb):
            if time.time() > deadline - 20:
                break
            # Small random perturbation
            scale = 0.05 * (1 - p / num_perturb)  # decreasing perturbation
            h_pert = best_h + scale * np.random.randn(N)
            h_pert = np.clip(h_pert, 0.0, 1.0)
            diff = N / 2.0 - np.sum(h_pert)
            h_pert += diff / N
            h_pert = np.clip(h_pert, 0.0, 1.0)
            
            # Quick smooth refinement from perturbed start
            for beta in [20.0, 60.0, 120.0]:
                if time.time() > deadline - 15:
                    break
                try:
                    def smooth_obj(h, b=beta):
                        return _smooth_objective(h, b)
                    
                    res = minimize(
                        smooth_obj, 
                        h_pert, 
                        method='SLSQP', 
                        bounds=bounds, 
                        constraints=constraints, 
                        options={'maxiter': 150, 'ftol': 1e-12}
                    )
                    if np.all(np.isfinite(res.x)):
                        try_update(res.x)
                        h_pert = best_h.copy()
                except Exception:
                    break

    # Phase 3: Exact local refinement with tight tolerance
    if time.time() < deadline - 10:
        try:
            res = minimize(
                _objective, 
                best_h, 
                method='SLSQP', 
                bounds=bounds, 
                constraints=constraints, 
                options={'maxiter': 500, 'ftol': 1e-13}
            )
            if np.all(np.isfinite(res.x)):
                try_update(res.x)
        except Exception:
            pass

    # Phase 4: Final high-precision polish
    if time.time() < deadline - 3:
        try:
            res = minimize(
                _objective, 
                best_h, 
                method='SLSQP', 
                bounds=bounds, 
                constraints=constraints, 
                options={'maxiter': 1000, 'ftol': 1e-15}
            )
            if np.all(np.isfinite(res.x)):
                try_update(res.x)
        except Exception:
            pass

    # Final validation and cleanup
    best_h = np.clip(best_h, 0.0, 1.0)
    # Iteratively fix sum to handle clipping effects
    for _ in range(3):
        diff = N / 2.0 - np.sum(best_h)
        if abs(diff) < 1e-14:
            break
        best_h += diff / N
        best_h = np.clip(best_h, 0.0, 1.0)
    
    return best_h

# EVOLVE-BLOCK-END
