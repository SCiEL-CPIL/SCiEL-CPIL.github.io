"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 100

# EVOLVE-BLOCK-START

def _objective(h):
    # h is a numpy array of size N
    one_minus_h = 1.0 - h
    # Correlation of h and 1-h
    # np.correlate(h, 1-h, mode="full")
    # We want the max of this correlation
    corr = np.correlate(h, one_minus_h, mode="full")
    return np.max(corr)

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Initial best is the seed (all 0.5)
    # Value for seed: max_corr = 25.0
    best_h = np.full(N, 0.5)
    best_val = _objective(best_h)
    
    # Helper to check validity and update best
    def try_update(h_candidate):
        nonlocal best_h, best_val
        if not np.all(np.isfinite(h_candidate)):
            return
        # Enforce bounds [0,1] and sum constraint approximately
        # The optimizer should handle sum, but let's be safe
        h_candidate = np.clip(h_candidate, 0.0, 1.0)
        # Adjust sum to be exactly N/2 if slightly off
        # This is a projection step
        diff = N / 2.0 - np.sum(h_candidate)
        if abs(diff) > 1e-9:
            h_candidate += diff / N
            h_candidate = np.clip(h_candidate, 0.0, 1.0)
            
        val = _objective(h_candidate)
        if val < best_val:
            best_val = val
            best_h = h_candidate.copy()

    # 1. Try structured initializations
    # Sawtooth
    h_saw = np.linspace(0.0, 1.0, N, endpoint=False)
    # Adjust sum: sum is approx N/2 - 0.5. 
    # Let's just run optimizer on it.
    
    # 2. SLSQP Optimization
    bounds = [(0.0, 1.0)] * N
    constraints = [{'type': 'eq', 'fun': lambda h: np.sum(h) - N / 2.0}]
    
    # We will try several restarts
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
    
    # Normalize sums for inits to start closer to feasible region
    processed_inits = []
    for h_init in inits:
        h_init = np.clip(h_init, 0, 1)
        diff = N/2.0 - np.sum(h_init)
        h_init += diff/N
        h_init = np.clip(h_init, 0, 1)
        processed_inits.append(h_init)
        
    # Add random restarts
    num_random_restarts = 20
    for i in range(num_random_restarts):
        if time.time() > deadline - 20: # Leave time for final checks
            break
        h_rand = np.random.uniform(0, 1, N)
        diff = N/2.0 - np.sum(h_rand)
        h_rand += diff/N
        h_rand = np.clip(h_rand, 0, 1)
        processed_inits.append(h_rand)

    for i, h_init in enumerate(processed_inits):
        if time.time() > deadline - 10:
            break
            
        # Objective for SLSQP
        def obj_func(h):
            return _objective(h)
            
        try:
            res = minimize(
                obj_func, 
                h_init, 
                method='SLSQP', 
                bounds=bounds, 
                constraints=constraints, 
                options={'maxiter': 200, 'ftol': 1e-9}
            )
            if res.success:
                try_update(res.x)
            else:
                # Sometimes SLSQP fails to converge strictly but finds a good point
                # Check the result anyway
                try_update(res.x)
        except Exception:
            continue

    # Final polish with higher precision if time remains
    if time.time() < deadline - 5:
        try:
            def obj_func_precise(h):
                return _objective(h)
                
            res = minimize(
                obj_func_precise, 
                best_h, 
                method='SLSQP', 
                bounds=bounds, 
                constraints=constraints, 
                options={'maxiter': 500, 'ftol': 1e-12}
            )
            if res.success:
                try_update(res.x)
            else:
                try_update(res.x)
        except Exception:
            pass

    # Final validation and cleanup
    best_h = np.clip(best_h, 0.0, 1.0)
    # Ensure sum is exactly N/2 within tolerance
    diff = N / 2.0 - np.sum(best_h)
    if abs(diff) > 1e-12:
        best_h += diff / N
        best_h = np.clip(best_h, 0.0, 1.0)
        
    return best_h

# EVOLVE-BLOCK-END
