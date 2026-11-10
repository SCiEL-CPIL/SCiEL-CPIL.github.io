import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
import sys

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    start_time = time.time()
    TIME_LIMIT = 630.0 # Evaluator limit
    SAFETY_MARGIN = 10.0 # Seconds to spare for cleanup
    N = 16

    def get_cost_fast(params):
        """
        Calculate d_max_sq / d_min_sq. Lower is better.
        This is equivalent to minimizing (d_max / d_min)^2.
        Avoids square roots for speed.
        """
        pts = params.reshape(N, 2)
        # Vectorized squared distance calculation
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=-1)
        
        # Exclude diagonal (self-distance)
        # Using a large number instead of inf to avoid potential issues with max/min if all are inf? 
        # No, there are 120 pairs.
        np.fill_diagonal(dist_sq, np.inf)
        
        d_max_sq = np.max(dist_sq)
        d_min_sq = np.min(dist_sq)
        
        if d_min_sq <= 1e-12:
            return 1e12
        if d_max_sq <= 1e-12:
            return 1e12
            
        return d_max_sq / d_min_sq

    # Bounds for differential evolution
    # Scale invariant, so [-1, 1] is a good box.
    bounds = [(-1.0, 1.0)] * (N * 2)
    
    best_result = None
    best_cost = np.inf
    
    try:
        # Differential Evolution
        # popsize: 20-50 is typical. Higher means better exploration but slower per iter.
        # maxiter: We want to use most of the time.
        # Let's estimate: 1000 evaluations takes ~1-2 seconds?
        # If popsize=30, 1 iter = 30 evals.
        # If we run 1000 iters, that's 30,000 evals. ~30-60s.
        # We have 600s. We can run much longer.
        # Let's try popsize=40, maxiter=500. 20,000 evals.
        
        de_res = differential_evolution(
            get_cost_fast, 
            bounds, 
            maxiter=1000, 
            popsize=40, 
            tol=1e-10,
            mutation=(0.5, 1.5),
            recombination=0.7,
            seed=42,
            polish=False, # We will polish manually
            init='latinhypercube' # Better initial spread
        )
        
        if de_res.fun < best_cost:
            best_cost = de_res.fun
            best_result = de_res.x.copy()
            
    except Exception as e:
        print(f"DE failed: {e}", file=sys.stderr)

    # Polish the best result from DE
    if best_result is not None:
        time_elapsed = time.time() - start_time
        polish_budget = TIME_LIMIT - SAFETY_MARGIN - time_elapsed
        
        # Try Nelder-Mead first, it's robust for non-smooth
        if polish_budget > 5:
            try:
                nm_res = minimize(
                    get_cost_fast, 
                    best_result, 
                    method='Nelder-Mead',
                    options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
                )
                
                if nm_res.fun < best_cost:
                    best_cost = nm_res.fun
                    best_result = nm_res.x.copy()
            except Exception as e:
                print(f"NM failed: {e}", file=sys.stderr)
        
        # Try Powell, often better for high dimensions
        time_elapsed = time.time() - start_time
        polish_budget = TIME_LIMIT - SAFETY_MARGIN - time_elapsed
        if polish_budget > 5 and best_result is not None:
            try:
                pw_res = minimize(
                    get_cost_fast, 
                    best_result, 
                    method='Powell',
                    options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12}
                )
                
                if pw_res.fun < best_cost:
                    best_cost = pw_res.fun
                    best_result = pw_res.x.copy()
            except Exception as e:
                print(f"Powell failed: {e}", file=sys.stderr)

        # If we still have significant time, do a few local restarts
        # This helps escape shallow local minima that DE might have settled into
        time_elapsed = time.time() - start_time
        restart_budget = TIME_LIMIT - SAFETY_MARGIN - time_elapsed
        if restart_budget > 30 and best_result is not None:
            rng = np.random.default_rng(42)
            # Perform a few local optimizations from perturbed best points
            for i in range(10):
                if time.time() - start_time > TIME_LIMIT - SAFETY_MARGIN:
                    break
                
                # Perturb the best result
                # Scale of perturbation: relative to the spread of points
                spread = np.std(best_result)
                if spread < 1e-6:
                    spread = 0.1
                noise = rng.normal(0, spread * 0.1, best_result.shape)
                perturbed = best_result + noise
                
                try:
                    # Use Powell for local search
                    local_res = minimize(
                        get_cost_fast, 
                        perturbed, 
                        method='Powell',
                        options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10}
                    )
                    
                    if local_res.fun < best_cost:
                        best_cost = local_res.fun
                        best_result = local_res.x.copy()
                except Exception as e:
                    pass

    # Fallback if everything failed
    if best_result is None or not np.all(np.isfinite(best_result)):
        # Regular 16-gon
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_result = np.column_stack([np.cos(angles), np.sin(angles)])

    return best_result.reshape(16, 2)
