"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100
import numpy as np
import time
import math
import sys

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import differential_evolution, minimize

def objective(h):
    """
    Compute the evaluation value:
    value = max(np.correlate(h, 1 - h, mode="full")) / len(h) * 2
    """
    corr = np.correlate(h, 1.0 - h, mode="full")
    return np.max(corr) / N * 2.0

def construct():
    """
    Uses Differential Evolution for global exploration and L-BFGS-B for local polishing.
    Strategy:
    1. Run Differential Evolution (DE) to find a good global minimum for the max-correlation.
    2. Take the best result from DE and polish it with L-BFGS-B for high precision.
    3. Ensure constraints (sum=50, bounds [0,1]) are strictly satisfied.
    """
    start_time = time.time()
    time_limit = 145.0  # Safety margin before 150s kill
    
    # Initial best is the constant 0.5 solution
    best_h = np.full(N, 0.5)
    best_val = objective(best_h)
    
    # Define bounds for DE
    bounds = [(0.0, 1.0) for _ in range(N)]
    
    # DE Objective function wrapper
    def de_objective(h):
        # Ensure validity
        if not np.all(np.isfinite(h)):
            return 1e6
        if np.min(h) < -1e-9 or np.max(h) > 1.0 + 1e-9:
            return 1e6
        # DE might drift slightly from sum constraint due to numerical issues or bounds clipping
        # We enforce sum constraint by projection in the objective? 
        # No, DE explores the space. If we just return objective, it might find a point with sum != 50.
        # The task requires sum(h) == N/2.
        # DE doesn't handle linear equality constraints directly in standard scipy version.
        # We must project h to satisfy the sum constraint within the objective or use a penalty.
        # Better: Project h to the hyperplane sum(h)=N/2 and bounds [0,1] before evaluating?
        # Or just add a large penalty if sum is off.
        s = np.sum(h)
        penalty = 0.0
        if abs(s - N/2.0) > 1e-6:
            penalty = (s - N/2.0)**2 * 1e4
        return objective(h) + penalty

    # Run Differential Evolution
    # maxiter and popsize are tuned to fit within time budget
    # DE is slow, so we use a modest population size and iterations.
    # We'll run it for a fixed number of iterations and check time.
    
    try:
        # Using DE to find a global-ish minimum
        # tol controls convergence
        # maxiter is max number of generations
        # popsize is population size multiplier
        res = differential_evolution(
            de_objective, 
            bounds, 
            maxiter=200, 
            popsize=15, 
            tol=1e-4,
            seed=42,
            polish=False, # We will do our own polish
            init='latinhypercube'
        )
        
        if res.success or res.fun < best_val:
            h_de = res.x
            # Project h_de to satisfy constraints exactly
            # Simple projection: adjust by mean difference, then clip, then adjust again?
            # For DE, the penalty method usually keeps it close.
            # Let's do a proper projection:
            # 1. Clip to [0,1]
            h_de = np.clip(h_de, 0.0, 1.0)
            # 2. Adjust sum to N/2
            # We need to distribute the difference (N/2 - sum(h_de)) among elements
            # that have room to move.
            diff = N/2.0 - np.sum(h_de)
            if abs(diff) > 1e-9:
                # Elements that can be increased
                can_inc = h_de < 1.0 - 1e-12
                # Elements that can be decreased
                can_dec = h_de > 1e-12
                
                if diff > 0:
                    # Need to increase sum
                    # Distribute diff to elements that can increase
                    if np.any(can_inc):
                        # Simple: add diff / count to those that can
                        # But this might push some over 1.0
                        # Iterative projection is safer
                        for _ in range(10):
                            h_de[can_inc] += diff / np.sum(can_inc)
                            h_de = np.clip(h_de, 0.0, 1.0)
                            diff = N/2.0 - np.sum(h_de)
                            can_inc = h_de < 1.0 - 1e-12
                            if abs(diff) < 1e-9:
                                break
                else:
                    # Need to decrease sum
                    if np.any(can_dec):
                        for _ in range(10):
                            h_de[can_dec] += diff / np.sum(can_dec)
                            h_de = np.clip(h_de, 0.0, 1.0)
                            diff = N/2.0 - np.sum(h_de)
                            can_dec = h_de > 1e-12
                            if abs(diff) < 1e-9:
                                break
            
            val_de = objective(h_de)
            if val_de < best_val:
                best_val = val_de
                best_h = h_de.copy()
                
    except Exception:
        pass
    
    # Check time
    if time.time() - start_time > time_limit:
        # Return best found so far
        if not np.all(np.isfinite(best_h)) or np.abs(np.sum(best_h) - N/2.0) > 1e-9 * N:
            return np.full(N, 0.5)
        return best_h

    # Polishing phase with L-BFGS-B
    # L-BFGS-B requires a differentiable objective.
    # max(correlation) is not differentiable everywhere, but we can smooth it or use subgradient.
    # Alternatively, we can minimize the sum of squares of the top k correlations?
    # Or just use the raw objective and hope L-BFGS-B handles the kinks (it often does reasonably well).
    # Better: Use a smoothed max, e.g., log-sum-exp.
    # But for simplicity and speed, let's try standard L-BFGS-B on the raw objective.
    # We need to handle the sum constraint. L-BFGS-B doesn't support equality constraints directly.
    # We can eliminate one variable: h[99] = N/2 - sum(h[0:99]).
    # Then optimize over h[0:98].
    # Bounds for h[0:98] are [0,1].
    # h[99] must also be in [0,1], which implies sum(h[0:98]) in [N/2 - 1, N/2].
    # This creates a complex feasible region for the reduced problem.
    
    # Alternative: Use penalty method with L-BFGS-B on the full N variables.
    # Objective: value(h) + lambda * (sum(h) - N/2)^2 + mu * (bounds_violation)^2
    
    def l_bfgs_objective(h):
        if not np.all(np.isfinite(h)):
            return 1e6
        val = objective(h)
        sum_penalty = (np.sum(h) - N/2.0)**2 * 1e6
        bound_penalty = 0.0
        if np.min(h) < 0:
            bound_penalty += (np.min(h))**2 * 1e6
        if np.max(h) > 1:
            bound_penalty += (np.max(h) - 1.0)**2 * 1e6
        return val + sum_penalty + bound_penalty

    # Try polishing from the DE result and from the seed (0.5)
    candidates = [best_h, np.full(N, 0.5)]
    
    for h_init in candidates:
        if time.time() - start_time > time_limit - 10.0:
            break
            
        try:
            res_polish = minimize(
                l_bfgs_objective, 
                h_init, 
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if res_polish.success:
                h_polish = res_polish.x
                # Final projection to ensure strict validity
                h_polish = np.clip(h_polish, 0.0, 1.0)
                diff = N/2.0 - np.sum(h_polish)
                if abs(diff) > 1e-9:
                    # Simple adjustment
                    h_polish += diff / N
                    h_polish = np.clip(h_polish, 0.0, 1.0)
                    # Re-adjust if clipping changed sum
                    for _ in range(5):
                        diff = N/2.0 - np.sum(h_polish)
                        if abs(diff) < 1e-9:
                            break
                        h_polish += diff / N
                        h_polish = np.clip(h_polish, 0.0, 1.0)
                
                val_polish = objective(h_polish)
                if val_polish < best_val:
                    best_val = val_polish
                    best_h = h_polish.copy()
                    
        except Exception:
            pass

    # Final check
    if not np.all(np.isfinite(best_h)) or np.abs(np.sum(best_h) - N/2.0) > 1e-9 * N:
        return np.full(N, 0.5)
        
    return best_h

# EVOLVE-BLOCK-END
