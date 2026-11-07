import numpy as np
from scipy.optimize import differential_evolution

N = 100
TARGET_SUM = N / 2.0

def _objective(h):
    h = np.asarray(h).reshape(N)
    # Calculate the raw value
    corr = np.correlate(h, 1.0 - h, mode="full")
    val = np.max(corr) / N * 2.0
    
    # Penalty for sum constraint
    # We want sum(h) to be TARGET_SUM
    # The tolerance is 1e-9 * N = 1e-7
    # A quadratic penalty with a large coefficient should enforce this
    sum_err = np.sum(h) - TARGET_SUM
    penalty = 1e8 * sum_err**2
    
    return val + penalty

def construct():
    bounds = [(0.0, 1.0) for _ in range(N)]
    
    # Differential Evolution parameters
    # popsize: 15 * N is default, maybe too slow for N=100? 
    # Let's use a smaller popsize and more iterations
    # maxiter: 100
    # tol: 1e-8
    
    try:
        result = differential_evolution(
            _objective,
            bounds,
            maxiter=50,
            popsize=10,
            tol=1e-8,
            seed=100,
            polish=True # Local polish using L-BFGS-B
        )
        h_opt = result.x
    except Exception:
        # Fallback to seed if DE fails
        h_opt = np.full(N, 0.5)

    # Post-processing: Ensure strict validity
    # Clip to [0, 1]
    h_opt = np.clip(h_opt, 0.0, 1.0)
    
    # Adjust sum to exactly TARGET_SUM if possible
    # We need sum(h) = 50.0
    # If sum is not close, we need to fix it.
    # Since DE minimizes the penalty, sum should be very close.
    # Let's do a fine-tuning step: 
    # If sum(h) > TARGET, reduce largest elements.
    # If sum(h) < TARGET, increase smallest elements.
    # This is a simple repair.
    
    current_sum = np.sum(h_opt)
    diff = TARGET_SUM - current_sum
    
    if abs(diff) > 1e-9:
        if diff > 0:
            # Need to increase sum. Increase elements that are < 1.
            # Sort indices by value, pick smallest, increase them.
            # To minimize impact on objective, we should increase elements where 
            # the derivative of the objective is low? 
            # Simple heuristic: increase the smallest elements slightly.
            # But we need to be careful not to exceed 1.
            
            # Better: Distribute diff among elements that have room to grow.
            # Room = 1 - h_i
            rooms = 1.0 - h_opt
            # We can only add to elements where room > 0
            # If total room < diff, we can't satisfy constraint with just increasing.
            # But DE should have found a valid point.
            
            # Simple redistribution:
            # We want to add diff to h_opt.
            # To keep it in [0,1], we scale the "slack" or just add evenly to those < 1.
            # Let's just add evenly to all elements that are < 1 - epsilon.
            mask = h_opt < 1.0 - 1e-6
            if np.sum(mask) > 0:
                # Calculate how much we can add to each
                # We want to add diff / count(mask)
                add_amount = diff / np.sum(mask)
                # If add_amount is too large, some will exceed 1.
                # Iterate or just clip and re-normalize?
                # Let's use a simple iterative correction.
                
                # Actually, a robust way:
                # h_opt[mask] += add_amount
                # h_opt = np.clip(h_opt, 0, 1)
                # This might change the sum again.
                
                # Let's use a linear scaling on the "active" set.
                # This is getting complicated. 
                # Since the penalty was 1e8, the sum error should be < 1e-4 or so.
                # The required precision is 1e-7.
                # So a simple correction is needed.
                
                # Correction:
                # We want sum(h) = T.
                # Let h_new = h_old + delta * v, where v is a direction.
                # Simplest: h_new[i] = h_old[i] + diff/N.
                # Then clip. Then re-correct?
                
                # Let's just do:
                h_opt += diff / N
                h_opt = np.clip(h_opt, 0.0, 1.0)
                
                # If still not close, try again with remaining diff
                # This is a bit hacky but works for small diffs.
                # Let's check if this is sufficient.
                # If diff was 1e-5, diff/N is 1e-7. Clipping might remove some.
                # Let's do a proper projection.
                
                # Project h onto {x | 0<=x<=1, sum(x)=T}
                # This is a quadratic programming problem: min ||h - x||^2 s.t. constraints.
                # Solution:
                # x_i = clip(h_i + lambda, 0, 1)
                # Find lambda such that sum(clip(h_i + lambda, 0, 1)) = T.
                
                def sum_of_clipped(lam):
                    return np.sum(np.clip(h_opt + lam, 0.0, 1.0))
                
                # Binary search for lambda
                lo = -1.0
                hi = 1.0
                # sum(h) + N*lam is roughly linear.
                # If sum(h) < T, we need positive lam.
                # If sum(h) > T, we need negative lam.
                
                for _ in range(50):
                    mid = (lo + hi) / 2.0
                    s = sum_of_clipped(mid)
                    if s < TARGET_SUM:
                        lo = mid
                    else:
                        hi = mid
                
                h_opt = np.clip(h_opt + (lo + hi) / 2.0, 0.0, 1.0)
                
    # Final check
    if not np.all(np.isfinite(h_opt)) or abs(np.sum(h_opt) - TARGET_SUM) > 1e-9 * N:
        return np.full(N, 0.5)
        
    return h_opt
