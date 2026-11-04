import numpy as np
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # Initial guess: uniform
    h = np.full(N, 0.5)
    
    # Calculate overlaps
    def calc_overlaps(h):
        # Circular shifts
        overlaps = np.zeros(N)
        for k in range(N):
            shifted_h = np.roll(h, k)
            overlaps[k] = np.sum(h * (1 - shifted_h))
        return overlaps
    
    # Objective: soft max of overlaps
    def objective(h, beta=10.0):
        overlaps = calc_overlaps(h)
        # Use log-sum-exp for stability
        max_o = np.max(overlaps)
        if max_o > 0:
            soft_max = max_o + np.log(np.sum(np.exp(beta * (overlaps - max_o)))) / beta
        else:
            soft_max = 0.0
        return soft_max
    
    # Gradient of soft max
    def grad_objective(h, beta=10.0):
        overlaps = calc_overlaps(h)
        max_o = np.max(overlaps)
        exp_term = np.exp(beta * (overlaps - max_o))
        weights = exp_term / np.sum(exp_term)
        
        grad = np.zeros(N)
        for k in range(N):
            if weights[k] < 1e-6:
                continue
            shifted_h = np.roll(h, k)
            # d/dh_i [ sum_j h_j (1-h_{j+k}) ]
            # Term involving h_i:
            # 1. h_i (1-h_{i+k}) -> deriv: (1-h_{i+k})
            # 2. h_{i-k} (1-h_i) -> deriv: -h_{i-k}
            # So grad_i += (1-h_{i+k}) - h_{i-k}
            grad += weights[k] * ((1 - np.roll(h, -k)) - np.roll(h, k))
        return grad * beta

    # Projection onto simplex {h | sum(h)=N/2, 0<=h<=1}
    def project(h):
        # Use bisection on Lagrange multiplier for sum constraint
        # Then clip to [0,1] and adjust? 
        # Simpler: use scipy.optimize or manual projection.
        # For speed, simple clipping and scaling might drift.
        # Let's use a simple iterative projection:
        # 1. Clip to [0,1]
        # 2. Adjust sum to N/2 by adding/subtracting small amounts.
        
        h = np.clip(h, 0, 1)
        current_sum = np.sum(h)
        target_sum = N / 2.0
        diff = target_sum - current_sum
        
        if diff > 0:
            # Need to increase sum. Add to elements < 1.
            mask = h < 1.0
            if np.sum(mask) > 0:
                add_val = diff / np.sum(mask)
                h[mask] += add_val
                h = np.clip(h, 0, 1)
        else:
            # Need to decrease sum. Subtract from elements > 0.
            mask = h > 0.0
            if np.sum(mask) > 0:
                sub_val = -diff / np.sum(mask)
                h[mask] -= sub_val
                h = np.clip(h, 0, 1)
        
        # Final correction for exact sum (floating point errors)
        # Distribute residual to a single element or spread out
        residual = target_sum - np.sum(h)
        if abs(residual) > 1e-9:
            # Add residual to the element with most space
            if residual > 0:
                idx = np.argmin(1 - h) # Actually argmax of h? No, min of (1-h) is max h.
                # We want to add to an element that can take it.
                # Find element with smallest h? No, if h is 1, it can't take.
                # Find element with max (1-h).
                idx = np.argmax(1 - h)
                if 1 - h[idx] >= residual:
                    h[idx] += residual
                else:
                    # Spread it
                    # This case is rare if projection is good
                    pass
            else:
                idx = np.argmin(h)
                if h[idx] >= -residual:
                    h[idx] += residual # residual is negative
                else:
                    pass
        
        return h

    # Optimization loop
    beta = 5.0
    lr = 0.01
    best_h = h.copy()
    best_score = objective(h, beta)
    
    for step in range(5000):
        if time.time() > deadline:
            break
            
        # Adaptive beta: increase as we converge
        # beta = 5.0 + step * 0.01
        
        g = grad_objective(h, beta)
        h_new = h - lr * g
        h_new = project(h_new)
        
        # Line search / Check improvement
        score_new = objective(h_new, beta)
        if score_new < best_score:
            best_h = h_new.copy()
            best_score = score_new
        
        h = h_new
        
        # Periodic restart or perturbation?
        # Maybe just keep going.

    # Final polish: use a high beta for exact max minimization
    # Or just return best_h
    
    return best_h
# EVOLVE-BLOCK-END
