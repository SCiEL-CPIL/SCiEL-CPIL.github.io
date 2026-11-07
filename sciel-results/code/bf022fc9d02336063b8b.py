import numpy as np
from scipy.optimize import minimize
import time

N = 100

def compute_value(h):
    # h is a numpy array of length N
    # Check constraints
    if np.any(h < -1e-9) or np.any(h > 1 + 1e-9):
        return np.inf
    if abs(np.sum(h) - N / 2.0) > 1e-9 * N:
        return np.inf
    if not np.all(np.isfinite(h)):
        return np.inf
        
    k = 1 - h
    corr = np.correlate(h, k, mode="full")
    max_corr = np.max(corr)
    value = max_corr / N * 2
    return value

def objective(h):
    # We want to minimize max_corr
    # Use a smooth approximation or just max
    k = 1 - h
    corr = np.correlate(h, k, mode="full")
    return np.max(corr)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    best_h = np.full(N, 0.5)
    best_val = compute_value(best_h)
    
    # Generate diverse initial points
    inits = []
    inits.append(np.full(N, 0.5))
    
    # Random continuous
    np.random.seed(42)
    for _ in range(5):
        h = np.random.uniform(0, 1, N)
        # Project to sum constraint
        h = h - np.mean(h) + N / 2.0
        h = np.clip(h, 0, 1)
        # Re-project to sum (simple iterative)
        for _ in range(100):
            diff = N / 2.0 - np.sum(h)
            h += diff / N
            h = np.clip(h, 0, 1)
        inits.append(h)
        
    # Random binary
    for _ in range(5):
        h = np.zeros(N)
        indices = np.random.choice(N, N // 2, replace=False)
        h[indices] = 1.0
        inits.append(h)

    # Local optimization
    for h0 in inits:
        if time.time() > deadline:
            break
            
        # Use Nelder-Mead for non-smooth max objective
        # But Nelder-Mead in 100D is slow.
        # Let's try a simple gradient-free local search or L-BFGS-B with smooth approx.
        
        # Smooth approx: max(x) ~ log-sum-exp(x/T) * T
        # Or just minimize the max directly with a few iterations of subgradient?
        
        # Let's try scipy minimize with Nelder-Mead but with limited iterations
        # Or better, use a simple iterative improvement:
        # 1. Compute grad of objective w.r.t h (approx)
        # 2. Step in direction of negative grad
        # 3. Project to constraints
        
        # Given the time limit, a simpler "random restart + local search" might be safer.
        
        # Let's try a specific structured pattern that might work:
        # Alternating 0 and 1 was bad.
        # What about blocks?
        
        pass

    # If we don't find anything better, return best_h
    # But we need to actually run an optimizer.
    
    # Let's use a quick gradient descent with projection.
    # Objective: J(h) = max_m R[m]
    # Subgradient of max is the gradient of the active term.
    
    for h0 in inits:
        if time.time() > deadline:
            break
            
        h = h0.copy()
        for step in range(200):
            if time.time() > deadline:
                break
                
            k = 1 - h
            corr = np.correlate(h, k, mode="full")
            max_val = np.max(corr)
            max_idx = np.argmax(corr)
            
            # Compute gradient of R[max_idx] w.r.t h
            # R[m] = sum_{i} h_i k_{i+m} = sum h_i (1 - h_{i+m})
            # dR/dh_j = 1 (if j in first sum) - sum_{i} d(h_i h_{i+m})/dh_j
            # This is complex to vectorize perfectly for all lags, but we only need the active one.
            
            # Instead of analytical grad, use numerical? Too slow.
            # Let's just use scipy.optimize.minimize with Nelder-Mead on a reduced set of variables?
            # No, let's use a simple heuristic:
            # If R[m] is high, try to reduce h_i or increase h_{i+m} for i in the sum.
            
            # This is getting complicated. Let's rely on the fact that
            # "random perturbations with proper feasibility projection" worked.
            
            # Let's just run a few random restarts with a simple local search:
            # 1. Pick random indices i, j.
            # 2. Swap h_i and h_j (preserves sum).
            # 3. If value decreases, keep.
            
            # This is a swap-based local search.
            # It preserves sum and bounds (if both in [0,1]).
            
            current_val = compute_value(h)
            for _ in range(1000): # 1000 swaps per start
                if time.time() > deadline:
                    break
                i, j = np.random.choice(N, 2, replace=False)
                h_new = h.copy()
                h_new[i], h_new[j] = h[j], h[i]
                # Check bounds (swapping keeps values in [0,1] if they were)
                new_val = compute_value(h_new)
                if new_val < current_val:
                    h = h_new
                    current_val = new_val
                    
        val = compute_value(h)
        if val < best_val:
            best_val = val
            best_h = h

    return best_h
