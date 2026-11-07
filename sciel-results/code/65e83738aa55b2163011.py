import numpy as np
from scipy.optimize import minimize
import time

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s limit to be safe
    
    # Initial guess
    h0 = np.full(N, 0.5)
    
    # Objective: Smooth max of correlations
    # C_k = np.correlate(h, 1-h, mode="full")
    # We want to minimize max(C_k)
    
    def objective(h):
        # Check time
        if time.time() > deadline:
            return np.inf
        
        h_clipped = np.clip(h, 0, 1)
        corr = np.correlate(h_clipped, 1 - h_clipped, mode="full")
        # Smooth max approximation
        # Use log-sum-exp for numerical stability
        max_corr = np.max(corr)
        # If max_corr is very small, log-sum-exp might underflow/overflow?
        # corr values are non-negative.
        # To avoid overflow, subtract max
        shifted = corr - max_corr
        smooth_max = max_corr + np.log(np.sum(np.exp(shifted)))
        return smooth_max

    def objective_grad(h):
        if time.time() > deadline:
            return np.zeros(N)
        
        h_clipped = np.clip(h, 0, 1)
        # Gradient of smooth max
        corr = np.correlate(h_clipped, 1 - h_clipped, mode="full")
        max_corr = np.max(corr)
        shifted = corr - max_corr
        weights = np.exp(shifted) / np.sum(np.exp(shifted))
        
        # Gradient of C_k with respect to h
        # C_k = sum_i h_i (1 - h_{i-k})
        # dC_k/dh_j = (1 - h_{j-k}) - h_{j+k}
        # We need to sum weights_k * dC_k/dh_j
        
        # This is computationally expensive to do naively.
        # Let's approximate or use finite differences if needed, but analytic is better.
        # Actually, let's just use the optimizer without gradient first to see if it works.
        # L-BFGS-B supports gradients. If not provided, it uses finite differences.
        # With N=100, finite differences might be slow but acceptable for few iterations.
        # Let's try providing a gradient.
        
        grad = np.zeros(N)
        # For each k, weight is weights[k]
        # dC_k/dh_j = (1 - h_{j-k}) - h_{j+k}
        # Note: indices in correlate "full" mode:
        # corr[k] corresponds to lag k - (N-1) ?
        # np.correlate(a, v, mode='full')
        # The output length is N + N - 1 = 199.
        # Index 99 is lag 0.
        # Index 99 + m is lag m.
        # Index 99 - m is lag -m.
        
        # Let's compute the gradient by iterating over k where weights are significant.
        # Or just compute the full gradient matrix.
        # dC_k/dh_j is sparse.
        
        # Efficient way:
        # C_k = sum_i h_i (1 - h_{i-k})
        # Let's use the fact that we have the weights.
        
        # We can compute the gradient as:
        # grad[j] = sum_k weights[k] * ( (1 - h[j-k]) - h[j+k] )
        # Note: h[x] is 0 if x out of bounds.
        
        # We can compute this using correlations of weights with h?
        # Let w_k be the weights.
        # Term 1: sum_k w_k (1 - h_{j-k}) = sum_k w_k - sum_k w_k h_{j-k}
        # sum_k w_k is constant (1).
        # sum_k w_k h_{j-k} is the correlation of w and h at lag j.
        
        # Term 2: sum_k w_k h_{j+k} is the correlation of w and h at lag -j.
        
        # So grad[j] = 1 - corr(w, h)[j] - corr(w, h)[-j]
        # Note: np.correlate(w, h, mode="full") gives lags.
        
        wh_corr = np.correlate(weights, h_clipped, mode="full")
        # wh_corr[m] corresponds to lag m - (N-1)
        # We need to sum over all lags.
        # The formula sum_k w_k h_{j-k} is the cross-correlation.
        # Let's verify indices.
        # C_k = sum_i h_i (1-h_{i-k})
        # dC_k/dh_j = 1-h_{j-k} - h_{j+k}
        # Sum_k w_k (1-h_{j-k}) = 1 - Sum_k w_k h_{j-k}
        # Sum_k w_k h_{j-k} is the correlation of w and h.
        # Let R[j] = Sum_k w_k h_{j-k}.
        # Then grad[j] = 1 - R[j] - R[-j] (if symmetric?) No, h is not symmetric.
        # R[j] = sum_k w_k h_{j-k}.
        # R[-j] = sum_k w_k h_{-j-k}.
        
        # Let's just compute R = np.correlate(weights, h_clipped, mode="full")
        # R[idx] = sum_k weights[k] h_clipped[idx - k + (N-1)] ?
        # Standard correlate: (a*v)_n = sum_m a_m v_{n-m}
        # Here a=weights, v=h.
        # R_n = sum_m weights_m h_{n-m}
        # We want sum_k weights_k h_{j-k}. This is R_j.
        # And sum_k weights_k h_{j+k} = sum_k weights_k h_{-( -j - k)}?
        # Let l = -k. sum_l weights_-l h_{j-l}.
        # This is the correlation of weights[::-1] and h?
        
        # Let's use the property:
        # grad = 1 - (weights * h) - (weights[::-1] * h) ??
        
        # Let's just do it explicitly for safety, N=100 is small.
        # But 199 lags * 100 vars = 20k ops. Very fast.
        
        # grad[j] = sum_k weights[k] * (1 - h[j-k] - h[j+k])
        
        # We can vectorize this.
        # Create a matrix M where M[j, k] = 1 - h[j-k] - h[j+k]
        # This is 100x199.
        
        # Actually, let's just use the finite difference gradient provided by scipy if analytic is tricky.
        # Given the time budget, finite difference might be okay for a few iterations.
        # But let's try to implement the analytic gradient properly.
        
        # R1[j] = sum_k weights[k] h[j-k]
        # R2[j] = sum_k weights[k] h[j+k]
        
        # R1 is correlate(weights, h, mode="full")
        # R2 is correlate(weights[::-1], h, mode="full") ?
        
        r1 = np.correlate(weights, h_clipped, mode="full")
        r2 = np.correlate(weights[::-1], h_clipped, mode="full")
        
        # Align lengths. r1 and r2 have length 2N-1.
        # We need to extract the part corresponding to j in [0, N-1].
        # For correlate(a, v, mode="full"), the lag 0 is at index N-1 (len(a)-1? No, len(a)+len(v)-1).
        # If a=weights (len 199), v=h (len 100).
        # Full length 298.
        # Lag 0 is at index 198?
        # Let's check np.correlate docs.
        # "The output is the same length as a + v - 1."
        # "The lags are from -(len(v)-1) to len(a)-1."
        
        # This is getting complicated. Let's stick to the simple loop for gradient, it's fast enough.
        grad = np.zeros(N)
        for k in range(len(weights)):
            if weights[k] < 1e-6:
                continue
            lag = k - (N - 1) # This mapping is for correlate(h, 1-h) where both are N.
            # Wait, weights are from corr of h and 1-h.
            # corr length is 199.
            # Lag for index k is k - 99.
            lag = k - 99
            
            # dC_k/dh_j = (1 - h[j-lag]) - h[j+lag]
            # We need to sum weights[k] * dC_k/dh_j for all j.
            
            # h[j-lag] is non-zero for j in [max(0, lag), min(N-1, N-1+lag)]
            # h[j+lag] is non-zero for j in [max(0, -lag), min(N-1, N-1-lag)]
            
            # Let's just compute the vector for this k.
            term = np.zeros(N)
            
            # Part 1: (1 - h[j-lag])
            if lag >= 0:
                # j-lag in [0, N-1] => j in [lag, N-1+lag]
                # j in [0, N-1]
                start = lag
                end = N
                # h indices: 0 to N-1-lag
                h_part = h_clipped[:N-lag]
                term[start:end] += 1 - h_part
            else:
                # j-lag in [0, N-1] => j in [lag, N-1+lag]
                # lag is negative. let L = -lag
                L = -lag
                start = 0
                end = N - L
                # h indices: L to N-1
                h_part = h_clipped[L:]
                term[start:end] += 1 - h_part
            
            # Part 2: - h[j+lag]
            if lag >= 0:
                # j+lag in [0, N-1] => j in [-lag, N-1-lag]
                # j in [0, N-1]
                start = 0
                end = N - lag
                # h indices: lag to N-1
                h_part = h_clipped[lag:]
                term[start:end] -= h_part
            else:
                L = -lag
                # j+lag in [0, N-1] => j in [L, N-1+L]
                start = L
                end = N
                # h indices: 0 to N-1-L
                h_part = h_clipped[:N-L]
                term[start:end] -= h_part
            
            grad += weights[k] * term
        
        return grad

    # Constraints
    constraints = [
        {'type': 'eq', 'fun': lambda h: np.sum(h) - 50}
    ]
    
    # Bounds
    bounds = [(0, 1) for _ in range(N)]
    
    # Optimization
    result = None
    best_value = np.inf
    best_h = h0
    
    # Run L-BFGS-B
    try:
        res = minimize(
            objective, 
            h0, 
            method='L-BFGS-B', 
            bounds=bounds, 
            constraints=constraints,
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        if time.time() < deadline:
            best_h = res.x
            # Evaluate true objective
            corr = np.correlate(best_h, 1 - best_h, mode="full")
            best_value = np.max(corr) / N * 2
            
    except Exception as e:
        pass

    # If time remains, do some local search
    if time.time() < deadline - 10:
        h = best_h.copy()
        curr_val = np.max(np.correlate(h, 1 - h, mode="full"))
        
        # Simple local search
        for _ in range(10000):
            if time.time() > deadline - 5:
                break
            i = np.random.randint(N)
            delta = np.random.uniform(-0.1, 0.1)
            h_new = h.copy()
            h_new[i] += delta
            h_new[i] = np.clip(h_new[i], 0, 1)
            
            # Check sum constraint
            if abs(np.sum(h_new) - 50) > 1e-9:
                # Adjust another element to fix sum? Or reject.
                # Rejecting is safer for simple search.
                continue
                
            corr_new = np.correlate(h_new, 1 - h_new, mode="full")
            val_new = np.max(corr_new)
            
            if val_new < curr_val:
                h = h_new
                curr_val = val_new
                best_h = h
                best_value = curr_val / N * 2

    # Final check
    h_final = np.clip(best_h, 0, 1)
    # Ensure sum is exactly 50
    diff = 50 - np.sum(h_final)
    if abs(diff) > 1e-9:
        # Distribute difference
        h_final += diff / N
        h_final = np.clip(h_final, 0, 1)
        # Re-adjust if clipping broke sum
        # Simple iterative fix
        for _ in range(10):
            diff = 50 - np.sum(h_final)
            if abs(diff) < 1e-9:
                break
            h_final += diff / N
            h_final = np.clip(h_final, 0, 1)
            
    return h_final
