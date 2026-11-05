import numpy as np
import time
import sys

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    def compute_value(h):
        """Compute the objective value (lower is better)."""
        s = np.sum(h)
        if abs(s) < 1e-12:
            return np.inf
        conv = np.convolve(h, h)
        return abs(2.0 * N * np.max(conv) / (s * s))

    def energy_objective(h):
        """
        Minimize the L2 norm of the autoconvolution.
        This is a smooth proxy: if the total energy is low, the max is likely low too.
        min sum(conv^2) subject to sum(h) = 1.
        """
        conv = np.convolve(h, h)
        return np.sum(conv * conv)

    # Strategy:
    # 1. Parameterize h with low-frequency Fourier components to ensure smoothness.
    # 2. Minimize the L2 energy of the convolution (smooth quadratic).
    # 3. Refine using coordinate descent on the actual max-convolution objective.
    
    # Initial setup
    # We want to minimize max(conv). 
    # Note: sum(h) is in the denominator squared. We can fix sum(h) = N (mean=1) 
    # to simplify, or just let the optimizer handle it. 
    # Actually, the objective is scale invariant: 
    # value(h) = value(alpha * h) for any alpha != 0.
    # So we can constrain sum(h) = 1 or mean(h) = 1 to remove the degree of freedom.
    # Let's constrain mean(h) = 1.
    
    # Use a low-dimensional Fourier parameterization to keep the search space manageable
    # and ensure the signal is "smooth" (low frequency content).
    # h[k] = 1 + sum_{m=1}^{K} (a_m cos(2*pi*m*k/N) + b_m sin(2*pi*m*k/N))
    
    K = 10 # Number of harmonics
    num_params = 2 * K
    k = np.arange(N)
    
    def h_from_params(params):
        """Reconstruct h from Fourier parameters, ensuring mean is 1."""
        h = np.ones(N)
        for i in range(K):
            freq = i + 1
            a = params[2*i]
            b = params[2*i + 1]
            h += a * np.cos(2 * np.pi * freq * k / N)
            h += b * np.sin(2 * np.pi * freq * k / N)
        return h

    def energy_obj_params(params):
        h = h_from_params(params)
        return energy_objective(h)

    # Phase 1: Optimize Fourier coefficients to minimize L2 energy
    # We use L-BFGS-B for this smooth problem.
    best_params = np.zeros(num_params)
    best_h = np.ones(N)
    best_val = compute_value(best_h)
    
    # Try a few random starts for the Fourier optimization
    np.random.seed(42)
    for restart in range(5):
        params0 = np.random.uniform(-0.1, 0.1, size=num_params)
        
        try:
            from scipy.optimize import minimize
            res = minimize(
                energy_obj_params,
                params0,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10}
            )
            h_candidate = h_from_params(res.x)
            val_candidate = compute_value(h_candidate)
            
            if val_candidate < best_val:
                best_val = val_candidate
                best_h = h_candidate
                best_params = res.x
        except Exception:
            pass
        
        if time.time() > deadline - 60.0:
            break

    # Phase 2: Coordinate Descent on the actual objective (max of conv)
    # Now we have a good smooth starting point. We refine it by adjusting individual
    # coordinates to reduce the specific peak that is currently the maximum.
    
    h = best_h.copy()
    current_val = best_val
    
    # Precompute convolution
    conv = np.convolve(h, h)
    s = np.sum(h)
    
    # We will perform several passes of coordinate descent.
    # In each pass, we iterate over all coordinates.
    # For each coordinate j, we try to reduce the global max of the convolution.
    
    # Optimization: 
    # Changing h[j] by delta changes conv[k] for k in [j, j+N-1] (and symmetrically).
    # Specifically, conv_full has length 2N-1.
    # conv_full[i] = sum_{u+v=i} h[u]h[v]
    # If we change h[j] -> h[j] + delta:
    # The terms involving h[j] are h[j]*h[i-j].
    # So conv_full[i] changes by delta * (h[i-j] + h[j] + delta * (1 if i-j==j else 0? No, careful))
    # Actually, let's just recompute the affected part or use a fast update if possible.
    # Given N=400, full conv is 799 elements. Recomputing full conv is O(N^2) ~ 160k ops.
    # 400 coords * 160k = 64M ops per pass. 
    # In Python/NumPy, this might be slow if done naively in a loop.
    # However, np.convolve is optimized. Let's see if we can do better.
    
    # Alternative: Gradient-based local search?
    # The objective is non-differentiable at the max. 
    # But we can use subgradients. The subgradient of max(x) is a vector with 1 at argmax.
    # So gradient of objective w.r.t h is roughly proportional to the "active" convolution terms.
    
    # Let's stick to coordinate descent but make it efficient.
    # We only recompute the max if we find a candidate that might be better.
    
    # To speed up, we can track the current max and its index.
    # If we change h[j], we only need to update conv entries that involve h[j].
    # conv[i] involves h[j] if i-j is a valid index for h.
    # Indices i in conv (0 to 2N-2) such that 0 <= i-j <= N-1.
    # i ranges from j to j+N-1.
    
    max_iter = 50 # Number of full passes
    for it in range(max_iter):
        if time.time() > deadline - 5.0:
            break
            
        improved = False
        # Randomize order to avoid local minima in coordinate descent
        indices = np.random.permutation(N)
        
        for j in indices:
            if time.time() > deadline - 5.0:
                break
                
            old_hj = h[j]
            
            # Determine the range of conv indices affected by h[j]
            # conv_full[i] = sum_{u=0}^{N-1} h[u] h[i-u]
            # Term h[j] appears when u=j (v=i-j) or v=j (u=i-j).
            # So conv_full[i] changes for i in [j, j+N-1] (from u=j) and i in [j, j+N-1] (from v=j).
            # Wait, the full convolution index i goes from 0 to 2N-2.
            # If u=j, then v = i-j must be in [0, N-1] => i in [j, j+N-1].
            # If v=j, then u = i-j must be in [0, N-1] => i in [j, j+N-1].
            # So the affected range is exactly [j, j+N-1].
            
            start_idx = j
            end_idx = j + N - 1
            
            # Get current conv values in this range
            conv_range = conv[start_idx:end_idx+1]
            
            # We want to find a delta that reduces the global max.
            # Let M = current global max.
            # If the max is NOT in the affected range, changing h[j] won't reduce M (it might increase it elsewhere? No, it only changes this range).
            # If M is in the affected range, we can try to reduce it.
            
            # Strategy: Try small steps.
            # Let's try to reduce the max value in the affected range if it contributes to the global max.
            
            # Quick check: Is the global max in this range?
            # We can maintain a histogram or just check np.max(conv_range) vs global max.
            # To be safe and simple, let's just try a few deltas and check the new global max.
            
            # To make this fast, we can approximate.
            # If we decrease h[j], terms involving h[j] decrease (assuming h is positive).
            # If h has negative values, it's more complex.
            
            # Let's use a "line search" on a few points.
            # Since N=400, we can afford to recompute the full max for a few candidates.
            
            candidates = [old_hj - 0.01, old_hj + 0.01, old_hj - 0.005, old_hj + 0.005]
            
            best_delta = 0
            best_local_val = current_val
            
            for delta in candidates:
                new_hj = old_hj + delta
                if new_hj == old_hj: continue
                
                # Update conv in the affected range
                # conv_new[i] = conv[i] + delta * (h[i-j] + h[j]) + delta^2 (if i-j == j)
                # Wait, the cross terms:
                # Original: ... + h[j]h[i-j] + h[i-j]h[j] + ... (if i-j != j)
                # New: ... + (h[j]+delta)h[i-j] + h[i-j](h[j]+delta) + ...
                # Change: delta * h[i-j] + h[i-j] * delta = 2 * delta * h[i-j]
                # If i-j == j (i.e., i=2j):
                # Original: h[j]^2
                # New: (h[j]+delta)^2 = h[j]^2 + 2h[j]delta + delta^2
                # Change: 2h[j]delta + delta^2
                # The formula 2*delta*h[i-j] covers the 2h[j]delta part.
                # So we need to add delta^2 if i == 2j.
                
                # Let's just do it vectorized for the range
                i_vals = np.arange(start_idx, end_idx + 1)
                j_vals = i_vals - j # This is v
                # h[j_vals] is the other factor
                # Note: j_vals are indices into h.
                
                # Update
                conv_new_range = conv_range.copy()
                conv_new_range += 2.0 * delta * h[j_vals]
                
                # Handle the diagonal case where u=v=j
                diag_idx_in_range = 2 * j - start_idx
                if 0 <= diag_idx_in_range < len(conv_new_range):
                    conv_new_range[diag_idx_in_range] += delta * delta
                    
                # Now we have the new conv in the affected range.
                # We need the new GLOBAL max.
                # The max of the unaffected part is max(conv) if the global max was not in the range.
                # We can precompute the max of the unaffected part? 
                # Or just take max of (new_range) and (old max if it wasn't in range).
                
                # Efficient way:
                # max_new = max( np.max(conv_new_range), np.max(np.delete(conv, np.arange(start_idx, end_idx+1))) )
                # np.delete is slow.
                
                # Alternative: Keep track of the top 2 maxima? 
                # Or just recompute np.max(conv) after updating the conv array in place?
                # Updating in place:
                # conv[start_idx:end_idx+1] = conv_new_range
                # Then np.max(conv)
                
                # This is O(N) for update + O(N) for max. Total O(N) per delta.
                # 4 deltas * 400 coords * 400 size = 640k ops per pass. Very fast.
                
                # Let's do it in place but restore later.
                old_range = conv_range.copy()
                conv[start_idx:end_idx+1] = conv_new_range
                
                new_global_max = np.max(conv)
                new_s = s + delta
                if abs(new_s) < 1e-6:
                    conv[start_idx:end_idx+1] = old_range
                    continue
                
                new_val = abs(2.0 * N * new_global_max / (new_s * new_s))
                
                if new_val < best_local_val:
                    best_local_val = new_val
                    best_delta = delta
                    best_new_range = conv_new_range
                    best_new_s = new_s
                
                # Restore conv
                conv[start_idx:end_idx+1] = old_range
            
            if best_delta != 0:
                h[j] = old_hj + best_delta
                conv[start_idx:end_idx+1] = best_new_range
                s = best_new_s
                current_val = best_local_val
                improved = True
                if current_val < best_val:
                    best_val = current_val
                    best_h = h.copy()
        
        if not improved:
            break

    # Final check
    final_val = compute_value(h)
    if final_val < best_val:
        best_val = final_val
        best_h = h.copy()

    # Ensure validity
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-6:
        return np.ones(N)
    
    return best_h

if __name__ == "__main__":
    h = construct()
    val = compute_value(h)
    print(f"Result value: {val}")
