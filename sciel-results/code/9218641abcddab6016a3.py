<<<<<<< SEARCH
import numpy as np
import time
from scipy.optimize import minimize

N = 600
DEADLINE = 145  # safety margin before 150s hard limit
=======
import numpy as np
import time
from scipy.optimize import minimize

N = 600
DEADLINE = 142  # safety margin before 150s hard limit
>>>>>>> REPLACE
<<<<<<< SEARCH
    # Phase 2: Gradient-based optimization with smooth max
    # Parameterize h = softmax(x) * N to keep sum ~ N (scale doesn't matter for normalized metric)
    # Actually, since metric is scale-invariant, we can just use h = exp(x) and normalize
    
    def objective(x):
        if elapsed() > DEADLINE - 10:
            return 1e10
        # Convert to positive heights
        x_shifted = x - np.max(x)
        h = np.exp(x_shifted)
        return smooth_metric(h, temp=0.01)
    
    # Start from best candidate
    x0 = np.log(best_h + 1e-15)
    x0 = x0 - np.max(x0)  # normalize
    
    result = None
    try:
        result = minimize(
            objective, x0, method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 500}
        )
        if result.success or result.fun < best_val:
            h_opt = np.exp(result.x - np.max(result.x))
            v = metric(h_opt)
            if v < best_val:
                best_val = v
                best_h = h_opt.copy()
    except Exception:
        pass
=======
    # Phase 2: Gradient-based optimization with smooth max
    # Use log-sum-exp with adaptive temperature for better convergence
    
    def objective(x, temp=0.005):
        if elapsed() > DEADLINE - 10:
            return 1e10
        # Convert to positive heights
        x_shifted = x - np.max(x)
        h = np.exp(x_shifted)
        return smooth_metric(h, temp=temp)
    
    # Start from best candidate
    x0 = np.log(best_h + 1e-15)
    x0 = x0 - np.max(x0)  # normalize
    
    # Try multiple temperatures for smoother optimization
    for temp in [0.01, 0.005, 0.002]:
        if elapsed() > DEADLINE - 15:
            break
        try:
            result = minimize(
                lambda x: objective(x, temp), x0, method='L-BFGS-B',
                options={'maxiter': 150, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 400}
            )
            if result.success or result.fun < best_val:
                h_opt = np.exp(result.x - np.max(result.x))
                v = metric(h_opt)
                if v < best_val:
                    best_val = v
                    best_h = h_opt.copy()
                    x0 = result.x - np.max(result.x)  # update starting point
        except Exception:
            pass
>>>>>>> REPLACE
<<<<<<< SEARCH
    # Phase 3: Coordinate descent with exact max (not smooth)
    current_h = best_h.copy()
    current_val = best_val
    np.random.seed(123)
    
    # Precompute for fast conv updates
    # When we change h[i], the convolution changes in a structured way
    # c[k] = sum_j h[j] h[j+k], changing h[i] affects c[i-j] for all j and c[j-i] for all j
    
    def update_conv(h, conv, i, old_val, new_val):
        """Update convolution when h[i] changes from old_val to new_val."""
        delta = new_val - old_val
        if abs(delta) < 1e-15:
            return conv
        # c[k] = sum_j h[j] h[j+k]
        # When h[i] changes by delta:
        # c[i-j] += h[j]*delta for all valid j (where i-j is valid lag)
        # c[j-i] += h[j]*delta for all valid j
        # Actually: c[k] = sum_j h[j] * h[j+k]
        # If h[i] changes, it contributes to c[k] when i = j (h[i]*h[i+k]) or i = j+k (h[j]*h[i])
        # So c[k] gets delta*h[i+k] from the first term (j=i) and delta*h[i-k] from second term (j=i-k)
        # Wait, let me be more careful.
        # c[k] = sum_{j: 0<=j<N, 0<=j+k<2N-1... } h[j]*h[j+k] for linear conv of length 2N-1
        # Actually np.convolve(h,h,'full') gives c[k] = sum_j h[j]*h[j+k] where k ranges from -(N-1) to N-1
        # Let's index c from 0 to 2N-2, where c[k] = sum_j h[j]*h[j+k-N+1]... 
        # This is getting complicated. Let me just recompute for small changes.
        return None  # Skip this optimization, just recompute
        
    # Simpler: just recompute conv for each coordinate update
    no_improve = 0
    max_no_improve = 3000
    
    while elapsed() < DEADLINE - 20 and no_improve < max_no_improve:
        # Random coordinate
        idx = np.random.randint(N)
        orig = current_h[idx]
        
        # Try several candidate values
        # Scale: current mean height
        mean_h = np.mean(current_h)
        candidates_val = [0.0, orig*0.5, orig*0.75, orig, orig*1.25, orig*1.5, orig*2.0, mean_h*0.1, mean_h*0.5, mean_h, mean_h*2.0, mean_h*5.0]
        candidates_val = np.array([max(0.0, v) for v in candidates_val])
        
        best_local_val = current_val
        best_local_v = orig
        
        for cv in candidates_val:
            if cv == orig:
                continue
            current_h[idx] = cv
            v = metric(current_h)
            if v < best_local_val:
                best_local_val = v
                best_local_v = cv
        
        if best_local_val < current_val - 1e-12:
            current_h[idx] = best_local_v
            current_val = best_local_val
            no_improve = 0
            if current_val < best_val:
                best_val = current_val
                best_h = current_h.copy()
        else:
            current_h[idx] = orig
            no_improve += 1
        
        # Periodic perturbation
        if no_improve % 200 == 0:
            pert = current_h.copy()
            pert += 0.1 * np.random.randn(N) * np.mean(current_h)
            pert = np.clip(pert, 0, None)
            if pert.sum() > 0:
                pv = metric(pert)
                if pv < current_val:
                    current_h = pert
                    current_val = pv
                    no_improve = 0
                    if current_val < best_val:
                        best_val = current_val
                        best_h = current_h.copy()
=======
    # Phase 3: Coordinate descent with exact max (not smooth)
    current_h = best_h.copy()
    current_val = best_val
    np.random.seed(123)
    
    # Use FFT-based convolution for faster computation
    def fast_metric(h):
        """Compute normalized max using FFT for speed."""
        s = np.sum(h)
        if s < 1e-15:
            return 1e10
        # Use FFT for autoconvolution
        nfft = 2 * N
        H = np.fft.rfft(h, n=nfft)
        C = np.fft.irfft(H * H, n=nfft)
        # The linear convolution is in the first 2N-1 elements
        c = C[:2*N-1]
        return np.max(c) / (s * s)
    
    # Simpler: just recompute conv for each coordinate update
    no_improve = 0
    max_no_improve = 2000
    
    while elapsed() < DEADLINE - 20 and no_improve < max_no_improve:
        # Random coordinate
        idx = np.random.randint(N)
        orig = current_h[idx]
        
        # Try several candidate values with finer granularity
        mean_h = np.mean(current_h)
        # Focus on local changes around current value
        step_sizes = [0.1, 0.25, 0.5, 1.0, 2.0] * mean_h
        candidates_val = [orig + s for s in step_sizes] + [orig - s for s in step_sizes] + [0.0, orig]
        candidates_val = np.array([max(0.0, v) for v in candidates_val])
        
        best_local_val = current_val
        best_local_v = orig
        
        for cv in candidates_val:
            if abs(cv - orig) < 1e-10:
                continue
            current_h[idx] = cv
            v = fast_metric(current_h)
            if v < best_local_val:
                best_local_val = v
                best_local_v = cv
        
        if best_local_val < current_val - 1e-12:
            current_h[idx] = best_local_v
            current_val = best_local_val
            no_improve = 0
            if current_val < best_val:
                best_val = current_val
                best_h = current_h.copy()
        else:
            current_h[idx] = orig
            no_improve += 1
        
        # Periodic perturbation with smaller steps
        if no_improve % 150 == 0:
            pert = current_h.copy()
            pert += 0.05 * np.random.randn(N) * np.mean(current_h)
            pert = np.clip(pert, 0, None)
            if pert.sum() > 0:
                pv = fast_metric(pert)
                if pv < current_val:
                    current_h = pert
                    current_val = pv
                    no_improve = 0
                    if current_val < best_val:
                        best_val = current_val
                        best_h = current_h.copy()
>>>>>>> REPLACE
<<<<<<< SEARCH
    # Phase 4: Fine-tune with small steps
    if elapsed() < DEADLINE - 5:
        step_frac = 0.01
        for _ in range(500):
            if elapsed() > DEADLINE - 3:
                break
            i, j = np.random.choice(N, 2, replace=False)
            delta = step_frac * np.random.uniform(-1, 1) * np.mean(best_h)
            new_h = best_h.copy()
            new_h[i] += delta
            new_h[j] -= delta
            if new_h[i] >= 0 and new_h[j] >= 0:
                v = metric(new_h)
                if v < best_val:
                    best_val = v
                    best_h = new_h
=======
    # Phase 4: Fine-tune with small steps using FFT
    if elapsed() < DEADLINE - 5:
        step_frac = 0.005
        for _ in range(800):
            if elapsed() > DEADLINE - 3:
                break
            i, j = np.random.choice(N, 2, replace=False)
            delta = step_frac * np.random.uniform(-1, 1) * np.mean(best_h)
            new_h = best_h.copy()
            new_h[i] += delta
            new_h[j] -= delta
            if new_h[i] >= 0 and new_h[j] >= 0:
                v = fast_metric(new_h)
                if v < best_val:
                    best_val = v
                    best_h = new_h
>>>>>>> REPLACE
<<<<<<< SEARCH
    # Ensure nonnegative
    best_h = np.maximum(best_h, 0)
    if np.sum(best_h) < 1e-15:
        best_h = np.ones(N)
    
    return best_h.tolist()
=======
    # Ensure nonnegative
    best_h = np.maximum(best_h, 0)
    if np.sum(best_h) < 1e-15:
        best_h = np.ones(N)
    
    # Final verification with exact metric
    final_val = metric(best_h)
    if final_val < best_val:
        best_val = final_val
    
    return best_h.tolist()
>>>>>>> REPLACE
