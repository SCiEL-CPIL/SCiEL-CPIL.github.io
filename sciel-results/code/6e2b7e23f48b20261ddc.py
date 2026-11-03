import numpy as np
import time
import random
from scipy.optimize import differential_evolution

N = 600
TIME_LIMIT = 135  # Safety margin

def compute_score(f):
    """Compute the normalized maximum of the autoconvolution."""
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    conv = np.convolve(f, f, mode='full')
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    return np.max(conv) / (s * s)

def parametric_to_step(params):
    """
    Convert parametric representation to a 600-step vector.
    Params: [center1, width1, amp1, center2, width2, amp2, ...]
    We use a sum of Gaussian bumps.
    """
    x = np.linspace(-0.25, 0.25, N, endpoint=False)
    f = np.zeros(N)
    n_bumps = len(params) // 3
    for i in range(n_bumps):
        c = params[3*i]
        w = params[3*i+1]
        a = params[3*i+2]
        # Ensure width is positive and not too small
        w = max(abs(w), 0.01)
        f += a * np.exp(-0.5 * ((x - c) / w) ** 2)
    f = np.maximum(f, 0.0)
    # Normalize to sum 1 to remove scale invariance issues during optimization
    s = np.sum(f)
    if s > 1e-15:
        f = f / s
    return f

def objective_parametric(params):
    """
    Objective function for parametric optimization.
    Returns the score (lower is better).
    """
    try:
        f = parametric_to_step(params)
        return compute_score(f)
    except Exception:
        return 1e10

def local_refinement(f0, max_iter=2000, time_limit=10.0):
    """
    Direct coordinate descent on the 600-step vector to minimize the max autoconvolution.
    """
    f = f0.copy()
    f = np.maximum(f, 0.0)
    s = np.sum(f)
    if s < 1e-15:
        return f
    
    # Precompute FFT size
    size = 1024 # Power of 2 >= 2*N-1
    
    start_time = time.time()
    best_score = compute_score(f)
    
    # Initialize gradient-like information or just do coordinate search
    # Coordinate descent: for each bin j, find optimal f_j given others fixed.
    # The objective is max_k (conv_k) / s^2.
    # This is non-convex in f_j due to s^2 in denominator.
    # However, for a fixed set of other bins, conv_k is linear in f_j (for k involving j) 
    # but s is also linear.
    # Let f_j = x. s = S0 + x.
    # conv_k = C_k + 2 * f_k' * x (if k corresponds to lag involving j).
    # Actually, conv is quadratic in f.
    # Let's use a simpler heuristic: 
    # 1. Compute current conv and max.
    # 2. For each bin j, try increasing/decreasing it slightly.
    # This is slow.
    
    # Better: Use a subgradient method or projected gradient on the raw vector.
    # Since we have a good starting point from parametric, we can use FISTA-like 
    # but with a simpler objective: min max(conv) subject to sum(f)=1, f>=0.
    # Wait, the objective is max(conv)/s^2. If we fix sum(f)=1, it's just min max(conv).
    # The parametric step already normalizes to sum=1.
    # So we can solve: min max(conv) s.t. f>=0, sum(f)=1.
    
    # Let's use a simple iterative water-filling / projection approach.
    # Or just a few steps of FISTA with beta=inf approximation (max).
    
    # Let's try a direct "peak reduction" local search.
    # Identify the lag k* where conv is max.
    # Reduce mass in bins that contribute to conv_k*.
    # Increase mass in bins that contribute to lower conv values?
    
    # A robust method: 
    # 1. Compute conv, find max lag k*.
    # 2. Compute sensitivity of conv_k* to each f_j.
    #    d conv_k* / d f_j = f_{k*-j} + f_{j-k*} (with appropriate indexing).
    # 3. Move mass from high sensitivity bins to low sensitivity bins.
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
            
        conv = np.convolve(f, f, mode='full')
        k_max = np.argmax(conv)
        max_val = conv[k_max]
        
        # Compute sensitivity of conv[k_max] to each f[j]
        # conv[k] = sum_i f[i] f[k-i]
        # d conv[k_max] / d f[j] = f[k_max-j] + f[j-k_max] (if indices valid)
        sens = np.zeros(N)
        for j in range(N):
            idx1 = k_max - j
            idx2 = j - k_max
            val = 0.0
            if 0 <= idx1 < N:
                val += f[idx1]
            if 0 <= idx2 < N:
                val += f[idx2]
            sens[j] = val
            
        # We want to decrease f[j] where sens[j] is high, and increase where low.
        # But we must keep f>=0 and sum(f)=1.
        # Let's pick the top 10% most sensitive bins and bottom 10% least sensitive.
        # Move a small amount of mass from high to low.
        
        sorted_indices = np.argsort(sens)
        n_move = max(1, N // 10)
        high_sens_indices = sorted_indices[-n_move:]
        low_sens_indices = sorted_indices[:n_move]
        
        # Calculate average value in high and low sets
        avg_high = np.mean(f[high_sens_indices])
        avg_low = np.mean(f[low_sens_indices])
        
        # If high sens bins have more mass than low sens bins, move some.
        if avg_high > avg_low:
            # Target: equalize or slightly reduce high sens
            # Step size
            step = 0.05 * (avg_high - avg_low)
            
            # Reduce high sens bins
            f[high_sens_indices] -= step
            # Increase low sens bins
            f[low_sens_indices] += step
            
            # Enforce non-negativity
            f = np.maximum(f, 0.0)
            # Renormalize
            s = np.sum(f)
            if s > 1e-15:
                f = f / s
            else:
                break
                
            # Check if score improved
            new_score = compute_score(f)
            if new_score < best_score:
                best_score = new_score
            else:
                # Revert if not improved? Or just continue?
                # Let's revert to be safe
                f = np.maximum(f0 if it==0 else f, 0.0) # This revert logic is flawed if we don't store previous state
                # Let's store previous state
                break # If this simple move doesn't help, maybe we're stuck. 
                # Actually, let's just not revert and let the next iteration try different indices.
                # But if it oscillates, we waste time.
                # Let's implement a proper line search or just accept if it doesn't get much worse.
                # For simplicity in this time-constrained task, let's just do a few random restarts of this local search.
    
    return f

def construct():
    start_time = time.time()
    
    # 1. Global Search: Parametric Optimization
    # Use Differential Evolution on a small number of Gaussian bumps.
    # Let's try 3 bumps. 9 parameters.
    n_bumps = 3
    n_params = n_bumps * 3
    
    # Bounds for parameters:
    # Center: [-0.25, 0.25]
    # Width: [0.01, 0.25]
    # Amp: [0.1, 1.0]
    bounds = []
    for i in range(n_bumps):
        bounds.append((-0.25, 0.25)) # Center
        bounds.append((0.01, 0.25))   # Width
        bounds.append((0.1, 1.0))     # Amp
        
    # Run DE
    best_f = None
    best_score = np.inf
    
    # Time budget for DE: 30 seconds
    de_time_limit = 30.0
    
    try:
        # differential_evolution can be slow. Use maxiter and popsize to control.
        # We will run it in a way that stops early if time runs out.
        # Since DE doesn't have a direct time limit, we'll run a fixed number of iterations.
        # N=9 params, popsize=10, maxiter=50 is ~500 evaluations.
        # Each eval is fast (FFT conv).
        
        res = differential_evolution(
            objective_parametric, 
            bounds, 
            seed=42, 
            maxiter=100, 
            popsize=15,
            tol=1e-5,
            init='sobol'
        )
        
        if res.success or res.fun < best_score:
            best_score = res.fun
            best_f = parametric_to_step(res.x)
            
    except Exception as e:
        print(f"DE failed: {e}")
        
    # If DE didn't find anything good, use a default
    if best_f is None:
        # Default: single Gaussian in center
        params_default = [0.0, 0.1, 1.0] * n_bumps
        best_f = parametric_to_step(params_default)
        best_score = compute_score(best_f)

    # 2. Local Refinement: Direct Optimization on 600 bins
    # Time budget for refinement: remaining time - safety
    remaining_time = TIME_LIMIT - (time.time() - start_time) - 5.0
    
    if remaining_time > 5.0:
        # Try a few different local refinement strategies or restarts
        # Strategy: Run the "peak reduction" local search.
        # My previous local_refinement was a bit sketchy. Let's use a simpler, more robust one:
        # Projected Gradient Descent on min max(conv) s.t. sum(f)=1, f>=0.
        
        def proj_grad_step(f, step_size):
            conv = np.convolve(f, f, mode='full')
            k_max = np.argmax(conv)
            # Gradient of conv[k_max] w.r.t f
            grad = np.zeros(N)
            for j in range(N):
                idx1 = k_max - j
                idx2 = j - k_max
                if 0 <= idx1 < N:
                    grad[j] += f[idx1]
                if 0 <= idx2 < N:
                    grad[j] += f[idx2]
            
            # Update
            f_new = f - step_size * grad
            # Project onto sum(f)=1, f>=0
            # This is a quadratic programming problem.
            # Simple approach: iterative clipping and normalization.
            # 1. Clip to 0
            f_new = np.maximum(f_new, 0.0)
            # 2. Normalize to sum 1
            s = np.sum(f_new)
            if s > 1e-15:
                f_new = f_new / s
            else:
                f_new = f
            return f_new, k_max

        current_f = best_f.copy()
        current_score = best_score
        step_size = 0.01
        
        for i in range(1000):
            if time.time() - start_time > TIME_LIMIT - 5.0:
                break
                
            f_new, _ = proj_grad_step(current_f, step_size)
            new_score = compute_score(f_new)
            
            if new_score < current_score:
                current_f = f_new
                current_score = new_score
                # Decrease step size slowly
                step_size *= 0.99
            else:
                # Increase step size? Or try smaller?
                step_size *= 0.9
                if step_size < 1e-6:
                    break
                    
        if current_score < best_score:
            best_f = current_f
            best_score = current_score
            
    # Final check and return
    best_f = np.maximum(best_f, 0.0)
    s = np.sum(best_f)
    if s > 1e-15:
        best_f = best_f / s
        
    return [float(x) for x in best_f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
