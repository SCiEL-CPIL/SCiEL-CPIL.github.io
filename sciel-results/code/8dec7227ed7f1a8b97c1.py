"""Program for Erdos minimum overlap task using model-and-optimize approach."""
import numpy as np
import time
from scipy.optimize import minimize

N = 100
DEADLINE = 145.0
START = time.time()

def remaining():
    return DEADLINE - (time.time() - START)

def objective(h):
    """Compute the value to minimize."""
    c = np.correlate(h, 1.0 - h, mode="full")
    return float(np.max(c)) / N * 2.0

def objective_smooth(h, beta=100.0):
    """Smooth surrogate using log-sum-exp for the max."""
    c = np.correlate(h, 1.0 - h, mode="full")
    c_max = np.max(c)
    shifted = beta * (c - c_max)
    log_sum = np.log(np.sum(np.exp(shifted))) + beta * c_max
    return float(log_sum / N * 2.0)

def project(h):
    """Project to [0,1]^N with sum = N/2."""
    h = np.asarray(h, dtype=np.float64).copy()
    target = N / 2.0
    
    # Bisection to find the right offset
    lo, hi = -1.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        h_trial = np.clip(h + mid, 0.0, 1.0)
        s = h_trial.sum()
        if s < target:
            lo = mid
        else:
            hi = mid
    h = np.clip(h + (lo + hi) / 2.0, 0.0, 1.0)
    
    # Final fine-tuning
    for _ in range(20):
        s = h.sum()
        if abs(s - target) < 1e-14:
            break
        diff = target - s
        free = (h > 1e-15) & (h < 1.0 - 1e-15)
        if np.any(free):
            h[free] += diff / np.sum(free)
            h = np.clip(h, 0.0, 1.0)
        else:
            break
    return h

def local_search_binary(h_init, max_iters=10000):
    """Perform local search on a binary sequence."""
    h = h_init.copy()
    best_val = objective(h)
    
    ones = np.where(h == 1.0)[0]
    zeros = np.where(h == 0.0)[0]
    
    for _ in range(max_iters):
        if remaining() < 5.0:
            break
            
        c = np.correlate(h, 1.0 - h, mode="full")
        max_val = np.max(c)
        
        # Find all 1s and 0s
        ones = np.where(h == 1.0)[0]
        zeros = np.where(h == 0.0)[0]
        
        best_swap = None
        best_new_max = max_val
        
        # Number of swaps is 50*50 = 2500. This is fast enough to check all.
        for i in ones:
            for j in zeros:
                # Simulate swap
                h[i], h[j] = 0.0, 1.0
                c_new = np.correlate(h, 1.0 - h, mode="full")
                new_max = np.max(c_new)
                
                if new_max < best_new_max:
                    best_new_max = new_max
                    best_swap = (i, j)
                
                # Revert
                h[i], h[j] = 1.0, 0.0
        
        if best_swap is not None:
            i, j = best_swap
            h[i], h[j] = 0.0, 1.0
            best_val = best_new_max
        else:
            # No improvement, try a random swap to escape local minimum
            if len(ones) > 0 and len(zeros) > 0:
                i = np.random.choice(ones)
                j = np.random.choice(zeros)
                h[i], h[j] = 0.0, 1.0
                # Accept the swap to explore new regions
    return h, best_val

def construct():
    t0 = time.time()
    i = np.arange(N)
    
    best_h = np.full(N, 0.5)
    best_val = objective(best_h)
    
    # Strategy 1: Model-based approach - parameterize as 0.5 + sum of sines/cosines
    # and optimize the Fourier coefficients directly using smooth objective
    
    # Try different numbers of Fourier modes
    for n_modes in range(1, 10):
        if remaining() < 5:
            break
        
        # Parameters: amplitude and phase for each mode
        np.random.seed(42 + n_modes)
        params_init = np.random.randn(2 * n_modes) * 0.1
        
        def params_to_h(params):
            h = np.full(N, 0.5)
            for k in range(n_modes):
                a = params[2*k]
                b = params[2*k+1]
                h += a * np.cos(2 * np.pi * k * i / N)
                h += b * np.sin(2 * np.pi * k * i / N)
            return project(h)
        
        def obj_params(params):
            h = params_to_h(params)
            return objective_smooth(h, beta=50.0)
        
        try:
            result = minimize(
                obj_params,
                params_init,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            
            h_candidate = params_to_h(result.x)
            val_candidate = objective(h_candidate)
            
            if val_candidate < best_val:
                best_h = h_candidate
                best_val = val_candidate
                
        except Exception:
            pass
    
    # Strategy 2: Multi-start with various patterns
    for k in range(1, 20):
        if remaining() < 3:
            break
        
        for A_init in [0.1, 0.2, 0.3, 0.4, 0.45, 0.5]:
            if remaining() < 2:
                break
            
            # Try cosine and sine
            for phase in [0.0, np.pi/2]:
                if remaining() < 1:
                    break
                
                h_init = 0.5 + A_init * np.cos(2 * np.pi * k * i / N + phase)
                h_init = project(h_init)
                
                def obj_func(x):
                    return objective_smooth(project(x), beta=30.0)
                
                try:
                    result = minimize(
                        obj_func,
                        h_init,
                        method='L-BFGS-B',
                        bounds=[(0.0, 1.0)] * N,
                        options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
                    )
                    
                    h_candidate = project(result.x)
                    val_candidate = objective(h_candidate)
                    
                    if val_candidate < best_val:
                        best_h = h_candidate
                        best_val = val_candidate
                        
                except Exception:
                    pass
    
    # Strategy 3: Random perturbations around best solution
    if remaining() > 10:
        np.random.seed(123)
        for _ in range(20):
            if remaining() < 3:
                break
            
            # Vary perturbation size
            scale = 0.01 + 0.03 * np.random.rand()
            h_pert = best_h + scale * np.random.randn(N)
            h_pert = project(h_pert)
            
            def obj_func2(x):
                return objective_smooth(project(x), beta=30.0)
            
            try:
                result = minimize(
                    obj_func2,
                    h_pert,
                    method='L-BFGS-B',
                    bounds=[(0.0, 1.0)] * N,
                    options={'maxiter': 150, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                
                h_candidate = project(result.x)
                val_candidate = objective(h_candidate)
                
                if val_candidate < best_val:
                    best_h = h_candidate
                    best_val = val_candidate
                    
            except Exception:
                pass
            
    # Strategy 4: Discrete binary local search
    if remaining() > 15:
        np.random.seed(999)
        for restart in range(10):
            if remaining() < 5:
                break
            
            # Create random binary sequence with 50 ones
            h_bin = np.zeros(N)
            ones_indices = np.random.choice(N, N//2, replace=False)
            h_bin[ones_indices] = 1.0
            
            h_candidate, val_candidate = local_search_binary(h_bin, max_iters=2000)
            
            if val_candidate < best_val:
                best_h = h_candidate
                best_val = val_candidate

    # Final polish with multiple refinement passes
    if remaining() > 5:
        for polish_pass in range(3):
            if remaining() < 3:
                break
            
            def obj_final(x):
                return objective_smooth(project(x), beta=20.0)
            
            try:
                result = minimize(
                    obj_final,
                    best_h,
                    method='L-BFGS-B',
                    bounds=[(0.0, 1.0)] * N,
                    options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-12}
                )
                
                h_candidate = project(result.x)
                val_candidate = objective(h_candidate)
                
                if val_candidate < best_val:
                    best_h = h_candidate
                    best_val = val_candidate
                    
            except Exception:
                pass
        
        # Exact objective refinement
        if remaining() > 2:
            def obj_exact(x):
                return objective(project(x))
            
            try:
                result = minimize(
                    obj_exact,
                    best_h,
                    method='L-BFGS-B',
                    bounds=[(0.0, 1.0)] * N,
                    options={'maxiter': 100, 'ftol': 1e-15, 'gtol': 1e-12}
                )
                
                h_candidate = project(result.x)
                val_candidate = objective(h_candidate)
                
                if val_candidate < best_val:
                    best_h = h_candidate
                    best_val = val_candidate
                    
            except Exception:
                pass
    
    best_h = project(best_h)
    return best_h
