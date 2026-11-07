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

def objective_epigraph(params, beta=50.0):
    """Epigraph formulation: minimize t subject to c_i <= t for all i."""
    h = params[:N]
    t = params[N]
    c = np.correlate(h, 1.0 - h, mode="full")
    # Smooth max via log-sum-exp: approx max(c_i)
    c_max = np.max(c)
    shifted = beta * (c - c_max)
    smooth_max = np.log(np.sum(np.exp(shifted))) / beta + c_max
    # Penalize violation: t should be >= smooth_max
    penalty = np.maximum(0, smooth_max - t) ** 2 * 100.0
    return float(t / N * 2.0 + penalty)

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

def construct():
    t0 = time.time()
    i = np.arange(N)
    
    best_h = np.full(N, 0.5)
    best_val = objective(best_h)
    
    # Strategy 1: Epigraph formulation with L-BFGS-B
    # Parameters: h (N vars) + t (1 var)
    if remaining() > 10:
        for n_modes in range(1, 6):
            if remaining() < 5:
                break
            
            np.random.seed(42 + n_modes)
            # Initialize h with low-frequency structure
            h_init = np.full(N, 0.5)
            for k in range(n_modes):
                a = 0.1 * np.random.randn()
                b = 0.1 * np.random.randn()
                h_init += a * np.cos(2 * np.pi * k * i / N)
                h_init += b * np.sin(2 * np.pi * k * i / N)
            h_init = project(h_init)
            
            # Initialize t with current max correlation
            c_init = np.correlate(h_init, 1.0 - h_init, mode="full")
            t_init = np.max(c_init) + 0.01
            
            params_init = np.concatenate([h_init, [t_init]])
            
            def obj_epigraph_local(params):
                return objective_epigraph(params, beta=50.0)
            
            bounds = [(0.0, 1.0)] * N + [(-1.0, 1.0)]
            
            try:
                result = minimize(
                    obj_epigraph_local,
                    params_init,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                
                h_candidate = project(result.x[:N])
                val_candidate = objective(h_candidate)
                
                if val_candidate < best_val:
                    best_h = h_candidate
                    best_val = val_candidate
                    
            except Exception:
                pass
    
    # Strategy 2: Direct optimization with adaptive beta
    for k in range(1, 15):
        if remaining() < 3:
            break
        
        for A_init in [0.2, 0.3, 0.4, 0.45]:
            if remaining() < 2:
                break
            
            for phase in [0.0, np.pi/3, 2*np.pi/3]:
                if remaining() < 1:
                    break
                
                h_init = 0.5 + A_init * np.cos(2 * np.pi * k * i / N + phase)
                h_init = project(h_init)
                
                # Use progressively higher beta
                for beta in [30.0, 50.0, 80.0]:
                    if remaining() < 1:
                        break
                    
                    def obj_func(x, b=beta):
                        return objective_smooth(project(x), beta=b)
                    
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
                            # Update h_init for next beta
                            h_init = h_candidate.copy()
                            
                    except Exception:
                        pass
    
    # Strategy 3: Random perturbations around best solution with epigraph
    if remaining() > 10:
        np.random.seed(123)
        for _ in range(15):
            if remaining() < 3:
                break
            
            scale = 0.02 + 0.04 * np.random.rand()
            h_pert = best_h + scale * np.random.randn(N)
            h_pert = project(h_pert)
            
            c_pert = np.correlate(h_pert, 1.0 - h_pert, mode="full")
            t_pert = np.max(c_pert) + 0.01
            
            params_pert = np.concatenate([h_pert, [t_pert]])
            bounds = [(0.0, 1.0)] * N + [(-1.0, 1.0)]
            
            try:
                result = minimize(
                    lambda p: objective_epigraph(p, beta=60.0),
                    params_pert,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                
                h_candidate = project(result.x[:N])
                val_candidate = objective(h_candidate)
                
                if val_candidate < best_val:
                    best_h = h_candidate
                    best_val = val_candidate
                    
            except Exception:
                pass
    
    # Final polish: high-precision refinement with exact objective
    if remaining() > 5:
        # First pass with smooth objective at high beta
        def obj_high_beta(x):
            return objective_smooth(project(x), beta=100.0)
        
        try:
            result = minimize(
                obj_high_beta,
                best_h,
                method='L-BFGS-B',
                bounds=[(0.0, 1.0)] * N,
                options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            
            h_candidate = project(result.x)
            val_candidate = objective(h_candidate)
            
            if val_candidate < best_val:
                best_h = h_candidate
                best_val = val_candidate
                
        except Exception:
            pass
        
        # Second pass with exact objective
        if remaining() > 3:
            def obj_exact(x):
                return objective(project(x))
            
            try:
                result = minimize(
                    obj_exact,
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
    
    # Final validation and projection
    best_h = project(best_h)
    
    # Ensure validity
    if not np.all(np.isfinite(best_h)):
        best_h = np.full(N, 0.5)
    if abs(best_h.sum() - N/2) > 1e-9 * N:
        best_h = project(best_h)
    if not np.all((best_h >= 0) & (best_h <= 1)):
        best_h = project(best_h)
    
    return best_h
