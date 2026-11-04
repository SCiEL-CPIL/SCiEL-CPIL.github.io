"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _autoconvolution_max_abs(f):
    """Compute max |f*f| over all lags using FFT."""
    n = len(f)
    conv = np.fft.ifft(np.fft.fft(f, 2*n) ** 2).real
    return np.max(np.abs(conv))


def _compute_objective(x, support):
    """Compute max |autoconv| for given amplitudes on fixed support."""
    f = np.zeros(N)
    f[support] = x
    return _autoconvolution_max_abs(f)


def construct():
    t0 = time.time()
    deadline = t0 + 115.0  # safety margin before 120s limit
    
    best_f = None
    best_val = np.inf
    
    # Structural change: Use continuous amplitudes on a fixed support,
    # optimized via scipy's L-BFGS-B. This is a different optimization
    # method (gradient-based continuous optimization) vs. discrete
    # sign-flip local search.
    
    # Try different support sizes
    support_sizes = [50, 80, 120, 160, 200]
    
    np.random.seed(42)
    
    for K in support_sizes:
        if time.time() > deadline - 30:
            break
        
        # Multiple random supports
        for trial in range(3):
            if time.time() > deadline - 15:
                break
            
            np.random.seed(42 + trial * 1000 + K)
            support = np.random.choice(N, size=K, replace=False)
            
            # Initialize with random small amplitudes in [-1, 1]
            x0 = np.random.uniform(-0.5, 0.5, size=K)
            
            # Use L-BFGS-B to minimize max |autoconv|
            # We need to define a smooth objective; max is non-smooth,
            # so we use a log-sum-exp approximation or just try it.
            
            # Actually, let's use a simpler approach: coordinate descent
            # on continuous amplitudes
            current_x = x0.copy()
            current_score = _compute_objective(current_x, support)
            
            if current_score < best_val:
                best_val = current_score
                best_f = np.zeros(N)
                best_f[support] = current_x
            
            # Coordinate descent: optimize one amplitude at a time
            improved = True
            while improved and time.time() < deadline - 15:
                improved = False
                for i in range(K):
                    if time.time() > deadline - 15:
                        break
                    
                    # Try to optimize x[i] by scanning a small range
                    old_xi = current_x[i]
                    
                    # Grid search around current value
                    best_xi = old_xi
                    best_score = current_score
                    
                    # Try values in a neighborhood
                    for dx in np.linspace(-0.3, 0.3, 13):
                        candidate = old_xi + dx
                        if candidate < -1.0 or candidate > 1.0:
                            continue
                        current_x[i] = candidate
                        s = _compute_objective(current_x, support)
                        if s < best_score:
                            best_score = s
                            best_xi = candidate
                    
                    if best_xi != old_xi:
                        current_x[i] = best_xi
                        current_score = best_score
                        improved = True
                        
                        if current_score < best_val:
                            best_val = current_score
                            best_f = np.zeros(N)
                            best_f[support] = current_x
                    else:
                        current_x[i] = old_xi
            
            # Final refinement: try scipy minimize on the continuous problem
            if time.time() < deadline - 10:
                def objective(x):
                    return _compute_objective(x, support)
                
                # Use L-BFGS-B with bounds
                bounds = [(-1.0, 1.0)] * K
                result = minimize(objective, current_x, method='L-BFGS-B', 
                                  bounds=bounds, options={'maxiter': 50, 
                                                         'ftol': 1e-6,
                                                         'gtol': 1e-4})
                
                if result.fun < current_score:
                    current_x = result.x
                    current_score = result.fun
                    
                    if current_score < best_val:
                        best_val = current_score
                        best_f = np.zeros(N)
                        best_f[support] = current_x
    
    if best_f is None:
        return np.ones(N)
    
    return best_f
# EVOLVE-BLOCK-END
