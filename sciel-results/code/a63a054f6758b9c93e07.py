import numpy as np
import time
from scipy.optimize import minimize

N = 600

def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin before 120s hard limit (evaluator kills at 150s)
    
    def elapsed():
        return time.time() - start_time
    
    def metric(h):
        """Compute the normalized maximum autoconvolution metric."""
        s = np.sum(h)
        if s < 1e-15:
            return 1e10
        hn = h / s
        # Use FFT for speed
        F = np.fft.fft(hn, n=2*N)
        conv = np.fft.ifft(F * F).real[:2*N-1]
        return np.max(conv)
    
    # Initial guess: uniform
    h0 = np.ones(N)
    best_h = h0.copy()
    best_val = metric(best_h)
    
    # Phase 1: Try structural candidates
    candidates = []
    
    # Uniform
    candidates.append(('uniform', np.ones(N)))
    
    # Triangular
    center = N // 2
    tri = np.array([1.0 - abs(i - center) / center if center > 0 else 1.0 for i in range(N)])
    tri = np.clip(tri, 0, None)
    candidates.append(('triangular', tri))
    
    # Gaussian
    x_grid = np.linspace(-3, 3, N)
    gauss = np.exp(-x_grid**2)
    candidates.append(('gaussian', gauss))
    
    # Inverted Gaussian
    inv_gauss = np.ones(N) - 0.5 * np.exp(-3.0 * np.linspace(-1, 1, N)**2)
    inv_gauss = np.clip(inv_gauss, 0, None)
    candidates.append(('inv_gauss', inv_gauss))
    
    # Two peaks
    two_peak = np.zeros(N)
    two_peak[N//4] = 1.0
    two_peak[3*N//4] = 1.0
    candidates.append(('two_peak', two_peak))
    
    # Plateaus
    for width_frac in [0.3, 0.5, 0.7]:
        w = int(N * width_frac)
        plateau = np.zeros(N)
        plateau[N//2 - w//2 : N//2 + w//2 + 1] = 1.0
        candidates.append((f'plateau_{width_frac}', plateau))
    
    # Ramp
    ramp = np.linspace(1, 0, N)
    ramp[ramp < 0] = 0
    candidates.append(('ramp', ramp))
    
    # Evaluate all candidates
    for name, h in candidates:
        if elapsed() > time_limit:
            break
        val = metric(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    # Phase 2: Gradient-based optimization using log-parameters for non-negativity
    # h_i = exp(x_i) / sum(exp(x_j))  (softmax)
    
    def softmax(x):
        x_shifted = x - np.max(x)
        e = np.exp(x_shifted)
        return e / np.sum(e)
    
    def objective_from_params(x):
        if elapsed() > time_limit:
            return 1e10
        h = softmax(x)
        return metric(h)
    
    # Initial params: all zeros (uniform)
    x0 = np.zeros(N)
    
    # Use L-BFGS-B for gradient-based optimization
    # We need to provide gradient or let scipy approximate it
    # For N=600, approximating gradient is expensive. Let's try Nelder-Mead first
    # but it's very slow in high dimensions.
    
    # Alternative: Coordinate descent on the logits
    # Or: Use a simpler approach - optimize the heights directly with penalty for negatives
    
    def objective_direct(h):
        if elapsed() > time_limit:
            return 1e10
        # Penalty for negative values
        penalty = np.sum(np.minimum(h, 0)**2) * 1e6
        return metric(np.maximum(h, 0)) + penalty
    
    # Try a few random restarts with local optimization
    for restart in range(5):
        if elapsed() > time_limit - 30:
            break
        
        if restart == 0:
            x_start = np.zeros(N)
        else:
            x_start = np.random.normal(0, 0.5, N)
        
        try:
            result = minimize(
                objective_from_params,
                x_start,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            
            if elapsed() > time_limit - 10:
                break
                
            h_result = softmax(result.x)
            val_result = metric(h_result)
            
            if val_result < best_val:
                best_val = val_result
                best_h = h_result.copy()
        except Exception:
            continue
    
    # Phase 3: Fine-tuning with simulated annealing on the best solution
    if elapsed() < time_limit - 20:
        f = best_h.copy()
        current_score = best_val
        T = 0.1
        cooling_rate = 0.999
        min_T = 1e-4
        
        while T > min_T:
            if elapsed() > time_limit:
                break
            
            # Perturb
            old_f = f.copy()
            idx = np.random.randint(0, N, size=10)
            f[idx] += np.random.normal(0, 0.1, size=10)
            f[idx] = np.maximum(f[idx], 0)
            
            norm = np.sum(f)
            if norm < 1e-10:
                f = np.ones(N) / N
            else:
                f /= norm
            
            new_score = metric(f)
            
            if new_score < current_score:
                current_score = new_score
                if new_score < best_val:
                    best_val = new_score
                    best_h = f.copy()
            else:
                delta = new_score - current_score
                if np.random.rand() < np.exp(-delta / T):
                    current_score = new_score
                else:
                    f = old_f
            
            T *= cooling_rate
    
    return list(best_h)
