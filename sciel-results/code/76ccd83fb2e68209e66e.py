"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin before 150s kill

    def autocorr_vector(h):
        """
        Compute the full autocorrelation vector (length N) using FFT.
        Returns acf[0..N-1] where acf[k] = sum_i h[i]*h[i+k].
        """
        n_fft = 1 << (2 * N - 1).bit_length()
        F = np.fft.rfft(h, n=n_fft)
        power = F * np.conj(F)
        acf = np.fft.irfft(power, n=n_fft).real[:N]
        return acf

    def smooth_max_abs_acf(h, temperature=0.01):
        """
        Smooth approximation of max(|acf[1]|, ..., |acf[N-1]|) using softmax.
        This makes the objective differentiable for gradient-based methods.
        """
        acf = autocorr_vector(h)
        abs_acf = np.abs(acf[1:])
        # Softmax approximation of max
        scaled = abs_acf / temperature
        max_scaled = np.max(scaled)
        exp_scaled = np.exp(scaled - max_scaled)
        log_sum = np.log(np.sum(exp_scaled)) + max_scaled
        return log_sum * temperature

    def smooth_objective_and_grad(x):
        """
        Compute smooth objective and its gradient.
        x is continuous vector; h = tanh(x) is the actual signal.
        """
        h = np.tanh(x)
        val = smooth_max_abs_acf(h)
        
        # Numerical gradient (L-BFGS-B can handle this)
        return val

    def discrete_objective(h):
        """
        Exact objective: max |acf[k]| for k=1..N-1.
        """
        acf = autocorr_vector(h)
        return np.max(np.abs(acf[1:]))

    def simulated_annealing_discrete(h, max_iters=5000):
        """
        Simulated annealing for discrete ±1 optimization.
        """
        current_h = h.copy()
        current_val = discrete_objective(current_h)
        best_h = current_h.copy()
        best_val = current_val
        
        T = 1.0
        cooling = 0.999
        iters = 0
        
        while iters < max_iters and time.time() < deadline:
            iters += 1
            
            # Pick a random index to flip
            i = np.random.randint(N)
            old_val = current_h[i]
            current_h[i] = -old_val
            new_val = discrete_objective(current_h)
            delta = new_val - current_val
            
            # Accept if better, or with probability exp(-delta/T)
            if delta < 0 or np.random.rand() < np.exp(-delta / T):
                current_val = new_val
                if new_val < best_val:
                    best_val = new_val
                    best_h = current_h.copy()
            else:
                current_h[i] = old_val
            
            T *= cooling
        
        return best_h, best_val

    # Strategy:
    # 1. Use continuous optimization with smooth max approximation to find good starting points.
    # 2. Project to ±1 and refine with simulated annealing.
    
    rng = np.random.RandomState(42)
    
    best_h = None
    best_val = np.inf
    
    # Phase 1: Continuous optimization from multiple starts
    num_continuous_starts = 8
    
    for s in range(num_continuous_starts):
        if time.time() > deadline - 30:
            break
        
        # Random start
        x0 = rng.randn(N) * 0.3
        
        # Optimize with L-BFGS-B
        try:
            res = minimize(
                smooth_objective_and_grad,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 500, 'maxfun': 20000, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            
            # Project to ±1
            h_proj = np.sign(res.x)
            h_proj[h_proj == 0] = 1.0
            
            # Evaluate discrete objective
            val_proj = discrete_objective(h_proj)
            
            if val_proj < best_val:
                best_val = val_proj
                best_h = h_proj.copy()
                
        except Exception as e:
            continue
    
    # Phase 2: Simulated annealing refinement on best solution
    if best_h is not None and time.time() < deadline - 10:
        h_sa, val_sa = simulated_annealing_discrete(best_h, max_iters=10000)
        if val_sa < best_val:
            best_val = val_sa
            best_h = h_sa
    
    # Phase 3: If time permits, try additional continuous starts and combine
    if time.time() < deadline - 5:
        for s in range(5):
            if time.time() > deadline - 2:
                break
            
            x0 = rng.randn(N) * 0.2
            try:
                res = minimize(
                    smooth_objective_and_grad,
                    x0,
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'maxfun': 10000, 'ftol': 1e-8, 'gtol': 1e-6}
                )
                
                h_proj = np.sign(res.x)
                h_proj[h_proj == 0] = 1.0
                
                val_proj = discrete_objective(h_proj)
                
                # Combine with best via crossover-like operation
                if val_proj < best_val:
                    best_val = val_proj
                    best_h = h_proj.copy()
                else:
                    # Try mixing
                    mixed = np.where(rng.rand(N) < 0.5, h_proj, best_h)
                    val_mixed = discrete_objective(mixed)
                    if val_mixed < best_val:
                        best_val = val_mixed
                        best_h = mixed.copy()
                        
            except Exception:
                continue
    
    # Fallback
    if best_h is None:
        best_h = rng.choice([1, -1], N).astype(np.float64)

    return best_h
# EVOLVE-BLOCK-END
