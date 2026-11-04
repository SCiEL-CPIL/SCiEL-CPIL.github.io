import numpy as np
import time
from scipy.optimize import minimize

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin from 150s limit
    
    N = 100
    target_sum = N / 2.0
    
    # Precompute FFT window for speed
    # We need circular correlation to use FFT efficiently, but the problem asks for linear.
    # However, for N=100, linear is fast enough. But let's use a vectorized linear approach.
    # C(k) = sum_{i=0}^{N-1-k} h[i] * (1 - h[i+k])
    # This is equivalent to the diagonal of the outer product h[:, None] * (1-h)[None, :] shifted.
    # Or simply:
    # c[k] = np.dot(h[:N-k], 1-h[k:])
    
    def get_correlations_linear(h):
        g = 1.0 - h
        # Vectorized computation of all lags
        # We want c[k] for k=1..N-1
        # Using np.correlate is O(N^2) but in C, which is fast.
        # np.correlate(h, g, mode='full')
        # Index N-1 is lag 0.
        # Index N+k is lag k.
        # Index N-1-k is lag -k.
        # We want lags 1..N-1 and -(N-1)..-1.
        c_full = np.correlate(h, g, mode='full')
        # Positive lags 1..N-1 are at indices N..2N-2
        c_pos = c_full[N:2*N-1]
        # Negative lags -1..-(N-1) are at indices N-2..0
        c_neg = c_full[0:N-1]
        return np.concatenate([c_pos, c_neg])

    def max_overlap(h):
        c = get_correlations_linear(h)
        return np.max(c)
    
    # Smooth max using log-sum-exp with temperature
    def smooth_max(c, temp=0.05):
        scaled = c / temp
        m = np.max(scaled)
        if m > 700:  # Avoid overflow
            return m * temp
        return temp * (m + np.log(np.sum(np.exp(scaled - m))))
    
    def objective_smooth(h, temp=0.05):
        c = get_correlations_linear(h)
        val = smooth_max(c, temp)
        # Penalty for sum constraint
        sum_err = np.sum(h) - target_sum
        return val + 10.0 * sum_err**2

    # Projection onto feasible set: 0 <= h_i <= 1, sum(h) = N/2
    def project(h):
        h = np.clip(h, 0, 1)
        # Water-filling or iterative projection for sum constraint
        # Iterative is fine for N=100
        for _ in range(50):
            diff = target_sum - np.sum(h)
            if abs(diff) < 1e-12:
                break
            h += diff / N
            h = np.clip(h, 0, 1)
        return h
    
    # Initialize
    rng = np.random.default_rng(42)
    
    best_h = np.full(N, 0.5)
    best_score = max_overlap(best_h)
    
    bounds = [(0, 1)] * N
    
    # Multi-Start Local Search
    # We will run several independent optimizations starting from different random initializations
    # or structured initializations (e.g., sinusoidal)
    
    n_starts = 20
    temps = [0.1, 0.05, 0.02, 0.01]
    
    # 1. Constant start
    h_starts = [np.full(N, 0.5)]
    
    # 2. Sinusoidal starts to break symmetry
    for k in range(1, 6):
        x = np.linspace(0, 2*np.pi*k, N)
        h_sin = 0.5 + 0.4 * np.sin(x)
        h_starts.append(project(h_sin))
        
    # 3. Random starts
    for _ in range(n_starts - len(h_starts)):
        h_rand = rng.uniform(0, 1, N)
        h_starts.append(project(h_rand))
        
    for h_init in h_starts:
        if time.time() - start_time > time_limit - 10:
            break
            
        # Try different temperatures for the smooth max
        for temp in temps:
            if time.time() - start_time > time_limit - 5:
                break
                
            def obj_scipy(x, t=temp):
                return objective_smooth(x, t)
            
            try:
                # Use L-BFGS-B for speed and bound handling
                res = minimize(obj_scipy, h_init, method='L-BFGS-B', bounds=bounds, 
                               options={'maxiter': 500, 'maxfun': 5000, 'gtol': 1e-6})
                h_opt = res.x
                h_opt = project(h_opt)
                score_opt = max_overlap(h_opt)
                
                if score_opt < best_score:
                    best_score = score_opt
                    best_h = h_opt.copy()
            except Exception:
                pass

    # Final High-Precision Polish
    # Use a very low temperature to approximate the max more closely
    if time.time() - start_time < time_limit - 2:
        h_init = best_h.copy()
        
        # Run a few iterations with very small temp
        for temp in [0.005, 0.001, 0.0005]:
            if time.time() - start_time > time_limit - 1:
                break
            
            def obj_scipy_final(x, t=temp):
                return objective_smooth(x, t)
            
            try:
                res = minimize(obj_scipy_final, h_init, method='L-BFGS-B', bounds=bounds, 
                               options={'maxiter': 1000, 'maxfun': 20000, 'gtol': 1e-9})
                h_opt = res.x
                h_opt = project(h_opt)
                score_opt = max_overlap(h_opt)
                
                if score_opt < best_score:
                    best_score = score_opt
                    best_h = h_opt.copy()
                    h_init = h_opt.copy()
            except Exception:
                pass

    # Ensure final solution is valid
    best_h = project(best_h)
    
    return best_h
# EVOLVE-BLOCK-END
