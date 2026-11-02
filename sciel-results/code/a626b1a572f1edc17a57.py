"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    """
    Construct step heights h in [0,1] with sum = N/2 (i.e., mean 0.5)
    to minimize the maximum overlap of h with shifted (1-h).
    
    The overlap at shift k is:
        O_k = sum_i h[i] * (1 - h[(i+k) % N])
    
    We want to minimize max_k O_k.
    
    Note: sum_i h[i]*(1-h[(i+k)%N]) = sum_i h[i] - sum_i h[i]*h[(i+k)%N]
    Since sum_i h[i] = N/2, this is N/2 - C_k where C_k = sum_i h[i]*h[(i+k)%N]
    
    So minimizing max_k O_k is equivalent to minimizing max_k (N/2 - C_k),
    which is equivalent to maximizing min_k C_k.
    
    So we want to maximize the minimum autocorrelation C_k over all non-zero shifts k.
    
    Alternatively, we can think of it as: we want h to be as "uncorrelated" as possible
    with its shifts, but since h has mean 0.5, the autocorrelation will have a baseline.
    
    Let's define g[i] = h[i] - 0.5, so sum g[i] = 0 and g[i] in [-0.5, 0.5].
    Then C_k = sum_i h[i]*h[(i+k)%N] = sum_i (0.5+g[i])*(0.5+g[(i+k)%N])
           = N*0.25 + 0.5*sum_i g[i] + 0.5*sum_i g[(i+k)%N] + sum_i g[i]*g[(i+k)%N]
           = N/4 + 0 + 0 + sum_i g[i]*g[(i+k)%N]
           = N/4 + A_k
    
    where A_k = sum_i g[i]*g[(i+k)%N] is the autocorrelation of the zero-mean component.
    
    So O_k = N/2 - C_k = N/2 - N/4 - A_k = N/4 - A_k.
    
    We want to minimize max_k O_k = N/4 - min_k A_k.
    So we want to maximize min_k A_k over k=1,...,N-1 (k=0 gives A_0 = sum g[i]^2).
    
    This is a max-min problem. We can use a bisection approach or direct optimization.
    
    Let's try to find h that maximizes min_{k=1}^{N-1} A_k.
    """
    start_time = time.time()
    deadline = start_time + 120  # 120 second limit
    
    # Use a parameterization that automatically satisfies the constraints.
    # h[i] = 0.5 + 0.5 * sin(2*pi*i*freq + phase) won't work well for arbitrary N.
    
    # Instead, let's use a direct optimization approach.
    # We'll minimize -min_k A_k subject to 0 <= h[i] <= 1 and sum h[i] = N/2.
    
    # For efficiency, let's try a few different initializations and methods.
    
    def compute_A(h):
        """Compute autocorrelation A_k for k=0,...,N-1."""
        # A_k = sum_i g[i]*g[(i+k)%N] where g = h - 0.5
        g = h - 0.5
        # Use FFT for efficiency
        G = np.fft.fft(g)
        A = np.fft.ifft(G * np.conj(G)).real
        return A
    
    def objective(x):
        """Objective to minimize: -min_{k=1}^{N-1} A_k"""
        h = x
        A = compute_A(h)
        # Exclude k=0
        min_A = np.min(A[1:])
        return -min_A
    
    def objective_with_constraints(x):
        """Same as objective but we handle constraints separately."""
        return objective(x)
    
    # Try a few different initializations
    best_h = None
    best_score = np.inf
    
    # Initialization 1: Alternating pattern
    h1 = np.array([0.5 + 0.5 * (1 if i % 2 == 0 else -1) for i in range(N)])
    # But this gives h in {0, 1}, sum = N/2 if N is even.
    # Let's check: for N=100, 50 ones and 50 zeros, sum = 50 = N/2. Good.
    
    # Initialization 2: Smooth variation
    t = np.linspace(0, 2*np.pi, N, endpoint=False)
    h2 = 0.5 + 0.25 * np.sin(t)
    # Adjust to make sum exactly N/2
    h2 = h2 - np.mean(h2) + 0.5
    
    # Initialization 3: Random with mean 0.5
    np.random.seed(42)
    h3 = np.random.uniform(0, 1, N)
    h3 = h3 - np.mean(h3) + 0.5
    # Clip to [0,1] and re-adjust
    h3 = np.clip(h3, 0, 1)
    # Re-adjust mean
    h3 = h3 - np.mean(h3) + 0.5
    
    candidates = [h1, h2, h3]
    
    for init_h in candidates:
        # Ensure sum is exactly N/2
        init_h = init_h - np.mean(init_h) + 0.5
        init_h = np.clip(init_h, 0, 1)
        
        # Use SLSQP with constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - N/2},
        ]
        bounds = [(0, 1)] * N
        
        try:
            result = minimize(objective, init_h, method='SLSQP', bounds=bounds,
                             constraints=constraints, options={'maxiter': 1000, 'ftol': 1e-10})
            
            if result.success or result.fun < best_score:
                h_opt = result.x
                # Ensure constraints are satisfied
                h_opt = h_opt - np.mean(h_opt) + 0.5
                h_opt = np.clip(h_opt, 0, 1)
                # Final adjustment
                h_opt = h_opt - np.mean(h_opt) + 0.5
                
                A = compute_A(h_opt)
                score = -np.min(A[1:])
                if score < best_score:
                    best_score = score
                    best_h = h_opt.copy()
        except Exception as e:
            pass
        
        # Check time
        if time.time() > deadline - 10:
            break
    
    # If we found a good solution, return it; otherwise return the alternating pattern
    if best_h is not None:
        # Final cleanup to ensure exact sum
        best_h = best_h - np.mean(best_h) + 0.5
        return best_h
    else:
        # Fallback: alternating pattern
        h_alt = np.array([1.0 if i % 2 == 0 else 0.0 for i in range(N)])
        return h_alt
# EVOLVE-BLOCK-END
