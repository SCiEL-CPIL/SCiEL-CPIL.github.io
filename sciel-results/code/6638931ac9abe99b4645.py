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
    
    Let's define g[i] = h[i] - 0.5, so sum g[i] = 0 and g[i] in [-0.5, 0.5].
    Then C_k = N/4 + A_k where A_k = sum_i g[i]*g[(i+k)%N].
    
    So O_k = N/4 - A_k.
    
    We want to minimize max_k O_k = N/4 - min_k A_k.
    So we want to maximize min_k A_k over k=1,...,N-1.
    
    This is a max-min problem. We can use a bisection approach or direct optimization.
    """
    start_time = time.time()
    deadline = start_time + 115  # 115 second limit (safety margin)
    
    def compute_A(h):
        """Compute autocorrelation A_k for k=0,...,N-1."""
        g = h - 0.5
        G = np.fft.fft(g)
        A = np.fft.ifft(G * np.conj(G)).real
        return A
    
    def compute_min_A(h):
        """Compute min_{k=1}^{N-1} A_k."""
        A = compute_A(h)
        return np.min(A[1:])
    
    def compute_max_O(h):
        """Compute max_{k=1}^{N-1} O_k = N/4 - min_{k=1}^{N-1} A_k."""
        A = compute_A(h)
        return N/4 - np.min(A[1:])
    
    def objective(x):
        """Objective to minimize: max_k O_k = N/4 - min_{k=1}^{N-1} A_k."""
        A = compute_A(x)
        return N/4 - np.min(A[1:])
    
    def objective_grad(x):
        """Compute objective and gradient."""
        g = x - 0.5
        G = np.fft.fft(g)
        A = np.fft.ifft(G * np.conj(G)).real
        min_A = np.min(A[1:])
        obj = N/4 - min_A
        
        # Gradient: we need d/dx of (N/4 - min(A[1:]))
        # = -d/dx min(A[1:])
        # The min is achieved at some k*, so gradient = -dA_k*/dx
        # dA_k/dx_i = g[(i-k)%N] + g[(i+k)%N] (by symmetry, but let's be careful)
        # Actually A_k = sum_j g[j]*g[(j+k)%N]
        # dA_k/dg_i = g[(i-k)%N] + g[(i+k)%N] (if both terms exist, which they do in cyclic)
        # But since g = x - 0.5, dg/dx = I, so dA_k/dx_i = g[(i-k)%N] + g[(i+k)%N]
        
        # Find argmin
        A_nonzero = A[1:]
        k_star = np.argmin(A_nonzero) + 1
        
        # Compute gradient of A_{k_star} with respect to x
        idx = np.arange(N)
        grad_A = g[(idx - k_star) % N] + g[(idx + k_star) % N]
        
        # d obj/dx = -d A_{k_star}/dx
        grad_obj = -grad_A
        
        return obj, grad_obj
    
    # Try multiple initializations and optimization strategies
    best_h = None
    best_obj = np.inf
    
    # Initialization 1: Alternating pattern (0, 1, 0, 1, ...)
    h1 = np.array([1.0 if i % 2 == 0 else 0.0 for i in range(N)])
    
    # Initialization 2: Smooth sinusoidal
    t = np.linspace(0, 2*np.pi, N, endpoint=False)
    h2 = 0.5 + 0.3 * np.sin(t)
    h2 = h2 - np.mean(h2) + 0.5
    h2 = np.clip(h2, 0, 1)
    h2 = h2 - np.mean(h2) + 0.5
    
    # Initialization 3: Random
    np.random.seed(42)
    h3 = np.random.uniform(0, 1, N)
    h3 = h3 - np.mean(h3) + 0.5
    h3 = np.clip(h3, 0, 1)
    h3 = h3 - np.mean(h3) + 0.5
    
    # Initialization 4: Multi-frequency
    h4 = 0.5 + 0.2*np.sin(t) + 0.15*np.sin(2*t + 0.5) + 0.1*np.sin(3*t + 1.0)
    h4 = h4 - np.mean(h4) + 0.5
    h4 = np.clip(h4, 0, 1)
    h4 = h4 - np.mean(h4) + 0.5
    
    # Initialization 5: Another random
    np.random.seed(123)
    h5 = np.random.uniform(0, 1, N)
    h5 = h5 - np.mean(h5) + 0.5
    h5 = np.clip(h5, 0, 1)
    h5 = h5 - np.mean(h5) + 0.5
    
    # Initialization 6: Block pattern
    h6 = np.zeros(N)
    h6[::3] = 1.0
    h6 = h6 - np.mean(h6) + 0.5
    h6 = np.clip(h6, 0, 1)
    h6 = h6 - np.mean(h6) + 0.5
    
    candidates = [h1, h2, h3, h4, h5, h6]
    
    for init_h in candidates:
        # Ensure sum is exactly N/2
        init_h = init_h - np.mean(init_h) + 0.5
        init_h = np.clip(init_h, 0, 1)
        init_h = init_h - np.mean(init_h) + 0.5
        
        if time.time() > deadline - 5:
            break
        
        # Use SLSQP with custom objective
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - N/2},
        ]
        bounds = [(0, 1)] * N
        
        try:
            result = minimize(objective, init_h, method='SLSQP', bounds=bounds,
                             constraints=constraints, options={'maxiter': 500, 'ftol': 1e-12})
            
            h_opt = result.x
            # Ensure constraints are satisfied
            h_opt = h_opt - np.mean(h_opt) + 0.5
            h_opt = np.clip(h_opt, 0, 1)
            h_opt = h_opt - np.mean(h_opt) + 0.5
            
            obj_val = objective(h_opt)
            if obj_val < best_obj:
                best_obj = obj_val
                best_h = h_opt.copy()
                
        except Exception as e:
            pass
        
        # Also try with gradient-based method (L-BFGS-B)
        if time.time() > deadline - 5:
            break
        
        try:
            # For L-BFGS-B, we need to handle the sum constraint differently
            # We'll use a penalty method or parameterize to satisfy the constraint
            
            # Parameterize: h = 0.5 + g where sum(g) = 0
            # We can use a transformation that ensures sum = 0
            
            # Actually, let's just use SLSQP with more iterations
            result2 = minimize(objective, init_h, method='SLSQP', bounds=bounds,
                              constraints=constraints, options={'maxiter': 1000, 'ftol': 1e-14})
            
            h_opt2 = result2.x
            h_opt2 = h_opt2 - np.mean(h_opt2) + 0.5
            h_opt2 = np.clip(h_opt2, 0, 1)
            h_opt2 = h_opt2 - np.mean(h_opt2) + 0.5
            
            obj_val2 = objective(h_opt2)
            if obj_val2 < best_obj:
                best_obj = obj_val2
                best_h = h_opt2.copy()
                
        except Exception as e:
            pass
    
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
