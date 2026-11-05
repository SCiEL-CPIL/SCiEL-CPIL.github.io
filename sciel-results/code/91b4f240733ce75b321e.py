import numpy as np
import time
import scipy.special as sp

N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.special as sp

def construct():
    deadline = time.time() + 115  # Safety margin for 120s limit
    
    # Precompute Chebyshev basis on the grid [-1/4, 1/4] mapped to [-1, 1] for T_n
    # The task says intervals on [-1/4, 1/4]. 
    # Let's use the midpoints of the intervals as the sample points for the step function values.
    # Or just linspace? "step function with N equal intervals".
    # Usually h[i] is the height on interval i.
    # The convolution is np.convolve(h, h).
    # The metric is scale invariant.
    
    # Let's define the grid points x in [-1/4, 1/4]
    # The intervals are [ -1/4 + i/N * 1/2, -1/4 + (i+1)/N * 1/2 ]
    # Let's use the center of each interval for the basis evaluation.
    # Centers: -1/4 + (i + 0.5) * (0.5 / N)
    x = -0.25 + (np.arange(N) + 0.5) * (0.5 / N)
    
    # Map x to t in [-1, 1] for Chebyshev polynomials
    # x in [-0.25, 0.25] -> t in [-1, 1]
    # t = 2 * (x + 0.25) - 1 = 2x + 0.5 - 1 = 2x - 0.5
    t = 2 * x - 0.5
    
    # Number of Chebyshev modes to use. 
    # Higher modes allow more complex shapes (negative lobes) but might increase peak.
    # Let's start with K=12 modes (indices 0 to 11).
    K = 12
    # Precompute the basis matrix B of shape (N, K)
    # B[i, k] = T_k(t[i])
    B = np.zeros((N, K))
    for k in range(K):
        B[:, k] = sp.eval_cheby(k, t)
        
    # Normalize basis columns to have unit L2 norm? Or just use raw values.
    # The optimization will handle scaling. But normalization helps with condition number.
    norms = np.linalg.norm(B, axis=0)
    norms[norms == 0] = 1.0
    B_norm = B / norms
    
    def score_from_coeffs(coeffs):
        # Reconstruct h
        h = B_norm @ coeffs
        s = np.sum(h)
        if s == 0 or not np.isfinite(s):
            return float('inf')
        # Convolution
        z = np.convolve(h, h)
        # Max of signed entries
        max_z = np.max(z)
        val = abs(2 * N * max_z / (s * s))
        if not np.isfinite(val):
            return float('inf')
        return val

    # Simulated Annealing
    # Initial guess: All zeros except a small positive bump at center?
    # Or random small values.
    # Let's try a few random initializations and keep the best.
    
    best_coeffs = np.zeros(K)
    best_score = float('inf')
    
    # Try a few deterministic starts
    starts = []
    # 1. Pure Gaussian-like (low freq dominant)
    c1 = np.zeros(K)
    c1[0] = 1.0
    c1[2] = 0.5
    starts.append(c1)
    
    # 2. With some negative lobes (higher freq)
    c2 = np.zeros(K)
    c2[0] = 1.0
    c2[4] = -0.5
    starts.append(c2)
    
    # 3. Random
    np.random.seed(42)
    c3 = np.random.randn(K) * 0.5
    starts.append(c3)
    
    # 4. Another random
    c4 = np.random.randn(K) * 0.5
    starts.append(c4)
    
    for start in starts:
        current_coeffs = start.copy()
        current_score = score_from_coeffs(current_coeffs)
        
        if current_score < best_score:
            best_score = current_score
            best_coeffs = current_coeffs.copy()
            
        T = 0.1 # Initial temperature
        T_min = 1e-6
        alpha = 0.99 # Cooling rate
        
        # Number of steps
        # We have ~115s. 
        # Score calculation is fast (convolve 400x400).
        # Let's do 2000 steps per start? 
        # 2000 * 4 = 8000 steps.
        # Each step: matrix mult (400x12 @ 12), sum, convolve (400x400).
        # Convolve is O(N^2) = 160,000 ops. Very fast in NumPy.
        # 8000 * 160k = 1.28e9 ops. Should be well under 1s.
        # We can afford more steps or more starts.
        
        steps = 5000
        for i in range(steps):
            if time.time() > deadline - 5:
                break
                
            # Perturb
            # Add Gaussian noise to coefficients
            # Scale noise by temperature
            noise_scale = T * 0.5
            new_coeffs = current_coeffs + np.random.randn(K) * noise_scale
            
            new_score = score_from_coeffs(new_coeffs)
            
            # Metropolis criterion
            if new_score < current_score:
                current_coeffs = new_coeffs
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_coeffs = current_coeffs.copy()
            else:
                # Accept with probability exp(-(new - old) / T)
                delta = new_score - current_score
                if delta < 0: # Should be caught above, but safe
                    current_coeffs = new_coeffs
                    current_score = new_score
                else:
                    prob = np.exp(-delta / T)
                    if np.random.rand() < prob:
                        current_coeffs = new_coeffs
                        current_score = new_score
                        
            T *= alpha
            if T < T_min:
                T = T_min
                
    # Final refinement: Local Nelder-Mead on the best coefficients found
    # This might polish the SA result.
    if time.time() < deadline - 10:
        try:
            def objective(coeffs):
                return score_from_coeffs(coeffs)
            
            from scipy.optimize import minimize
            res = minimize(objective, best_coeffs, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
            
            if res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x
        except Exception:
            pass
            
    # Construct final h
    h = B_norm @ best_coeffs
    
    # Ensure sum is not zero
    s = np.sum(h)
    if s == 0:
        # Fallback to seed
        return np.ones(N)
        
    return h

# EVOLVE-BLOCK-END
