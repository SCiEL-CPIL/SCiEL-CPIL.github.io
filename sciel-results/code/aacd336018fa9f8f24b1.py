import numpy as np
import time
import random
from scipy.optimize import minimize_scalar

def construct():
    N = 24
    start_time = time.time()
    max_time = 115.0
    
    # Grid for fast evaluation
    M_fast = 1024
    theta_fast = np.linspace(0, 2 * np.pi, M_fast, endpoint=False)
    z_fast = np.exp(1j * theta_fast)
    V_fast = np.vander(z_fast, N=N, increasing=True)
    
    # Grid for exact evaluation (can be same or finer, let's use 4096 for safety in peak detection)
    M_exact = 4096
    theta_exact = np.linspace(0, 2 * np.pi, M_exact, endpoint=False)
    z_exact = np.exp(1j * theta_exact)
    V_exact = np.vander(z_exact, N=N, increasing=True)
    
    def fast_sup_norm(c):
        """
        Fast approximation of sup norm using coarse grid.
        Used for SA acceptance criteria.
        """
        p_vals = V_fast @ c
        return np.max(np.abs(p_vals))

    def exact_sup_norm(c):
        """
        Compute the maximum of |P(z)| for |z|=1 with high accuracy.
        Uses fine grid for peak location, then minimization.
        """
        # 1. Evaluate on fine grid
        p_vals = V_exact @ c
        mags = np.abs(p_vals)
        idx_max = np.argmax(mags)
        max_val = mags[idx_max]
        
        # 2. Refine the peak
        h = 2 * np.pi / M_exact
        theta_peak = theta_exact[idx_max]
        
        def obj(t):
            z_loc = np.exp(1j * (theta_peak + t))
            # Compute P(z_loc)
            # Using dot product with powers is fine for N=24
            z_powers = np.ones(N, dtype=complex)
            for k in range(1, N):
                z_powers[k] = z_powers[k-1] * z_loc
            val = np.dot(c, z_powers)
            return -np.abs(val)
        
        res = minimize_scalar(obj, bounds=(-h, h), method='bounded', options={'xatol': 1e-10})
        
        if res.success:
            refined_val = -res.fun
            return max(max_val, refined_val)
        else:
            return max_val

    # Simulated Annealing
    # Initialize
    c = np.random.choice([-1, 1], N).astype(float)
    curr_fast = fast_sup_norm(c)
    best_c = c.copy()
    best_exact = exact_sup_norm(c)
    best_fast = curr_fast
    
    # SA Parameters
    T = 1.0
    cooling_rate = 0.995
    min_T = 1e-5
    
    # We run until time limit
    while time.time() - start_time < max_time:
        # Pick random index to flip
        k = np.random.randint(N)
        c[k] *= -1
        
        new_fast = fast_sup_norm(c)
        delta_E = new_fast - curr_fast
        
        # Acceptance criterion
        accept = False
        if delta_E < 0:
            accept = True
        else:
            if T > 0:
                prob = np.exp(-delta_E / T)
                if np.random.rand() < prob:
                    accept = True
        
        if accept:
            curr_fast = new_fast
            # If this is a new best on the fast grid, check exact
            if new_fast < best_fast:
                best_fast = new_fast
                exact_val = exact_sup_norm(c)
                if exact_val < best_exact:
                    best_exact = exact_val
                    best_c = c.copy()
                    # Early exit if we hit the known optimum closely
                    if best_exact < 1.258376:
                        break
        else:
            # Revert
            c[k] *= -1
            
        # Cool down
        T *= cooling_rate
        if T < min_T:
            T = min_T
            # If we are stuck at min_T, we might want to restart or just continue
            # For simplicity, we continue with min_T which effectively becomes greedy
            # but with random moves.
            
    return [int(x) for x in best_c]
