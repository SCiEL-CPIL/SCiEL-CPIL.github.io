# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 50

def construct():
    # We maximize R = ||h||_2^2 / ||h*h||_inf
    # Equivalent to minimizing f(h) = ||h*h||_inf / ||h||_2^2
    # We parameterize h = softmax(x) to ensure h >= 0 and sum(h) = 1 automatically.
    
    def h_from_x(x):
        # Subtract max for numerical stability
        x_shifted = x - np.max(x)
        h = np.exp(x_shifted)
        h = h / np.sum(h)
        return h

    def objective_x(x):
        h = h_from_x(x)
        conv = np.convolve(h, h, mode='full')
        max_conv = np.max(conv)
        l2_sq = np.sum(h**2)
        if l2_sq < 1e-10:
            return 1e10
        return max_conv / l2_sq

    # Global Search using Differential Evolution
    # Bounds for x: 
    # Lower bound -10 allows for sparse vectors (exp(-10) ~ 4.5e-5)
    # Upper bound 0 ensures max component <= 1
    bounds = [(-10, 0)] * N
    
    best_h = np.ones(N) / N
    best_score = 0.0

    try:
        # Run Differential Evolution
        # maxiter=150, popsize=15 for better exploration within time limit
        res_de = differential_evolution(
            objective_x, 
            bounds=bounds, 
            maxiter=150, 
            popsize=15, 
            tol=1e-6, 
            seed=42,
            polish=False,
            init='sobol' # Sobol sequence for better initialization coverage
        )
        
        if res_de.success:
            best_x = res_de.x
            best_val = res_de.fun
            best_h = h_from_x(best_x)
            best_score = 1.0 / best_val if best_val > 0 else 0.0
            
            # Local Polish with Nelder-Mead
            # Nelder-Mead is robust to non-smooth functions
            local_starts = [best_x]
            
            # Add a few random starts to escape local minima
            for _ in range(10):
                local_starts.append(np.random.uniform(-10, 0, N))
            
            for x0 in local_starts:
                try:
                    res_nm = minimize(
                        objective_x, 
                        x0, 
                        method='Nelder-Mead', 
                        options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8}
                    )
                    if res_nm.success:
                        h_opt = h_from_x(res_nm.x)
                        val = objective_x(res_nm.x)
                        if val > 0:
                            score = 1.0 / val
                            if score > best_score:
                                best_score = score
                                best_h = h_opt
                except Exception:
                    pass
        else:
            # If DE fails, try a few random NM starts
            for _ in range(20):
                x0 = np.random.uniform(-10, 0, N)
                try:
                    res_nm = minimize(objective_x, x0, method='Nelder-Mead', options={'maxiter': 500})
                    if res_nm.success:
                        h_opt = h_from_x(res_nm.x)
                        val = objective_x(res_nm.x)
                        if val > 0:
                            score = 1.0 / val
                            if score > best_score:
                                best_score = score
                                best_h = h_opt
                except Exception:
                    pass

    except Exception:
        # Fallback
        best_h = np.ones(N) / N
        conv = np.convolve(best_h, best_h, mode='full')
        max_conv = np.max(conv)
        l2_sq = np.sum(best_h**2)
        best_score = l2_sq / max_conv if max_conv > 0 else 0.0

    return best_h
# EVOLVE-BLOCK-END
