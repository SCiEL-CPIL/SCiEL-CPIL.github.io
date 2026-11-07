"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100
import numpy as np
import time
import math
from scipy.optimize import minimize

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

def construct():
    """
    Discrete combinatorial optimization over binary vectors with simulated annealing.
    
    Strategy:
    1. The objective is to minimize max_k C_k(h) where C_k(h) = sum_i h_{i+k}(1-h_i).
    2. For binary h, C_k is an integer count of "1 followed by 0" pairs at distance k.
    3. We use simulated annealing on the binary space to find a good binary configuration.
    4. Then we project to the continuous space to satisfy sum(h) = N/2 exactly.
    5. Finally, we do a local continuous refinement.
    """
    start_time = time.time()
    time_limit = 140.0  # Safety margin before 150s kill

    def compute_corrs(h):
        return np.correlate(h, 1.0 - h, mode="full")

    def objective(h):
        corr = compute_corrs(h)
        return np.max(corr) / N * 2.0

    # Start with a random binary vector with sum = N/2
    def random_binary():
        h = np.zeros(N)
        indices = np.random.permutation(N)
        h[indices[:N//2]] = 1.0
        return h

    def repair_continuous(h):
        h = np.clip(h, 0.0, 1.0)
        diff = N / 2.0 - np.sum(h)
        if abs(diff) > 1e-9:
            for _ in range(100):
                h = np.clip(h, 0.0, 1.0)
                current_sum = np.sum(h)
                rem = N / 2.0 - current_sum
                if abs(rem) < 1e-9:
                    break
                if rem > 0:
                    mask = h < 1.0
                else:
                    mask = h > 0.0
                if np.sum(mask) == 0:
                    break
                h[mask] += rem / np.sum(mask)
        return h

    # Simulated annealing on binary space
    def simulated_annealing(h_init, n_iterations=50000):
        h = h_init.copy()
        current_val = objective(h)
        best_h = h.copy()
        best_val = current_val
        
        T_start = 1.0
        T_end = 0.01
        cooling_rate = (T_end / T_start) ** (1.0 / n_iterations)
        
        for i in range(n_iterations):
            if time.time() - start_time > time_limit - 10.0:
                break
            T = T_start * (cooling_rate ** i)
            
            # Random move: flip a random bit
            idx = np.random.randint(N)
            h_new = h.copy()
            h_new[idx] = 1.0 - h_new[idx]
            
            new_val = objective(h_new)
            delta = new_val - current_val
            
            if delta < 0 or np.random.random() < np.exp(-delta / max(T, 1e-10)):
                h = h_new
                current_val = new_val
                if new_val < best_val:
                    best_val = new_val
                    best_h = h_new.copy()
        
        return best_h, best_val

    # Continuous refinement using SLSQP
    def continuous_refinement(h0):
        def objective_func(x):
            h = x[:N]
            t = x[N]
            return t

        def constraints_func(x):
            h = x[:N]
            t = x[N]
            corrs = compute_corrs(h)
            return t - corrs

        def sum_constraint(x):
            h = x[:N]
            return np.sum(h) - N / 2.0

        constraints = [
            {'type': 'ineq', 'fun': constraints_func},
            {'type': 'eq', 'fun': sum_constraint}
        ]

        bounds = [(0.0, 1.0) for _ in range(N)]
        bounds.append((0.0, None))

        corrs0 = compute_corrs(h0)
        t0 = np.max(corrs0) + 0.1
        x0 = np.concatenate([h0, [t0]])

        try:
            res = minimize(
                objective_func,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 200, 'ftol': 1e-9, 'disp': False}
            )
            h_opt = repair_continuous(res.x[:N])
            return h_opt
        except Exception:
            return h0

    # Main search loop
    best_h = np.full(N, 0.5)
    best_val = objective(best_h)

    # Phase 1: Simulated annealing with multiple restarts
    n_sa_restarts = 10
    for i in range(n_sa_restarts):
        if time.time() - start_time > time_limit - 20.0:
            break
        
        h_init = random_binary()
        h_sa, val_sa = simulated_annealing(h_init, n_iterations=20000)
        
        if val_sa < best_val:
            best_val = val_sa
            best_h = h_sa.copy()
    
    # Phase 2: Continuous refinement from best binary solution
    if time.time() - start_time < time_limit - 10.0:
        h_refined = continuous_refinement(best_h)
        val_refined = objective(h_refined)
        if val_refined < best_val:
            best_val = val_refined
            best_h = h_refined.copy()
    
    # Phase 3: More continuous refinements from other good binary solutions
    n_continuous_restarts = 5
    for i in range(n_continuous_restarts):
        if time.time() - start_time > time_limit - 5.0:
            break
        
        h_init = random_binary()
        h_sa, _ = simulated_annealing(h_init, n_iterations=10000)
        h_refined = continuous_refinement(h_sa)
        val_refined = objective(h_refined)
        if val_refined < best_val:
            best_val = val_refined
            best_h = h_refined.copy()

    # Final check
    if not np.all(np.isfinite(best_h)) or np.abs(np.sum(best_h) - N/2.0) > 1e-9 * N:
        return np.full(N, 0.5)
        
    return best_h

# EVOLVE-BLOCK-END
