"""Optimizer for BBOB-style f3 (Rastrigin) on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Model-and-optimize: fit a Rastrigin surrogate (center + offset) then optimize it."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 330s kill

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = np.zeros(dim)
        best_f = np.inf
        call_count = 0

        def call_func(x):
            nonlocal call_count, best_x, best_f
            if call_count >= budget:
                return None
            val = func(x)
            call_count += 1
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        # Strategy: Multi-start CMA-ES
        # 1. Sample several starting points to find good basins
        # 2. Run CMA-ES from the best start
        # 3. If stagnation, restart from next best start or new random
        
        n_starts = 10
        starts = []
        start_vals = []
        
        # Generate diverse starts
        # Include origin and random points
        starts.append(np.zeros(dim))
        for _ in range(n_starts - 1):
            starts.append(rng.uniform(-4.5, 4.5, dim))
            
        # Evaluate starts
        for s in starts:
            if call_count >= budget or time.time() > deadline:
                break
            val = call_func(np.clip(s, -5, 5))
            if val is None:
                break
            starts[-1] = np.clip(s, -5, 5) # Store clipped version
            start_vals.append(val)
            
        # Sort starts by fitness
        if len(start_vals) > 0:
            sorted_indices = np.argsort(start_vals)
            sorted_starts = [starts[i] for i in sorted_indices]
            sorted_vals = [start_vals[i] for i in sorted_indices]
        else:
            sorted_starts = [np.zeros(dim)]
            sorted_vals = [np.inf]

        # Run CMA-ES on the best start, with restarts
        current_start_idx = 0
        max_restarts = 5
        
        for restart in range(max_restarts):
            if call_count >= budget or time.time() > deadline:
                break
            
            # Pick start
            if current_start_idx < len(sorted_starts):
                x0 = sorted_starts[current_start_idx]
                f0 = sorted_vals[current_start_idx]
            else:
                # If we run out of pre-computed starts, generate a new random one
                x0 = rng.uniform(-4.5, 4.5, dim)
                val = call_func(np.clip(x0, -5, 5))
                if val is None:
                    break
                f0 = val

            # CMA-ES parameters
            x = x0.copy()
            sigma = 1.0  # Start with larger sigma to explore basin
            mu = 4
            lambda_ = 12
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights**2)
            
            c_sigma = (mueff + 2) / (dim + mueff + 5)
            c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
            c_1 = 2 / ((dim + 1.3)**2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
            
            C = np.eye(dim)
            C_inv_sqrt = np.eye(dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            
            gen = 0
            stagnation = 0
            max_stagnation = 10 # Restart if no improvement for 10 gens
            last_best_f = best_f
            
            while call_count < budget and time.time() < deadline and stagnation < max_stagnation:
                if gen % 10 == 0 and gen > 0:
                    try:
                        eigvals, eigvecs = np.linalg.eigh(C)
                        eigvals = np.maximum(eigvals, 1e-12)
                        C_inv_sqrt = eigvecs @ np.diag(1 / np.sqrt(eigvals)) @ eigvecs.T
                    except:
                        C_inv_sqrt = np.eye(dim)
                
                offspring = []
                off_f = []
                for k in range(lambda_):
                    if call_count >= budget or time.time() > deadline:
                        break
                    z = rng.standard_normal(dim)
                    y = x + sigma * (C_inv_sqrt @ z)
                    y = np.clip(y, -5, 5)
                    offspring.append(y)
                    fy = call_func(y)
                    if fy is None:
                        break
                    off_f.append(fy)
                
                if len(off_f) < mu:
                    break
                
                sorted_idx = np.argsort(off_f)
                selected_x = [offspring[i] for i in sorted_idx[:mu]]
                selected_f = [off_f[i] for i in sorted_idx[:mu]]
                
                # Check for improvement
                if selected_f[0] < last_best_f - 1e-10:
                    last_best_f = selected_f[0]
                    stagnation = 0
                else:
                    stagnation += 1
                
                m = sum(w * xk for w, xk in zip(weights, selected_x))
                
                diff = m - x
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (C_inv_sqrt @ diff)
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mueff) * diff
                
                h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (gen + 1)))) / chi_n
                if h_sigma < (1.4 + 2 / (dim + 1)):
                    sigma *= np.exp(c_sigma / chi_n * (h_sigma - 1))
                
                delta_d = (1 + c_mu * (mueff - 1)) * diff
                rank_mu = np.outer(delta_d, delta_d)
                p_c_norm_sq = np.linalg.norm(p_c)**2
                h_c = 1.0 if p_c_norm_sq > 1 / (1 - c_c)**2 else 0.0
                C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + h_c * np.eye(dim)) + c_mu * rank_mu
                
                x = m
                gen += 1
                
            # Move to next start for next restart
            current_start_idx += 1

        # Final fine-tuning with coordinate descent on the best point
        if call_count < budget and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            
            for step in [0.1, 0.05, 0.01, 0.005]:
                if call_count >= budget or time.time() > deadline:
                    break
                improved = True
                while improved and call_count < budget and time.time() < deadline:
                    improved = False
                    for d in range(dim):
                        if call_count >= budget or time.time() > deadline:
                            break
                        for direction in [1, -1]:
                            x_new = x.copy()
                            x_new[d] += direction * step
                            x_new = np.clip(x_new, -5, 5)
                            fx_new = call_func(x_new)
                            if fx_new is None:
                                break
                            if fx_new < fx - 1e-12:
                                x = x_new
                                fx = fx_new
                                improved = True
                                if fx < best_f:
                                    best_f = fx
                                    best_x = x.copy()
                                break
        
        return best_x, best_f
# EVOLVE-BLOCK-END
