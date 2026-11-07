"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-start CMA-ES with adaptive restarts for BBOB f18."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 325.0  # 5s safety margin
        
        best_x = None
        best_f = np.inf
        total_calls = 0
        n = self.dim
        
        # Standard CMA-ES parameters
        lam = max(4, int(4 + 3 * np.log(n)))
        # Ensure lambda is not too large for small budgets
        if lam > self.budget // 10:
            lam = max(2, self.budget // 10)
            
        # IPOP-CMA-ES parameters
        sigma_init = 2.0
        max_generations = 100
        min_sigma = 1e-8
        sigma_scale = 0.5
        max_sigma = 10.0
        
        # Success rate threshold for stagnation detection
        success_rate_window = 10
        success_rate_threshold = 0.2
        
        def run_cma_es(x0, sigma0, remaining_budget, current_calls, current_best_x, current_best_f):
            nonlocal total_calls, best_x, best_f
            
            # Initialize CMA-ES state
            x = x0.copy()
            sigma = sigma0
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            
            # CMA-ES parameters
            mu = lam // 2
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w = w / np.sum(w)
            mu_eff = 1.0 / np.sum(w**2)
            
            c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
            c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            c_1 = 2 / ((n + 1.3)**2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))
            
            # Precompute B and D for initial C
            D, B = np.linalg.eigh(C)
            D = np.sqrt(D)
            
            # History for success rate
            success_history = []
            
            gen = 0
            while total_calls < self.budget and time.time() < deadline:
                if self.budget - total_calls < lam:
                    break
                    
                # Sample population
                ys = []
                fys = []
                for i in range(lam):
                    z = rng.standard_normal(n)
                    y = x + sigma * (B @ (D * z))
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    total_calls += 1
                    ys.append(y)
                    fys.append(fy)
                    
                    if fy < current_best_f:
                        current_best_f = fy
                        current_best_x = y.copy()
                        
                # Sort by fitness
                idx = np.argsort(fys)
                ys = np.array(ys)[idx]
                fys = np.array(fys)[idx]
                
                # Update best
                if fys[0] < current_best_f:
                    current_best_f = fys[0]
                    current_best_x = ys[0].copy()
                    
                # Update mean
                y_mean = np.sum(w[:, np.newaxis] * ys[:mu], axis=0)
                
                # Check stagnation: if best of last 'success_rate_window' generations hasn't improved significantly
                # Simple check: if current best is not better than best from 'success_rate_window' steps ago
                # For IPOP, we usually restart if sigma becomes too small or progress is slow.
                # Let's use a simple progress check: if no improvement in last K generations
                # We'll track best so far in this CMA-ES run
                if gen == 0:
                    local_best = fys[0]
                else:
                    if fys[0] < local_best - 1e-10:
                        local_best = fys[0]
                        last_improve_gen = gen
                    elif gen - last_improve_gen > 20: # No improvement for 20 generations
                        break
                        
                # Update step size
                y_sigma = (y_mean - x) / sigma
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma
                chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (24 * n**2))
                
                if np.linalg.norm(ps) / chi_n < (1.4 + 2 / (n + 1)) * np.sqrt((1 - (1 - c_sigma)**(gen + 1))):
                    sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
                    
                # Update covariance
                pc = (1 - c_c) * pc + c_c * (y_mean - x) / sigma
                C = (1 + c_1 * c_mu - c_1) * C + c_1 * np.outer(pc, pc)
                for i in range(mu):
                    zi = (ys[i] - x) / sigma
                    C += c_mu * w[i] * np.outer(zi, zi)
                    
                x = y_mean
                sigma *= sigma_scale # IPOP scaling? No, IPOP restarts with scaled sigma. 
                # Wait, standard CMA-ES doesn't scale sigma every gen like this. 
                # IPOP-CMA-ES restarts with sigma = sigma * 0.5 when stagnation is detected.
                # Let's remove the sigma_scale from the main loop and handle it in restart logic.
                sigma /= sigma_scale # Undo the scale for the main loop logic, we'll handle restart separately
                
                # Re-eigen-decompose C periodically (e.g., every 10 gens) to save time
                if gen % 10 == 0:
                    D, B = np.linalg.eigh(C)
                    D = np.sqrt(np.maximum(D, 0)) # Ensure non-negative
                    
                gen += 1
                
            return current_best_x, current_best_f

        rng = np.random.default_rng(0)
        
        # Initial best
        x0 = np.zeros(n)
        f0 = func(x0)
        total_calls += 1
        best_x = x0
        best_f = f0
        
        current_sigma = sigma_init
        current_x0 = x0.copy()
        
        while total_calls < self.budget and time.time() < deadline:
            # Run one CMA-ES instance
            best_x, best_f = run_cma_es(current_x0, current_sigma, self.budget - total_calls, total_calls, best_x, best_f)
            
            # Check if we should restart
            # IPOP-CMA-ES restart condition: if sigma is very small or no progress
            # Here we simply restart with a smaller sigma and a new random start if we haven't converged
            # A simple heuristic: if we used up a significant chunk of budget, restart
            # Let's restart every 100 calls or if sigma becomes too small (handled inside by break)
            
            # For IPOP, we restart with sigma = sigma * 0.5 and a new random center
            # But we need to be careful not to waste budget on bad starts.
            # Let's try a few restarts.
            if total_calls >= self.budget:
                break
                
            # If we broke out of run_cma_es due to stagnation, restart with smaller sigma
            # If we broke out due to budget, we are done.
            
            # Simple restart strategy:
            # 1. New random start point
            # 2. Smaller sigma
            # 3. Limit number of restarts to avoid overhead
            
            if total_calls < self.budget - lam: # Need at least one more generation
                current_x0 = rng.uniform(-5, 5, n)
                f_check = func(current_x0)
                total_calls += 1
                if f_check < best_f:
                    best_f = f_check
                    best_x = current_x0.copy()
                    
                current_sigma *= 0.5
                if current_sigma < min_sigma:
                    break
            else:
                break
                
        return best_x, best_f

# EVOLVE-BLOCK-END
