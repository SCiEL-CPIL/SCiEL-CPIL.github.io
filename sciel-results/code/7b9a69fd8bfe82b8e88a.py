"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    CMA-ES with restarts for BBOB f21 (Gallagher 101 peaks).
    
    Strategy:
    1. Run CMA-ES with small population (lambda=10) from random starts.
    2. Restart when stagnation is detected (no improvement for 20 generations).
    3. Use remaining budget for final refinement with (1+1)-ES.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cma_es_step(self, x, sigma, C, Cinv, pc, ps, lam, mu, weights, mu_eff, c_sigma, d_sigma, c_c, c_m, safe_call, rng, dim):
        """One CMA-ES generation: sample, evaluate, update."""
        # Sample lambda points
        B = rng.standard_normal((lam, dim))
        D = np.sqrt(np.diag(C))
        # Whitening transform: C^0.5 = Cinv^T * diag(sqrt(diag(C)))
        C05 = np.linalg.cholesky(C)
        Y = np.array([x + sigma * (B[i] @ C05) for i in range(lam)])
        Y = np.clip(Y, -5, 5)
        
        # Evaluate
        F = []
        for i in range(lam):
            fy = safe_call(Y[i])
            if fy is None:
                F.append(float('inf'))
            else:
                F.append(fy)
        
        # Sort by fitness
        idx = np.argsort(F)
        Y_sorted = Y[idx]
        F_sorted = F[idx]
        
        # Update mean
        w = weights[:mu]
        x_new = np.sum(w[:, None] * Y_sorted[:mu], axis=0)
        
        # Update covariance
        yw = np.array([(Y_sorted[i] - x) / sigma for i in range(mu)])
        C = (1 - c_m) * C + c_m * (np.outer(yw.T @ w, yw) + (1 - mu_eff * c_c) * np.outer(pc, pc))
        C = (C + C.T) / 2  # Symmetrize
        
        # Update pc
        pc = (1 - c_c) * pc + c_c * np.sqrt(mu_eff) * yw.T @ w
        
        # Update ps
        ps = (1 - c_sigma) * ps + c_sigma * np.sqrt(mu_eff) * (C05.T @ yw.T @ w)
        
        # Update sigma
        hsig = min(1, np.sqrt((dim+1.7)/(dim-1)) * np.linalg.norm(ps) / np.sqrt(1-(1-c_sigma)**2) - 1)
        sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / np.sqrt(2*dim) - 1))
        sigma = max(sigma, 1e-10)
        sigma = min(sigma, 5.0)
        
        return x_new, sigma, C, pc, ps, F_sorted[0]

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 305.0
        budget = self.budget
        dim = self.dim

        def check_time():
            return time.time() < deadline

        calls_made = 0
        best_x = None
        best_f = float('inf')

        def safe_call(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return None
            try:
                val = float(func(x))
            except Exception:
                return None
            calls_made += 1
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        rng = np.random.default_rng(42)
        
        # CMA-ES parameters
        lam = 10
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_m = (2 + 2 * mu_eff / dim) / (dim + 1.5 + 2 * mu_eff / dim)
        
        # Phase 1: CMA-ES with restarts
        max_restarts = 5
        gens_per_restart = 20  # Restart if no improvement for this many generations
        
        for restart in range(max_restarts):
            if not check_time() or calls_made >= budget:
                break
            
            # Initialize CMA-ES state
            if restart == 0:
                x = rng.uniform(-4, 4, dim)
            else:
                # Perturb from best found
                x = best_x + 1.0 * rng.standard_normal(dim)
                x = np.clip(x, -5, 5)
            
            sigma = 1.0
            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            
            best_in_run = float('inf')
            no_improve_gens = 0
            
            for gen in range(lam * gens_per_restart):  # Max gens for this run
                if not check_time() or calls_made >= budget:
                    break
                
                x, sigma, C, pc, ps, f_best_gen = self._cma_es_step(
                    x, sigma, C, Cinv if (Cinv := np.linalg.inv(C)) is not None else np.eye(dim),
                    pc, ps, lam, mu, weights, mu_eff, c_sigma, d_sigma, c_c, c_m,
                    safe_call, rng, dim
                )
                
                if f_best_gen < best_in_run:
                    best_in_run = f_best_gen
                    no_improve_gens = 0
                else:
                    no_improve_gens += 1
                
                if no_improve_gens >= gens_per_restart:
                    break
            
            # Update global best
            if best_in_run < best_f:
                best_f = best_in_run
        
        # Phase 2: Final refinement with (1+1)-ES
        if best_x is not None and check_time() and calls_made < budget:
            x = best_x.copy()
            fx = best_f
            
            sigma = 0.3
            no_improve = 0
            max_no_improve = 50
            total_steps = 0
            success_count = 0
            window = 10
            
            while check_time() and calls_made < budget:
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = safe_call(y)
                if fy is None:
                    break
                
                total_steps += 1
                
                if fy < fx:
                    x, fx = y, fy
                    success_count += 1
                    no_improve = 0
                    sigma *= 1.05
                else:
                    sigma *= 0.95
                    no_improve += 1
                
                if total_steps >= window:
                    success_rate = success_count / total_steps
                    if success_rate > 0.25:
                        sigma *= 1.15
                    elif success_rate < 0.15:
                        sigma *= 0.85
                    success_count = 0
                    total_steps = 0
                
                sigma = max(sigma, 1e-5)
                sigma = min(sigma, 2.0)
                
                if no_improve > max_no_improve:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
