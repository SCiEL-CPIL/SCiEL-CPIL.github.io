"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """
    Optimizer for BBOB-style f18 (Schaffers F7 ill-conditioned) on [-5,5]^5.
    
    Strategy: Population-based global exploration with CMA-ES local refinement.
    - Maintains a diverse population to explore the rugged landscape
    - Uses CMA-ES for local refinement around promising points
    - Stagnation detection triggers new exploration
    - Adaptive budget allocation between exploration and exploitation
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(12345)
        dim = self.dim
        budget = self.budget
        
        best_x = None
        best_f = np.inf
        calls_used = 0
        
        def evaluate(x):
            nonlocal calls_used, best_f, best_x
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx
        
        # Phase 1: Broad initial sampling to understand landscape
        n_initial = min(50, budget // 10)
        points = []
        for i in range(n_initial):
            x = rng.uniform(-5, 5, dim)
            fx = evaluate(x)
            points.append((x, fx))
        
        # Sort by fitness
        points.sort(key=lambda t: t[1])
        
        # Phase 2: Iterative improvement with CMA-ES on top points
        # Keep a pool of candidate starting points
        n_candidates = min(5, len(points))
        candidates = [p[0].copy() for p in points[:n_candidates]]
        
        # CMA-ES parameters
        def cma_es_local(start_x, local_budget, sigma_init=1.0):
            nonlocal calls_used, best_f, best_x
            
            if calls_used >= budget - 5:
                return start_x, best_f
            
            # CMA-ES state
            x = start_x.copy()
            sigma = sigma_init
            C = np.eye(dim)
            B = np.eye(dim)
            D = np.ones(dim)
            pc = np.zeros(dim)
            cs = 1.0 + 2 * max(0, (np.sqrt(dim) - 1) / (dim + 3))
            cc = (4 + dim / 4) / (dim + 4 + 2 * cs)
            cmu = min(0.5, 2 / ((dim + 1)**2 + 1))
            lam = max(4, int(4 + 3 * dim))
            weights = np.log(np.arange(lam, 0, -1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights[:lam])**2
            alpha_mu = 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) * (mueff + 2 - 2 / mueff))
            alpha_sigma = (mueff + 2) / (dim + mueff + 5)
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
            
            population = []
            for i in range(lam):
                z = rng.standard_normal(dim)
                y = B @ (D * z)
                xi = x + sigma * y
                population.append((xi, z, y))
            
            iters = 0
            max_iters = local_budget // lam
            
            while calls_used < budget - 5 and iters < max_iters:
                # Evaluate population
                pop_f = []
                for xi, z, y in population:
                    xi_clipped = np.clip(xi, -5, 5)
                    fi = func(xi_clipped)
                    calls_used += 1
                    pop_f.append((xi_clipped, fi, z, y))
                    if fi < best_f:
                        best_f = fi
                        best_x = xi_clipped.copy()
                
                # Sort by fitness
                pop_f.sort(key=lambda t: t[1])
                
                # Update mean
                x_new = np.zeros(dim)
                for i in range(lam):
                    x_new += weights[i] * pop_f[i][0]
                x = x_new
                
                # Update covariance
                delta_x = (x - start_x) / sigma if sigma > 0 else np.zeros(dim)
                pc = (1 - cc) * pc + cc * delta_x
                C = (1 - (cmu + 2 * cc * (1 - cc))) * C \
                    + cmu * np.outer(delta_x, delta_x) \
                    + 2 * cc * (1 - cc) * np.outer(pc, pc)
                
                # Update sigma
                sigma *= np.exp((alpha_sigma / chi_n) * (np.linalg.norm(pc) / np.sqrt(1 - (1 - cc)**2) - 1))
                sigma = np.clip(sigma, 1e-10, 10.0)
                
                # Recompute B and D
                D, B = np.linalg.eigh(C)
                D = np.sqrt(np.maximum(D, 1e-12))
                B = B[:, ::-1]
                D = D[::-1]
                
                # Generate new population
                population = []
                for i in range(lam):
                    z = rng.standard_normal(dim)
                    y = B @ (D * z)
                    xi = x + sigma * y
                    population.append((xi, z, y))
                
                iters += 1
                
                # Stagnation check
                if iters % 5 == 0 and calls_used < budget - 5:
                    recent_best = min(p[1] for p in pop_f[:3])
                    if abs(recent_best - best_f) < 1e-8:
                        # No improvement in recent iterations, break
                        break
            
            return x, best_f
        
        # Phase 2: CMA-ES on top candidates
        remaining_budget = budget - calls_used - 10
        n_cma_runs = min(3, n_candidates)
        budget_per_cma = remaining_budget // n_cma_runs
        
        for i in range(n_cma_runs):
            if calls_used >= budget - 5:
                break
            
            start = candidates[i].copy()
            sigma_init = 1.5 if i == 0 else 1.0
            cma_es_local(start, budget_per_cma, sigma_init)
        
        # Phase 3: Final polish with fine-grained search
        if best_x is not None and calls_used < budget:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            remaining = budget - calls_used
            
            for _ in range(remaining):
                if calls_used >= budget:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_used += 1
                
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.05
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.95
            
            # One more round with even smaller steps
            if calls_used < budget:
                sigma = 0.01
                for _ in range(min(20, budget - calls_used)):
                    if calls_used >= budget:
                        break
                    y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                    fy = func(y)
                    calls_used += 1
                    
                    if fy < fx:
                        x, fx = y, fy
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
        
        return best_x, best_f
# EVOLVE-BLOCK-END
