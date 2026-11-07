"""Rewritten optimizer: CMA-ES with restarts and (1+1)-ES polish."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """CMA-ES optimizer for BBOB f13 (Sharp Ridge)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t0 = time.time()
        deadline = t0 + 295.0  # Safety margin

        def elapsed_ok():
            return time.time() < deadline

        rng = np.random.default_rng(12345)
        best_x = None
        best_f = np.inf
        evaluated = 0

        def evaluate(x):
            nonlocal evaluated, best_f, best_x
            if evaluated >= budget or not elapsed_ok():
                return np.inf
            x_clipped = np.clip(x, -5.0, 5.0)
            f = func(x_clipped)
            evaluated += 1
            if f < best_f:
                best_f = f
                best_x = x_clipped.copy()
            return f

        # Phase 1: Initial Sampling
        # Use 5 random starts to locate the basin
        n_init = 5
        init_xs = rng.uniform(-4.0, 4.0, size=(n_init, dim))
        init_xs = np.vstack([init_xs, np.zeros(dim)])
        init_fs = []
        for x in init_xs:
            f = evaluate(x)
            init_fs.append(f)
        
        if best_x is None:
            return None, np.inf

        # Phase 2: CMA-ES
        # Parameters
        lambda_ = 4 + int(3 * np.log(dim))  # ~12 for dim=5
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * ((mu_eff - 2 + 1. / mu_eff) / ((dim + 2)**2 + mu_eff)))
        chi_n = np.sqrt(dim) * (1 - 1. / (4 * dim) + 1. / (21 * dim**2))
        
        max_restarts = 5
        restarts = 0
        
        while evaluated < budget and elapsed_ok() and restarts < max_restarts:
            # Initialize CMA-ES state
            mean = best_x.copy()
            sigma = 0.5  # Initial step size
            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            
            # Run CMA-ES for a fixed number of generations or until convergence
            gen = 0
            max_gen = 40  # ~40 * 12 = 480 evals, might need to adjust
            # We have budget - evaluated remaining. 
            # Let's run until budget is low or converged.
            
            while evaluated < budget and elapsed_ok():
                gen += 1
                # Generate population
                zs = rng.multivariate_normal(np.zeros(dim), C, size=lambda_)
                ys = mean + sigma * zs
                fs = np.array([evaluate(y) for y in ys])
                
                if np.all(np.isinf(fs)):
                    break
                
                # Sort indices
                indices = np.argsort(fs)
                ys = ys[indices]
                fs = fs[indices]
                
                # Update mean
                new_mean = np.sum(weights[:, None] * ys[:mu], axis=0)
                
                # Update ps
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (new_mean - mean)
                
                # Update pc
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * (gen + 1)))) < (1.4 + 2 / (dim + 1)) * chi_n
                pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mu_eff) * (new_mean - mean) / sigma
                
                # Update C
                C = (1 + c_1 * hsig * (c_c + 1 - c_c) - c_1 - c_mu) * C \
                    + c_1 * np.outer(pc, pc) \
                    + c_mu / sigma**2 * np.sum(weights[:, None, None] * np.outer(ys[:mu] - mean, ys[:mu] - mean), axis=0)
                
                # Ensure C is symmetric and positive definite
                C = (C + C.T) / 2
                # Eigenvalue check for stability
                eigvals = np.linalg.eigvalsh(C)
                if np.any(eigvals < 1e-12):
                    C = C + 1e-12 * np.eye(dim)
                
                # Update sigma
                sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
                
                # Update mean
                mean = new_mean
                
                # Check convergence
                # 1. Step size too small
                # 2. No improvement in last 10 generations
                # 3. Population collapsed
                if sigma < 1e-8:
                    break
                if gen > 10 and fs[0] >= fs[0] and np.std(ys[:mu]) < 1e-6:
                    break
                if gen % 10 == 0 and np.max(fs) - np.min(fs) < 1e-6:
                    break
            
            # If we exited the inner loop without using all budget, restart
            if evaluated < budget and elapsed_ok():
                # Try a new restart from the best found so far with a larger sigma
                restarts += 1
                # If we are stuck, maybe try a random restart?
                # For now, just continue with a new CMA-ES instance from best_x
                # But if sigma collapsed, we might be at a local min.
                # Let's try a random restart if we haven't improved much
                if best_f > -10: # Heuristic: if value is not very low, try random
                    x_new = rng.uniform(-4, 4, size=dim)
                    f_new = evaluate(x_new)
                    if f_new < best_f:
                        best_x = x_new.copy()
                        best_f = f_new
        
        # Phase 3: (1+1)-ES Polish
        remaining = budget - evaluated
        if remaining > 0 and best_x is not None and elapsed_ok():
            x = best_x.copy()
            fx = best_f
            sigma = 0.01
            successes = 0
            window = 0
            WIN = 5
            for _ in range(remaining):
                if not elapsed_ok() or evaluated >= budget:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5.0, 5.0)
                fy = func(y)
                evaluated += 1
                window += 1
                if fy < fx:
                    x, fx = y, fy
                    successes += 1
                    if fx < best_f:
                        best_f, best_x = fx, x.copy()
                if window >= WIN:
                    if successes / window > 0.2:
                        sigma *= 1.2
                    else:
                        sigma *= 0.8
                    sigma = float(np.clip(sigma, 1e-8, 0.1))
                    successes = 0
                    window = 0

        return best_x, best_f
# EVOLVE-BLOCK-END
