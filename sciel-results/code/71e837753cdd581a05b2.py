"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES with periodicity-aware restarts for BBOB f15 (Rastrigin rotated)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit
        
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        # CMA-ES parameters
        # For dim=5, lambda is usually 4 + floor(3*log(dim)) = 4 + 3*1.6 = 8
        # Let's use lambda=10 for slightly more exploration
        lambda_ = 10
        mu = lambda_ // 2
        
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= np.sum(w)
        mu_eff = 1.0 / np.sum(w**2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # Initial sigma
        # Rastrigin has local minima every 2 units. Box is [-5, 5].
        # Starting sigma should be large enough to cover a few periods but not the whole box initially.
        # 2.0 seems reasonable to explore one period width.
        sigma0 = 2.0
        
        best_x_global = None
        best_y_global = np.inf
        
        # Multi-start strategy:
        # We will run CMA-ES for a certain number of evaluations, then restart if stagnation.
        # Given budget=500, we can afford ~5-10 restarts.
        
        # To handle the "hidden rotation", CMA-ES adapts its covariance matrix to the shape of the function.
        # The "periodicity-aware" part: if we stagnate, we don't just random restart.
        # We try to jump by a multiple of the expected period (2.0) in random directions, 
        # or simply restart with a larger sigma centered on the best found so far.
        
        # Strategy:
        # 1. Start at origin (or random) with sigma=2.0.
        # 2. Run CMA-ES for `steps_per_phase` evaluations.
        # 3. If no improvement in last `stagnation_window` steps, restart.
        #    Restart center = current best.
        #    Restart sigma = previous_sigma * 1.5 or fixed larger value.
        
        steps_per_phase = 50
        stagnation_window = 20
        
        x_mean = np.zeros(dim)
        # Initial mean: random point in [-2, 2] to avoid boundary issues initially?
        # Or just 0.0. Since minimizer is in [-4, 4], 0 is safe.
        x_mean = rng.uniform(-2, 2, dim)
        
        C = np.eye(dim)
        sigma = sigma0
        
        # State variables
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Precompute
        hs = np.sqrt(1 - np.exp(-2 / (dim + 1)))
        hs = np.sqrt(1 - np.exp(-2 / (dim + 1))) # Correction: hs for sigma update
        # Actually, standard CMA-ES formulas:
        # h_sigma = 1 - exp(-1 / (dim + 1)) * (1 - ... ) ... it's complex.
        # Let's use a simplified robust version or standard library if available?
        # scipy.optimize.differential_evolution is available but might be slow or not adaptive enough for this specific structure.
        # Let's implement a robust CMA-ES.
        
        # For simplicity and robustness in a short script, we can use a simplified CMA-ES
        # or rely on the fact that for dim=5, a well-tuned random search with local refinement (Nelder-Mead) 
        # might compete, but CMA-ES is generally superior for rotated landscapes.
        
        # Let's implement a standard CMA-ES step.
        
        def cma_step(func, x_mean, C, sigma, p_sigma, p_c, budget_remaining):
            nonlocal best_x_global, best_y_global
            
            # Generate population
            D = np.sqrt(np.diag(C))
            B = np.linalg.inv(np.sqrtm(C)) if not np.allclose(C, np.eye(dim)) else np.eye(dim)
            # Actually, we need B such that B @ B.T = C.
            # Using eigendecomposition for stability
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            B = eigvecs @ np.diag(np.sqrt(eigvals))
            
            z = rng.standard_normal((lambda_, dim))
            pop = x_mean + sigma * (z @ B.T)
            
            # Clip to box? The task says func clips, but we should probably keep points reasonable.
            # If we go out of bounds, func clips. This is fine.
            
            fvals = np.array([func(x) for x in pop])
            
            # Sort by fitness
            order = np.argsort(fvals)
            pop = pop[order]
            fvals = fvals[order]
            
            # Update best global
            if fvals[0] < best_y_global:
                best_y_global = fvals[0]
                best_x_global = pop[0].copy()
            
            # Update mean
            x_new = np.sum(pop[:mu] * w[:, None], axis=0)
            
            # Update step size control variables
            y_sigma = (B.T @ (x_new - x_mean)) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma
            
            # Hessian of the norm of p_sigma
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * budget_remaining / (dim + 1))) < (1.4 + 2 / (dim + 1)) * np.trace(np.eye(dim)))
            
            # Update path evolution path
            y_c = (B.T @ (x_new - x_mean)) / sigma
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_c
            
            # Update covariance matrix
            C = (1 + c_1 * (h_sigma - 1) + c_mu) * C \
                - c_1 * (p_c @ p_c - np.eye(dim)) \
                - c_mu * np.sum([np.outer(pop[i] - x_mean, pop[i] - x_mean) for i in range(mu)], axis=0) * w[i] # This line is wrong syntax/logic
            
            # Correct C update:
            delta_h = h_sigma * (1 - c_c)
            rank_one = delta_h * (p_c @ p_c - np.eye(dim))
            rank_mu = np.sum([w[i] * np.outer(pop[i] - x_mean, pop[i] - x_mean) for i in range(mu)], axis=0)
            C = (1 + c_1 * (h_sigma - 1) + c_mu - c_1) * C + c_1 * (p_c @ p_c) + c_mu * rank_mu
            # Wait, the standard update is:
            # C = (1 - c_1 - c_mu) * C + c_1 * (p_c @ p_c) + c_mu * sum(w_i * (x_i - x_old)(x_i - x_old)^T)
            # Let's re-implement carefully.
            
            # Re-calculate C update correctly
            C_new = (1 - c_1 - c_mu) * C
            C_new += c_1 * np.outer(p_c, p_c)
            for i in range(mu):
                C_new += c_mu * w[i] * np.outer(pop[i] - x_mean, pop[i] - x_mean)
            
            # Ensure symmetry and positive definiteness
            C_new = (C_new + C_new.T) / 2.0
            # Add jitter if needed
            try:
                np.linalg.cholesky(C_new)
            except np.linalg.LinAlgError:
                C_new += 1e-6 * np.eye(dim)
            
            # Update sigma
            sigma = sigma * np.exp(c_sigma / dsigma * (np.linalg.norm(p_sigma) / dsigma - 1))
            # dsigma = 1 + 2*max(0, (mu_eff-dim-1)/(dim**2+dim-1))
            dsigma = 1 + 2 * max(0, (mu_eff - dim - 1) / (dim**2 + dim - 1))
            sigma = sigma * np.exp(c_sigma / dsigma * (np.linalg.norm(p_sigma) / dsigma - 1))
            
            return x_new, C_new, sigma, p_sigma, p_c, fvals[0]

        # The above inline CMA-ES implementation is prone to bugs in this short format.
        # A more robust approach for this specific task (dim=5, budget=500) is to use
        # a simpler, highly robust variant or a different algorithm entirely if CMA-ES is too complex to write correctly from scratch in this context without testing.
        
        # Alternative: Use `scipy.optimize.minimize` with Nelder-Mead starting from multiple points?
        # Or a simple Random Search with Local Refinement?
        
        # Let's try a "Multi-start Nelder-Mead" approach. It's simpler to implement correctly.
        # Rastrigin is multimodal. Nelder-Mead is good for local minima.
        # Strategy:
        # 1. Generate K random starting points.
        # 2. Run Nelder-Mead from each, with a small budget per point.
        # 3. If time remains, re-run Nelder-Mead from the best found points with tighter bounds or more iterations.
        
        # However, the directive asks for CMA-ES. Let's try to implement a minimal, robust CMA-ES.
        
        # Minimal CMA-ES
        def run_cmaes(x0, sigma_init, max_evals):
            nonlocal best_x_global, best_y_global
            nonlocal budget_remaining_global # Need to track global budget
            
            # Initialize
            x_mean = x0.copy()
            C = np.eye(dim)
            sigma = sigma_init
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            
            # Parameters
            lambda_ = 10
            mu = 5
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w /= np.sum(w)
            mu_eff = 1.0 / np.sum(w**2)
            
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((dim + 2)**2 + mu_eff))
            
            dsigma = 1 + 2 * max(0, (mu_eff - dim - 1) / (dim**2 + dim - 1))
            
            evals_used = 0
            best_local_y = np.inf
            best_local_x = None
            
            while evals_used < max_evals:
                # Generate population
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                B = eigvecs @ np.diag(np.sqrt(eigvals))
                
                z = rng.standard_normal((lambda_, dim))
                pop = x_mean + sigma * (z @ B.T)
                
                # Evaluate
                fvals = np.array([func(x) for x in pop])
                evals_used += lambda_
                
                # Update global best
                if fvals[0] < best_y_global:
                    best_y_global = fvals[0]
                    best_x_global = pop[np.argmin(fvals)].copy()
                
                if fvals[0] < best_local_y:
                    best_local_y = fvals[0]
                    best_local_x = pop[np.argmin(fvals)].copy()
                
                # Sort
                order = np.argsort(fvals)
                pop = pop[order]
                fvals = fvals[order]
                
                # Update mean
                x_new = np.sum(pop[:mu] * w[:, None], axis=0)
                
                # Update paths
                y_sigma = (B.T @ (x_new - x_mean)) / sigma
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma
                
                y_c = (B.T @ (x_new - x_mean)) / sigma
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_c
                
                # Hessian check for sigma update
                # Simplified: always update sigma
                sigma = sigma * np.exp(c_sigma / dsigma * (np.linalg.norm(p_sigma) / dsigma - 1))
                
                # Update Covariance
                C = (1 - c_1 - c_mu) * C + c_1 * np.outer(p_c, p_c)
                for i in range(mu):
                    C += c_mu * w[i] * np.outer(pop[i] - x_mean, pop[i] - x_mean)
                
                C = (C + C.T) / 2.0
                # Regularize
                try:
                    np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C += 1e-6 * np.eye(dim)
                
                x_mean = x_new
                
                # Check for stagnation in local run
                if evals_used > 20 and fvals[0] >= best_local_y - 1e-5:
                    # If we haven't improved in the last few steps, break and restart
                    break
                
            return best_local_x, best_local_y

        # Main Loop: Multi-start with adaptive restarts
        # We divide the budget into phases.
        # Phase 1: Broad exploration.
        # Phase 2: Local refinement around best found.
        
        # Since we don't know the function, we use a heuristic:
        # Start with 5 random points. Run CMA-ES for 40 evals each (200 total).
        # Then take the best 2 points and run CMA-ES for 50 evals each (100 total).
        # Then take the best 1 point and run CMA-ES for 100 evals (100 total).
        # Total: 400 evals. 100 left for final polish or random search.
        
        # Let's make it dynamic based on time and stagnation.
        
        # Initial random points
        n_starts = 5
        starts = [rng.uniform(-4, 4, dim) for _ in range(n_starts)]
        
        # Allocate budget
        # 500 total.
        # 5 starts * 40 evals = 200.
        # 2 best * 50 evals = 100.
        # 1 best * 100 evals = 100.
        # 100 remaining for "periodicity jump" or final Nelder-Mead.
        
        # To satisfy "periodicity-aware restart", we will check if the best value is close to the theoretical minimum of Rastrigin (which is 0, but offset/rotated).
        # We don't know the offset. But we know the landscape has period 2.
        # If we stagnate, we try shifting the center by a random vector of magnitude ~2.
        
        # Let's implement the loop.
        
        remaining_budget = budget
        
        # Phase 1
        for i in range(n_starts):
            if time.time() > deadline or remaining_budget < 40:
                break
            x0 = starts[i]
            best_x, best_y = run_cmaes(x0, 2.0, 40)
            remaining_budget -= 40
            
        # Phase 2: Refine top 2
        # We need to track the best points from Phase 1. 
        # My run_cmaes updates global best, but doesn't return a list of all good points.
        # Let's modify strategy: just run CMA-ES from the current global best, but with a larger sigma if we suspect we are in a local min.
        
        # Simpler, more robust approach for this specific constraint:
        # Use a single CMA-ES instance but with "restarts" that expand the search radius.
        
        # Let's restart the whole logic with a cleaner, single-instance CMA-ES that adapts sigma on stagnation.
        
        # Reset state
        best_x_global = None
        best_y_global = np.inf
        remaining_budget = budget
        
        x_mean = rng.uniform(-4, 4, dim)
        C = np.eye(dim)
        sigma = 2.0
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        lambda_ = 10
        mu = 5
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= np.sum(w)
        mu_eff = 1.0 / np.sum(w**2)
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((dim + 2)**2 + mu_eff))
        dsigma = 1 + 2 * max(0, (mu_eff - dim - 1) / (dim**2 + dim - 1))
        
        last_improvement_call = 0
        current_call = 0
        stagnation_threshold = 30 # If no improvement in 30 calls, restart logic
        
        while current_call < remaining_budget and time.time() < deadline:
            # Generate population
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            B = eigvecs @ np.diag(np.sqrt(eigvals))
            
            z = rng.standard_normal((lambda_, dim))
            pop = x_mean + sigma * (z @ B.T)
            
            # Evaluate
            fvals = np.array([func(x) for x in pop])
            current_call += lambda_
            
            # Update global best
            min_idx = np.argmin(fvals)
            if fvals[min_idx] < best_y_global:
                best_y_global = fvals[min_idx]
                best_x_global = pop[min_idx].copy()
                last_improvement_call = current_call
            
            # Sort
            order = np.argsort(fvals)
            pop = pop[order]
            fvals = fvals[order]
            
            # Update mean
            x_new = np.sum(pop[:mu] * w[:, None], axis=0)
            
            # Update paths
            y_sigma = (B.T @ (x_new - x_mean)) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma
            
            y_c = (B.T @ (x_new - x_mean)) / sigma
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_c
            
            # Update sigma
            sigma = sigma * np.exp(c_sigma / dsigma * (np.linalg.norm(p_sigma) / dsigma - 1))
            
            # Update Covariance
            C = (1 - c_1 - c_mu) * C + c_1 * np.outer(p_c, p_c)
            for i in range(mu):
                C += c_mu * w[i] * np.outer(pop[i] - x_mean, pop[i] - x_mean)
            
            C = (C + C.T) / 2.0
            try:
                np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C += 1e-6 * np.eye(dim)
            
            x_mean = x_new
            
            # Periodicity-Aware Restart Mechanism
            # If we have been stagnant for a while, assume we are in a local minimum.
            # Rastrigin has period 2.
            # We will shift the mean by a random vector of length ~2 in a random direction.
            # And increase sigma slightly to explore the new basin.
            
            if current_call - last_improvement_call > stagnation_threshold:
                # Stagnation detected
                # Generate a random direction
                dir = rng.normal(0, 1, dim)
                dir /= np.linalg.norm(dir)
                
                # Shift by 2.0 (one period)
                x_mean = x_mean + 2.0 * dir
                
                # Increase sigma to explore
                sigma *= 1.5
                
                # Reset paths to prevent bias from previous basin
                p_sigma = np.zeros(dim)
                p_c = np.zeros(dim)
                
                # Reset last improvement to avoid immediate re-trigger
                last_improvement_call = current_call
                
                # Optional: Reset C to identity if it became too narrow?
                # Let's keep C but maybe regularize it more.
                C = C * 0.5 + 0.5 * np.eye(dim)

        return best_x_global, best_y_global

# EVOLVE-BLOCK-END
