"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES with restarts for black-box continuous optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        total_budget = self.budget
        
        # Deadline guard
        start_time = time.time()
        max_time = 285.0  # Safety margin before 300s kill

        # Global best tracking
        global_best_x = None
        global_best_f = np.inf
        
        # We will use a multi-start strategy with CMA-ES.
        # Lunacek bi-Rastrigin has a secondary global minimum.
        # A single run might get stuck in the local minimum or the secondary global.
        # We will split the budget into a few phases or runs.
        # Given dim=5 and budget=500, we can afford 2-3 runs of ~200 evaluations.
        # Let's do 2 runs: one aggressive, one fine-tuning.
        # Or better: one main CMA-ES run with a large population, and if it stagnates, restart.
        
        # CMA-ES parameters
        # Standard CMA-ES uses lambda = 4 + floor(3 log(n))
        # For n=5, lam = 4 + floor(3*1.6) = 4 + 4 = 8.
        # Let's use lam = 10 for slightly more exploration.
        lam = 10
        mu = max(1, lam // 2)  # mu = 5
        
        # Initial step size
        # Domain is [-5, 5]. Width 10.
        # sigma0 = 2.0 is a good start to cover the space quickly.
        
        # We will implement a simplified CMA-ES loop.
        # Since we have a hard budget of 500 calls, we can't use the full adaptive lambda logic easily without overhead,
        # but standard CMA-ES is efficient.
        
        # Strategy: 
        # 1. Start with a random initial mean.
        # 2. Run CMA-ES until stagnation or budget.
        # 3. If stagnation detected (no improvement for K generations), restart with a new random mean 
        #    but keep the best found so far as a reference? No, CMA-ES state is local.
        #    Better: Just do 2 independent CMA-ES runs with different initializations.
        #    Run 1: 300 evals. Run 2: 200 evals starting from best of Run 1? 
        #    Actually, Lunacek's secondary minimum is far. 
        #    Let's try: 
        #    Phase 1: 100 random samples to find a decent basin.
        #    Phase 2: CMA-ES starting from the best of Phase 1, with budget 400.
        
        calls_made = 0
        rng = np.random.default_rng(42)
        
        # Phase 1: Random Sampling / Latin Hypercube to initialize
        # 50 random points to get a feel for the landscape and find a good starting point for CMA-ES.
        n_init = 50
        if total_budget < n_init:
            n_init = total_budget // 2
            
        init_pop = rng.uniform(-5, 5, size=(n_init, dim))
        init_fvals = np.array([func(x) for x in init_pop])
        calls_made += n_init
        
        best_idx = np.argmin(init_fvals)
        current_mean = init_pop[best_idx].copy()
        global_best_f = init_fvals[best_idx]
        global_best_x = init_pop[best_idx].copy()
        
        # Phase 2: CMA-ES
        # Remaining budget
        remaining_budget = total_budget - calls_made
        
        # CMA-ES Setup
        # lambda
        lam = 10
        mu = lam // 2
        
        # Step size
        sigma = 2.0
        
        # Covariance matrix
        C = np.eye(dim)
        
        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Adaptation rates (standard values for n=5)
        # cs = (mu_eff + 2) / (n + mu_eff + 5)
        # cm = (2 + 2/min(mu_eff, (lam-mu_eff)/2)) / (n^2 + 2)
        # mu_eff = mu^2 / sum(wi^2)
        
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1 / np.sum(weights**2)
        
        cs = (mu_eff + 2) / (dim + mu_eff + 5)
        ds = 1 + 2 * max(0, (mu_eff - 1) / (dim + 1)) + cs
        cm = (2 + 2 / max(mu_eff, (lam - mu_eff) / 2)) / (dim**2 + 2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        
        # Main loop
        # We will run until remaining_budget is exhausted or stagnation is severe.
        
        # Keep track of population for selection
        # We don't need to store all history, just the current generation.
        
        # To handle the "secondary global minimum" of Lunacek, 
        # if we get stuck, we might need to jump. 
        # But CMA-ES with sigma=2.0 initially should explore well.
        
        # Let's add a simple stagnation check: 
        # If no improvement for 50 generations, restart CMA-ES state from global_best_x with larger sigma?
        # Or just let it run. 500 evals is quite a lot for dim=5.
        
        # We will loop while calls_made < total_budget and time is safe.
        
        # To ensure we don't get stuck in a local minimum that isn't the global one,
        # we can occasionally (every 50 gens) inject a random point far away?
        # No, that breaks CMA-ES assumptions. 
        # Instead, we rely on the initial random sampling to pick a good basin.
        # Lunacek's secondary minimum is at x* = (2.5, ..., 2.5) or similar? 
        # Actually, the secondary minimum is usually far from the main one.
        # Random sampling with 50 points in [-5,5]^5 has a decent chance of hitting near both.
        
        # Let's refine: Instead of just one CMA-ES run, let's do a "Restart CMA-ES" logic.
        # If the improvement per generation drops below a threshold, restart the CMA-ES state
        # centered at the current global best, but with a new random rotation of the covariance?
        # Simpler: Just restart the whole CMA-ES process with a new random mean if stuck.
        
        # Given the budget, let's just run one robust CMA-ES.
        # If it fails, the random init phase might have missed the global basin.
        # To mitigate, we can do 2 independent CMA-ES runs of 250 evals each, 
        # starting from the top 2 distinct points in the initial random sample?
        # That's complex. 
        # Let's stick to: 50 random -> pick best -> CMA-ES for 450.
        # This is standard and effective.
        
        # Optimization Loop
        # We need to be careful with the number of function calls.
        # Each generation consumes 'lam' calls.
        
        # We'll store the best point found in this CMA-ES run locally, 
        # but update global_best if better.
        
        local_best_x = current_mean.copy()
        local_best_f = global_best_f
        
        # Pre-compute Cholesky factor for efficiency? 
        # For dim=5, np.linalg.cholesky is fast.
        
        # We'll run for max_generations = remaining_budget // lam
        max_generations = remaining_budget // lam
        
        for gen in range(max_generations):
            if time.time() - start_time > max_time:
                break
            if calls_made + lam > total_budget:
                # If we can't fit a full generation, do a partial one?
                # CMA-ES requires a full population for selection.
                # So we stop if we can't do a full generation.
                # Or we can do a smaller population? 
                # Let's just break to be safe and valid.
                break
                
            # 1. Sample population
            # C = L @ L.T
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Fallback to identity if Cholesky fails (shouldn't happen often with regularization)
                C = np.eye(dim) + 1e-6 * np.eye(dim)
                L = np.linalg.cholesky(C)
                
            z = rng.standard_normal((lam, dim))
            # Transform to x
            # x = mean + sigma * L @ z.T
            pop = current_mean + sigma * (z @ L.T)
            
            # Clip to domain? 
            # The problem says func clips. So we can pass unclipped, 
            # but it's better to clip to avoid wasting calls on out-of-bounds if func didn't clip?
            # "func(x) clips a finite length-5 vector to the box" -> So we don't need to clip.
            # But clipping in the optimizer might help keep the mean inside?
            # CMA-ES mean can drift outside. 
            # Let's not clip the samples, let func handle it.
            
            # 2. Evaluate
            fvals = np.array([func(x) for x in pop])
            calls_made += lam
            
            # 3. Update global best
            gen_best_idx = np.argmin(fvals)
            if fvals[gen_best_idx] < global_best_f:
                global_best_f = fvals[gen_best_idx]
                global_best_x = pop[gen_best_idx].copy()
            
            if fvals[gen_best_idx] < local_best_f:
                local_best_f = fvals[gen_best_idx]
                local_best_x = pop[gen_best_idx].copy()
            
            # 4. Selection
            # Sort indices by fitness
            order = np.argsort(fvals)
            # Take top mu
            selected_indices = order[:mu]
            
            # 5. Update Mean
            # Weighted mean of selected
            new_mean = np.sum(weights[:, None] * pop[selected_indices], axis=0)
            
            # 6. Update Covariance and Step Size
            # Recenter selected points
            y = (pop[selected_indices] - current_mean) / sigma
            y = y.T  # dim x mu
            
            # Update p_sigma
            p_sigma = (1 - c_sigma) * p_sigma + c_sigma * np.sqrt(mu_eff) * (weights @ y)
            
            # Update p_c
            p_c = (1 - c_c) * p_c + c_c * np.sqrt(mu_eff) * (weights @ y)
            
            # Update C
            # h_sigma = (||p_sigma|| / sqrt(1 - (1-c_sigma)^{2(gen+1)})) / sqrt(1 - (1-c_sigma)^{2(gen+1)}) < 1.4 + 2/(dim+1)
            # Simplified h_sigma check
            h_sigma = 1.0
            # For short runs, we can skip the exact h_sigma calculation or use a heuristic.
            # Standard heuristic: if ||p_sigma|| / sqrt(1 - (1-c_sigma)^(2*gen+1)) < (1.4 + 2/(dim+1)) * sqrt(1 - (1-c_c)^(2*gen+1))
            # Let's use a simple threshold: if ||p_sigma|| < 1.4 * sqrt(dim * (1 - (1-c_c)^(2*gen+1)))?
            # Actually, just using h_sigma = 1 is often fine for short runs.
            # Let's use the standard condition:
            # denom = np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1)))
            # if denom > 0:
            #     h_sigma = 1 if (np.linalg.norm(p_sigma) / denom < (1.4 + 2/(dim+1)) * np.sqrt(1 - (1-c_c)**(2*(gen+1)))) else 0
            # else:
            #     h_sigma = 1
            
            # Let's implement the standard h_sigma for correctness
            denom_sigma = np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1)))
            denom_c = np.sqrt(1 - (1 - c_c) ** (2 * (gen + 1)))
            
            if denom_sigma > 1e-10:
                threshold = (1.4 + 2 / (dim + 1)) * denom_c
                if np.linalg.norm(p_sigma) / denom_sigma < threshold:
                    h_sigma = 1.0
                else:
                    h_sigma = 0.0
            else:
                h_sigma = 1.0
            
            # Rank-one update
            C = (1 - cm * (1 + (1 - h_sigma) * cs * c_sigma)) * C + cm * h_sigma * np.outer(p_sigma, p_sigma)
            
            # Rank-mu update
            # C = (1 - cm) * C + cm * (1 + (1 - h_sigma) * cs * c_sigma) * sum(wi * yi @ yi.T)
            # Actually the formula is:
            # C += cm * ( (1 + (1 - h_sigma) * cs * c_sigma) * sum(wi * yi yi^T) - (1 - cm * (1 - h_sigma) * cs * c_sigma) * C )? No.
            # Standard:
            # C = (1 - mu * cm * (1 + (1 - h_sigma) * cs * c_sigma)) * C + mu * cm * (1 + (1 - h_sigma) * cs * c_sigma) * sum(wi * yi yi^T)
            # Wait, the rank-one update already modified C.
            # The combined update is:
            # C_new = (1 - cm) * C_old + cm * ( (1 + (1 - h_sigma) * cs * c_sigma) * Y_weighted + (1 - (1 + (1 - h_sigma) * cs * c_sigma)) * p_sigma p_sigma^T? No.
            
            # Let's stick to the sequential updates as implemented in standard libraries:
            # 1. Update C with rank-one (p_sigma)
            # 2. Update C with rank-mu (weighted sum of y_i y_i^T)
            
            # My previous line did rank-one. Now rank-mu:
            # Y_weighted = sum(weights[i] * y[:, i] @ y[:, i].T)
            Y_weighted = np.zeros((dim, dim))
            for i in range(mu):
                Y_weighted += weights[i] * np.outer(y[:, i], y[:, i])
            
            # Correction factor
            correction = 1 + (1 - h_sigma) * cs * c_sigma
            C = (1 - cm * correction) * C + cm * correction * Y_weighted
            
            # Ensure C is positive definite (symmetrize and add small diag if needed)
            C = (C + C.T) / 2
            # Check eigenvalues? For dim=5, we can just add small regularization if Cholesky fails next time.
            
            # Update sigma
            sigma *= np.exp((cs / ds) * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
            
            # Clamp sigma to reasonable bounds
            sigma = np.clip(sigma, 1e-8, 10.0)
            
            # Update mean
            current_mean = new_mean
            
            # Optional: If mean is far outside [-5,5], pull it back?
            # The function clips, so it's okay, but it might waste exploration.
            # Let's keep it as is.
            
            # Stagnation Check:
            # If local_best_f hasn't improved in 20 generations, and sigma is very small,
            # we might be stuck. 
            # We can restart the CMA-ES state from global_best_x with a larger sigma.
            # Let's implement a simple restart:
            # Track generations since improvement.
            if not hasattr(self, '_gen_since_improve'):
                self._gen_since_improve = 0
                self._last_improve_f = local_best_f
            
            if local_best_f < self._last_improve_f - 1e-8:
                self._gen_since_improve = 0
                self._last_improve_f = local_best_f
            else:
                self._gen_since_improve += 1
                
            # If stuck for 30 generations and sigma < 0.1:
            if self._gen_since_improve > 30 and sigma < 0.1:
                # Restart CMA-ES state
                # Keep global_best_x as the new center
                current_mean = global_best_x.copy()
                sigma = 1.0  # Reset step size
                C = np.eye(dim)
                p_sigma = np.zeros(dim)
                p_c = np.zeros(dim)
                self._gen_since_improve = 0
                self._last_improve_f = local_best_f
                # Reset local best for this restart? No, keep global best.
                # But local_best_f is used for improvement tracking.
                # Let's reset local tracking to current global best.
                local_best_f = global_best_f
                local_best_x = global_best_x.copy()

        return global_best_x, global_best_f

# EVOLVE-BLOCK-END
