import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin (330s limit)
        d = self.dim
        budget = self.budget
        
        best_x = np.zeros(d)
        best_f = float('inf')
        calls_made = 0

        def check_time():
            return time.time() > deadline

        def update_best(x, f):
            nonlocal best_x, best_f, calls_made
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Phase 1: Coarse Grid Search to find promising regions
        # Schaffers F7 has a global minimum at the origin (before transformation)
        # The transformation shifts/rotates it. A coarse grid helps locate it.
        # We have 500 calls. Let's use ~100 for grid, ~400 for CMA-ES.
        
        # Create a grid of points. 3^5 = 243 points is too many if we want 400 left.
        # 2^5 = 32 points is too few.
        # Let's use a randomized Latin Hypercube or just random points for initial exploration?
        # Actually, for Schaffers F7, the landscape is very ill-conditioned.
        # A simple random search might get stuck in local minima.
        # Let's try a hybrid:
        # 1. 50 random points to get a rough idea of the scale and location.
        # 2. Pick the best point.
        # 3. Run CMA-ES from there with the remaining budget.
        
        # Alternative: Use a larger population CMA-ES from the start?
        # The current program uses multi-start CMA-ES with 5 runs of 100.
        # The inspiration suggests that adapting to ill-conditioning is key.
        # A single, well-tuned CMA-ES run might be better than multiple short ones.
        # However, Schaffers F7 has many local minima. Multi-start is good.
        
        # Let's try:
        # 1. 100 evaluations: Random sampling to find a good starting point.
        # 2. 400 evaluations: CMA-ES starting from the best random point.
        
        # Or even better:
        # 1. 20 evaluations: Random sampling.
        # 2. Pick top 2.
        # 3. Run CMA-ES on both in parallel? No, sequential.
        
        # Let's stick to a robust strategy:
        # 1. Random search for 50 calls to find a decent basin.
        # 2. CMA-ES with budget 450.
        
        # Actually, the current program scores 1.29. The inspiration scores 1.87.
        # The inspiration likely uses a more efficient CMA-ES implementation or better initialization.
        # Let's implement a cleaner, faster CMA-ES and use a small random search to initialize it.
        
        # Step 1: Random Initialization
        rng = np.random.default_rng(42)
        num_init = 20
        for _ in range(num_init):
            if check_time():
                break
            x = rng.uniform(-5, 5, d)
            f = func(x)
            calls_made += 1
            update_best(x, f)
        
        # Step 2: CMA-ES
        # Parameters for CMA-ES
        lam = 10  # Population size
        mu = lam // 2
        
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        
        # mu_eff
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
        c_c = (4 + mu_eff / lam) / (d + 4 + 2 * mu_eff / lam)
        c_1 = 2 / ((d + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1.0 / mu_eff) / ((d + 2)**2 + mu_eff))
        
        # Initialization
        m = best_x.copy()
        # Estimate sigma from the spread of the initial random points?
        # Or just start with a reasonable sigma.
        # Schaffers F7 has a sharp minimum. Start small?
        # The box is [-5, 5]. The minimum is in [-4, 4].
        # Let's start with sigma = 1.0.
        sigma = 1.0
        C = np.eye(d)
        pc = np.zeros(d)
        ps = np.zeros(d)
        
        # Precompute C^(1/2)
        eigvals, eigvecs = np.linalg.eigh(C)
        sqrt_C = eigvecs @ np.diag(np.sqrt(np.maximum(eigvals, 0)))
        
        remaining_budget = budget - calls_made
        if remaining_budget <= lam:
            return best_x, best_f
            
        # CMA-ES Loop
        for gen in range(remaining_budget // lam):
            if check_time():
                break
            
            # Sample lambda_ individuals
            # Use standard normal and transform
            z = rng.standard_normal((lam, d))
            # z @ sqrt_C.T is equivalent to sqrt_C @ z.T for each row
            # But we need m + sigma * z_i @ C^(1/2)
            # Let's do: samples = m + sigma * (z @ sqrt_C.T)
            samples = m[None, :] + sigma * (z @ sqrt_C.T)
            
            # Clip to box
            samples = np.clip(samples, -5, 5)
            
            # Evaluate
            fitnesses = np.array([func(s) for s in samples])
            calls_made += lam
            
            # Update global best
            min_idx = np.argmin(fitnesses)
            if fitnesses[min_idx] < best_f:
                best_f = fitnesses[min_idx]
                best_x = samples[min_idx].copy()
                
            # Sort indices by fitness
            sorted_indices = np.argsort(fitnesses)
            
            # Update mean
            m_new = np.sum(weights[:, np.newaxis] * samples[sorted_indices[:mu]], axis=0)
            
            # Update covariance matrix
            # y = (m_new - m) / sigma
            y = (m_new - m) / sigma
            
            # Cumulation step for sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (sqrt_C @ y)
            
            # h_sigma check (simplified: just update)
            # Proper check:
            # h_sigma = 1 if ||ps|| / sqrt(1 - (1-c_sigma)^(2*(gen+1))) < (1.4 + 2/(d+1)) * E[||N(0, I)||]
            # E[||N(0, I)||] = sqrt(d) * (1 - 1/(4d) + 1/(32d^2))
            c_sigma = (mu_eff + 2) / (d + mu_eff + 5) # Re-define for clarity in check
            # Actually c_sigma is fixed.
            # Let's implement the check properly.
            # E_norm = sqrt(d) * (1 - 1/(4*d) + 1/(32*d**2))
            E_norm = np.sqrt(d) * (1 - 1/(4*d) + 1/(32*d**2))
            denom = np.sqrt(1 - (1 - c_sigma)**(2 * (gen + 1)))
            if denom < 1e-10:
                denom = 1e-10
            h_sigma = 1 if (np.linalg.norm(ps) / denom) < (1.4 + 2/(d+1)) * E_norm else 0
            
            sigma *= np.exp(c_sigma / c_c * (np.linalg.norm(ps) / np.sqrt(d) - 1) * h_sigma)
            # Wait, the standard formula is:
            # sigma = sigma * exp(c_sigma * (||ps|| / (sqrt(d) * E||N||) - 1))
            # My previous code had a simplified version. Let's use the standard one.
            # sigma *= np.exp(c_sigma * (np.linalg.norm(ps) / (np.sqrt(d) * E_norm) - 1))
            # But h_sigma is a gate.
            # Let's stick to the simplified version from the current program if it worked, 
            # or use the standard one. The standard one is better.
            # Let's re-calculate sigma with the standard formula.
            # Actually, the line above was: sigma *= np.exp(c_sigma / c_sigma * ...) which is just exp(...).
            # Let's use:
            sigma_factor = np.exp(c_sigma * (np.linalg.norm(ps) / (np.sqrt(d) * E_norm) - 1))
            sigma *= sigma_factor
            
            # Cumulation step for C
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * y
            
            # h_pc check
            # h_pc = 1 if ||pc|| / sqrt(1 - (1-c_c)^(2*(gen+1))) < (1.4 + 2/(d+1)) * sqrt(d+1)
            denom_c = np.sqrt(1 - (1 - c_c)**(2 * (gen + 1)))
            if denom_c < 1e-10:
                denom_c = 1e-10
            h_pc = 1 if (np.linalg.norm(pc) / denom_c) < (1.4 + 2/(d+1)) * np.sqrt(d+1) else 0
            
            # Update C
            rank_mu = np.outer(y, y)
            rank_one = np.outer(pc, pc)
            C = (1 + c_1 * h_pc * (1 - c_c) - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu
            
            # Update mean
            m = m_new
            
            # Update sqrt_C
            eigvals, eigvecs = np.linalg.eigh(C)
            # Ensure positive definite
            eigvals = np.maximum(eigvals, 1e-10)
            sqrt_C = eigvecs @ np.diag(np.sqrt(eigvals))
            
            # Check for convergence
            if sigma < 1e-10:
                break

        return best_x, best_f
