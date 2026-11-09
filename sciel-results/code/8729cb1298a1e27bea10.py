import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bound_lo = -5.0
        self.bound_hi = 5.0
        self.start_time = None
        self.deadline = 325.0  # Safety margin

    def __call__(self, func):
        self.start_time = time.time()
        d = self.dim
        budget = self.budget
        lo = self.bound_lo
        hi = self.bound_hi

        rng = np.random.default_rng(42)
        
        def check_time():
            return time.time() - self.start_time > self.deadline

        # --- Phase 1: Differential Evolution (Global Search) ---
        # DE is robust and good for global exploration.
        # Use a significant portion of the budget, e.g., 40%.
        de_budget = int(budget * 0.4)
        if de_budget < 50:
            de_budget = 50
            
        npop = 10  # Population size
        if npop > de_budget:
            npop = de_budget
            
        # Initialize population
        pop = rng.uniform(lo, hi, (npop, d))
        fpop = np.array([func(x) for x in pop])
        
        best_x = pop[np.argmin(fpop)].copy()
        best_f = np.min(fpop)
        
        # DE Parameters
        F = 0.8  # Mutation factor
        CR = 0.9  # Crossover rate
        
        calls_made = npop
        while calls_made + npop <= de_budget and not check_time():
            for i in range(npop):
                if check_time():
                    break
                # Select two distinct random indices
                a = rng.integers(0, npop)
                b = rng.integers(0, npop)
                while a == b or a == i or b == i:
                    a = rng.integers(0, npop)
                    b = rng.integers(0, npop)
                    
                # Mutation
                v = pop[i] + F * (pop[a] - pop[b])
                
                # Crossover
                trial = np.empty(d)
                j_rand = rng.integers(0, d)
                for j in range(d):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                    else:
                        trial[j] = pop[i][j]
                
                # Bounds
                trial = np.clip(trial, lo, hi)
                
                # Evaluation
                f_trial = func(trial)
                calls_made += 1
                
                # Selection
                if f_trial <= fpop[i]:
                    pop[i] = trial
                    fpop[i] = f_trial
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()
            
            # Early stopping if no improvement? Maybe not, DE can jump.
            # But if budget is tight, we should move to CMA-ES.
            
        # --- Phase 2: CMA-ES (Local Refinement) ---
        remaining_budget = budget - calls_made
        if remaining_budget < 20:
            return best_x, best_f

        # CMA-ES Parameters
        lam = 4 + int(3 * np.log(d))
        if lam < 6:
            lam = 6
        if lam > remaining_budget:
            lam = max(4, remaining_budget) # Ensure at least 1 generation
            
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (d + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        c_1 = 2 / ((d + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((d + 2)**2 + mu_eff))
        
        chi_n = np.sqrt(d) * (1 - 1.0/(4*d) + 1.0/(21*d*d))

        # Initialize CMA-ES state
        m = best_x.copy()
        
        # Estimate initial sigma from DE population spread
        # Use the std of the best 50% of DE population
        sorted_idx = np.argsort(fpop)
        top_half = pop[sorted_idx[:max(1, npop//2)]]
        sigma = np.linalg.norm(np.std(top_half, axis=0), axis=0) * 0.5
        if sigma < 0.1:
            sigma = 0.1
        elif sigma > 2.0:
            sigma = 2.0

        C = np.eye(d)
        pc = np.zeros(d)
        ps = np.zeros(d)
        
        stagnation_count = 0
        stagnation_threshold = 15 # Generations without improvement
        
        while remaining_budget >= lam and not check_time():
            # Sample candidates
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = np.eye(d)
                L = np.eye(d)
                
            Z = rng.standard_normal((lam, d))
            Y = Z @ L.T
            X = m[None, :] + sigma * Y
            X = np.clip(X, lo, hi)
            
            F_vals = np.array([func(x) for x in X])
            remaining_budget -= lam
            
            min_idx = np.argmin(F_vals)
            if F_vals[min_idx] < best_f:
                best_f = F_vals[min_idx]
                best_x = X[min_idx].copy()
                stagnation_count = 0
            else:
                stagnation_count += 1
            
            # Stagnation Handling
            if stagnation_count > stagnation_threshold:
                # Restart mean to a random point or keep best?
                # For local refinement, keeping best is safer.
                # But if stuck in a local min, we might need to jump.
                # Given the "Step" nature, we might be on a plateau.
                # Let's try to increase sigma significantly.
                sigma *= 2.0
                if sigma > 3.0:
                    # Full restart from random point
                    new_x = rng.uniform(lo, hi, d)
                    new_f = func(new_x)
                    remaining_budget -= 1
                    if new_f < best_f:
                        best_f = new_f
                        best_x = new_x.copy()
                    m = best_x.copy()
                    sigma = 1.5
                    C = np.eye(d)
                    pc = np.zeros(d)
                    ps = np.zeros(d)
                    stagnation_count = 0
                else:
                    # Just reset covariance
                    C = np.eye(d)
                    pc = np.zeros(d)
                    ps = np.zeros(d)
                    stagnation_count = 0

            sort_idx = np.argsort(F_vals)
            X_sorted = X[sort_idx]
            
            m_new = np.sum(weights[:, None] * X_sorted[:mu], axis=0)
            
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-16, None)
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            
            step = (m_new - m) / sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ step)
            
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * step
            
            h_sigma = 1 if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**2) < (1.4 + 2 / (d + 1)) * chi_n else 0
            
            C = (1 + c_1 * (h_sigma - 1) - c_1 - c_mu) * C \
                + c_1 * np.outer(pc, pc)
                
            Y_delta = X_sorted[:mu] - m
            Y_delta = Y_delta / sigma
            rank_mu_term = np.sum(weights[:, None, None] * Y_delta[:, :, None] * Y_delta[:, None, :], axis=0)
            C += c_mu * rank_mu_term
            
            C = (C + C.T) / 2.0
            if not np.all(np.linalg.eigvalsh(C) > 0):
                C = C + 1e-4 * np.eye(d)
                
            m = m_new
            
        return best_x, best_f
