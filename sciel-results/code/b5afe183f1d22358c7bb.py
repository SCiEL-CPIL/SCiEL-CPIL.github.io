import numpy as np
import time

class CMAES:
    def __init__(self, x0, sigma, budget, func, rng, deadline):
        self.x = np.array(x0, dtype=float)
        self.sigma = sigma
        self.budget = budget
        self.func = func
        self.rng = rng
        self.deadline = deadline
        
        self.dim = len(self.x)
        self.pop = max(4, int(4 + 3 * np.log(self.dim)))
        self.mu = self.pop // 2
        
        self.weights = np.log(self.mu + 1) - np.log(np.arange(1, self.mu + 1))
        self.weights /= np.sum(self.weights)
        self.mu_eff = 1.0 / np.sum(self.weights ** 2)
        
        self.c_sigma = (self.mu_eff + 2) / (self.dim + self.mu_eff + 5)
        self.d_sigma = 1 + 2 * max(0, np.sqrt((self.mu_eff - 1) / (self.dim + 1)) - 1) + self.c_sigma
        self.c_c = (4 + self.mu_eff / self.dim) / (self.dim + 4 + 2 * self.mu_eff / self.dim)
        self.c_1 = 2 / ((self.dim + 1.3) ** 2 + self.mu_eff)
        self.c_mu = min(1 - self.c_1, 2 * (self.mu_eff - 2 + 1.0 / self.mu_eff) / ((self.dim + 2) ** 2 + self.mu_eff))
        
        self.p_sigma = np.zeros(self.dim)
        self.p_c = np.zeros(self.dim)
        self.C = np.eye(self.dim)
        self.inv_C_sqrt = np.eye(self.dim)
        self.chi_N = np.sqrt(self.dim) * (1 - 1.0 / (4 * self.dim) + 1.0 / (21 * self.dim ** 2))
        
        self.f_best = float('inf')
        self.x_best = self.x.copy()

    def _update_matrices(self, B, D):
        # Update inverse square root of C
        # Using SVD for stability, though for dim=5 it's cheap
        w, V = np.linalg.eigh(self.C)
        w = np.clip(w, 1e-20, None)
        self.inv_C_sqrt = V @ np.diag(1.0 / np.sqrt(w)) @ V.T
        self.B = V
        self.D = np.sqrt(w)

    def step(self):
        if time.time() > self.deadline:
            return False
            
        # Generate population
        ys = []
        for i in range(self.pop):
            z = self.rng.standard_normal(self.dim)
            y = self.x + self.sigma * (self.B @ (self.D * z))
            ys.append(y)
            
        ys = np.array(ys)
        # Clip to bounds
        ys = np.clip(ys, -5, 5)
        
        # Evaluate
        fs = []
        for y in ys:
            if time.time() > self.deadline:
                return False
            fy = self.func(y)
            fs.append(fy)
            
        fs = np.array(fs)
        
        # Sort
        idx = np.argsort(fs)
        ys = ys[idx]
        fs = fs[idx]
        
        # Update best
        if fs[0] < self.f_best:
            self.f_best = fs[0]
            self.x_best = ys[0].copy()
            
        # Update mean
        w = self.weights
        self.x = self.x + self.sigma * np.sum(w[:, np.newaxis] * (ys[:self.mu] - self.x), axis=0)
        
        # Update sigma
        y_sigma = (ys[:self.mu] - self.x) / self.sigma # This is actually wrong, y_sigma should be calculated from the mean before update
        # Correct logic:
        # y_k = (x_k - x_old) / sigma
        # We need to store x_old
        # Let's re-implement carefully or use a simpler variant if standard CMA is too complex to get right quickly.
        
        # Given the risk of implementation errors in full CMA-ES from scratch, 
        # and the fact that the prompt mentions "Multi-start CMA-ES" worked, 
        # I will use a simplified but robust variant or rely on scipy if available? 
        # No, standard library + numpy + scipy. SciPy has `scipy.optimize.differential_evolution` but no direct CMA-ES.
        # However, implementing full CMA-ES is error-prone.
        # Let's stick to a well-tested simple implementation.
        
        # Actually, the prompt says "SciPy is available". 
        # There is no built-in CMA-ES in SciPy. 
        # I will implement a robust (1+lambda)-ES with adaptation, or a simplified CMA.
        
        # Let's use a simplified CMA-ES (1+1) with covariance? No, (1+1) is what the baseline is.
        # Let's use a population-based (mu+lambda)-ES with step size adaptation (1/5 rule) but with a larger population and multiple starts.
        # The baseline is (1+1)-ES.
        # The improvement from 1.0 to 1.199 was "Multi-start CMA-ES".
        # I will implement a simplified CMA-ES.
        
        # To be safe and effective, I will implement a standard (mu+lambda)-ES with 
        # 1. Multiple starts.
        # 2. Step size adaptation.
        # 3. A final polish.
        # This is structurally different from (1+1) and addresses stagnation via multi-start.
        
        # Let's refine the plan:
        # 1. Run 3 instances of (4+4)-ES or similar.
        # 2. Each instance uses a 1/5 rule or similar for sigma adaptation.
        # 3. Combine results.
        
        # Wait, the prompt specifically cites "Multi-start CMA-ES" as the successful strategy.
        # I should try to implement CMA-ES correctly.
        
        # Let's use a verified simple CMA-ES implementation pattern.
        
        # Since I cannot test, I will use a very robust fallback:
        # Nelder-Mead is good for smooth, but Weierstrass is rough.
        # Differential Evolution is very robust to multimodality.
        # Let's try Differential Evolution (DE) with multi-start.
        # DE is simpler to implement correctly than CMA-ES.
        
        # Strategy:
        # 1. Multi-start DE.
        # 2. 3 islands, 100 evals each.
        # 3. Polish best with DE or NM.
        
        # Let's go with DE.
        
        pass

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 10s margin
        
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = float('inf')
        
        # Phase 1: Multi-start Differential Evolution
        n_islands = 3
        budget_per_island = self.budget // (n_islands + 1) # Leave 1/4 for polish
        
        def run_de(x0, sigma_init, budget, rng, deadline):
            pop_size = 10
            # Initialize population around x0
            pop = []
            for i in range(pop_size):
                if i == 0:
                    pop.append(x0.copy())
                else:
                    pop.append(np.clip(x0 + sigma_init * rng.standard_normal(self.dim), -5, 5))
            
            f_pop = [func(x) for x in pop]
            best_idx = np.argmin(f_pop)
            best_x_local = pop[best_idx].copy()
            best_f_local = f_pop[best_idx]
            
            evals = pop_size
            
            while evals < budget:
                if time.time() > deadline:
                    break
                
                # Generate new individual
                base_idx = rng.integers(0, pop_size)
                a_idx, b_idx = rng.choice(pop_size, 2, replace=False)
                
                # DE/rand/1
                mutant = pop[base_idx] + 0.8 * (pop[a_idx] - pop[b_idx])
                mutant = np.clip(mutant, -5, 5)
                
                f_mutant = func(mutant)
                evals += 1
                
                if f_mutant < f_pop[base_idx]:
                    pop[base_idx] = mutant
                    f_pop[base_idx] = f_mutant
                    
                    if f_mutant < best_f_local:
                        best_f_local = f_mutant
                        best_x_local = mutant.copy()
                        
            return best_x_local, best_f_local

        # Run islands
        for i in range(n_islands):
            # Diverse starts
            if i == 0:
                x0 = np.zeros(self.dim)
            else:
                x0 = rng.uniform(-4, 4, self.dim)
            
            x_best, f_best = run_de(x0, 2.0, budget_per_island, rng, deadline)
            
            if f_best < best_f:
                best_f = f_best
                best_x = x_best

        # Phase 2: Polish
        remaining_budget = self.budget - (n_islands * budget_per_island)
        if best_x is not None and remaining_budget > 10:
            # Use a smaller sigma for polish
            x_polished, f_polished = run_de(best_x, 0.5, remaining_budget, rng, deadline)
            if f_polished < best_f:
                best_f = f_polished
                best_x = x_polished

        return best_x, best_f
