import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # Deterministic setup
        rng = np.random.default_rng(42)
        
        # Parameters
        N = 10  # Population size
        mu = N // 2  # Number of best individuals for mean update
        # Weights for mean update (standard CMA-ES weights)
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1 / np.sum(w**2)
        
        # CMA-ES step size control parameters
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * ((mu_eff + 2) / (dim + 2) - 1))
        
        # State
        m = np.zeros(dim)
        C = np.eye(dim)
        B = np.eye(dim) # Eigenvectors
        D = np.ones(dim) # Eigenvalues
        sigma = 4.0
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        
        best_x = m.copy()
        best_fx = func(m)
        calls_made = 1
        
        # Schedule: We will run for a fixed number of generations
        # Budget 500. N=10. 40 generations = 400 calls.
        # Remaining 100 for fine tuning or more generations.
        max_generations = (budget - 1) // N
        
        # Pre-generate random vectors for determinism? 
        # CMA-ES uses random vectors each generation. 
        # To be "deterministic" in the sense of the directive, we use a fixed seed.
        
        for gen in range(max_generations):
            if calls_made >= budget:
                break
                
            # Generate population
            zs = rng.standard_normal((N, dim))
            # Transform to current distribution
            # x = m + sigma * B @ z.T * D
            # Note: B @ (z * D)
            pop = m + sigma * (zs * D[None, :]) @ B.T
            
            # Clip to box
            pop = np.clip(pop, -5, 5)
            
            # Evaluate
            fvals = np.array([func(x) for x in pop])
            calls_made += N
            
            # Update best
            idx_best = np.argmin(fvals)
            if fvals[idx_best] < best_fx:
                best_fx = fvals[idx_best]
                best_x = pop[idx_best].copy()
                
            # Sort indices by fitness
            order = np.argsort(fvals)
            y_best = pop[order[:mu]]
            f_best = fvals[order[:mu]]
            
            # Update mean
            m_old = m.copy()
            m = np.sum(w[:, None] * y_best, axis=0)
            
            # Update path states
            y_sigma = (m - m_old) / (sigma * np.sqrt(mu_eff))
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (gen + 1))) < (1.4 + 2 / (dim + 1)) * np.sqrt(c_sigma * (2 - c_sigma))
            
            pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_sigma
            
            # Update Covariance Matrix
            delta_hsig = (1 - hsig) * c_c * (2 - c_c)
            rank_one = np.outer(pc, pc)
            
            # Rank-mu update
            y_w = (y_best - m_old) / sigma
            rank_mu = np.sum(w[:, None, None] * (y_w[:, :, None] * y_w[:, None, :]), axis=0)
            
            C = (1 + c_1 * delta_hsig - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu
            
            # Eigen-decomposition to ensure positive definiteness and get B, D
            eigvals, eigvecs = np.linalg.eigh(C)
            # Clip eigenvalues to avoid numerical issues
            eigvals = np.clip(eigvals, 1e-12, None)
            B = eigvecs
            D = np.sqrt(eigvals)
            
            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / np.sqrt(mu_eff) - 1))
            
            # Optional: Enforce box constraints on m? 
            # m is already implicitly constrained by the clip of pop, but m itself can drift out.
            # If m goes out of box, it's bad. Clip m?
            m = np.clip(m, -5, 5)
            
            # Plateau detection / Restart logic
            # If sigma becomes too small or no improvement for a while, restart.
            # Simple heuristic: if best_fx hasn't improved in last 5 generations, restart.
            # We need to track history.
            if not hasattr(self, '_history'):
                self._history = []
            self._history.append(best_fx)
            
            if len(self._history) > 5:
                if self._history[-1] == self._history[-6]:
                    # Restart
                    # Move to a different deterministic point
                    # Use a hash of gen to pick a point
                    rng_restart = np.random.default_rng(gen * 1000 + 42)
                    m = rng_restart.uniform(-4, 4, size=dim)
                    C = np.eye(dim)
                    B = np.eye(dim)
                    D = np.ones(dim)
                    sigma = 2.0
                    ps = np.zeros(dim)
                    pc = np.zeros(dim)
                    self._history = []
                    # Evaluate new m
                    fm = func(m)
                    calls_made += 1
                    if fm < best_fx:
                        best_fx = fm
                        best_x = m.copy()

        # Final fine-tuning with remaining budget
        remaining = budget - calls_made
        if remaining > 0:
            # Local coordinate search
            step = 1e-4
            for _ in range(remaining):
                if calls_made >= budget:
                    break
                improved = False
                for i in range(dim):
                    if calls_made >= budget:
                        break
                    # Try +
                    y = best_x.copy()
                    y[i] += step
                    if np.all(y >= -5) and np.all(y <= 5):
                        fy = func(y)
                        calls_made += 1
                        if fy < best_fx:
                            best_x, best_fx = y, fy
                            improved = True
                            break
                    # Try -
                    if calls_made >= budget:
                        break
                    y = best_x.copy()
                    y[i] -= step
                    if np.all(y >= -5) and np.all(y <= 5):
                        fy = func(y)
                        calls_made += 1
                        if fy < best_fx:
                            best_x, best_fx = y, fy
                            improved = True
                            break
                if not improved:
                    step *= 0.5
                    if step < 1e-7:
                        break

        return best_x, best_fx
