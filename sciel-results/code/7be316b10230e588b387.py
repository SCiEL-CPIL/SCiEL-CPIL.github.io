import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_val = np.inf
        self.best_x = None
        self.calls = 0
        self.start_time = 0

    def __call__(self, func):
        self.start_time = time.time()
        # Safety margin: 330s limit, keep 30s buffer for cleanup/overhead
        deadline = self.start_time + 300
        
        # Wrapper to count calls and update best
        def wrapped_func(x):
            self.calls += 1
            # Ensure x is a numpy array of correct shape for safety
            x = np.asarray(x, dtype=float).flatten()
            val = func(x)
            if val < self.best_val:
                self.best_val = val
                self.best_x = x.copy()
            return val

        # CMA-ES Parameters
        # Population size lambda: 4 + floor(3 * log(n)) is a common heuristic
        # For n=5, 4 + 3*1.6 = 8.8 -> 8 or 9. Let's use 10 for robustness.
        lam = 10
        mu = lam // 2  # Number of parents for recombination
        
        # Initial step size (sigma)
        # Box is [-5, 5], optimum in [-4, 4]. 
        # Starting sigma around 1.0-2.0 allows good global exploration.
        sigma = 1.5
        
        # Initial mean: Origin is a reasonable start, or slightly random?
        # Origin is 0. If optimum is shifted, CMA-ES will move.
        # Let's start at 0.
        mean = np.zeros(self.dim)
        
        # Covariance matrix: Identity
        C = np.eye(self.dim)
        
        # Weights for recombination
        # Standard weights: proportional to rank
        weights = np.zeros(mu)
        for i in range(mu):
            weights[i] = np.log(mu + 1) - np.log(i + 1)
        weights /= np.sum(weights)
        
        # Variance-effective selection weight
        mueff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        cc = (4 + mueff / self.dim) / (self.dim + 4 + 2 * mueff / self.dim)
        cs = (mueff + 2) / (self.dim + mueff + 5)
        c1 = 2 / ((self.dim + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((self.dim + 2)**2 + mueff))
        
        # Damping
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (self.dim + 1)) - 1) + cs
        
        # State variables
        pc = np.zeros(self.dim)
        ps = np.zeros(self.dim)
        
        # Cholesky factor of C (for sampling)
        # Initially C = I, so Ch = I
        Ch = np.eye(self.dim)
        
        # Cache function values for the current generation
        # We will generate lambda points, evaluate them, sort, update.
        
        def cholesky_update(C):
            # Compute Cholesky factor with jitter if needed
            for jitter in [1e-12, 1e-10, 1e-8, 1e-6, 1e-4]:
                try:
                    Ch = np.linalg.cholesky(C + jitter * np.eye(self.dim))
                    return Ch
                except np.linalg.LinAlgError:
                    continue
            # Fallback to identity if all fail (shouldn't happen)
            return np.eye(self.dim)
        
        Ch = cholesky_update(C)
        
        # Main loop
        while self.calls < self.budget and time.time() < deadline:
            # 1. Sample population
            pop = np.zeros((lam, self.dim))
            for i in range(lam):
                z = np.random.randn(self.dim)
                pop[i] = mean + sigma * (Ch @ z)
            
            # 2. Evaluate population
            vals = np.zeros(lam)
            for i in range(lam):
                if self.calls >= self.budget or time.time() > deadline:
                    # Fill remaining with best known or break
                    # Actually, if we break here, we might have partial evaluation.
                    # It's better to just break the outer loop.
                    # But we need to handle the partial array.
                    # Let's just stop sampling and break out.
                    # We'll set a flag.
                    break
                vals[i] = wrapped_func(pop[i])
            
            if self.calls >= self.budget or time.time() > deadline:
                break
            
            # 3. Sort by fitness (ascending)
            order = np.argsort(vals)
            # We only use the top mu
            x_best = pop[order[:mu]]
            f_best = vals[order[:mu]]
            
            # 4. Update mean
            mean = np.sum(weights[:, np.newaxis] * x_best, axis=0)
            
            # 5. Update step size (CSA)
            # ps = (1-cs)*ps + sqrt(cs*(2-cs)*damps) * C^{-1/2} * (mean - old_mean)/sigma
            # We need C^{-1/2}. We have Ch such that C = Ch @ Ch.T.
            # C^{-1/2} is not directly Ch^{-1}. 
            # Standard CMA-ES uses C^{-1/2} for the evolution path.
            # We can approximate or compute it. 
            # For simplicity and stability, often people just use the normalized difference.
            # Let's use the standard formula:
            # C^{-1/2} is the matrix square root inverse.
            # Computing this every iteration is expensive.
            # Alternative: Use the "simple" CMA-ES variant or approximate.
            # Given the small dim (5), we can compute it.
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            except:
                C_inv_sqrt = np.eye(self.dim)
                
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * damps) * (C_inv_sqrt @ (mean - mean_prev) / sigma)
            
            # hsig condition
            hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * (self.calls // lam + 1))) / np.sqrt(self.dim) < 1.4 + 2 / (self.dim + 1)
            
            # 6. Update covariance
            delta_sigma = (np.linalg.norm(ps) / np.sqrt(self.dim) - 1) * cs * damps
            # Sigma update (optional, often fixed or simple)
            # Let's keep sigma constant or use a simple adaptation.
            # For robustness, let's keep sigma fixed unless we implement full CSA.
            # Actually, let's implement the sigma update.
            # sigma = sigma * exp(cs / damps * (norm(ps)/sqrt(dim) - 1))
            sigma = sigma * np.exp(cs / damps * (np.linalg.norm(ps) / np.sqrt(self.dim) - 1))
            
            # Rank-one update
            if hsig:
                pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * damps) * (mean - mean_prev) / sigma
                rank_one = np.outer(pc, pc)
            else:
                rank_one = np.zeros((self.dim, self.dim))
            
            # Rank-mu update
            y = (x_best - mean_prev) / sigma
            rank_mu = np.sum(weights[:, np.newaxis] * (y * y.T), axis=0)
            
            C = (1 + c1 * (cc * (2 - cc) + (1 - hsig) * 0) + cmu * np.sum(weights) - c1 - cmu) * C \
                + c1 * rank_one + cmu * rank_mu
            
            # Ensure symmetry
            C = (C + C.T) / 2
            
            # Update Cholesky
            Ch = cholesky_update(C)
            
            # Save mean_prev for next iteration
            mean_prev = mean.copy()
            
            # Check for stagnation or convergence
            # If sigma becomes too small, we might be stuck.
            # With 500 evals and lam=10, we have 50 generations.
            # This should be enough.
            
        return self.best_x, self.best_val
