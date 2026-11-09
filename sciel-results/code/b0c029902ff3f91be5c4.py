"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES optimizer for BBOB f12 (Bent Cigar)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s limit
        
        d = self.dim
        # CMA-ES parameters
        # Lambda is population size
        lam = 4 + int(3 * np.log(d))
        # Weights for ranking
        mu = lam // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1 / weights.sum()
        
        # Step size adaptation parameters
        cs = (mueff + 2) / (d + mueff + 5)
        ds = 1 + 2 * max(0, (mueff - 1) / (d + 1)) - cs
        c1 = (2 + 2 / (d + 1.3)) / ((d + 1) ** 2 + mueff + 2)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((d + 2) ** 2 + mueff))
        
        # Initial state
        x = np.zeros(d)
        sigma = 2.0
        C = np.eye(d)
        C_sqrt = np.eye(d)
        pc = np.zeros(d)
        ps = np.zeros(d)
        
        # Precompute square root of C (initially identity)
        # We will update C and recompute sqrt each step for simplicity and robustness
        # given small d=5
        
        best_x = x
        best_f = float('inf')
        
        evals_used = 0
        max_evals = self.budget
        
        # Initial population
        # We need to evaluate the initial point x=0? Or just start loop?
        # CMA-ES usually samples lambda points.
        
        # To ensure we use the budget efficiently and handle the "first call" constraint,
        # we will evaluate the initial mean first if possible, or just let the first generation
        # do the work. The task says "exception before the first call makes result invalid".
        # So we must call func at least once.
        
        # Let's do a standard loop.
        # We track the best solution found so far.
        
        # Initial evaluation of the mean to have a baseline
        # This counts as 1 eval.
        f_x = func(x)
        evals_used += 1
        if f_x < best_f:
            best_f = f_x
            best_x = x.copy()
            
        # If we hit deadline immediately, return
        if time.time() > deadline:
            return best_x, best_f

        while evals_used < max_evals and time.time() < deadline:
            # Check if we have enough budget for a full generation
            # If not, we might do a partial generation or just break.
            # CMA-ES works best with full generations.
            # If remaining evals < lam, we can just do random local search or break.
            # Given 500 budget and lam=8, we have ~62 full generations.
            
            if max_evals - evals_used < lam:
                break
                
            # Sample population
            # Compute square root of C
            # For d=5, eigendecomposition is fast
            eigvals, eigvecs = np.linalg.eigh(C)
            # Clip negative eigenvalues due to numerical errors
            eigvals = np.clip(eigvals, 1e-12, None)
            C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals))
            
            z = np.random.standard_normal((lam, d))
            y = z @ C_sqrt
            x_pop = x + sigma * y
            
            # Evaluate population
            f_pop = np.zeros(lam)
            for i in range(lam):
                if evals_used >= max_evals:
                    break
                # Clip to box [-5, 5]
                xi = np.clip(x_pop[i], -5, 5)
                f_pop[i] = func(xi)
                evals_used += 1
                
                if f_pop[i] < best_f:
                    best_f = f_pop[i]
                    best_x = xi.copy()
            
            # If we ran out of budget mid-generation, stop
            if evals_used >= max_evals:
                break
                
            # Sort by fitness
            idx = np.argsort(f_pop)
            y_sorted = y[idx]
            x_sorted = x_pop[idx]
            f_sorted = f_pop[idx]
            
            # Update mean
            w = weights
            x_new = np.sum(w[:, None] * x_sorted[:mu], axis=0)
            
            # Update evolution paths
            y_mean = np.sum(w[:, None] * y_sorted[:mu], axis=0)
            
            # Update pc
            pc = (1 - cs) * pc + cs / np.sqrt(ds) * (y_mean / sigma)
            
            # Update ps
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * ds) * (y_mean / sigma)
            
            # Heffron condition for C update
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (evals_used // lam + 1)))) < \
                   (1 + 2 / (d + 1)) * np.sqrt(1 + (1 - cs) / cs)
            
            # Update C
            rank_mu = np.sum(w[:, None, None] * np.outer(y_sorted[:mu], y_sorted[:mu]), axis=0)
            rank_one = np.outer(ps, ps)
            
            if hsig:
                C = (1 + c1 * (2 - c1) - c1 * cmu) * C + cmu * rank_mu + c1 * (rank_one - (1 - cs) * C)
            else:
                C = (1 + c1 * (2 - c1) - c1 * cmu) * C + cmu * rank_mu
            
            # Ensure C is symmetric
            C = (C + C.T) / 2
            
            # Update sigma
            sigma = sigma * np.exp(cs / ds * (np.linalg.norm(ps) / np.sqrt(2 * d) - 1))
            
            # Update mean
            x = x_new
            
            # Optional: Restart or keep going.
            # For this task, a single run is usually sufficient.
            
        return best_x, best_f
# EVOLVE-BLOCK-END
