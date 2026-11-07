"""Optimizer using CMA-ES (Covariance Matrix Adaptation Evolution Strategy) for box-constrained minimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for box-constrained minimization on [-5,5]^5."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        call_count = [0]

        def wrapped_func(x):
            if time.time() > deadline:
                raise TimeoutError()
            call_count[0] += 1
            val = float(func(x))
            return val

        def time_ok():
            return time.time() < deadline

        # CMA-ES Parameters
        lam = 9
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mueff = 1.0 / np.sum(w**2)
        
        # Init
        mean = np.zeros(dim)
        sigma = 1.0
        C = np.eye(dim)
        C_sqrt = np.eye(dim)
        C_inv_sqrt = np.eye(dim)
        
        # Control params
        cs = (mueff + 1) / (dim + mueff + 5)
        ds = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
        cmu = min(1 - cs, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) * (mueff + 5)))
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c1 = (2 - mueff / (dim + 2)) / ((dim + 2) ** 2 + mueff)
        
        # Norm of standard normal
        chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(32*dim**2))
        
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        
        # Initial eval
        if call_count[0] < budget:
            best_f = wrapped_func(mean)
            best_x = mean.copy()
        else:
            return mean, best_f

        gen = 0
        while call_count[0] < budget and time_ok():
            gen += 1
            # Sample
            pop = []
            for i in range(lam):
                z = np.random.randn(dim)
                x = mean + sigma * (C_sqrt @ z)
                x = np.clip(x, -5, 5)
                pop.append(x)
            
            # Evaluate
            f_pop = []
            for x in pop:
                if call_count[0] >= budget or not time_ok():
                    break
                f = wrapped_func(x)
                f_pop.append(f)
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
            
            if len(f_pop) < lam:
                break

            idx = np.argsort(f_pop)
            f_sorted = np.array(f_pop)[idx]
            x_sorted = np.array(pop)[idx]
            
            # Mean update
            mean_old = mean
            mean = np.sum(w[:, np.newaxis] * x_sorted[:mu], axis=0)
            
            # Step size update
            y = (mean - mean_old) / sigma
            # hsig check
            norm_ps = np.linalg.norm(ps)
            hsig = 1.0 if norm_ps < (1.4 + 2.0/(dim+1.0)) * np.sqrt((mueff+1.0)/mueff - 1.0) else 0.0
            
            # ps update
            ps = (1 - cs) * ps + hsig * np.sqrt(cs * (2 - cs) * mueff) * (C_inv_sqrt @ y)
            
            # sigma update
            sigma *= np.exp(cs/ds * (np.linalg.norm(ps) / chi_n - 1))
            
            # Covariance update
            # pc update
            norm_pc = np.linalg.norm(pc)
            hc = 1.0 if norm_pc < (1.2 + 2.0/(dim+1.0)) * np.sqrt((mueff+1.0)/mueff - 1.0) else 0.0
            pc = (1 - cc) * pc + hc * np.sqrt(cc * (2 - cc) * mueff) * y
            
            # Rank-one and rank-mu updates
            y_parents = (x_sorted[:mu] - mean_old) / sigma
            
            rank_one = np.outer(y, y)
            if hsig == 0:
                rank_one += cs * (2 - cs) * np.outer(ps, ps)
            
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                rank_mu += w[i] * np.outer(y_parents[i], y_parents[i])
                
            C = (1 - c1 - cmu) * C + c1 * rank_one + cmu * rank_mu
            
            # Update C_sqrt
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-10)
            C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            C_inv_sqrt = eigvecs @ np.diag(1.0/np.sqrt(eigvals)) @ eigvecs.T

        if best_x is None:
            best_x = np.zeros(dim)
            if call_count[0] < budget:
                best_f = wrapped_func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
