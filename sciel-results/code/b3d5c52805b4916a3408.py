import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    """
    Optimizer using Sobol global sampling followed by CMA-ES local refinement.
    """
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin for 300s limit
        
        d = self.dim
        total_budget = self.budget
        
        # Phase 1: Global Exploration with Sobol Sequence
        # We allocate ~60% of the budget to global exploration to ensure good coverage
        # 300 points is a reasonable number for 5D to get a decent landscape overview.
        n_global = min(300, total_budget // 2)
        
        if n_global > 0:
            sobol = qmc.Sobol(d, scramble=True)
            # Generate points in [0, 1]^d
            # Sobol requires the number of points to be a power of 2 for the generator,
            # but we can just generate a large batch and take the first n_global.
            # Actually, qmc.Sobol.random(n) works for any n, but it's faster if n is power of 2.
            # Let's just use random(n_global) which handles it internally.
            points_unit = sobol.random(n_global)
            # Scale to [-5, 5]^d
            points = points_unit * 10.0 - 5.0
            
            x_best = None
            f_best = float('inf')
            
            for p in points:
                if time.time() > deadline:
                    break
                fp = func(p)
                if fp < f_best:
                    f_best = fp
                    x_best = p.copy()
        
        if x_best is None:
            # Fallback if no points were evaluated
            x_best = np.zeros(d)
            f_best = func(x_best)
            n_global = 1 # Adjust budget accounting

        # Phase 2: Local Refinement with CMA-ES
        remaining_budget = total_budget - n_global
        
        if remaining_budget > 0:
            self._run_cma_es(func, x_best, f_best, remaining_budget, deadline)
            
        return x_best, f_best

    def _run_cma_es(self, func, x0, f0, budget, deadline):
        """
        Lightweight CMA-ES implementation for low dimensions.
        """
        d = self.dim
        # Population size lambda
        lam = max(4 + int(3 * np.log(d)), 10)
        
        # Initial step size
        # Since we started from a good global candidate, a moderate step size is good.
        sigma = 0.5
        
        # Mean
        m = x0.copy()
        
        # Covariance matrix
        C = np.eye(d)
        
        # Evolution path
        ps = np.zeros(d)
        pc = np.zeros(d)
        
        # Strategy parameters
        alpha_mu = 1.0
        mu = lam // 2  # Number of selected parents
        
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mu_eff = 1.0 / np.sum(w**2)
        
        # Damping parameters
        c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
        c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        c_1 = 2 / ((d + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((d + 2)**2 + mu_eff))
        
        # Initial values
        chi_d = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d**2))
        
        # Loop
        calls = 0
        best_x = x0.copy()
        best_f = f0
        
        while calls < budget:
            if time.time() > deadline:
                break
            
            # Generate population
            population = []
            for i in range(lam):
                if time.time() > deadline:
                    break
                # Standard normal
                z = np.random.default_rng().standard_normal(d)
                # Transform using Cholesky of C
                # For small d, we can just do a simple update or use a cached Cholesky
                # To keep it simple and fast, we approximate the sampling
                # Actually, for d=5, we can afford a proper Cholesky every step or every few steps.
                # Let's update Cholesky every step for correctness in this low dim.
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # If C is not positive definite, reset it
                    C = np.eye(d)
                    L = np.eye(d)
                
                y = L @ z
                x_new = m + sigma * y
                # Clip to box
                x_new = np.clip(x_new, -5, 5)
                fx = func(x_new)
                calls += 1
                population.append((fx, x_new))
                
                if fx < best_f:
                    best_f = fx
                    best_x = x_new.copy()
            
            if not population:
                break
                
            # Sort by fitness
            population.sort(key=lambda x: x[0])
            
            # Update mean
            m_new = m.copy()
            for i in range(mu):
                m_new += w[i] * population[i][1]
            m = m_new
            
            # Update covariance matrix
            # Rank-mu update
            delta_s = (m - m_new) / sigma
            # Rank-one update
            C += (c_1 + c_mu) * (np.outer(delta_s, delta_s) - C)
            
            # Ensure C is symmetric
            C = (C + C.T) / 2
            
            # Update step size (CSA)
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (C ** -0.5) @ delta_s
            # Note: C**-0.5 is expensive. For d=5, it's okay.
            # But we can approximate or use the evolution path trick.
            # Let's stick to a simpler step size adaptation for robustness and speed:
            # If success rate is around 1/3, keep sigma.
            # This is a simplification, but CMA-ES with fixed sigma often works well if sigma is chosen well.
            # Let's use a simple success rule for sigma to avoid complex matrix inversions.
            
            # Success rule:
            # If more than 1/3 of the population improved the best, increase sigma slightly?
            # No, standard (1+1) logic doesn't apply directly.
            # Let's use a fixed sigma or a very slow adaptation.
            # Given the budget is small, a fixed sigma of 0.1-0.5 might be best.
            # Let's decay sigma slowly to ensure convergence.
            sigma *= 0.99
            
            # If we are close to convergence, break
            if sigma < 1e-8:
                break

        # Return the best found
        # The caller keeps the best_x, best_f from its own scope, 
        # but we need to update them. 
        # Since we can't easily return them to the parent scope without modifying the signature,
        # we will store them in self or return them.
        # The __call__ method returns x_best, f_best. 
        # We need to update the local variables in __call__.
        # To do this cleanly, we can return the best from _run_cma_es.
        
        return best_x, best_f

# Modify __call__ to use the return value of _run_cma_es
class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        
        d = self.dim
        total_budget = self.budget
        
        n_global = min(300, total_budget // 2)
        
        x_best = None
        f_best = float('inf')
        
        if n_global > 0:
            sobol = qmc.Sobol(d, scramble=True)
            points_unit = sobol.random(n_global)
            points = points_unit * 10.0 - 5.0
            
            for p in points:
                if time.time() > deadline:
                    break
                fp = func(p)
                if fp < f_best:
                    f_best = fp
                    x_best = p.copy()
        
        if x_best is None:
            x_best = np.zeros(d)
            f_best = func(x_best)
            n_global = 1

        remaining_budget = total_budget - n_global
        
        if remaining_budget > 0:
            x_best, f_best = self._run_cma_es(func, x_best, f_best, remaining_budget, deadline)
            
        return x_best, f_best

    def _run_cma_es(self, func, x0, f0, budget, deadline):
        d = self.dim
        lam = max(4 + int(3 * np.log(d)), 10)
        sigma = 0.5
        m = x0.copy()
        C = np.eye(d)
        
        # Simplified CMA-ES with fixed covariance adaptation to avoid heavy matrix ops
        # We will use a simplified version that updates the mean and a simple step size.
        # This is effectively a random walk with a bias towards the best points.
        
        best_x = x0.copy()
        best_f = f0
        
        # Weights for recombination
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        
        calls = 0
        while calls < budget:
            if time.time() > deadline:
                break
            
            population = []
            for i in range(lam):
                if time.time() > deadline:
                    break
                z = np.random.default_rng().standard_normal(d)
                # Simple sampling: m + sigma * z
                # To adapt to rotation, we should update C. 
                # Let's do a very light C update.
                # For d=5, Cholesky is fast.
                try:
                    L = np.linalg.cholesky(C)
                except:
                    L = np.eye(d)
                
                x_new = m + sigma * (L @ z)
                x_new = np.clip(x_new, -5, 5)
                fx = func(x_new)
                calls += 1
                population.append((fx, x_new))
                if fx < best_f:
                    best_f = fx
                    best_x = x_new.copy()
            
            if not population:
                break
                
            population.sort(key=lambda x: x[0])
            
            # Update mean
            m_new = np.zeros(d)
            for i in range(mu):
                m_new += w[i] * population[i][1]
            
            # Update Covariance (Simplified Rank-Mu)
            # C = (1 - c1) * C + c1 * sum(wi * (xi - m_new)(xi - m_new)^T)
            # This is a simplified version without the rank-1 term from the mean update.
            C_new = np.zeros((d, d))
            for i in range(mu):
                diff = population[i][1] - m_new
                C_new += w[i] * np.outer(diff, diff)
            
            # Blend with old C to maintain diversity
            C = 0.9 * C + 0.1 * C_new
            C = (C + C.T) / 2 # Symmetrize
            
            # Update Step Size
            # Simple success rule: if the best point is significantly better than the mean, we might be on the right track.
            # Let's just decay sigma slowly to ensure convergence.
            sigma *= 0.95
            
            m = m_new
            
            if sigma < 1e-8:
                break
                
        return best_x, best_f
