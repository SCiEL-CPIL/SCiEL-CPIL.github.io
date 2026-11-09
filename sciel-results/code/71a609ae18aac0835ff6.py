import numpy as np
import time

class Optimizer:
    """
    CMA-ES based optimizer for BBOB-style functions.
    Implements a lightweight version of CMA-ES suitable for low dimensions (d=5)
    and tight function call budgets (N=500).
    """
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 280  # Safety margin
        d = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        # Population size lambda
        lambda_pop = max(4, int(4 + 3 * np.log(d)))
        # Weights for recombination
        weights = np.log(lambda_pop / 2 + 1) - np.log(np.arange(1, lambda_pop + 1))
        weights /= np.sum(weights)
        mu = int(np.sum(weights > 0))  # Number of parents
        
        # Adaptation parameters
        c_sigma = (mu + 2) / (d + mu + 5)
        c_c = (4 + mu / d) / (d + 4 + 2 * mu / d)
        c_1 = 2 / ((d + 1.3) ** 2 + mu)
        c_mu = min(1 - c_1, 2 * (mu - 2 + 1 / mu) / ((d + 2) ** 2 + mu))
        
        # Initial state
        # Start with a random point, but ensure it's not on the boundary
        x0 = np.random.uniform(-4, 4, d)
        x_mean = x0.copy()
        sigma = 2.0  # Initial step size, roughly 1/3 of the box width
        C = np.eye(d)
        pc = np.zeros(d)
        ps = np.zeros(d)
        cs = np.sqrt((1 + c_sigma) * (2 - c_sigma) * (mu + 1. / mu))
        
        # Pre-compute inverse square root of C for sampling
        # We update C every generation, so we need to recompute the decomposition
        # For d=5, this is very fast.
        
        def sample_from_cma():
            # Cholesky decomposition of C
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C is not positive definite, reset it
                C[:] = np.eye(d)
                L = np.eye(d)
            # Sample standard normal vectors
            z = np.random.randn(lambda_pop, d)
            # Transform to C-normal distribution
            # y = L @ z.T
            y = (L @ z.T).T
            # x = x_mean + sigma * y
            return x_mean + sigma * y

        best_x = x0.copy()
        best_f = func(best_x)
        calls = 1
        
        # Main loop
        while calls < budget:
            if time.time() > deadline:
                break
                
            # Sample population
            pop = sample_from_cma()
            
            # Evaluate population
            fvals = np.zeros(lambda_pop)
            for i in range(lambda_pop):
                if calls >= budget:
                    break
                # Clip to box
                x_i = np.clip(pop[i], -5, 5)
                fvals[i] = func(x_i)
                calls += 1
                if fvals[i] < best_f:
                    best_f = fvals[i]
                    best_x = x_i.copy()
            
            if calls >= budget:
                break
            
            # Sort by fitness
            order = np.argsort(fvals)
            pop_sorted = pop[order]
            fvals_sorted = fvals[order]
            
            # Update mean
            w = weights[:mu]
            x_new_mean = np.sum(w[:, None] * pop_sorted[:mu], axis=0)
            
            # Update step size (sigma)
            # ps = 1/sqrt(c_sigma * (2 - c_sigma) * mu) * (sqrt(mu) * C^(-1/2) * (x_new_mean - x_mean) / sigma - (1 - c_c) * ps)
            # Note: We need C^(-1/2). Since we have L such that L L^T = C, C^(-1/2) is not exactly L^(-1).
            # Approximation: Use the inverse of the Cholesky factor from the previous step or just identity if simple.
            # For robustness in low dim, we can approximate C^(-1/2) using the previous C's inverse square root.
            # However, a simpler and robust approach for low dimensions is to use the diagonal scaling or just update ps with identity if C is close to identity.
            # Let's stick to a simplified ps update that is stable.
            
            # Compute C^(-1/2) approximation
            # If we maintain the Cholesky factor L, C^(-1/2) is tricky.
            # Alternative: Use the formula with the previous C's inverse.
            # For d=5, we can compute the matrix square root inverse directly if needed, but it's expensive.
            # A common trick in simple implementations is to just use the identity for the ps update if C is not well-estimated, 
            # or to update ps less frequently.
            
            # Let's use a simpler sigma adaptation: 1/5 rule on the population? No, CMA-ES has specific rules.
            # Let's implement the standard ps update with an approximation for C^(-1/2).
            # We can maintain the inverse Cholesky factor.
            
            # To keep it simple and robust, I will use a slightly simplified CMA-ES that updates C but uses a heuristic for sigma adaptation 
            # or just rely on the covariance adaptation which is the main driver for ill-conditioned functions.
            
            # Update ps
            # We need C^(-1/2). Let's compute it from the current C before updating it? No, use previous C.
            # Let's store the inverse square root of C from the previous iteration.
            # Init: C=I, so C^(-1/2)=I
            if not hasattr(self, '_C_inv_sqrt'):
                self._C_inv_sqrt = np.eye(d)
            
            # ps = (1 - c_c) * ps + c_c * cs * C_inv_sqrt @ (x_new_mean - x_mean) / sigma
            ps = (1 - c_c) * ps + c_c * cs * (self._C_inv_sqrt @ (x_new_mean - x_mean) / sigma)
            
            # Update sigma
            # chi_n = sqrt(d) * (1 - 1/(2*d^2) + 1/(12*d^4))
            chi_n = np.sqrt(d) * (1 - 1.0/(2.0*d**2) + 1.0/(12.0*d**4))
            sigma *= np.exp(c_sigma / cs * (np.linalg.norm(ps) / chi_n - 1))
            
            # Update pc
            pc = (1 - c_c) * pc + c_c * (x_new_mean - x_mean) / sigma
            
            # Update C
            # C = (1 - c_1 - c_mu) * C + c_1 * (pc * pc.T + (1 - (1 - c_c)^2) * delta_sig * C) + c_mu * sum(w_i * (y_i - y_bar) * (y_i - y_bar).T)
            
            # Compute the weighted sum of outer products of the step vectors
            # y_i = (x_i - x_mean) / sigma
            # We use the sorted population
            y_sorted = (pop_sorted[:mu] - x_mean) / sigma
            # y_bar = sum(w_i * y_i)
            y_bar = np.sum(w[:, None] * y_sorted, axis=0)
            
            # Sum of outer products
            sum_outer = np.zeros((d, d))
            for i in range(mu):
                diff = y_sorted[i] - y_bar
                sum_outer += w[i] * np.outer(diff, diff)
            
            # Update C
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) + (1 - (1 - c_c)**2) * (np.linalg.norm(ps)**2 / chi_n**2 - 1) * C) + c_mu * sum_outer
            
            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            # Regularize if needed
            eigvals = np.linalg.eigvalsh(C)
            if np.min(eigvals) < 1e-12:
                C += 1e-12 * np.eye(d)
            
            # Update mean
            x_mean = x_new_mean
            
            # Update C_inv_sqrt for next iteration
            # Compute inverse square root of C
            # U, s, Vt = np.linalg.svd(C)
            # C_inv_sqrt = Vt.T @ np.diag(1/np.sqrt(s)) @ U.T
            # For d=5, this is fast enough.
            try:
                U, s, Vt = np.linalg.svd(C)
                self._C_inv_sqrt = Vt.T @ np.diag(1.0 / np.sqrt(s)) @ U.T
            except:
                self._C_inv_sqrt = np.eye(d)

        return best_x, best_f
