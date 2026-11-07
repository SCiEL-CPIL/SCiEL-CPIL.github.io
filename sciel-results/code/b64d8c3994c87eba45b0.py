"""Optimized program for BBOB f21 (Gallagher 101 peaks) using population-based CMA-ES."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES with restarts for BBOB f21."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320  # Safety margin before 330s kill
        
        n_calls = 0
        best_f = float('inf')
        best_x = None
        dim = self.dim
        budget = self.budget

        def wrapped_func(x):
            nonlocal n_calls
            if n_calls >= budget:
                return 1e18
            n_calls += 1
            return func(x)

        rng = np.random.default_rng(42)

        # CMA-ES parameters
        # Population size lambda
        lambda_pop = 10
        # Number of best solutions to select
        mu = lambda_pop // 2
        
        # Weights for selected solutions
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        
        # Variance reduction factor
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * ((mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff)))
        
        # Chi-square expectation
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        
        def run_cma_es(x0, sigma0, max_evals):
            nonlocal n_calls, best_f, best_x
            nonlocal deadline, start_time
            
            # State variables
            x_mean = x0.copy()
            sigma = sigma0
            B = np.eye(dim)
            D = np.ones(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            
            # Count evaluations used in this run
            local_calls = 0
            
            while local_calls < max_evals and n_calls < budget and time.time() < deadline - 5:
                # Generate population
                ys = []
                fs = []
                
                for i in range(lambda_pop):
                    if n_calls >= budget or time.time() > deadline - 5:
                        break
                    # Sample from N(0, I)
                    z = rng.standard_normal(dim)
                    y = z
                    # Transform to x-space
                    x = x_mean + sigma * (B @ (D * y))
                    # Clip to box [-5, 5]
                    x = np.clip(x, -5, 5)
                    
                    fx = wrapped_func(x)
                    ys.append(y)
                    fs.append((fx, x))
                    local_calls += 1
                
                if len(fs) < lambda_pop:
                    break
                
                # Sort by fitness
                fs.sort(key=lambda t: t[0])
                
                # Update best
                if fs[0][0] < best_f:
                    best_f = fs[0][0]
                    best_x = fs[0][1].copy()
                    
                # Update mean
                y_w = np.array([y for f, y in [(f, None) for f, _ in fs[:mu]]]) # Placeholder
                # Actually we need the y corresponding to the best mu solutions
                # Let's re-extract properly
                # fs is list of (f, x). We need y.
                # We stored y in ys, but sorted fs. We need to sort ys accordingly.
                # Better approach: sort indices.
                
                # Redo sorting with indices
                indices = np.argsort([f for f, _ in fs])
                y_w = np.array([ys[i] for i in indices[:mu]])
                x_w = np.array([fs[i][1] for i in indices[:mu]])
                
                new_x_mean = x_mean + sigma * (B @ (D * (weights @ y_w)))
                
                # Update path
                # Inverse of B@D
                # Since B is orthogonal and D is diagonal, (B@D)^-1 = D^-1 @ B.T
                # Let's compute B_inv_D_inv
                # Actually, it's easier to compute the step in the transformed space
                # y_mean = (B@D)^-1 @ (new_x_mean - x_mean) / sigma
                # y_mean = D^-1 @ B.T @ (new_x_mean - x_mean) / sigma
                
                y_mean = (1.0 / D) * (B.T @ ((new_x_mean - x_mean) / sigma))
                
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_mean
                norm_ps = np.linalg.norm(ps)
                
                if norm_ps / chi_n < (1.4 + 2 / (dim + 1)) * c_sigma:
                    pc = (1 - c_c) * pc + c_c * y_mean
                    H = np.outer(pc, pc)
                    H = (1 - c_1 - c_mu) * H + c_1 * (np.eye(dim) - pc @ pc.T)
                    # Wait, standard update is:
                    # H = (1 - c_1 - c_mu) * C + c_1 * (1 + (1-c_c)*c_c*(1+norm_ps^2/chi_n^2 - 1) * outer(pc,pc)) + c_mu * sum(wi * outer(yi, yi))
                    # Let's stick to a simplified but robust version or use the standard formula.
                    
                    # Standard CMA-ES update for Covariance Matrix C
                    # C = (1 - c_1 - c_mu) * C + c_1 * (1 + (1-c_c)*c_c*(1+||ps||^2/chi_n^2 - 1) * pc*pc^T) + c_mu * sum(wi * yi*yi^T)
                    
                    # We are using B and D decomposition.
                    # Let's update C directly and then decompose.
                    # This might be slower but more standard.
                    
                    # Let's switch to maintaining C directly for simplicity and correctness.
                    pass # Placeholder, we will rewrite the loop to use C directly.

            return

        # Since the above inline implementation of CMA-ES with B/D decomposition is getting complex and error-prone in a short script,
        # and considering the strict time limit and small dimension (5),
        # I will implement a robust CMA-ES using the standard covariance matrix C directly.
        # This is computationally heavier but for dim=5, C is 5x5, so decomposition is trivial.
        
        def run_cma_es_v2(x0, sigma0, max_evals):
            nonlocal n_calls, best_f, best_x
            nonlocal deadline, start_time
            
            x_mean = x0.copy()
            sigma = sigma0
            C = np.eye(dim)
            B = np.eye(dim)
            D = np.ones(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            
            local_calls = 0
            last_best_f = best_f
            
            while local_calls < max_evals and n_calls < budget and time.time() < deadline - 5:
                # Generate population
                ys = []
                xs = []
                fs = []
                
                for i in range(lambda_pop):
                    if n_calls >= budget or time.time() > deadline - 5:
                        break
                    z = rng.standard_normal(dim)
                    # Cholesky or Eigen decomposition of C
                    # For dim=5, we can just do eigen decomposition every step? 
                    # Or maintain B and D.
                    # Let's use B and D to generate samples efficiently.
                    y = B @ (D * z)
                    x = x_mean + sigma * y
                    x = np.clip(x, -5, 5)
                    
                    fx = wrapped_func(x)
                    ys.append(y)
                    xs.append(x)
                    fs.append(fx)
                    local_calls += 1
                
                if len(fs) < lambda_pop:
                    break
                
                # Sort
                indices = np.argsort(fs)
                y_w = np.array([ys[i] for i in indices[:mu]])
                x_w = np.array([xs[i] for i in indices[:mu]])
                f_w = np.array([fs[i] for i in indices[:mu]])
                
                # Update best
                if f_w[0] < best_f:
                    best_f = f_w[0]
                    best_x = x_w[0].copy()
                    
                # Update mean
                new_x_mean = x_mean + sigma * (weights @ y_w)
                
                # Update paths
                # y_mean in transformed space
                # y_w are already in the transformed space (z -> B@D@z)
                # So y_mean = weights @ y_w
                y_mean = weights @ y_w
                
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_mean
                norm_ps = np.linalg.norm(ps)
                
                if norm_ps / chi_n < (1.4 + 2 / (dim + 1)) * c_sigma:
                    pc = (1 - c_c) * pc + c_c * y_mean
                    hsig = (norm_ps / chi_n) < (1.4 + 2 / (dim + 1)) * c_sigma
                    delta_sig = hsig
                else:
                    pc = (1 - c_c) * pc
                    delta_sig = 0
                
                # Update Covariance Matrix C
                # C = (1 - c_1 - c_mu) * C + c_1 * (1 + (1-c_c)*c_c*(1+delta_sig) * pc*pc^T) + c_mu * sum(wi * yi*yi^T)
                
                # Note: The term with pc is in the transformed space?
                # Yes, pc and ps are in the transformed space (space of z).
                # But C is the covariance in the z-space?
                # In standard CMA-ES, C is the covariance of the distribution in the z-space (standard normal space).
                # The step is sigma * B @ D @ z.
                # So C is the covariance of z.
                
                rank_one = np.outer(pc, pc)
                rank_mu = np.zeros((dim, dim))
                for i in range(mu):
                    rank_mu += weights[i] * np.outer(y_w[i], y_w[i])
                    
                C = (1 - c_1 - c_mu) * C + c_1 * (1 + (1 - c_c) * c_c * (1 + delta_sig)) * rank_one + c_mu * rank_mu
                
                # Ensure symmetry
                C = (C + C.T) / 2
                
                # Decompose C = B @ diag(D^2) @ B.T
                # Eigen decomposition
                eigenvalues, eigenvectors = np.linalg.eigh(C)
                # Clip small negative eigenvalues
                eigenvalues = np.clip(eigenvalues, 1e-10, None)
                D = np.sqrt(eigenvalues)
                B = eigenvectors
                
                # Update sigma
                sigma *= np.exp((c_sigma / chi_n) * (norm_ps - chi_n))
                
                x_mean = new_x_mean
                
                # Restart condition
                if best_f == last_best_f and local_calls > 5 * dim:
                    break
                last_best_f = best_f

        # Main loop: Multiple restarts
        n_restarts = 20
        evals_per_restart = max(50, budget // n_restarts)
        
        for r in range(n_restarts):
            if n_calls >= budget or time.time() > deadline - 10:
                break
            
            # Random start
            x0 = rng.uniform(-4, 4, dim)
            # Start with a moderate step size
            sigma0 = 1.5
            
            run_cma_es_v2(x0, sigma0, evals_per_restart)
            
            # If we found a very good solution, we might want to exploit more
            # But for f21, finding *any* peak is hard.
            # If best_f is still very high, we continue restarting.
            
        return best_x, best_f
# EVOLVE-BLOCK-END
