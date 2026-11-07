"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Single robust CMA-ES with adaptive population size and time bounding."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320  # 10s safety margin before 330s kill

        budget = self.budget
        dim = self.dim
        
        best_x = None
        best_f = np.inf
        calls_made = 0

        def call_func(x):
            nonlocal calls_made, best_f, best_x
            if calls_made >= budget:
                return np.inf
            calls_made += 1
            fx = float(func(x))
            if fx < best_f:
                best_f = fx
                best_x = np.array(x, copy=True)
            return fx

        def check_deadline():
            return (time.time() - start_time) > deadline

        # Initialize at origin, which is a reasonable starting point for BBOB instances
        x0 = np.zeros(dim)
        f0 = call_func(x0)
        
        if best_x is None:
            return np.zeros(dim), f0

        # CMA-ES Parameters
        # We start with a small population and increase it if we have budget left
        # Standard CMA-ES population size: lambda = 4 + floor(3 * log(n))
        # For n=5, this is 4 + floor(3*1.6) = 4 + 4 = 8.
        # We will start with 8 and potentially increase to 16 or 24 if we have time.
        
        # Adaptive strategy:
        # Run CMA-ES with a fixed pop_size. If we finish early (unlikely with 500 budget) or if we stagnate, 
        # we could restart, but a single run with good parameters is often best for BBOB.
        
        # Let's use a single robust CMA-ES run.
        # To maximize exploration early, we might use a slightly larger population initially.
        # However, larger populations require more evaluations per iteration.
        # With 500 evals and dim=5:
        # Pop 8: ~62 iterations.
        # Pop 16: ~31 iterations.
        
        # Let's stick to a standard, well-tuned CMA-ES.
        
        def run_cma_es(x_start, f_start, max_calls, seed=123):
            nonlocal calls_made, best_f, best_x, budget
            
            if calls_made >= budget or check_deadline():
                return x_start, f_start
            
            rng = np.random.default_rng(seed)
            
            # CMA-ES hyperparameters
            pop_size = 4 + int(3 * np.log(dim))
            # For dim=5, pop_size = 8. Let's try 12 for more exploration? 
            # Actually, standard is usually sufficient. Let's use 4 + 3*log(5) = 8.
            # But for Weierstrass (multi-modal), a larger population might help.
            # Let's use pop_size = 12.
            pop_size = 12
            
            cc = (4 + dim / (dim + 4)) / ((dim + 4) + 4 + dim / (dim + 4))
            cs = (dim + 1.7) / (dim + 22)
            c1 = 2 / ((dim + 1.3) ** 2 + 10)
            cmu = min(0.3 - c1, 2 / ((dim + 2) ** 2 - 1))
            dsigma = 1 + 2 * max(0, np.sqrt(dim) - 1) / (dim + 3)
            
            # Initialize
            x_mean = x_start.copy()
            sigma = 1.0  # Initial step size. For [-5,5] box, 1.0 is reasonable.
            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
            
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                L = np.eye(dim)
                
            # Weights
            w = np.log(pop_size/2 + 1) - np.arange(1, pop_size+1)
            w = w / w.sum()
            mu_eff = 1 / np.sum(w[:int(2*pop_size/2)]**2)
            
            it = 0
            while calls_made < budget and not check_deadline():
                if f_start < best_f:
                    best_f = f_start
                    best_x = x_mean.copy()
                    
                # Generate offspring
                y_sorted = []
                f_y = []
                for k in range(pop_size):
                    if calls_made >= budget or check_deadline():
                        break
                    z = rng.standard_normal(dim)
                    y = x_mean + sigma * (L @ z)
                    y = np.clip(y, -5, 5)
                    fy = call_func(y)
                    y_sorted.append(y)
                    f_y.append(fy)
                
                if len(y_sorted) < 2:
                    break
                
                # Sort by fitness
                indices = np.argsort(f_y)
                y_sorted = np.array([y_sorted[i] for i in indices])
                f_y = [f_y[i] for i in indices]
                
                # Update mean
                y_mean_new = np.sum(w[:int(2*pop_size/2)][:, None] * y_sorted[:int(2*pop_size/2)], axis=0)
                
                # Update path cumulation
                pc = (1 - cc) * pc + cc * (1/sigma) * (y_mean_new - x_mean)
                hsig = (np.linalg.norm(pc) / np.sqrt(1 - (1-cc)**(2*(it+1)))) / chi_n < (1.4 + 2/(dim+1))
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * hsig) * (1/sigma) * (y_mean_new - x_mean)
                
                # Update covariance matrix
                delta = (1 + hsig * cs * (2 - cs)) * c1 * (np.outer(pc, pc) - C)
                rank_mu = (y_sorted[:int(2*pop_size/2)] - x_mean)
                delta += cmu * np.sum(w[:int(2*pop_size/2)] * np.outer(rank_mu, rank_mu), axis=0)
                C = (1 + c1 + cmu - c1) * C + delta
                
                # Ensure C is positive definite and symmetric
                C = (C + C.T) / 2
                eigvals = np.linalg.eigvalsh(C)
                if eigvals.min() < 1e-12:
                    C += (1e-12 - eigvals.min()) * np.eye(dim)
                
                # Update sigma
                sigma *= np.exp(1/dsigma * (np.linalg.norm(ps)/chi_n - 1))
                sigma = np.clip(sigma, 1e-12, 1e12)
                
                # Update mean
                x_mean = y_mean_new
                f_start = f_y[0]
                
                # Update best
                if f_start < best_f:
                    best_f = f_start
                    best_x = x_mean.copy()
                    
                # Recompute Cholesky
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    L = np.eye(dim)
                    
                it += 1
                
                # Stopping criteria
                if sigma < 1e-10 or sigma > 1e10:
                    break
                if it > 200:
                    break
                
            return x_mean, f_start

        # Run the single CMA-ES
        cma_x, cma_f = run_cma_es(x0, f0, budget - calls_made, seed=42)
        
        if cma_f < best_f:
            best_f = cma_f
            best_x = cma_x.copy()
            
        # If we have significant budget left, do a second restart from a different point
        # This helps if the first run got stuck in a local minimum.
        if calls_made < budget * 0.8 and not check_deadline() and best_x is not None:
            rng_restart = np.random.default_rng(1234)
            # Start from a random point or slightly perturbed best
            x_start2 = np.clip(best_x + 2.0 * rng_restart.standard_normal(dim), -5, 5)
            f_start2 = call_func(x_start2)
            
            if f_start2 < best_f:
                best_f = f_start2
                best_x = x_start2.copy()
            
            if calls_made < budget and not check_deadline():
                cma_x2, cma_f2 = run_cma_es(x_start2, f_start2, budget - calls_made, seed=456)
                if cma_f2 < best_f:
                    best_f = cma_f2
                    best_x = cma_x2.copy()

        return best_x, best_f

# EVOLVE-BLOCK-END
