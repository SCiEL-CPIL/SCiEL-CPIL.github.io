"""CMA-ES optimizer for BBOB f14."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer with adaptive population size and boundary handling."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 320.0  # Safety margin before 330s kill
        dim = self.dim
        budget = self.budget
        calls = 0

        # CMA-ES parameters
        # Standard CMA-ES lambda is 4 + floor(3*log(n))
        # For n=5, lambda = 4 + 3*1.6 = ~8.8 -> 12 or 13 is often used, but 4+3*1=7 for n=2?
        # Let's use a robust population size. For n=5, lambda=10 is a good start.
        # Actually, standard formula: 4 + floor(3 * ln(n)) -> 4 + floor(3*1.609) = 4+4=8.
        # Let's use lambda = 10 for better exploration in 5D.
        pop_size = 10
        
        # Initial sigma: large enough to cover the box [-5, 5]
        # The box is size 10. A sigma of 2.5 or 3.0 is reasonable.
        sigma = 2.5
        
        # Mean start at center
        x_mean = np.zeros(dim)
        
        # Covariance matrix C = I
        C = np.eye(dim)
        
        # Eigendecomposition of C
        D = np.ones(dim)  # Eigenvalues
        B = np.eye(dim)   # Eigenvectors
        
        # Step size control parameters
        mu = pop_size // 2  # Weights for top mu individuals
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        
        # Path lengths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        best_x = x_mean.copy()
        best_f = np.inf
        
        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_x = x.copy()
                best_f = f
                return True
            return False

        def time_left():
            return deadline - time.time() > 1.0

        def calls_left():
            return budget - calls

        # Initial evaluation of mean
        f_mean = func(x_mean)
        calls += 1
        update_best(x_mean, f_mean)
        
        # Main loop
        while time_left() and calls < budget:
            # Sample population
            x_population = np.zeros((pop_size, dim))
            f_population = np.zeros(pop_size)
            
            for i in range(pop_size):
                if calls >= budget or not time_left():
                    break
                # Sample y ~ N(0, C)
                y = B @ (D * np.random.standard_normal(dim))
                x_pop = np.clip(x_mean + sigma * y, -5, 5)
                
                f_val = func(x_pop)
                calls += 1
                x_population[i] = x_pop
                f_population[i] = f_val
                update_best(x_pop, f_val)
                
            # If we ran out of budget mid-generation, break
            if calls >= budget or not time_left():
                break
            
            # Sort by fitness
            sorted_indices = np.argsort(f_population)
            x_sorted = x_population[sorted_indices]
            f_sorted = f_population[sorted_indices]
            
            # Update best
            if f_sorted[0] < best_f:
                best_x = x_sorted[0].copy()
                best_f = f_sorted[0]
            
            # Compute new mean
            x_new_mean = np.zeros(dim)
            for i in range(mu):
                x_new_mean += weights[i] * x_sorted[i]
            x_mean = x_new_mean
            
            # Update path p_sigma
            c_sigma_sqrt = np.sqrt(c_sigma * (2 - c_sigma) * mu_eff)
            # Need inverse of B and D
            # B is orthogonal, so B_inv = B.T
            # D_inv = 1/D
            y_mean = np.dot(B.T, x_new_mean - x_mean) / (sigma * D)
            p_sigma = (1 - c_sigma) * p_sigma + c_sigma_sqrt * y_mean
            
            # Hessian correction
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls // pop_size + 1)))) / chi_n < (1.4 + 2 / (dim + 1))
            h_sigma = 1.0 if h_sigma else 0.0
            
            # Update path p_c
            p_c = (1 - c_c) * p_c + h_sigma * c_c * c_sigma_sqrt * y_mean
            
            # Update covariance matrix
            delta_diag = (h_sigma - 1) * c_c * (2 - c_c)
            C = (1 + c_1 * delta_diag - c_1) * C + c_1 * np.outer(p_c, p_c)
            
            for i in range(mu):
                y_i = np.dot(B.T, x_sorted[i] - x_mean) / (sigma * D)
                C += c_mu * weights[i] * np.outer(y_i, y_i)
            
            # Update sigma
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Update eigenvalues and eigenvectors
            eigvals, eigvecs = np.linalg.eigh(C)
            # Ensure positive eigenvalues
            eigvals = np.maximum(eigvals, 1e-12)
            D = np.sqrt(eigvals)
            B = eigvecs
            
            # Boundary handling: if mean is out of bounds, project it?
            # The sampling clips, so mean can drift outside, but that's okay for now.
            # However, if mean is far outside, the effective step size shrinks due to clipping.
            # A simple reflection or clipping of the mean itself can help.
            x_mean = np.clip(x_mean, -5, 5)

        return best_x, best_f

# EVOLVE-BLOCK-END
