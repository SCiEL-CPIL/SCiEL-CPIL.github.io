"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Strategy: CMA-ES (Covariance Matrix Adaptation Evolution Strategy).
    
    The Bent Cigar function (BBOB f12) is highly ill-conditioned and has a narrow valley
    that is rotated relative to the coordinate axes. CMA-ES is the state-of-the-art
    derivative-free optimizer for such problems because it adapts the covariance matrix
    of the search distribution to align with the eigenvectors of the function's Hessian.
    
    Key structural differences from Nelder-Mead:
    1. Population-based: Maintains a population of lambda candidates per generation,
       providing a more robust global search than a single simplex.
    2. Covariance Adaptation: Learns the shape of the valley (anisotropy) and the
       rotation, allowing it to converge efficiently even if the initial distribution
       is isotropic.
    3. Step-size Adaptation: Uses a path-based mechanism to adjust the step size
       dynamically, balancing exploration and exploitation.
    
    For the 500 budget in 5D:
    - We use a population size lambda = 10 (approx 2*dim).
    - This allows for roughly 500/10 = 50 generations.
    - CMA-ES typically converges well within this budget for 5D problems, especially
      with the help of the initial random sampling which covers the box [-5, 5]^5.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.start_time = None
        self.deadline = None
        self.best_x = None
        self.best_f = float('inf')

    def __call__(self, func):
        self.start_time = time.time()
        # Safety margin: 300s limit, stop at 290s
        self.deadline = self.start_time + 290.0
        self.calls = 0
        self.best_x = None
        self.best_f = float('inf')

        dim = self.dim
        budget = self.budget

        def wrapped_func(x):
            if time.time() > self.deadline:
                raise TimeoutError("Time limit exceeded")
            self.calls += 1
            if self.calls > self.budget:
                raise ValueError("Budget exceeded")
            # Clip to box [-5, 5]
            x_clipped = np.clip(x, -5, 5)
            fx = func(x_clipped)
            if fx < self.best_f:
                self.best_f = fx
                self.best_x = x_clipped.copy()
            return fx

        # CMA-ES Parameters
        # Population size: lambda = 4 + floor(3 * log(n)) is standard, but for small n,
        # lambda = 2*n is often a good balance for budget constraints.
        # For dim=5, lambda=10 gives ~50 generations.
        lambda_ = 10 
        # Initial step size: Heuristic suggests sigma0 = 0.5 * range
        # But since we start in [-5,5], and the minimum is in [-4,4], 
        # a sigma of 2.0 covers the space well initially.
        sigma = 2.0
        
        # Mean vector initialization: Center of the box
        mean = np.zeros(dim)
        
        # Covariance matrix: Identity
        C = np.eye(dim)
        # Cholesky decomposition for sampling
        C_chol = np.linalg.cholesky(C)
        
        # Path variables for step-size adaptation
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Parameters
        mu = lambda_ // 2  # Number of selected parents
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Adaptation rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_sigma = c_sigma * (1 + 2 * max(0, sqrt(mu_eff / dim - 1)))  # simplified
        c_sigma = 0.008 + c_sigma  # ensure positive
        
        c_cov = (2 + 2 * mu_eff / dim) / (dim + 2 + mu_eff / dim)
        c_cov_pos = c_cov
        c_cov_neg = c_cov * 2  # slightly faster adaptation for negative curvature? Standard is same.
        
        # Damping for sigma update
        d_sigma = 1 + 2 * max(0, sqrt(mu_eff / dim - 1)) + c_sigma
        
        # Tolerance
        tol = 1e-12
        
        # Main Loop
        while self.calls < budget and time.time() < self.deadline:
            # Check time
            if time.time() > self.deadline - 1.0:
                break
                
            # Sample lambda candidates
            ys = np.random.randn(lambda_, dim)
            xs = np.tile(mean, (lambda_, 1)) + sigma * (ys @ C_chol.T)
            
            # Evaluate
            fs = np.array([wrapped_func(x) for x in xs])
            
            # Sort by fitness
            sort_idx = np.argsort(fs)
            ys = ys[sort_idx]
            xs = xs[sort_idx]
            fs = fs[sort_idx]
            
            # Update mean
            weighted_sum = np.sum(weights[:, np.newaxis] * ys[:mu], axis=0)
            mean = mean + sigma * weighted_sum
            
            # Update p_sigma
            # h_sigma = 1 if ||p_sigma|| / sqrt(1 - (1 - 2 * gen / budget)) < c(1 + c_sigma) * sqrt(mu_eff - 1)
            # Simplified: just update p_sigma
            norm_p_sigma = np.linalg.norm(p_sigma)
            # Threshold for h_sigma
            # Using a simple approximation for the threshold
            threshold = (1 + c_sigma) * np.sqrt(mu_eff - 1)
            h_sigma = 1.0 if norm_p_sigma / np.sqrt(1 - (1 - 2 * (self.calls / budget)) ** 2) < threshold * np.sqrt(1 + 2 * max(0, np.sqrt(mu_eff / dim - 1))) else 0.0
            
            p_sigma = (1 - c_sigma) * p_sigma + h_sigma * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (weighted_sum / sigma)
            
            # Update sigma
            # chi_n = sqrt(n) * (1 - 1/(4n) + 1/(21n^2))
            chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Update p_c
            norm_p_c = np.linalg.norm(p_c)
            # h_c = 1 if ||p_c|| / sqrt(1 - (1 - c_c)^2) < (1 + c_c) * sqrt(mu_eff - 1)
            threshold_c = (1 + c_c) * np.sqrt(mu_eff - 1)
            h_c = 1.0 if norm_p_c / np.sqrt(1 - (1 - c_c) ** 2) < threshold_c else 0.0
            
            p_c = (1 - c_c) * p_c + h_c * np.sqrt(c_c * (2 - c_c) * mu_eff) * (weighted_sum / sigma)
            
            # Update C
            # C = (1 - c_cov) * C + c_cov * (outer(p_c, p_c) + (1 - h_c) * sum(weights * outer(y, y)))
            outer_pc = np.outer(p_c, p_c)
            sum_ys = np.zeros((dim, dim))
            for i in range(mu):
                sum_ys += weights[i] * np.outer(ys[i], ys[i])
            
            C = (1 - c_cov) * C + c_cov * (outer_pc + (1 - h_c) * sum_ys)
            
            # Symmetrize C to avoid numerical drift
            C = (C + C.T) / 2.0
            
            # Update Cholesky
            try:
                C_chol = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C becomes singular or non-PSD, restart or adjust
                # For robustness, we can just add a small identity
                C = C + 1e-12 * np.eye(dim)
                try:
                    C_chol = np.linalg.cholesky(C)
                except:
                    break
            
            # Check convergence
            # If the spread of the population is very small
            spread = np.std(xs, axis=0)
            if np.max(spread) < 1e-8:
                break

        return self.best_x, self.best_f

# EVOLVE-BLOCK-END
