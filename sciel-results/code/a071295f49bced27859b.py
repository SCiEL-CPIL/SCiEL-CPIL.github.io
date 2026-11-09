"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES based optimizer for BBOB f11 (Discus)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Deadline guard: 330s wall-clock limit, keep 10s margin
        start_time = time.time()
        deadline = start_time + 320
        
        # State tracking
        self.call_count = 0
        self.best_x = None
        self.best_f = np.inf

        # CMA-ES Parameters
        dim = self.dim
        budget = self.budget
        
        # Initial point: center of the box
        x_mean = np.zeros(dim)
        
        # Initial sigma: heuristic based on domain size
        # Domain is [-5, 5], so width is 10.
        # A common heuristic is sigma0 = 0.25 * (xmax - xmin) / dim or similar.
        # For 5D, let's start with a reasonable exploration size.
        sigma = 1.0 
        
        # Population size: default lambda = 4 + floor(3 * log(dim))
        # For dim=5, log(5)~1.6, 3*1.6=4.8, floor=4, lambda=8.
        # Let's use a slightly larger population for better coverage in 5D.
        lambda_pop = 10
        
        # Step sizes for mean update
        cmu = 1.0 / (dim + 3)
        c = 1.0 / (dim + 3)
        dsigma = 1.0 + 2 * max(0, (sqrt(dim + 1.3) - 1) / (dim + 3)) if dim > 1 else 1.0
        
        # Pre-compute constants for efficiency
        sqrt_dim = np.sqrt(dim)
        
        # Cholesky decomposition of the covariance matrix (initially identity)
        C = np.eye(dim)
        C_inv_sqrt = np.eye(dim) # C^(-1/2)
        
        # History for sigma adaptation
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Evolution path for covariance adaptation
        # We need the normalized step vector y
        
        # Wrapper for function evaluation
        def wrapped_func(x):
            if self.call_count >= self.budget:
                raise ValueError("Budget exceeded")
            if time.time() > deadline:
                raise TimeoutError("Time limit exceeded")
            
            self.call_count += 1
            fx = func(x)
            if fx < self.best_f:
                self.best_f = fx
                self.best_x = x.copy()
            return fx

        # Main CMA-ES Loop
        try:
            while self.call_count < self.budget:
                if time.time() > deadline:
                    break
                
                # 1. Sample population
                samples = []
                for i in range(lambda_pop):
                    # Standard normal sample
                    z = np.random.randn(dim)
                    # Transform using Cholesky factor
                    # x = x_mean + sigma * C^(1/2) z
                    # Since C is symmetric, C^(1/2) is the Cholesky factor L such that C = L L^T
                    # We maintain L directly or recompute?
                    # For simplicity and robustness in this implementation, we recompute L if C changes significantly,
                    # or we can just use the standard normal if we assume isotropic initially.
                    # Proper CMA-ES updates C. Let's implement a simplified version that tracks C.
                    
                    # Actually, for a robust and fast implementation in Python without external libs,
                    # full CMA-ES is complex. 
                    # However, the "Discus" function is smooth and unimodal.
                    # A simpler, highly effective alternative for low dimensions is 
                    # **Nelder-Mead with multi-start and adaptive restarts** or 
                    # **Powell's Method**.
                    
                    # But the directive asks for a DIFFERENT strategy from the current (Nelder-Mead).
                    # Let's implement a robust **Simplex Method (Nelder-Mead) variant with adaptive scaling**
                    # OR **CMA-ES**. 
                    
                    # Given the constraint of standard library + NumPy, implementing full CMA-ES is doable.
                    # Let's stick to CMA-ES as it's the gold standard for BBOB.
                    
                    pass # Placeholder for logic below
                
                # --- Simplified CMA-ES Implementation ---
                
                # Generate lambda_pop samples
                X = np.zeros((lambda_pop, dim))
                for i in range(lambda_pop):
                    z = np.random.randn(dim)
                    X[i] = x_mean + sigma * np.dot(C_inv_sqrt, z)
                
                # Evaluate population
                F = np.zeros(lambda_pop)
                for i in range(lambda_pop):
                    try:
                        F[i] = wrapped_func(X[i])
                    except (ValueError, TimeoutError):
                        break
                
                # Sort by fitness
                order = np.argsort(F)
                X = X[order]
                F = F[order]
                
                # Update mean
                # Weights: default weights are proportional to rank
                # w_i = log((lambda/2 + 1) - i) - mean(log(lambda/2 + 1 - j))
                # Let's use standard weights
                weights = np.zeros(lambda_pop)
                mu_eff = 0
                for i in range(lambda_pop):
                    # Only top mu individuals contribute to mean update
                    # mu = lambda / 2
                    mu = lambda_pop // 2
                    if i < mu:
                        weights[i] = 1.0
                    else:
                        weights[i] = 0.0
                        continue
                    mu_eff += weights[i]
                
                # Normalize weights
                weights = weights / mu_eff
                
                # New mean
                x_mean_new = np.sum(X[:mu] * weights[:, np.newaxis], axis=0)
                
                # Step vector for covariance update
                y = (x_mean_new - x_mean) / sigma if sigma > 0 else np.zeros(dim)
                
                # Update sigma
                # p_sigma = (1 - c_sigma) p_sigma + sqrt(c_sigma (2 - c_sigma) mu_eff) / sigma * y
                c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
                c_sigma = 1.0 / (dim + 3) # Simplified
                c_sigma = 1.0 / (dim + 3)
                # Standard c_sigma
                c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
                
                # We need C^(-1/2) y for the norm
                # Since we used C_inv_sqrt to generate, y is in the transformed space?
                # No, y is in the original space.
                # p_sigma update requires the norm of y in the "natural" scale.
                # Let's use a simplified sigma update:
                # sigma *= exp(c_sigma / mu_eff * (||y|| / sigma - 1))
                # ||y|| is the norm of the step in the original space.
                # But y is scaled by 1/sigma. So ||y|| is dimensionless.
                
                norm_y = np.linalg.norm(y)
                # Target norm is 1 (for standard normal)
                # If norm_y > 1, sigma should increase? No, if the step was large and successful, maybe keep it.
                # Standard CMA-ES:
                # p_sigma = (1-c_sigma)p_sigma + sqrt(c_sigma(2-c_sigma)mu_eff)/sigma * y
                # sigma = sigma * exp(c_sigma/mu_eff * (||p_sigma||/sigma - 1))
                
                # Let's implement the standard update properly.
                # We need C_inv_sqrt to transform y for the p_sigma update?
                # The update rule for p_sigma uses the normalized step.
                
                # To keep this robust and within time limits, I will use a **Simplified CMA-ES**
                # that tracks the covariance matrix but uses a simplified update rule.
                
                # 1. Update Mean
                x_mean = x_mean_new
                
                # 2. Update Covariance Matrix C
                # C = (1 - mu_c) C + mu_c * (y y^T + h_sigma p_sigma p_sigma^T)
                # where mu_c = 2 / (dim^2 + 1)
                mu_c = 2.0 / (dim**2 + 1.0)
                
                # We need to be careful with the term h_sigma.
                # h_sigma = 1 if ||p_sigma|| < (1.4 + 2/(dim+1)) * sqrt((1-c_sigma)/(1+c_sigma))
                
                # Let's update p_sigma first
                # The vector y is in the original coordinate system.
                # The update for p_sigma is:
                # p_sigma = (1 - c_sigma) * p_sigma + sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * y
                
                # However, this formula assumes y is the step in the original space.
                # In CMA-ES, the step is z = C^(1/2) u.
                # Our y = (x_new - x_old) / sigma.
                # So y = C^(1/2) u.
                # The standard update uses the "natural" step.
                
                # For simplicity and stability, let's use a **Nelder-Mead** approach but with
                # **Adaptive Restart** and **Global Search** which is often competitive with CMA-ES in low dimensions
                # and much easier to implement correctly without external libraries.
                # Wait, the current program IS Nelder-Mead. The directive says "DIFFERENT strategy".
                # So I MUST implement CMA-ES or another population-based method.
                
                # Let's implement a **Simple Genetic Algorithm** with **Elitism** and **Adaptive Mutation**?
                # Or **Differential Evolution**?
                
                # **Differential Evolution (DE)** is robust, simple, and effective for BBOB functions.
                # It uses a population and vector differences for mutation.
                # Strategy: DE/rand/1/bin
                # F (scale factor) = 0.8
                # CR (crossover rate) = 0.9
                
                # Let's switch to Differential Evolution. It's a distinct family from Nelder-Mead.
                
                break # Break out of the complex CMA-ES attempt to implement DE
                
        except Exception:
            pass

        # --- Implementing Differential Evolution ---
        
        # Reset state if we broke out early or to start fresh
        self.call_count = 0
        self.best_f = np.inf
        self.best_x = None
        
        # Population size
        pop_size = 20
        # Initialize population
        pop = np.random.uniform(-5, 5, (pop_size, dim))
        
        # Evaluate initial population
        fits = np.zeros(pop_size)
        for i in range(pop_size):
            try:
                fits[i] = wrapped_func(pop[i])
            except (ValueError, TimeoutError):
                break
        
        if self.best_x is None:
            # Fallback if no evaluations happened
            x0 = np.zeros(dim)
            try:
                self.best_f = wrapped_func(x0)
                self.best_x = x0
            except:
                return np.zeros(dim), np.inf

        # DE Parameters
        F = 0.8  # Scale factor
        CR = 0.9 # Crossover rate
        
        # Main Loop
        while self.call_count < self.budget:
            if time.time() > deadline:
                break
            
            # For each individual in the population
            for i in range(pop_size):
                if self.call_count >= self.budget or time.time() > deadline:
                    break
                
                # Select 3 distinct random indices different from i
                while True:
                    r1, r2, r3 = np.random.choice(pop_size, 3, replace=False)
                    if i not in [r1, r2, r3]:
                        break
                
                # Mutation: v = x[r1] + F * (x[r2] - x[r3])
                v = pop[r1] + F * (pop[r2] - pop[r3])
                
                # Clip to bounds
                v = np.clip(v, -5, 5)
                
                # Crossover: binomial crossover
                # Trial vector u
                u = v.copy()
                # Random crossover point
                j_rand = np.random.randint(dim)
                
                for j in range(dim):
                    if np.random.rand() < CR or j == j_rand:
                        u[j] = v[j]
                    else:
                        u[j] = pop[i][j]
                
                # Evaluation
                try:
                    f_u = wrapped_func(u)
                except (ValueError, TimeoutError):
                    break
                
                # Selection
                if f_u <= fits[i]:
                    pop[i] = u
                    fits[i] = f_u
                    if f_u < self.best_f:
                        self.best_f = f_u
                        self.best_x = u.copy()

        return self.best_x, self.best_f

# EVOLVE-BLOCK-END
