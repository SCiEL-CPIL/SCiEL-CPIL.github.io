"""CMA-ES optimizer for BBOB f20 (Schwefel) with explicit deadline guard."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """
    CMA-ES (Covariance Matrix Adaptation Evolution Strategy) for BBOB f20 (Schwefel).
    Uses a population-based approach to explore the rotated and shifted landscape.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin before 330s kill
        rng = np.random.default_rng(42)

        dim = self.dim
        budget = self.budget

        # CMA-ES parameters
        # Initial mean: center of the box
        mean = np.zeros(dim)
        
        # Initial sigma: roughly 1/4 of the domain width (10) -> 2.5
        # Standard CMA-ES initialization often uses 1.0 or based on box.
        # For Schwefel, local minima are deep. Starting wide is good.
        sigma = 2.0
        
        # Population size
        # Standard lambda = 4 + floor(3 * log(dim))
        # dim=5 -> 4 + 3*1.6 = 8.8 -> 9
        # Let's use 10 for better exploration in 5D with 500 budget.
        # 500 / 10 = 50 generations.
        mu = 5  # Number of parents
        lambda_pop = 10
        
        # Weights
        # Standard weights: w_i = log(mu+0.5) - log(i)
        # Normalized to sum to mu
        weights = np.array([np.log(mu + 0.5) - np.log(i + 1) for i in range(mu)])
        weights /= np.sum(weights)
        
        # Variance-effective population size
        mueff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        # Standard CMA-ES parameters
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mueff)
        c_mu = min(1 - c_1, 2 * ((mueff - 2 + 1/mueff) / ((dim + 2)**2 + mueff)))
        
        # Step size control parameters
        chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        
        # State variables
        ps = np.zeros(dim)  # Evolution path for sigma
        pc = np.zeros(dim)  # Evolution path for covariance
        
        # Initial covariance matrix
        C = np.eye(dim)
        
        # Precompute inverse square root of C
        # Use eigen decomposition for stability and speed in 5D
        # Since C starts as Identity, we can start fast and update as needed
        # For simplicity and robustness in a fixed budget, we'll update every generation
        # 50 generations * eigen decomp of 5x5 is very fast.
        
        # Pre-compute for initial step (C=I)
        # eigvals, eigvecs = np.linalg.eigh(C)
        # inv_sqrt_C = (eigvecs / np.sqrt(eigvals)) @ eigvecs.T
        # But we can just use identity initially.
        
        best_x = mean.copy()
        best_f = func(best_x)
        call_count = 1
        
        # Track best seen so far
        def update_best(x, fx):
            nonlocal best_x, best_f
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        # Main loop
        while call_count < budget:
            if time.time() > deadline:
                break
                
            # 1. Sample population
            # We need to generate lambda_pop points
            # If we don't have enough budget for a full population, we stop
            if call_count + lambda_pop > budget:
                # Generate only as many as we can? 
                # CMA-ES requires a full population for proper update.
                # If we can't complete a generation, we might do a final polish or just stop.
                # Given 500 budget and lambda=10, we have exactly 50 generations.
                # 1 (initial) + 50*10 = 501. 
                # Wait, 1 initial + 49*10 = 491. We have 9 left.
                # Let's adjust lambda or just break if we can't do a full gen.
                # Actually, let's set lambda=9. 1 + 55*9 = 500. Perfect.
                # But lambda is fixed at start. 
                # If we hit this condition, it means budget is running out mid-gen.
                # We will just break to avoid partial updates which might destabilize.
                break

            # Generate samples
            # x_k = mean + sigma * z_k
            # z_k ~ N(0, C)
            
            # Efficient sampling from N(0, C):
            # z = L @ u, where L L^T = C, u ~ N(0, I)
            # Or use eigendecomposition: C = V D V^T, z = V (sqrt(D) * u)
            
            # Since dim=5, eigen decomp is cheap.
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                # Clip small negative eigenvalues due to numerical error
                eigvals = np.clip(eigvals, 1e-10, None)
                sqrt_eigvals = np.sqrt(eigvals)
            except np.linalg.LinAlgError:
                # Fallback to identity if C becomes singular/invalid
                C = np.eye(dim)
                eigvals = np.ones(dim)
                eigvecs = np.eye(dim)
                sqrt_eigvals = np.ones(dim)
            
            # Generate lambda_pop random vectors u ~ N(0, I)
            U = rng.standard_normal((lambda_pop, dim))
            # Scale by sqrt eigenvalues
            # Z = U * sqrt_eigvals (broadcast)
            Z = U * sqrt_eigvals
            # Rotate by eigenvectors
            # X = Z @ eigvecs.T
            X = Z @ eigvecs.T
            
            # Evaluate
            offspring = []
            fitnesses = []
            
            for i in range(lambda_pop):
                if call_count >= budget:
                    break
                if time.time() > deadline:
                    break
                    
                x_off = mean + sigma * X[i]
                # Clip to box? 
                # CMA-ES usually handles boundaries by reflection or clipping.
                # Clipping is simple. If clipped, the distribution is distorted, 
                # but for 5D and box [-5,5], and sigma starting at 2.0, 
                # most points will be inside.
                x_off = np.clip(x_off, -5, 5)
                
                fx_off = func(x_off)
                call_count += 1
                
                offspring.append(x_off)
                fitnesses.append(fx_off)
                update_best(x_off, fx_off)
                
            if not offspring:
                break

            # Sort by fitness
            fitnesses = np.array(fitnesses)
            offspring = np.array(offspring)
            sort_idx = np.argsort(fitnesses)
            offspring = offspring[sort_idx]
            fitnesses = fitnesses[sort_idx]
            
            # Take top mu
            parents = offspring[:mu]
            
            # 2. Update mean
            # new_mean = sum(w_i * x_i)
            new_mean = np.sum(weights[:, np.newaxis] * parents, axis=0)
            
            # 3. Update evolution paths
            # ps = (1-c_sigma)*ps + sqrt(c_sigma*(2-c_sigma)*mueff) * C^(-1/2) * (x_1 - mean)/sigma
            # Note: x_1 is the best individual (index 0 in sorted parents)
            # C^(-1/2) is needed. We have eigvecs and eigvals.
            # C^(-1/2) = eigvecs @ diag(1/sqrt(eigvals)) @ eigvecs.T
            
            # Compute C^(-1/2)
            inv_sqrt_eigvals = 1.0 / sqrt_eigvals
            C_inv_sqrt = (eigvecs * inv_sqrt_eigvals) @ eigvecs.T
            
            # x_1 is parents[0]
            x1 = parents[0]
            
            # Update ps
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (C_inv_sqrt @ (x1 - mean) / sigma)
            
            # Update pc
            # pc = (1-c_c)*pc + h_sigma * sqrt(c_c*(2-c_c)*mueff) * (new_mean - mean)/sigma
            # h_sigma = (||ps|| / chi_n) < c_1 + (1+c_1)*sqrt((mueff+1)/dim - 1)
            norm_ps = np.linalg.norm(ps)
            h_sigma = 1.0 if (norm_ps / chi_n) < (c_1 + (1 + c_1) * np.sqrt((mueff + 1) / dim - 1)) else 0.0
            
            pc = (1 - c_c) * pc + h_sigma * np.sqrt(c_c * (2 - c_c) * mueff) * (new_mean - mean) / sigma
            
            # 4. Update covariance
            # C = (1 - c_1 - c_mu)*C + c_1*(pc*pc.T + (1-h_sigma)*(1-c_c)*C) + c_mu * sum(w_i * (y_i * y_i.T))
            # where y_i = (x_i - mean)/sigma
            
            # Compute y_i for parents
            Y = (parents - mean) / sigma
            
            # Rank-mu update
            rank_mu_update = np.zeros((dim, dim))
            for i in range(mu):
                rank_mu_update += weights[i] * np.outer(Y[i], Y[i])
                
            # Rank-1 update
            rank_1_update = np.outer(pc, pc)
            if h_sigma < 1.0:
                rank_1_update += (1 - h_sigma) * (1 - c_c) * C
                
            # Combine
            C = (1 - c_1 - c_mu) * C + c_1 * rank_1_update + c_mu * rank_mu_update
            
            # 5. Update sigma
            # sigma = sigma * exp(c_sigma / chi_n * (||ps||/chi_n - 1))
            sigma *= np.exp(c_sigma / chi_n * (norm_ps / chi_n - 1))
            
            # Keep sigma in reasonable bounds
            sigma = np.clip(sigma, 0.01, 5.0)
            
            # 6. Update mean
            mean = new_mean
            
            # Optional: If sigma becomes very small, we might be stuck.
            # CMA-ES has restart strategies, but with fixed budget, we just let it run.
            # If it converges, it will waste calls on the same point.
            # However, the adaptive sigma should prevent premature convergence 
            # if the landscape is multimodal.
            
        return best_x, best_f
# EVOLVE-BLOCK-END
