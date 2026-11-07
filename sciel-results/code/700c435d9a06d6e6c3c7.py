import numpy as np

class Optimizer:
    """
    CMA-ES (Covariance Matrix Adaptation Evolution Strategy) implementation.
    Optimizes a black-box function in 5 dimensions with a budget of 500 evaluations.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        
        # Initialize
        # Start at the origin, which is a reasonable guess for the center of the domain
        x_mean = np.zeros(d)
        
        # Initial step size. 
        # For BBOB f21, the landscape is complex. A moderate sigma is good.
        # The domain is [-5, 5]. The minimum is in [-4, 4].
        # Sigma = 1.0 or 2.0 is a common starting point.
        sigma = 1.5
        
        # Population size
        # Standard CMA-ES uses lambda = 4 + floor(3 * log(d))
        # For d=5, lambda = 4 + floor(3 * 1.609) = 4 + 4 = 8.
        # Let's use lambda = 10 to be safe and allow for more exploration.
        lambda_ = 10
        
        # Max weight for rank-mu update
        # mu = lambda_ // 2
        mu = lambda_ // 2
        
        # Weights for rank-mu update
        # Default weights: w_i = log((mu+1)/2) - log(i)
        weights = np.log((mu + 1) / 2) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        
        # Effective population size for variance reduction
        mueff = 1.0 / np.sum(weights**2)
        
        # Covariance matrix
        C = np.eye(d)
        C_inv = np.eye(d)
        C_sqrt = np.eye(d)
        
        # Step size control parameters
        c_sigma = (mueff + 2) / (d + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (d + 1)) - 1) + c_sigma
        
        # Covariance adaptation parameters
        c_c = (4 + mueff / d) / (d + 4 + 2 * mueff / d)
        c_1 = 2 / ((d + 1.3)**2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((d + 2)**2 + mueff))
        
        # Cumulation parameters
        c_c = (4 + mueff / d) / (d + 4 + 2 * mueff / d)
        c_sigma = (mueff + 2) / (d + mueff + 5)
        
        # State variables
        pc = np.zeros(d)
        ps = np.zeros(d)
        chi_n = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d**2))
        
        # Counter for evaluations
        evals = 0
        
        # First evaluation
        x0 = np.clip(x_mean, -5, 5)
        f_mean = func(x0)
        evals += 1
        
        # Main loop
        # We have budget - 1 evaluations left.
        # Each iteration consumes lambda_ evaluations.
        # We can run until we run out of budget.
        
        while evals + lambda_ <= budget:
            # Generate population
            Y = np.zeros((lambda_, d))
            for i in range(lambda_):
                # Sample from N(0, C)
                z = np.random.randn(d)
                Y[i] = C_sqrt @ z
            
            # Update mean
            # We use the weighted average of the best mu points
            # First, we need to evaluate all lambda points
            
            X = np.zeros((lambda_, d))
            F = np.zeros(lambda_)
            
            for i in range(lambda_):
                x_i = x_mean + sigma * Y[i]
                # Clip to bounds before evaluation? 
                # The problem says func clips. But keeping internal state bounded is better.
                # However, CMA-ES assumes unconstrained. 
                # Since the optimum is in [-4,4] and we start at 0 with sigma=1.5, 
                # we are unlikely to hit bounds immediately. 
                # If we do, clipping might distort the gradient. 
                # Given the constraint "func(x) clips", we can just pass x_i.
                x_i_clipped = np.clip(x_i, -5, 5)
                F[i] = func(x_i_clipped)
                X[i] = x_i_clipped
                evals += 1
                
            # Sort by fitness
            idx = np.argsort(F)
            F_sorted = F[idx]
            X_sorted = X[idx]
            
            # Update mean
            x_mean_new = np.sum(weights[:, None] * X_sorted[:mu], axis=0)
            
            # Update step size
            # D = C_sqrt @ C_sqrt_inv
            # We need C_inv and C_sqrt. 
            # For simplicity and robustness in this small dimension, 
            # we can recompute C_inv and C_sqrt each iteration or use Cholesky.
            
            # Recompute C_inv and C_sqrt from C
            # C is symmetric positive definite.
            try:
                C_inv = np.linalg.inv(C)
                C_sqrt, _ = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Regularize C
                C += 1e-4 * np.eye(d)
                C_inv = np.linalg.inv(C)
                C_sqrt, _ = np.linalg.cholesky(C)
                
            # ps update
            ps = np.sqrt(1 - c_sigma**2) * ps + \
                 np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (C_sqrt @ (C_sqrt_inv @ (x_mean_new - x_mean))) / sigma
            
            # Check if ps is too large
            if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * (evals / lambda_))) > (1.4 + 2 / (d + 1)) * chi_n:
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
                
            # pc update
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mueff) * (x_mean_new - x_mean) / sigma
            
            # Update C
            # Rank-one update
            v = x_mean_new - x_mean
            C = (1 + c_c * c_mu + c_1 - c_1 * np.trace(C_inv @ C)) * C \
                + c_1 * np.outer(v, v) / sigma**2 \
                + c_mu * np.sum(weights[:, None] * np.outer(Y_sorted[:mu], Y_sorted[:mu]), axis=0)
                
            # Ensure symmetry and positive definiteness
            C = (C + C.T) / 2
            
            # Regularize if necessary
            try:
                C_inv = np.linalg.inv(C)
                C_sqrt, _ = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C += 1e-4 * np.eye(d)
                C_inv = np.linalg.inv(C)
                C_sqrt, _ = np.linalg.cholesky(C)
                
            x_mean = x_mean_new
            f_mean = F_sorted[0]
            
            # Optional: If we are very close to the optimum, reduce sigma?
            # CMA-ES handles this automatically.
            
        # If we have remaining evaluations, use a local search or just keep the best
        # Since CMA-ES is adaptive, the last mean is a good estimate.
        # We can use the remaining budget for a simple local refinement if needed.
        # But with 500 budget and lambda=10, we do 49 iterations.
        # 49 * 10 = 490. We have 10 left.
        
        # Use remaining budget for a simple (1+1) ES refinement from the best point found
        best_x = x_mean
        best_f = f_mean
        
        # Sort all evaluated points to find the actual best
        # We didn't store all, but we can just run a few more steps
        # Or simply, the x_mean is a good candidate.
        
        # Let's use the remaining evaluations for a simple local search
        while evals < budget:
            # Generate a few candidates around x_mean
            # Simple Gaussian perturbation
            y = np.clip(x_mean + sigma * np.random.randn(d), -5, 5)
            fy = func(y)
            evals += 1
            if fy < best_f:
                best_x = y
                best_f = fy
                # Shrink sigma slightly for local refinement
                sigma *= 0.9
            else:
                sigma *= 0.95

        return best_x, best_f

# Helper for C_sqrt_inv
# In the code above, I used C_sqrt_inv but didn't define it. 
# C_sqrt_inv is the inverse of C_sqrt. 
# Since C_sqrt is lower triangular, its inverse is also lower triangular.
# However, for simplicity and to avoid bugs in the draft above, 
# I will rewrite the core loop more carefully.

class Optimizer:
    """
    CMA-ES (Covariance Matrix Adaptation Evolution Strategy) implementation.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        
        # Initialize
        x_mean = np.zeros(d)
        sigma = 1.5
        
        # Population size
        lambda_ = 10
        mu = lambda_ // 2
        
        # Weights
        weights = np.log((mu + 1) / 2) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        
        # Parameters
        c_sigma = (mueff + 2) / (d + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (d + 1)) - 1) + c_sigma
        c_c = (4 + mueff / d) / (d + 4 + 2 * mueff / d)
        c_1 = 2 / ((d + 1.3)**2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((d + 2)**2 + mueff))
        
        # State
        C = np.eye(d)
        pc = np.zeros(d)
        ps = np.zeros(d)
        chi_n = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d**2))
        
        evals = 0
        
        # First evaluation
        x0 = np.clip(x_mean, -5, 5)
        f_mean = func(x0)
        evals += 1
        best_x = x0
        best_f = f_mean
        
        while evals + lambda_ <= budget:
            # Generate population
            # Compute C_sqrt and C_inv
            try:
                C_inv = np.linalg.inv(C)
                C_sqrt, _ = np.linalg.cholesky(C)
                # C_sqrt_inv is the inverse of C_sqrt
                C_sqrt_inv = np.linalg.inv(C_sqrt)
            except np.linalg.LinAlgError:
                C += 1e-4 * np.eye(d)
                C_inv = np.linalg.inv(C)
                C_sqrt, _ = np.linalg.cholesky(C)
                C_sqrt_inv = np.linalg.inv(C_sqrt)
                
            Y = np.zeros((lambda_, d))
            X = np.zeros((lambda_, d))
            F = np.zeros(lambda_)
            
            for i in range(lambda_):
                z = np.random.randn(d)
                Y[i] = C_sqrt @ z
                x_i = x_mean + sigma * Y[i]
                x_i_clipped = np.clip(x_i, -5, 5)
                X[i] = x_i_clipped
                F[i] = func(x_i_clipped)
                evals += 1
                
                if F[i] < best_f:
                    best_f = F[i]
                    best_x = x_i_clipped
                    
            # Sort
            idx = np.argsort(F)
            F_sorted = F[idx]
            X_sorted = X[idx]
            Y_sorted = Y[idx]
            
            # Update mean
            x_mean_new = np.sum(weights[:, None] * X_sorted[:mu], axis=0)
            
            # Update ps
            # v = (x_mean_new - x_mean) / sigma
            v = (x_mean_new - x_mean) / sigma
            # C_sqrt_inv @ v
            tmp = C_sqrt_inv @ v
            ps = np.sqrt(1 - c_sigma**2) * ps + \
                 np.sqrt(c_sigma * (2 - c_sigma) * mueff) * tmp
            
            # Step size update
            if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * (evals / lambda_))) > (1.4 + 2 / (d + 1)) * chi_n:
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
                
            # Update pc
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mueff) * v
            
            # Update C
            # Rank-one
            C = (1 + c_c * c_mu + c_1 - c_1 * np.trace(C_inv @ C)) * C \
                + c_1 * np.outer(v, v)
                
            # Rank-mu
            C += c_mu * np.sum(weights[:, None] * np.outer(Y_sorted[:mu], Y_sorted[:mu]), axis=0)
            
            # Symmetrize
            C = (C + C.T) / 2
            
            # Regularize
            try:
                C_inv = np.linalg.inv(C)
                C_sqrt, _ = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C += 1e-4 * np.eye(d)
                C_inv = np.linalg.inv(C)
                C_sqrt, _ = np.linalg.cholesky(C)
                
            x_mean = x_mean_new
            
        # Use remaining budget for local refinement
        while evals < budget:
            y = np.clip(x_mean + sigma * np.random.randn(d), -5, 5)
            fy = func(y)
            evals += 1
            if fy < best_f:
                best_f = fy
                best_x = y
                sigma *= 0.9
            else:
                sigma *= 0.95

        return best_x, best_f
