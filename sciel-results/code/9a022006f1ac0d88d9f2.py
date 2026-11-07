import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin for 300s limit
        dim = self.dim
        budget = self.budget

        eval_count = 0
        best_x = np.zeros(dim)
        best_f = np.inf

        def time_check():
            return (time.time() - start_time) > deadline

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        def wrapped_func(x):
            nonlocal eval_count
            if time_check():
                return 1e100
            eval_count += 1
            val = func(x)
            update_best(x, val)
            return val

        # 1. Initialization: Origin + Random Points
        # This helps identify the general region and provides initial diversity.
        rng = np.random.default_rng(123)
        init_points = [np.zeros(dim)]
        for _ in range(4):
            init_points.append(rng.uniform(-5, 5, dim))

        for p in init_points:
            if eval_count >= budget or time_check():
                break
            wrapped_func(p)

        # 2. CMA-ES Phase
        # CMA-ES is chosen for its ability to handle rotated and shifted landscapes effectively,
        # which is crucial for the BBOB f3 (Rastrigin) instances described.
        # It adapts the covariance matrix to the local geometry of the function.

        # CMA-ES Parameters
        # For dim=5, lambda=12 is a standard choice (4 + 3*ln(5) approx 8, but 12 is often better for multimodal)
        lam = 12
        mu = lam // 2  # 6
        
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Step size control
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        
        # Covariance matrix adaptation
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1.0 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # State variables
        x_mean = best_x.copy()
        sigma = 2.0  # Initial step size, covers a good portion of [-5,5]^n
        C = np.eye(dim)
        C_sqrt = np.eye(dim)
        B = np.eye(dim)
        D = np.ones(dim)
        
        # Path variables
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Precompute constants
        chi_n = np.sqrt(dim) * (1 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim**2))
        c_hs = 1.1 + 2.0 / (dim + 1.7)
        
        # Main CMA-ES Loop
        while eval_count < budget and not time_check():
            # 1. Sample new points
            ys = []
            for k in range(lam):
                z = np.random.standard_normal(dim)
                y = B @ (D * z)
                ys.append(y)
            
            xs = x_mean + sigma * np.array(ys)
            
            # 2. Evaluate
            remaining = budget - eval_count
            num_evals = min(lam, remaining)
            if num_evals <= 0:
                break
            
            fitnesses = []
            eval_xs = []
            for i in range(num_evals):
                x = xs[i]
                f = wrapped_func(x)
                fitnesses.append(f)
                eval_xs.append(x)
                
                if eval_count >= budget:
                    break
            
            if eval_count >= budget:
                break
                
            # If we ran out of budget mid-evaluation, we can't do a full generation update
            if num_evals < lam:
                break

            fitnesses = np.array(fitnesses)
            eval_xs = np.array(eval_xs)
            
            # 3. Sort by fitness
            order = np.argsort(fitnesses)
            xs_sorted = eval_xs[order]
            fitnesses_sorted = fitnesses[order]
            ys_array = np.array(ys)
            ys_sorted = ys_array[order][:mu]
            
            # 4. Update mean
            new_mean = np.sum(weights[:, None] * xs_sorted[:mu], axis=0)
            
            # 5. Update sigma (step size)
            y_mean = (new_mean - x_mean) / sigma
            
            norm_y_mean = np.linalg.norm(y_mean)
            sqrt_trace_C = np.sqrt(np.trace(C))
            ratio = norm_y_mean / (chi_n * sqrt_trace_C) if sqrt_trace_C > 0 else 0
            
            p_sigma = (1 - c_sigma) * p_sigma + c_hs * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * ratio * y_mean
            
            # h_sigma calculation
            # Note: The exponent in the power term depends on the number of generations, but for simplicity 
            # and robustness in a short run, we approximate the decay factor.
            # A more precise implementation would track generations. Here we use a simplified check.
            norm_p_sigma = np.linalg.norm(p_sigma)
            h_sigma = (norm_p_sigma / np.sqrt(1 - (1 - c_sigma)**(2 * (eval_count / lam))) < (1.4 + 2 / (dim + 1)) * chi_n)
            
            sigma = sigma * np.exp(c_sigma / d_sigma * (norm_p_sigma / chi_n - 1))
            
            # 6. Update covariance matrix C
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_mean
            
            # Rank-one update
            rank_one = np.outer(p_c, p_c)
            if not h_sigma:
                rank_one += c_c * (2 - c_c) * C
            
            # Rank-mu update
            # Sum w_i (y_i - y_mean)(y_i - y_mean)^T ? 
            # Standard CMA-ES uses the steps y_i themselves in the rank-mu update relative to the old mean?
            # Actually, the standard formula is:
            # C = (1 - c_1 - c_mu) C + c_1 (p_c p_c^T + (1-h_sigma) c_c (2-c_c) C) + c_mu sum w_i (y_i)(y_i)^T
            # Where y_i are the steps (x_i - x_mean_old)/sigma.
            # So we use ys_sorted directly.
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                rank_mu += weights[i] * np.outer(ys_sorted[i], ys_sorted[i])
            
            C = (1 - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu
            
            # Update x_mean
            x_mean = new_mean
            
            # 7. Decompose C for next iteration
            # Eigendecomposition
            try:
                D2, B = np.linalg.eigh(C)
                # Ensure positive eigenvalues
                D2 = np.clip(D2, 1e-10, None)
                D = np.sqrt(D2)
                C_sqrt = B @ np.diag(D)
            except np.linalg.LinAlgError:
                # Fallback if decomposition fails
                C = np.eye(dim)
                B = np.eye(dim)
                D = np.ones(dim)
                C_sqrt = np.eye(dim)
            
            # Stagnation check: If sigma becomes too small, restart?
            # For now, let it run. CMA-ES usually handles this well.
            
            # Check time
            if time_check():
                break

        return best_x, best_f
