import numpy as np

class Optimizer:
    """
    CMA-ES optimizer for Black-box continuous optimization.
    """
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # CMA-ES Parameters
        # Default population size for dim=5 is approx 9
        # Formula: 4 + floor(3 * ln(dim))
        lambda_ = max(2, int(4 + 3 * np.log(self.dim)))
        
        # Step size initialization
        # A good heuristic is to start with a step size that covers a significant portion of the domain
        # but not the whole thing. 2.0 is a safe start for [-5,5].
        sigma = 2.0
        
        # Initial mean (centroid)
        # Start at origin, as it's the center of the box and often near the optimum in BBOB
        x_mean = np.zeros(self.dim)
        
        # Covariance matrix
        C = np.eye(self.dim)
        
        # Inverse square root of C for efficient sampling
        # We maintain C and its eigen-decomposition or Cholesky factor
        # For simplicity and robustness in a short script, we can use Cholesky or eigen decomposition each step
        # but maintaining the decomposition is more efficient.
        # Let's use a simplified CMA-ES that updates C directly and recomputes the Cholesky factor.
        
        # History variables
        p_sigma = np.zeros(self.dim)
        p_c = np.zeros(self.dim)
        
        # Adaptation parameters
        # mu (number of best solutions to update the mean)
        mu = lambda_ // 2
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        # mu_eff
        mu_eff = 1.0 / np.sum(w**2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (self.dim + mu_eff + 5)
        c_c = (4 + mu_eff / self.dim) / (self.dim + 4 + 2 * mu_eff / self.dim)
        c_1 = 2 / ((self.dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((self.dim + 2)**2 + mu_eff))
        
        # Damping for p_sigma
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (self.dim + 1)) - 1) + c_sigma
        
        # Initial Cholesky factor of C
        # C = I, so Cholesky is I
        B = np.eye(self.dim)
        D = np.ones(self.dim)
        
        # Evaluation counter
        count = 0
        
        # First evaluation to set initial best
        x0 = np.clip(x_mean, -5, 5)
        f0 = func(x0)
        count += 1
        
        best_x = x0
        best_f = f0
        
        while count < self.budget:
            # Generate population
            pop_x = []
            pop_f = []
            
            # Determine how many to generate in this batch
            # We can generate up to lambda_ individuals
            # If remaining budget < lambda_, we generate fewer
            n_gen = min(lambda_, self.budget - count)
            
            for i in range(n_gen):
                # Sample from N(x_mean, sigma^2 * C)
                # y = B * (D * z)
                z = np.random.standard_normal(self.dim)
                y = x_mean + sigma * (B @ (D * z))
                
                # Clip to box
                y = np.clip(y, -5, 5)
                
                pop_x.append(y)
                count += 1
                
                if count >= self.budget:
                    break
            
            # Evaluate population
            for y in pop_x:
                f_val = func(y)
                pop_f.append(f_val)
                
                if f_val < best_f:
                    best_f = f_val
                    best_x = y.copy()
            
            if count >= self.budget:
                break
                
            # Sort by fitness
            idx = np.argsort(pop_f)
            pop_x = [pop_x[i] for i in idx]
            pop_f = [pop_f[i] for i in idx]
            
            # Update mean
            y_w = np.zeros(self.dim)
            for i in range(mu):
                y_w += w[i] * pop_x[i]
            
            x_old = x_mean.copy()
            x_mean = x_mean + sigma * y_w
            
            # Update path length evolution path for step-size adaptation
            # p_sigma = (1 - c_sigma) * p_sigma + sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (B^{-1} / sigma) * (x_new - x_old)
            # Note: (x_new - x_old) = sigma * y_w
            # So (x_new - x_old) / sigma = y_w
            # We need B^{-1} * y_w
            # Since y = B * (D * z), the "unit" step in the transformed space is z.
            # The step in the original space is sigma * B * D * z.
            # The vector y_w is in the original space.
            # To get the step in the "unit" space (before scaling by sigma and C), we need to solve B * D * z = y_w / sigma.
            # Actually, the standard CMA-ES update for p_sigma is:
            # p_sigma = (1 - c_sigma) * p_sigma + sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (1/sigma) * B^{-1} * (x_new - x_old)
            
            # Compute B^{-1} * y_w
            # y_w is the weighted sum of steps in original coordinates.
            # Let v = y_w. We want z such that B * (D * z) = v.
            # D * z = B^{-1} v.
            # z = D^{-1} B^{-1} v.
            # We have B and D.
            # B is lower triangular (Cholesky).
            # We can solve B * u = v for u, then z = u / D.
            try:
                u = np.linalg.solve(B, y_w)
                z_step = u / D
            except:
                z_step = y_w # Fallback if solve fails
            
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * z_step
            
            # Hessian norm of C
            # ||C||_H = sqrt(Tr(C^2)) / sigma^2 ... wait, the formula uses the Hessian norm of the covariance matrix.
            # h_sigma = 1 if ||p_sigma|| / sqrt(1 - (1-c_sigma)^{2*(count/lambda_)/d_sigma}) < (1.4 + 2/(dim+1)) * E[||N(0,I)||]
            
            # We need to track the expected norm of standard normal
            # For dim=5, E[||N(0,I)||] is approx 2.5 (actually sqrt(5) * Gamma(3)/Gamma(2.5) ... approx 2.2-2.5)
            # Let's use a constant approximation or calculate it.
            # c_n = 1.0 for small dims is a safe approximation for the threshold multiplier base, but let's be more precise.
            # The threshold is (1.4 + 2/(dim+1)) * E[||N(0,I)||]
            
            # Calculate E[||N(0,I)||] for dim d
            # It is sqrt(2) * Gamma((d+1)/2) / Gamma(d/2)
            from math import gamma
            exp_norm = np.sqrt(2) * gamma((self.dim + 1) / 2.0) / gamma(self.dim / 2.0)
            
            # Check condition for h_sigma
            # We need to estimate the norm of p_sigma relative to the expected norm of the step.
            # The term (1 - (1-c_sigma)^{...}) accounts for the transient phase.
            # For simplicity and robustness in a short budget, we can approximate the denominator.
            # If count is small, the denominator is small, making the ratio large.
            # Let's use a simplified check: if ||p_sigma|| is not too large, h_sigma=1.
            
            # A more robust approach for short runs:
            # p_sigma_norm = np.linalg.norm(p_sigma)
            # threshold = (1.4 + 2.0 / (self.dim + 1)) * exp_norm
            # # The denominator term: sqrt(1 - (1 - c_sigma)**(2 * count / d_sigma))
            # # If count is 0, this is 0. We should avoid division by zero.
            # denom = np.sqrt(max(1e-10, 1 - (1 - c_sigma)**(2 * count / d_sigma)))
            # h_sigma = 1.0 if p_sigma_norm / denom < threshold else 0.0
            
            # Let's implement the standard check carefully.
            if count > 0:
                exp_factor = 1 - (1 - c_sigma) ** (2 * count / d_sigma)
                if exp_factor > 1e-10:
                    denom = np.sqrt(exp_factor)
                else:
                    denom = 1e-10
                p_sigma_norm = np.linalg.norm(p_sigma)
                threshold = (1.4 + 2.0 / (self.dim + 1)) * exp_norm
                h_sigma = 1.0 if p_sigma_norm / denom < threshold else 0.0
            else:
                h_sigma = 1.0
                p_sigma_norm = 0.0

            # Update step size
            sigma = sigma * np.exp(c_sigma / d_sigma * (p_sigma_norm / exp_norm - 1))
            
            # Update path length evolution path for covariance adaptation
            # p_c = (1 - c_c) * p_c + h_sigma * sqrt(c_c * (2 - c_c) * mu_eff) * (B^{-1} / sigma) * (x_new - x_old)
            # Again, (x_new - x_old)/sigma = y_w
            # We need B^{-1} * y_w, which is u calculated above.
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * u
            
            # Update covariance matrix
            # C = (1 - c_1 - c_mu) * C + c_1 * (p_c * p_c^T + (1 - h_sigma) * c_c * (2 - c_c) * C) + c_mu * sum(w_i * y_i * y_i^T)
            # where y_i are the steps in the "unit" space? No, y_i are the steps in the original space divided by sigma.
            # Let y_i = (x_i - x_old) / sigma.
            # In our loop, we have pop_x. x_old is the mean before update.
            # y_i = (pop_x[i] - x_old) / sigma.
            
            # Calculate the sum of outer products
            sum_yy = np.zeros((self.dim, self.dim))
            for i in range(mu):
                y_i = (pop_x[i] - x_old) / sigma
                sum_yy += w[i] * np.outer(y_i, y_i)
            
            # Update C
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 - h_sigma) * c_c * (2 - c_c) * C) + c_mu * sum_yy
            
            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            
            # Update Cholesky factor B and D
            # We need C = (B * D)^T (B * D) = D * B^T * B * D
            # Standard CMA-ES uses C = (B * diag(D))^2
            # We can compute the Cholesky decomposition of C.
            # However, CMA-ES maintains B and D such that C = (B * D)^2.
            # A simpler way is to just do a Cholesky decomposition of C to get L, then set B=L, D=1?
            # No, the standard algorithm uses an eigen-decomposition to update B and D to maintain orthogonality of B.
            # For a robust implementation without too much code, we can just use the Cholesky factor of C directly as the transformation matrix.
            # Let L be the Cholesky factor of C (L * L^T = C).
            # Then we sample z ~ N(0, I), and step = sigma * L * z.
            # This is equivalent to using B=L and D=1.
            # Let's switch to this simpler "Cholesky-only" approach for stability in a short script.
            # The update of C remains the same, but we just recompute L from C.
            
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C is not positive definite, add a small regularization
                C += 1e-6 * np.eye(self.dim)
                try:
                    L = np.linalg.cholesky(C)
                except:
                    L = np.eye(self.dim)
            
            # Update B and D for the next iteration's sampling
            # In the standard CMA-ES, B is orthogonal. Here L is not necessarily orthogonal.
            # But sampling from N(0, C) is done by L @ z.
            # So we set B = L and D = 1?
            # Wait, the update formulas for p_sigma and p_c relied on B^{-1}.
            # If we change the representation of C, we must adjust the inverse calculations.
            # If we use L such that C = L L^T, then the "unit" step z maps to L z.
            # The inverse map is L^{-1}.
            # The previous code used B and D such that C = (B D)^2.
            # Let's stick to the B, D representation to keep the update formulas consistent, 
            # OR simplify the whole thing to just use the Cholesky factor of C for sampling 
            # and adjust the p_sigma/p_c updates to use L^{-1}.
            
            # Let's refine:
            # We will maintain C.
            # We compute L = chol(C).
            # Sampling: y = x_mean + sigma * L @ z.
            # For p_sigma update: we need L^{-1} @ (x_new - x_old) / sigma.
            # For p_c update: we need L^{-1} @ (x_new - x_old) / sigma.
            
            # So we replace B, D with L.
            # And we replace the solve step.
            
            # Let's re-implement the loop logic slightly to use L.
            # Actually, it's easier to just keep the B, D structure and update it via eigen-decomposition of C.
            # This is the standard way.
            
            # Eigen-decomposition of C
            eigvals, eigvecs = np.linalg.eigh(C)
            
            # Ensure eigvals are positive
            eigvals = np.maximum(eigvals, 1e-10)
            
            # B = eigvecs
            # D = sqrt(eigvals)
            B = eigvecs
            D = np.sqrt(eigvals)
            
            # Re-calculate u (which was B^{-1} y_w) with the new B, D?
            # No, u was calculated using the OLD B, D before the update.
            # The update of p_sigma and p_c uses the OLD B, D.
            # So the calculation of u and z_step earlier is correct for the current step.
            # The new B, D are for the NEXT step's sampling.
            
            # So the flow is:
            # 1. Sample using old B, D.
            # 2. Evaluate.
            # 3. Update mean.
            # 4. Compute u = old_B^{-1} y_w.
            # 5. Update p_sigma, p_c using u.
            # 6. Update C.
            # 7. Decompose new C to get new B, D.
            # 8. Loop.
            
            # This matches the code above. I just need to make sure I don't use the new B, D for the current step's p_sigma update.
            # In my previous draft, I calculated u before updating C, so that is correct.
            # I then updated C and then computed new B, D.
            # This is correct.

        return best_x, best_f
