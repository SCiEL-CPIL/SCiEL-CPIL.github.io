"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with 1/5 success rule and time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290
        rng = np.random.default_rng(0)
        
        # CMA-ES initialization
        # Start in the center of the box
        x_mean = np.zeros(self.dim)
        sigma = 2.0  # Initial step size
        
        # Population size
        # Standard is 4 + floor(3 * log(N)), for N=5 -> 4 + 3*1.6 = 8.8 -> 8
        # But with budget 500 and N=5, we can be a bit larger for better exploration.
        # Let's use N=10 to balance exploration and exploitation within 500 evals.
        # 500 / 10 = 50 generations.
        lambda_pop = 10 
        mu = lambda_pop // 2  # Weighting of top mu individuals
        
        # Weights
        weights = np.zeros(mu)
        w_sum = 0
        for i in range(mu):
            weights[i] = np.log(mu + 1) - np.log(i + 1)
            w_sum += weights[i]
        weights /= w_sum
        
        # Variance-effective population size
        mueff = 1.0 / np.sum(weights ** 2)
        
        # CMA-ES parameters
        c_sigma = (mueff + 2) / (self.dim + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (self.dim + 1)) - 1) + c_sigma
        c_c = (4 + mueff / self.dim) / (self.dim + 4 + 2 * mueff / self.dim)
        c_1 = 2 / ((self.dim + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((self.dim + 2) ** 2 + mueff))
        
        # State variables
        pc = np.zeros(self.dim)
        ps = np.zeros(self.dim)
        C = np.eye(self.dim)
        # Eigen decomposition for Cholesky-like factor (or just use sqrtm if available, but we can update C directly or use Cholesky of C)
        # For simplicity and speed in Python, we will update C directly or use a simplified update.
        # Actually, standard CMA-ES updates C and then computes its Cholesky decomposition for sampling.
        # Computing Cholesky every generation for dim=5 is cheap.
        
        # Initial evaluation
        x0 = np.clip(x_mean, -5, 5)
        fvals = [func(x0)]
        x_best = x0
        f_best = fvals[0]
        calls_made = 1
        
        # Main loop
        while calls_made < self.budget:
            if time.time() > deadline:
                break
            
            # Generate population
            # We need the square root of C. Since C is symmetric positive definite, we can use np.linalg.cholesky
            # But C might become ill-conditioned. Let's use np.linalg.eigh to get the square root.
            # Actually, for dim=5, cholesky is fine if we add a small regularization if it fails.
            try:
                D = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Fallback to identity if C is broken
                D = np.eye(self.dim)
                
            # Sample lambda points
            ys = rng.standard_normal((lambda_pop, self.dim))
            xs = np.array([x_mean + sigma * y @ D.T for y in ys])
            # Clip to bounds
            xs = np.clip(xs, -5, 5)
            
            # Evaluate
            fs = np.zeros(lambda_pop)
            for i in range(lambda_pop):
                if time.time() > deadline:
                    break
                fs[i] = func(xs[i])
                calls_made += 1
                if fs[i] < f_best:
                    f_best = fs[i]
                    x_best = xs[i]
            
            if calls_made >= self.budget:
                break
                
            # Sort by fitness
            order = np.argsort(fs)
            xs_sorted = xs[order]
            fs_sorted = fs[order]
            
            # Update mean
            x_mean_new = np.sum(weights[:, None] * xs_sorted[:mu], axis=0)
            
            # Update step size control variables
            # ps = (1 - c_sigma) * ps + sqrt(c_sigma * (2 - c_sigma) * mueff) / sigma * D.T * (x_mean_new - x_mean)
            # Note: The vector (x_mean_new - x_mean) is in the "x-space". To update ps (which is in the "standard normal space" relative to the distribution), we need to map it back.
            # Actually, the standard update is:
            # ps = (1 - c_sigma) * ps + sqrt(c_sigma * (2 - c_sigma) * mueff) * (D^{-1} (x_mean_new - x_mean)) / sigma
            # Where D is the Cholesky factor such that C = D D^T.
            # So D^{-1} v is solving D w = v.
            
            diff = x_mean_new - x_mean
            try:
                # Solve D w = diff / sigma
                w = np.linalg.solve(D, diff / sigma)
            except np.linalg.LinAlgError:
                w = diff / sigma
                
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * w
            
            # Update pc
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls_made // lambda_pop + 1)))) < (1 + 2 / self.dim) * np.sqrt(1 - c_sigma)
            # Note: The exponent in the denominator depends on the number of generations, not calls. 
            # Let's track generations properly.
            # For simplicity, we approximate the generation count.
            gen_count = calls_made // lambda_pop
            denom = np.sqrt(1 - (1 - c_sigma) ** (2 * gen_count))
            if denom == 0:
                denom = 1e-8
            hsig = (np.linalg.norm(ps) / denom) < (1 + 2 / self.dim) * np.sqrt(1 - c_sigma)
            
            pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mueff) * w
            
            # Update covariance matrix C
            # C = (1 - c_1 - c_mu) * C + c_1 * (pc pc^T + (1 - hsig) * c_c * (2 - c_c) * I) + c_mu * sum(weights_i * (y_i - y_bar)(y_i - y_bar)^T)
            # Where y_i are the standard normal samples, and y_bar is the weighted mean of the top mu y's.
            
            # We need the top mu y's corresponding to the sorted x's.
            # We stored ys in the same order as xs.
            ys_sorted = ys[order]
            y_bar = np.sum(weights[:, None] * ys_sorted[:mu], axis=0)
            
            delta_h = hsig - (1 - c_c)
            # Term 1: (1 - c_1 - c_mu) * C
            C_new = (1 - c_1 - c_mu) * C
            # Term 2: c_1 * (pc pc^T + delta_h * I)
            C_new += c_1 * (np.outer(pc, pc) + delta_h * np.eye(self.dim))
            # Term 3: c_mu * sum(weights_i * (y_i - y_bar)(y_i - y_bar)^T)
            diff_ys = ys_sorted[:mu] - y_bar
            # Weighted sum of outer products
            # (diff_ys.T * weights) @ diff_ys is not quite right, it's sum_i w_i * v_i v_i^T
            C_new += c_mu * (diff_ys.T * weights) @ diff_ys
            
            # Ensure symmetry
            C_new = (C_new + C_new.T) / 2.0
            
            # Update state
            x_mean = x_mean_new
            C = C_new
            
            # Adaptive step size
            # sigma = sigma * exp(c_sigma / d_sigma * (||ps|| / chi_N - 1))
            chi_n = np.sqrt(self.dim) * (1 - 1 / (4 * self.dim) + 1 / (21 * self.dim ** 2))
            norm_ps = np.linalg.norm(ps)
            sigma = sigma * np.exp(c_sigma / d_sigma * (norm_ps / chi_n - 1))
            
            # Clip sigma to reasonable range
            sigma = np.clip(sigma, 1e-8, 10.0)
            
            # If sigma is too small, we might be stuck. 
            # But CMA-ES is generally robust.
            
        return x_best, f_best
# EVOLVE-BLOCK-END
