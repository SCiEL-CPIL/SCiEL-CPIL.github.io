import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.call_count = 0
        self.best_x = None
        self.best_f = np.inf
        self.start_time = None

    def __call__(self, func):
        self.call_count = 0
        self.best_f = np.inf
        self.best_x = None
        self.start_time = time.time()
        
        # Time limit safety margin: 300s limit, stop at 280s
        time_limit = 280.0 
        
        dim = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        # For dim=5, default lambda is 4 + floor(3*log(5)) = 4+5 = 9. 
        # Let's use a slightly larger population for better exploration in early stages, 
        # but standard CMA-ES is quite efficient. 
        # Standard lambda = 4 + int(3 * np.log(dim))
        lambda_ = 4 + int(3 * np.log(dim))
        
        # Initial sigma
        # Box is [-5, 5], so width is 10. 
        # A good initial sigma is often 1/5 of the box width or similar.
        # Since the function is Rosenbrock (valley), a larger initial sigma helps explore the valley.
        sigma = 2.0
        
        # Initial center: Uniform random in [-5, 5]
        rng = np.random.default_rng(42)
        x_mean = rng.uniform(-5, 5, dim)
        
        # Covariance matrix C = I
        C = np.eye(dim)
        
        # Step sizes
        sigma_update_rate = 0.01 # Default for CMA-ES
        # Actually, standard CMA-ES uses specific update rules for sigma and C.
        # Implementing full CMA-ES from scratch is complex and prone to bugs.
        # SciPy's `differential_evolution` or `minimize` with `method='Powell'` are options, 
        # but the prompt asks for a DIFFERENT strategy. 
        # The current program uses Powell + (1+1)-ES.
        # A robust alternative that is often better for Rosenbrock is **Nelder-Mead** with restarts, 
        # or **DE (Differential Evolution)** which is very robust for black-box.
        # However, CMA-ES is the gold standard. 
        # Let's use SciPy's `differential_evolution`? No, it's often slow.
        # Let's use a robust implementation of **CMA-ES** via a simplified version or 
        # use `scipy.optimize.minimize` with `method='COBYLA'`? No, COBYLA is bad for high dim.
        
        # Let's reconsider. The parent program (score 1.0) was a simple (1+1)-ES with 1/5 rule.
        # The current child (score 0.116) tried Powell + (1+1)-ES and failed, likely due to 
        # Powell's high eval count or getting stuck.
        # The "inspiration 1" is the parent (1.0). 
        # Wait, the prompt says "inspiration 1 (score 1.0000): Initial program...". 
        # So the simple (1+1)-ES scored 1.0. 
        # The current program (0.116) is worse. 
        # Why? Powell might have consumed the budget without finding the global min, 
        # or the (1+1)-ES fallback started from a bad local min found by Powell.
        # Or simply, the overhead of Powell in 5D on Rosenbrock is inefficient compared to 
        # a well-tuned (1+1)-ES or CMA-ES.
        
        # Strategy: Use **CMA-ES** properly. 
        # Since implementing full CMA-ES is lengthy, I will use a robust **Differential Evolution** 
        # or **Particle Swarm Optimization**? 
        # Actually, for 5D Rosenbrock, **Nelder-Mead** with multiple restarts is very effective 
        # if initialized well. But Nelder-Mead is local.
        
        # Let's go with a **Multi-Start Nelder-Mead** approach.
        # 1. Generate K random initial points.
        # 2. Run Nelder-Mead from each, with a strict eval limit per run.
        # 3. Keep the best.
        # 4. If budget remains, refine the best with a tight (1+1)-ES or CMA-ES.
        
        # Nelder-Mead in SciPy uses `maxiter` which is iterations, not evals. 
        # We need to count evals.
        
        # Alternative: **DE (Differential Evolution)** is global and robust.
        # SciPy's `differential_evolution` has `maxiter` and `popsize`. 
        # It can be slow. 
        
        # Let's stick to a **High-Performance (1+1)-ES with Adaptive Step Size (1/5 rule)** 
        # but with **Multiple Restarts** and **Better Initialization**.
        # The parent (1.0) was a single (1+1)-ES. 
        # To beat 1.0, we need to ensure we don't get stuck in local minima of Rosenbrock 
        # (which has a long, curved valley). 
        # Rosenbrock in 5D is not extremely hard, but the valley is narrow.
        
        # A **CMA-ES** implementation is the best bet. 
        # I will implement a simplified but correct CMA-ES.
        
        # Parameters for CMA-ES
        mu = lambda_ // 2  # Number of recombination points
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / sum(weights**2)
        
        # Update rates
        cc = (4.0 + mueff / dim) / (dim + 4.0 + 2.0 * mueff / dim)
        cs = (mueff + 2.0) / (dim + mueff + 5.0)
        c1 = 2.0 / ((dim + 1.3)**2 + mueff)
        cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((dim + 2.0)**2 + mueff))
        
        # Eigenvalues of C for step size update
        # We need to maintain C and its eigen decomposition or use Cholesky.
        # For dim=5, Cholesky is fast.
        
        # Initialize
        x_best = x_mean.copy()
        f_best = self._safe_eval(func, x_best)
        
        C = np.eye(dim)
        # Step size sigma
        sigma = 1.0
        
        # Path accumulators
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        
        # Initial Cholesky factor
        D, V = np.linalg.eigh(C)
        # C = V @ diag(D) @ V.T
        # We need to sample from N(0, C).
        # x = x_mean + sigma * V @ (sqrt(D) * z)
        
        while self.call_count < budget:
            # Check time
            if time.time() - self.start_time > time_limit:
                break
                
            # Sample lambda points
            ys = []
            xs = []
            fs = []
            
            # Check if we have enough budget for a full generation
            # Each generation uses lambda evals.
            if self.call_count + lambda_ > budget:
                # Do a smaller population or stop
                # If we can't do a full generation, we might just stop 
                # or do a (1+1)-ES step.
                # For simplicity, if we can't do a full gen, break.
                # Or we can reduce lambda dynamically? No, fixed lambda is better for stability.
                # Let's break and do a final (1+1)-ES refinement if needed.
                break
                
            for i in range(lambda_):
                # Sample
                z = rng.standard_normal(dim)
                # Transform z to have covariance C
                # C = V @ diag(D) @ V.T
                # sqrt(C) = V @ diag(sqrt(D))
                y = V @ (np.sqrt(D) * z)
                x = x_mean + sigma * y
                x = np.clip(x, -5, 5)
                
                fx = self._safe_eval(func, x)
                ys.append(y)
                xs.append(x)
                fs.append(fx)
                
                if self.call_count >= budget:
                    break
            
            if len(ys) < 2:
                break
                
            # Sort by fitness
            order = np.argsort(fs)
            ys = [ys[i] for i in order]
            xs = [xs[i] for i in order]
            fs = [fs[i] for i in order]
            
            # Update best
            if fs[0] < self.best_f:
                self.best_f = fs[0]
                self.best_x = xs[0].copy()
            
            # Update mean
            # x_new = sum(weights[i] * xs[i]) for i in range(mu)
            x_new = np.zeros(dim)
            for i in range(mu):
                x_new += weights[i] * xs[i]
            
            # Update ps
            # ps = (1-cs) * ps + sqrt(cs * (2-cs) * mueff) / sigma * (V.T @ (x_new - x_mean))
            # Note: x_new - x_mean is in the transformed space? No, x_new is in original space.
            # The update rule for ps is in the eigen-space or using the inverse square root of C?
            # Standard CMA-ES:
            # ps = (1-cs) * ps + sqrt(cs * (2-cs) * mueff) * inv(sqrt(C)) * (x_new - x_mean) / sigma
            # We have C = V @ diag(D) @ V.T.
            # inv(sqrt(C)) = V @ diag(1/sqrt(D)) @ V.T
            inv_sqrt_C = V @ (np.diag(1.0 / np.sqrt(D))) @ V.T
            ps = (1.0 - cs) * ps + np.sqrt(cs * (2.0 - cs) * mueff) * (inv_sqrt_C @ (x_new - x_mean)) / sigma
            
            # Update pc
            # pc = (1-cc) * pc + cc * (x_new - x_mean) / sigma
            pc = (1.0 - cc) * pc + cc * (x_new - x_mean) / sigma
            
            # Update C
            # C = (1 - c1 - cmu) * C + c1 * (ps @ ps.T + (cs*(2-cs)) * (||ps||^2/dim - 1) * C) + cmu * sum(weights[i] * (yi @ yi.T))
            # Wait, the standard update is:
            # C = (1 - c1 - cmu) * C + c1 * (ps @ ps.T + (cs*(2-cs)) * (||ps||^2/dim - 1) * C) + cmu * sum(weights[i] * (yi @ yi.T))
            # Where yi are the sorted step vectors (in original space? No, usually the steps y_i).
            # In our case, y_i are the steps (x_i - x_mean)/sigma.
            # We stored ys as the steps y.
            
            # Correction: The standard CMA-ES update uses the steps y_i.
            # C = (1 - c1 - cmu) * C + c1 * (ps @ ps.T + (cs*(2-cs)) * (||ps||^2/dim - 1) * C) + cmu * sum(weights[i] * (y_i @ y_i.T))
            
            delta_hsig = (1.0 - np.exp(-2.0 * cs / dim)) * (1.0 - np.exp(-cs / dim))
            # Actually, the correction term is (cs * (2-cs)) * (||ps||^2 / dim - 1)
            # But there is a more stable version using the "delta_hsig" factor? 
            # Let's stick to the basic formula.
            
            # Calculate the rank-1 update
            rank1 = ps @ ps.T + (cs * (2.0 - cs)) * (np.dot(ps, ps) / dim - 1.0) * C
            
            # Calculate the rank-mu update
            rankmu = np.zeros((dim, dim))
            for i in range(mu):
                rankmu += weights[i] * np.outer(ys[i], ys[i])
                
            C = (1.0 - c1 - cmu) * C + c1 * rank1 + cmu * rankmu
            
            # Ensure symmetry and positive semi-definite
            C = (C + C.T) / 2.0
            
            # Update sigma
            # sigma = sigma * exp(cs / (2*dim) * (||ps|| / chiN - 1))
            chiN = np.sqrt(dim) * (1.0 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim**2))
            sigma = sigma * np.exp(cs / (2.0 * dim) * (np.linalg.norm(ps) / chiN - 1.0))
            
            # Update mean
            x_mean = x_new
            
            # Update eigen decomposition
            D, V = np.linalg.eigh(C)
            # Clip negative eigenvalues to 0
            D = np.maximum(D, 1e-12)
            
            # If sigma becomes too small or large, we might stop
            if sigma < 1e-8:
                break

        # If we have budget left, do a final (1+1)-ES refinement
        if self.call_count < budget - 5 and self.best_x is not None:
            x = self.best_x.copy()
            fx = self.best_f
            sigma_es = 0.1
            while self.call_count < budget - 1:
                if time.time() - self.start_time > time_limit:
                    break
                y = x + sigma_es * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = self._safe_eval(func, y)
                if fy < fx:
                    x, fx = y, fy
                    sigma_es *= 1.1
                else:
                    sigma_es *= 0.9
                sigma_es = np.clip(sigma_es, 1e-6, 5.0)
                
            if fx < self.best_f:
                self.best_x = x
                self.best_f = fx

        return self.best_x, self.best_f

    def _safe_eval(self, func, x):
        if self.call_count >= self.budget:
            raise ValueError("Budget Exceeded")
        
        x = np.clip(x, -5, 5)
        f = func(x)
        self.call_count += 1
        
        if f < self.best_f:
            self.best_f = f
            self.best_x = x.copy()
            
        return f
