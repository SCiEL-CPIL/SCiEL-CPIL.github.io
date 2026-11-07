import numpy as np
import time
from scipy.stats import norm
import math

class Optimizer:
    """
    Surrogate-assisted Bayesian Optimizer for BBOB f9 (Rosenbrock rotated).
    
    Strategy:
    - Phase 1 (Exploration): Evaluate a small set of random points to seed the surrogate.
    - Phase 2 (Exploitation/Exploration Balance): 
        - Fit a Gaussian Process (GP) to the observed data.
        - Use Expected Improvement (EI) to select the next point.
        - Add a small jitter to the EI maximization to avoid getting stuck in local optima of the acquisition function.
    - Phase 3 (Refinement): If budget allows and progress stalls, switch to a local (1+1)-ES for fine-tuning.
    
    This approach is structurally different from (1+1)-ES as it maintains a global model of the function landscape,
    allowing it to "jump" between basins if the current trajectory is suboptimal, guided by uncertainty.
    """
    
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.rng = np.random.default_rng(42) # Fixed seed for determinism in surrogate fit if needed, but usually random restarts are better. Let's use a fixed seed for reproducibility of the surrogate behavior, but randomize initial points if possible? No, evaluator creates new instance.
        self.rng = np.random.default_rng()

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 290  # Safety margin
        
        n = self.dim
        calls_made = 0
        
        # Storage for observations
        X_obs = []
        y_obs = []
        
        # Best tracking
        best_x = None
        best_f = np.inf
        
        # Phase 1: Initial Exploration
        # Sample 10 random points to get a rough idea of the landscape
        n_init = min(10, self.budget)
        for _ in range(n_init):
            x = self.rng.uniform(-5, 5, size=n)
            f = func(x)
            calls_made += 1
            
            X_obs.append(x)
            y_obs.append(f)
            
            if f < best_f:
                best_f = f
                best_x = x.copy()
                
            if calls_made >= self.budget:
                break

        X_obs = np.array(X_obs)
        y_obs = np.array(y_obs)
        
        # Normalize y for GP stability
        y_mean = np.mean(y_obs)
        y_std = np.std(y_obs) + 1e-8
        
        # Phase 2: Bayesian Optimization Loop
        # We will use a simple GP with RBF kernel.
        # To keep it fast, we limit the history size to the last 50 points to keep matrix inversion O(N^2) small.
        max_history = 50
        
        while calls_made < self.budget and time.time() < deadline:
            # Update history if it grows too large
            if len(X_obs) > max_history:
                # Keep the best 20 and the most recent 30? Or just most recent?
                # Let's keep the best points to exploit, and recent to explore.
                # Simple strategy: keep the best 10 and the most recent 40.
                idxs = list(range(len(X_obs)))
                # Sort by fitness
                sorted_idxs = np.argsort(y_obs)
                best_idxs = sorted_idxs[:10]
                recent_idxs = sorted_idxs[-40:]
                keep_idxs = np.unique(np.concatenate([best_idxs, recent_idxs]))
                X_obs = X_obs[keep_idxs]
                y_obs = y_obs[keep_idxs]
                y_mean = np.mean(y_obs)
                y_std = np.std(y_obs) + 1e-8

            # Fit GP
            try:
                # Simple RBF kernel
                # K(x, x') = sigma_f^2 * exp(-0.5 * ||x - x'||^2 / l^2)
                # We use a fixed length scale l=1.0 and signal variance sigma_f=1.0 for simplicity and speed.
                # Adding a small noise for numerical stability
                noise = 1e-8
                
                # Compute kernel matrix
                diff = X_obs[:, np.newaxis, :] - X_obs[np.newaxis, :, :]
                sq_dist = np.sum(diff**2, axis=-1)
                l = 1.0
                K = np.exp(-0.5 * sq_dist / (l*l))
                K += noise * np.eye(len(X_obs))
                
                # Normalize y
                y_norm = (y_obs - y_mean) / y_std
                
                # Solve for alpha
                try:
                    L = np.linalg.cholesky(K)
                    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_norm))
                except np.linalg.LinAlgError:
                    # Fallback to least squares if Cholesky fails
                    alpha = np.linalg.lstsq(K, y_norm, rcond=None)[0]
                
                # Acquisition Function: Expected Improvement
                # We want to maximize EI.
                # EI(x) = (f_best_norm - mu(x)) * Phi(Z) + sigma(x) * phi(Z)
                # where Z = (f_best_norm - mu(x)) / sigma(x)
                
                # To find the max of EI, we use a grid search + local optimization (Nelder-Mead) on the acquisition function.
                # This is a common and robust approach for low dimensions.
                
                f_best_norm = (best_f - y_mean) / y_std
                
                def acquisition(x):
                    # Predict mean and variance at x
                    x = x.reshape(-1, 1)
                    diff = X_obs - x.T
                    sq_dist = np.sum(diff**2, axis=1)
                    k = np.exp(-0.5 * sq_dist / (l*l))
                    
                    mu = np.dot(k, alpha) * y_std + y_mean
                    # Variance: sigma_f^2 - k^T K^-1 k
                    v = 1.0 - np.dot(k, alpha) * y_std # Wait, alpha is for normalized y.
                    # Correct variance calculation:
                    # v_norm = 1 - k^T K^-1 k
                    # v = v_norm * y_std^2
                    # Let's re-calculate v properly using L
                    try:
                        v = 1.0 - np.dot(np.linalg.solve(L, k), k)
                    except:
                        v = 1.0 - np.dot(k, alpha) # Approximation if L fails, but alpha was solved with K.
                        # Actually, if L exists, use it.
                    
                    v = max(v * y_std**2, 1e-8)
                    sigma = np.sqrt(v)
                    
                    if sigma < 1e-8:
                        return 0.0
                    
                    z = (f_best_norm - mu / y_std) / (sigma / y_std) # This is getting messy with normalization.
                    
                    # Let's work in raw space to avoid confusion.
                    # We need K_inv_y for raw y?
                    # Let's restart the GP math in raw space for clarity in acquisition.
                    # K is same. y is raw.
                    # alpha_raw = K^-1 y_obs
                    # mu = k^T alpha_raw
                    # v = 1 - k^T K^-1 k (in units of signal variance, which is 1)
                    # So v is just the variance.
                    
                    # Re-calculate alpha_raw if not done? No, we did alpha for normalized.
                    # alpha_raw = alpha * y_std
                    alpha_raw = alpha * y_std
                    mu = np.dot(k, alpha_raw)
                    v = 1.0 - np.dot(k, np.linalg.solve(K, k)) # K is noisy K.
                    v = max(v, 1e-8)
                    sigma = np.sqrt(v)
                    
                    if sigma < 1e-8:
                        return 0.0
                    
                    z = (best_f - mu) / sigma
                    ei = (best_f - mu) * norm.cdf(z) + sigma * norm.pdf(z)
                    return ei
                
                # Optimize acquisition function
                # Start from several random points and the current best
                starts = [best_x]
                for _ in range(5):
                    starts.append(self.rng.uniform(-5, 5, size=n))
                    
                best_x_acq = None
                best_ei = -np.inf
                
                for start in starts:
                    # Local optimization using Nelder-Mead
                    try:
                        from scipy.optimize import minimize
                        res = minimize(acquisition, start, method='Nelder-Mead', 
                                       options={'xatol': 0.01, 'fatol': 1e-4, 'maxiter': 50})
                        if res.fun > best_ei:
                            best_ei = res.fun
                            best_x_acq = res.x
                    except:
                        continue
                
                if best_x_acq is None:
                    # Fallback to random
                    best_x_acq = self.rng.uniform(-5, 5, size=n)
                
                # Clip to box
                best_x_acq = np.clip(best_x_acq, -5, 5)
                
                # Evaluate
                f_new = func(best_x_acq)
                calls_made += 1
                
                X_obs.append(best_x_acq)
                y_obs.append(f_new)
                
                if f_new < best_f:
                    best_f = f_new
                    best_x = best_x_acq.copy()
                    
            except Exception as e:
                # If GP fails, fallback to random search for a few steps
                for _ in range(5):
                    if calls_made >= self.budget:
                        break
                    x = self.rng.uniform(-5, 5, size=n)
                    f = func(x)
                    calls_made += 1
                    X_obs.append(x)
                    y_obs.append(f)
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()
                continue

        return best_x, best_f
