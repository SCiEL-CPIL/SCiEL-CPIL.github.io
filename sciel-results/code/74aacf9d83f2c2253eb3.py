# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Surrogate-assisted optimizer for BBOB f7 (Step Ellipsoidal) on [-5,5]^d.
    
    Strategy:
    1. Initial exploration with a diverse set of points.
    2. Use a simple surrogate model (Gaussian Process approximation via RBF kernel)
       to guide the search.
    3. Combine expected improvement with random exploration.
    4. Final polish with (1+1)-ES on the best point found.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self._call_count = 0
        self._best_x = None
        self._best_f = np.inf
        self._X = []  # list of evaluated points
        self._Y = []  # list of function values

    def _count_call(self):
        self._call_count += 1

    def _rbf_kernel(self, X1, X2, length_scale=1.0):
        """Compute RBF kernel matrix between X1 and X2."""
        # X1: (n, d), X2: (m, d)
        sq_dists = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T
        sq_dists = np.maximum(sq_dists, 0)
        return np.exp(-sq_dists / (2 * length_scale**2))

    def _surrogate_predict(self, X_new):
        """Predict mean and variance at X_new using RBF kernel regression."""
        if len(self._X) == 0:
            return np.zeros(len(X_new)), np.ones(len(X_new)) * 10.0
        
        X = np.array(self._X)
        Y = np.array(self._Y)
        
        # Standardize Y for numerical stability
        Y_mean = np.mean(Y)
        Y_std = np.std(Y) + 1e-8
        Y_norm = (Y - Y_mean) / Y_std
        
        # Compute kernel matrices
        K = self._rbf_kernel(X, X, length_scale=0.8)
        K_new = self._rbf_kernel(X_new, X, length_scale=0.8)
        
        # Add small noise for numerical stability
        K += 1e-6 * np.eye(len(X))
        
        # Solve for weights
        try:
            alpha = np.linalg.solve(K, Y_norm)
        except np.linalg.LinAlgError:
            alpha = np.linalg.lstsq(K, Y_norm, rcond=None)[0]
        
        # Predict mean
        mean_norm = K_new @ alpha
        mean = Y_mean + mean_norm * Y_std
        
        # Predict variance (simplified: use kernel diagonal)
        K_new_diag = self._rbf_kernel(X_new, X_new, length_scale=0.8).diagonal()
        var = Y_std**2 * np.maximum(1.0 - K_new_diag, 0.01)
        
        return mean, var

    def _expected_improvement(self, mean, var, best_f, xi=0.01):
        """Compute expected improvement acquisition function."""
        sigma = np.sqrt(var)
        sigma[sigma < 1e-10] = 1e-10
        z = (best_f - mean - xi) / sigma
        from scipy.stats import norm
        EI = (best_f - mean - xi) * norm.cdf(z) + sigma * norm.pdf(z)
        EI[z < -10] = 0.0  # clip extreme negative z
        return EI

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # safety margin before 300s limit

        # Wrap func to count calls
        def wrapped(x):
            self._count_call()
            return func(x)

        rng = np.random.default_rng(42)
        
        # Phase 1: Initial exploration with diverse points
        # Use a space-filling design
        n_initial = min(20, self.budget // 5)
        
        # Latin hypercube-like initial points
        for i in range(n_initial):
            if self._call_count >= self.budget or time.time() > deadline:
                break
            
            if i == 0:
                x = np.zeros(self.dim)
            elif i == 1:
                x = np.full(self.dim, 2.5)
            elif i == 2:
                x = np.full(self.dim, -2.5)
            else:
                # Random point in [-4, 4]^d
                x = rng.uniform(-4, 4, self.dim)
            
            x = np.clip(x, -5, 5)
            fx = wrapped(x)
            
            self._X.append(x.copy())
            self._Y.append(fx)
            
            if fx < self._best_f:
                self._best_f = fx
                self._best_x = x.copy()

        # Phase 2: Surrogate-guided search
        # For each step, generate candidate points, evaluate surrogate, pick best
        n_candidates = 20  # number of candidate points to evaluate surrogate on
        length_scale = 0.8
        
        while self._call_count < self.budget and time.time() < deadline:
            # Generate candidate points: mix of random and local perturbations
            candidates = []
            
            # Random candidates
            for _ in range(n_candidates // 2):
                x = rng.uniform(-5, 5, self.dim)
                candidates.append(x)
            
            # Local perturbations around best point
            if self._best_x is not None:
                for _ in range(n_candidates // 2):
                    sigma = 0.5
                    x = self._best_x + sigma * rng.standard_normal(self.dim)
                    x = np.clip(x, -5, 5)
                    candidates.append(x)
            
            candidates = np.array(candidates)
            
            # Evaluate surrogate on candidates
            mean, var = self._surrogate_predict(candidates)
            
            # Compute expected improvement
            EI = self._expected_improvement(mean, var, self._best_f)
            
            # Select candidate with highest EI
            best_idx = np.argmax(EI)
            x_new = candidates[best_idx]
            
            # Evaluate the function
            fx = wrapped(x_new)
            
            self._X.append(x_new.copy())
            self._Y.append(fx)
            
            if fx < self._best_f:
                self._best_f = fx
                self._best_x = x_new.copy()
            
            # Periodic check: if no improvement in last 20 calls, do a random restart
            if len(self._Y) > 20:
                recent_best = min(self._Y[-20:])
                if recent_best >= self._best_f * 1.001:  # no significant improvement
                    # Try a few random points
                    for _ in range(3):
                        if self._call_count >= self.budget or time.time() > deadline:
                            break
                        x_rand = rng.uniform(-5, 5, self.dim)
                        fx_rand = wrapped(x_rand)
                        self._X.append(x_rand.copy())
                        self._Y.append(fx_rand)
                        if fx_rand < self._best_f:
                            self._best_f = fx_rand
                            self._best_x = x_rand.copy()

        # Phase 3: Final polish with (1+1)-ES on the best point
        if self._best_x is not None and self._call_count < self.budget and time.time() < deadline:
            x = self._best_x.copy()
            fx = self._best_f
            sigma = 0.3
            
            while self._call_count < self.budget and time.time() < deadline:
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = wrapped(y)
                
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.1
                    if fx < self._best_f:
                        self._best_f = fx
                        self._best_x = x.copy()
                else:
                    sigma *= 0.9
                    if sigma < 1e-5:
                        break

        return self._best_x, self._best_f
# EVOLVE-BLOCK-END
