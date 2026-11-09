"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import norm


class Optimizer:
    """Two-phase surrogate-assisted optimizer for Rosenbrock (f8)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Track all evaluations
        self.X = []
        self.Y = []
        self.call_count = 0

        def evaluate(x):
            x = np.clip(np.asarray(x, dtype=float), -5, 5)
            self.X.append(x.copy())
            self.Y.append(func(x))
            self.call_count += 1
            return self.Y[-1]

        # Phase 1: Initial exploration with Latin Hypercube + random
        n_phase1 = min(50, budget)
        
        # Latin Hypercube sampling for better coverage
        indices = np.arange(n_phase1)
        lhd = np.zeros((n_phase1, dim))
        for d in range(dim):
            perm = rng.permutation(n_phase1)
            lhd[:, d] = (perm + rng.uniform(size=n_phase1)) / n_phase1
        lhd = lhd * 10 - 5  # scale to [-5, 5]

        # Add some random points too
        n_lhd = n_phase1 // 2
        n_rand = n_phase1 - n_lhd
        random_pts = rng.uniform(-5, 5, (n_rand, dim))
        
        initial_points = np.vstack([lhd[:n_lhd], random_pts])

        best_x = None
        best_f = np.inf

        for pt in initial_points:
            if self.call_count >= budget:
                break
            if time.time() > deadline - 30:
                break
            f = evaluate(pt)
            if f < best_f:
                best_f = f
                best_x = pt.copy()

        # Phase 2: Use GP surrogate to find promising basins
        # Fit a simple GP surrogate
        X = np.array(self.X)
        Y = np.array(self.Y)
        
        # Normalize Y for GP
        Y_mean = Y.mean()
        Y_std = Y.std() + 1e-10
        Y_norm = (Y - Y_mean) / Y_std

        # Simple GP with RBF kernel
        def gp_predict(X_test):
            X_test = np.atleast_2d(X_test)
            # Compute kernel matrix
            K = self._rbf_kernel(X, X_test)
            K_noise = self._rbf_kernel(X, X) + 1e-6 * np.eye(len(X))
            alpha = np.linalg.solve(K_noise, Y_norm)
            mu = K.T @ alpha
            # Predictive variance (simplified)
            v = np.ones(len(X))
            var = 1.0 - K @ np.linalg.solve(K_noise, K.T)
            return mu, np.maximum(var, 0)

        def _rbf_kernel(X1, X2, length_scale=1.0):
            # Compute squared distances
            sq_dists = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T
            sq_dists = np.maximum(sq_dists, 0)
            return np.exp(-sq_dists / (2 * length_scale**2))

        # Try to find promising points using GP posterior mean
        n_phase2_candidates = 20
        candidate_points = []
        
        # Sample from GP posterior mean minimization
        for _ in range(100):
            if self.call_count >= budget:
                break
            if time.time() > deadline - 20:
                break
            # Random candidate
            cand = rng.uniform(-4, 4, dim)
            mu, var = gp_predict(cand)
            # Acquisition: minimize predicted mean
            candidate_points.append((mu[0], cand))
        
        # Sort by predicted mean and evaluate top candidates
        candidate_points.sort(key=lambda x: x[0])
        for _, cand in candidate_points[:n_phase2_candidates]:
            if self.call_count >= budget:
                break
            if time.time() > deadline - 15:
                break
            f = evaluate(cand)
            if f < best_f:
                best_f = f
                best_x = cand.copy()

        # Phase 3: Local optimization from best points
        # Use LBFGS-B from multiple starting points
        n_local_starts = min(10, budget - self.call_count)
        
        # Get top starting points from all evaluations
        sorted_indices = np.argsort(self.Y)
        start_points = [self.X[i].copy() for i in sorted_indices[:n_local_starts]]
        
        # Add the current best
        if best_x is not None:
            start_points.append(best_x.copy())
        
        for sp in start_points:
            if self.call_count >= budget:
                break
            if time.time() > deadline - 10:
                break
            
            # Count remaining budget for this local optimization
            remaining = budget - self.call_count
            if remaining < 5:
                break
            
            # Wrap func to count calls and respect budget
            local_calls = [0]
            def local_func(x):
                if self.call_count >= budget:
                    return 1e10
                if time.time() > deadline - 5:
                    return 1e10
                local_calls[0] += 1
                x = np.clip(x, -5, 5)
                f = func(x)
                self.X.append(x.copy())
                self.Y.append(f)
                self.call_count += 1
                return f
            
            try:
                res = minimize(local_func, sp, method='L-BFGS-B',
                             bounds=[(-5, 5)] * dim,
                             options={'maxiter': 100, 'maxfun': min(remaining, 200)})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5, 5).copy()
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
