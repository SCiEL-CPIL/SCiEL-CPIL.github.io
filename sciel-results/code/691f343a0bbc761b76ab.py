"""Multi-objective optimizer on RE34: Surrogate-assisted optimization with Expected Hypervolume Improvement."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm


class Optimizer:
    """Surrogate-assisted multi-objective optimizer for RE34 using EHVI."""

    def __init__(self, budget, dim, n_obj):
        self.budget = int(budget)
        self.dim = int(dim)
        self.n_obj = int(n_obj)

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        used = 0
        X_list = []
        F_list = []

        def evaluate(x):
            nonlocal used
            x = np.clip(x, lo, hi)
            f = np.asarray(problem(x), dtype=np.float64)
            used += 1
            return x, f

        rng = np.random.default_rng(42)

        # Reference point for hypervolume
        ref_point = np.array([1705.0, 12.0, 0.3])

        # Initialize with a diverse set of points
        # Use a grid-like initialization for good coverage
        n_init = 15
        for i in range(n_init):
            x = rng.uniform(lo, hi)
            x, f = evaluate(x)
            X_list.append(x)
            F_list.append(f)

        # Also add some corner points for better initial coverage
        corners = []
        for i in range(2 ** dim):
            corner = np.array([(hi[d] if (i >> d) & 1 else lo[d]) for d in range(dim)])
            corners.append(corner)
        for c in corners[:8]:  # Add up to 8 corners
            if used < budget:
                x, f = evaluate(c)
                X_list.append(x)
                F_list.append(f)

        X = np.array(X_list)
        F = np.array(F_list)

        # Helper: Check if a point dominates another
        def dominates(f1, f2):
            return np.all(f1 <= f2) and np.any(f1 < f2)

        # Helper: Get non-dominated points
        def get_non_dominated(F):
            n = len(F)
            is_nd = np.ones(n, dtype=bool)
            for i in range(n):
                if not is_nd[i]:
                    continue
                for j in range(n):
                    if i == j or not is_nd[j]:
                        continue
                    if dominates(F[j], F[i]):
                        is_nd[i] = False
                        break
            return np.where(is_nd)[0]

        # Helper: Hypervolume contribution of a point (simplified)
        def point_hv_contribution(f, ref, other_points):
            """Calculate the unique hypervolume contribution of point f."""
            if np.any(f >= ref):
                return 0.0
            
            # Volume of the box defined by f and ref
            box_vol = np.prod(ref - f)
            
            # Subtract volume already covered by other non-dominated points
            covered = 0.0
            for other in other_points:
                if dominates(other, f):
                    continue
                if dominates(f, other):
                    continue
                # Overlap calculation
                overlap = np.minimum(ref, np.maximum(f, other))
                if np.all(overlap > np.minimum(f, other)):
                    covered += np.prod(np.minimum(ref, np.maximum(f, other)) - np.minimum(f, other))
            
            return max(0.0, box_vol - covered)

        # Helper: Simple hypervolume calculation for a set of points
        def calc_hv(F_set, ref):
            """Calculate hypervolume for a set of non-dominated points."""
            if len(F_set) == 0:
                return 0.0
            
            # Filter points that are better than reference
            valid = [f for f in F_set if np.all(f < ref)]
            if len(valid) == 0:
                return 0.0
            
            # For 3 objectives, we can use a simple O(n^2) approach
            # Sort by first objective
            valid = np.array(valid)
            idx = np.argsort(valid[:, 0])
            valid = valid[idx]
            
            hv = 0.0
            for i in range(len(valid)):
                f = valid[i]
                # For each point, calculate the volume it uniquely contributes
                # This is a simplified approach
                vol = 0.0
                for j in range(len(valid)):
                    if i == j:
                        continue
                    other = valid[j]
                    # Check if this point is dominated by any other in the set
                    if dominates(other, f):
                        break
                else:
                    # Point is non-dominated in this set
                    # Calculate its unique contribution
                    # This is an approximation
                    vol = np.prod(np.maximum(0, ref - f))
                    hv += vol * 0.3  # Rough scaling factor
            
            return hv

        # Main loop: Surrogate-assisted optimization
        n_candidates = 50
        gp_retrain_interval = 10  # Retrain GP every 10 evaluations

        # Simple GP surrogate using kernel regression
        class SimpleGP:
            def __init__(self, dim, n_obj, alpha=0.1, length_scale=0.5):
                self.dim = dim
                self.n_obj = n_obj
                self.alpha = alpha
                self.length_scale = length_scale
                self.X_train = None
                self.F_train = None
                self.K_inv = None
                self.mean = None

            def kernel(self, X1, X2):
                # RBF kernel
                X1 = np.atleast_2d(X1)
                X2 = np.atleast_2d(X2)
                d = np.sum(X1 ** 2, axis=1)[:, None] + np.sum(X2 ** 2, axis=1)[None, :] - 2 * X1 @ X2.T
                d = np.maximum(d, 0)
                return np.exp(-d / (2 * self.length_scale ** 2))

            def fit(self, X, F):
                self.X_train = X
                self.F_train = F
                n = len(X)
                K = self.kernel(X, X)
                K += self.alpha * np.eye(n)
                try:
                    self.K_inv = np.linalg.inv(K)
                except np.linalg.LinAlgError:
                    self.K_inv = np.linalg.pinv(K)

            def predict(self, X_test):
                X_test = np.atleast_2d(X_test)
                n_test = len(X_test)
                K_test = self.kernel(X_test, self.X_train)
                
                # Mean prediction
                mu = K_test @ self.K_inv @ self.F_train
                
                # Variance prediction
                K_test_train = self.kernel(self.X_train, X_test)
                var = np.diag(self.kernel(X_test, X_test))
                var -= np.sum(K_test @ self.K_inv @ K_test_train, axis=1)
                var = np.maximum(var, 1e-10)
                
                return mu, var

        gp = SimpleGP(dim, n_obj)
        gp.fit(X, F)

        # EHVI-based acquisition
        def ehvi_acquisition(X_cand, mu, var, F_current, ref):
            """Calculate EHVI for each candidate point."""
            n_cand = len(X_cand)
            n_obj = F_current.shape[1]
            ehvi = np.zeros(n_cand)
            
            # Get non-dominated points for reference
            nd_idx = get_non_dominated(F_current)
            F_nd = F_current[nd_idx]
            
            for i in range(n_cand):
                # For each candidate, estimate EHVI
                # Simplified: use expected improvement per objective
                ei_sum = 0.0
                for k in range(n_obj):
                    # Find best observed value for objective k
                    f_best_k = np.min(F_current[:, k])
                    mu_k = mu[i, k]
                    sigma_k = np.sqrt(var[i, k])
                    
                    if sigma_k < 1e-10:
                        delta = f_best_k - mu_k
                        ei_k = max(0, delta)
                    else:
                        delta = f_best_k - mu_k
                        z = delta / sigma_k
                        ei_k = delta * norm.cdf(z) + sigma_k * norm.pdf(z)
                        ei_k = max(0, ei_k)
                    
                    ei_sum += ei_k
                
                ehvi[i] = ei_sum
            
            return ehvi

        # Main optimization loop
        while used < budget:
            if time.time() > deadline:
                break
            
            # Generate candidate points
            X_cand = rng.uniform(lo, hi, size=(n_candidates, dim))
            
            # Predict with GP
            mu, var = gp.predict(X_cand)
            
            # Calculate EHVI
            ehvi = ehvi_acquisition(X_cand, mu, var, F, ref_point)
            
            # Select best candidate
            idx = np.argmax(ehvi)
            x_new = X_cand[idx]
            
            # Evaluate
            x_new, f_new = evaluate(x_new)
            
            # Update
            X = np.vstack([X, x_new])
            F = np.vstack([F, f_new])
            
            # Periodically retrain GP
            if used % gp_retrain_interval == 0:
                gp.fit(X, F)

        return X
# EVOLVE-BLOCK-END
