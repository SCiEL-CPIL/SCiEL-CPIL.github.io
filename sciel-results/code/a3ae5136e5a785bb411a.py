"""Adaptive surrogate-based optimizer for BBOB f18 (Schaffers F7 ill-conditioned)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Surrogate-assisted optimizer with low-discrepancy initialization.
    
    Strategy:
    - Phase 1: Low-discrepancy (Halton) sequence for initial space-filling sampling
    - Phase 2: Kriging surrogate model with Expected Improvement acquisition
    - Phase 3: Local refinement with adaptive step size (1+1-ES)
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        n = self.dim
        budget = self.budget

        if budget < 20:
            rng = np.random.default_rng(0)
            best_x = rng.uniform(-5, 5, n)
            best_f = func(best_x)
            for _ in range(budget - 1):
                x = rng.uniform(-5, 5, n)
                f = func(x)
                if f < best_f:
                    best_f, best_x = f, x
            return best_x, best_f

        calls_made = 0

        def safe_call(x):
            nonlocal calls_made
            if calls_made >= budget:
                return None
            if time.time() > deadline:
                return None
            calls_made += 1
            return func(x)

        # Phase 1: Halton sequence for space-filling initial sampling
        # Use about 30% of budget for initial exploration
        n_halton = max(15, int(budget * 0.30))
        
        def halton_sequence(n_points, dim):
            """Generate Halton sequence points in [0,1]^dim."""
            points = np.zeros((n_points, dim))
            for d in range(dim):
                prime = self._nth_prime(d + 1)
                for i in range(n_points):
                    f = 1.0
                    i_val = i + 1
                    while i_val > 0:
                        f /= prime
                        i_val = i_val // prime
                    points[i, d] = f
            return points

        def _nth_prime(self, n):
            """Return the nth prime number."""
            primes = []
            candidate = 2
            while len(primes) < n:
                is_prime = True
                for p in primes:
                    if p * p > candidate:
                        break
                    if candidate % p == 0:
                        is_prime = False
                        break
                if is_prime:
                    primes.append(candidate)
                candidate += 1
            return primes[-1]

        # Generate Halton points and map to [-5, 5]^n
        halton_pts = halton_sequence(n_halton, n)
        halton_pts = halton_pts * 10.0 - 5.0  # Map [0,1] to [-5,5]

        best_x = None
        best_f = np.inf

        for x in halton_pts:
            if calls_made >= budget or time.time() > deadline:
                break
            x = np.clip(x, -5, 5)
            f = safe_call(x)
            if f is None:
                break
            if f < best_f:
                best_f = f
                best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(n)
            best_f = safe_call(best_x)
            if best_f is None:
                return best_x, np.inf

        # Phase 2: Kriging surrogate with Expected Improvement
        # Use about 50% of remaining budget for surrogate-guided search
        remaining_after_phase1 = budget - calls_made
        n_kriging = max(20, int(remaining_after_phase1 * 0.50))

        # Store observed points for kriging
        X_obs = [best_x.copy()]
        y_obs = [best_f]

        # Add all Halton points we evaluated
        for x in halton_pts[:min(len(halton_pts), calls_made)]:
            if calls_made <= 0:
                break
            # We need to re-evaluate or store; for simplicity, just use best
            pass

        # Actually, let's properly track all evaluated points
        X_all = []
        y_all = []
        
        # Re-do Phase 1 tracking properly
        # Reset and redo with proper tracking
        calls_made = 0
        best_x = None
        best_f = np.inf
        X_all = []
        y_all = []

        for x in halton_pts:
            if calls_made >= budget or time.time() > deadline:
                break
            x = np.clip(x, -5, 5)
            f = safe_call(x)
            if f is None:
                break
            X_all.append(x.copy())
            y_all.append(f)
            if f < best_f:
                best_f = f
                best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(n)
            best_f = safe_call(best_x)
            if best_f is None:
                return best_x, np.inf
            X_all.append(best_x.copy())
            y_all.append(best_f)

        # Kriging implementation
        def kriging_predict(X_train, y_train, X_test):
            """Simple Gaussian process with constant + linear mean and squared exponential kernel."""
            X_train = np.array(X_train)
            y_train = np.array(y_train)
            X_test = np.array(X_test)
            
            m = len(X_train)
            n_test = len(X_test)
            
            # Standardize y
            y_mean = np.mean(y_train)
            y_std = np.std(y_train) + 1e-10
            y_stdized = (y_train - y_mean) / y_std
            
            # Squared exponential kernel
            def kernel(A, B, length_scale=1.0, variance=1.0):
                # A: (m, d), B: (n, d)
                diff = A[:, np.newaxis, :] - B[np.newaxis, :, :]
                sq_dist = np.sum(diff ** 2, axis=2)
                return variance * np.exp(-0.5 * sq_dist / (length_scale ** 2))
            
            # Estimate length scale from data
            if m > 1:
                dists = []
                for i in range(m):
                    for j in range(i + 1, m):
                        dists.append(np.linalg.norm(X_train[i] - X_train[j]))
                length_scale = max(0.5, np.median(dists))
            else:
                length_scale = 1.0
            
            K = kernel(X_train, X_train, length_scale)
            K[np.diag_indices(m)] += 1e-6  # Numerical stability
            
            try:
                K_inv = np.linalg.inv(K)
            except np.linalg.LinAlgError:
                K_inv = np.linalg.pinv(K)
            
            # Predict at test points
            K_star = kernel(X_test, X_train, length_scale)
            
            mu = y_mean + K_star @ K_inv @ (y_stdized - 0)  # Mean is 0 after standardization
            mu = mu * y_std + y_mean
            
            # Variance
            v = 1.0 - K_star @ K_inv @ K_star.T
            v = np.maximum(v, 1e-6)
            std = np.sqrt(v * y_std ** 2)
            
            return mu, std

        def expected_improvement(mu, sigma, best_f, xi=0.01):
            """Expected improvement acquisition function."""
            if sigma < 1e-10:
                return 0.0
            z = (best_f - mu - xi) / sigma
            from scipy.stats import norm
            ei = (best_f - mu - xi) * norm.cdf(-z) + sigma * norm.pdf(-z)
            return np.maximum(ei, 0.0)

        # Surrogate-guided search
        rng = np.random.default_rng(123)
        
        for _ in range(n_kriging):
            if calls_made >= budget or time.time() > deadline:
                break
            
            if len(X_all) < 2:
                # Not enough data for kriging, use random
                x_new = rng.uniform(-5, 5, n)
            else:
                # Generate candidate points: mix of random and local perturbations
                n_candidates = 50
                candidates = []
                
                # Random candidates
                for _ in range(n_candidates // 2):
                    candidates.append(rng.uniform(-5, 5, n))
                
                # Local perturbations around best point
                for _ in range(n_candidates // 4):
                    x_cand = best_x + 0.5 * rng.standard_normal(n)
                    candidates.append(np.clip(x_cand, -5, 5))
                
                # Points near other observed good points
                if len(X_all) > 1:
                    for _ in range(n_candidates // 4):
                        idx = rng.integers(0, len(X_all))
                        x_cand = X_all[idx] + 0.3 * rng.standard_normal(n)
                        candidates.append(np.clip(x_cand, -5, 5))
                
                candidates = np.array(candidates)
                
                try:
                    mu, sigma = kriging_predict(X_all, y_all, candidates)
                    ei = expected_improvement(mu, sigma, best_f, xi=0.01)
                    best_idx = np.argmax(ei)
                    x_new = candidates[best_idx]
                except Exception:
                    x_new = rng.uniform(-5, 5, n)
            
            x_new = np.clip(x_new, -5, 5)
            f_new = safe_call(x_new)
            if f_new is None:
                break
            
            X_all.append(x_new.copy())
            y_all.append(f_new)
            
            if f_new < best_f:
                best_f = f_new
                best_x = x_new.copy()

        # Phase 3: Local refinement with adaptive (1+1)-ES
        remaining = budget - calls_made
        if remaining > 10 and best_x is not None:
            sigma = 0.3
            x = best_x.copy()
            fx = best_f
            fail_count = 0
            
            for _ in range(remaining):
                if calls_made >= budget or time.time() > deadline:
                    break
                
                z = rng.standard_normal(n)
                x_trial = x + sigma * z
                x_trial = np.clip(x_trial, -5, 5)
                
                f_trial = safe_call(x_trial)
                if f_trial is None:
                    break
                
                if f_trial < fx:
                    x = x_trial
                    fx = f_trial
                    fail_count = 0
                    sigma *= 1.1
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    fail_count += 1
                    sigma *= 0.95
                
                sigma = np.clip(sigma, 1e-5, 5.0)
                
                if fail_count > 15:
                    sigma = 0.5
                    fail_count = 0
                    # Try a larger jump
                    x_trial = x + 0.8 * rng.standard_normal(n)
                    x_trial = np.clip(x_trial, -5, 5)
                    f_trial = safe_call(x_trial)
                    if f_trial is not None and f_trial < fx:
                        x = x_trial
                        fx = f_trial
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
