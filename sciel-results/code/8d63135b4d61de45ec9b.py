"""Bayesian Optimization with Gaussian Process surrogate for Lunacek bi-Rastrigin."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Bayesian Optimization with GP surrogate and Expected Improvement acquisition."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        total_budget = self.budget
        
        # Deadline guard
        start_time = time.time()
        max_time = 285.0  # Safety margin before 300s kill

        # Global best tracking
        global_best_x = None
        global_best_f = np.inf
        
        # We will use a two-phase strategy:
        # Phase 1: Random initialization to build initial GP training set (50 points)
        # Phase 2: Bayesian Optimization loop with GP surrogate and EI acquisition
        
        calls_made = 0
        rng = np.random.default_rng(42)
        
        # Phase 1: Random Sampling to initialize
        n_init = 50
        if total_budget < n_init:
            n_init = total_budget // 2
            
        init_pop = rng.uniform(-5, 5, size=(n_init, dim))
        init_fvals = np.array([func(x) for x in init_pop])
        calls_made += n_init
        
        best_idx = np.argmin(init_fvals)
        global_best_x = init_pop[best_idx].copy()
        global_best_f = init_fvals[best_idx]
        
        # Store training data
        X_train = init_pop.copy()
        y_train = init_fvals.copy()
        
        # Phase 2: Bayesian Optimization Loop
        remaining_budget = total_budget - calls_made
        
        # GP Parameters (fixed for simplicity, tuned for dim=5)
        # We use a simple RBF kernel with length scales and signal variance
        # For 5D, we need to be careful with kernel parameters
        
        # Hyperparameters
        length_scales = np.ones(dim) * 1.0  # Initial length scales
        signal_var = 100.0  # Signal variance (variance of the function)
        noise_var = 1e-4    # Noise variance (small, since function is deterministic)
        
        # We will use a fixed kernel for efficiency, or adapt if time permits
        # For simplicity, use fixed kernel with reasonable defaults
        
        # Acquisition: Expected Improvement (EI)
        # We will use a grid-based or random-search based acquisition optimization
        # Since dim=5, we can do a local search around the best candidate
        
        # Main loop
        while calls_made < total_budget:
            if time.time() - start_time > max_time:
                break
            
            # If remaining budget is small, just do random sampling or local search
            if total_budget - calls_made < 10:
                # Do a few random samples around global best
                for _ in range(total_budget - calls_made):
                    if time.time() - start_time > max_time:
                        break
                    x_new = global_best_x + rng.normal(0, 0.1, size=dim)
                    x_new = np.clip(x_new, -5, 5)
                    f_new = func(x_new)
                    calls_made += 1
                    if f_new < global_best_f:
                        global_best_f = f_new
                        global_best_x = x_new.copy()
                    X_train = np.vstack([X_train, x_new])
                    y_train = np.append(y_train, f_new)
                break
            
            # Build GP model
            # We use a simple GP with RBF kernel
            # For efficiency, we'll use a fixed kernel and solve the linear system each time
            
            # Compute kernel matrix
            n_train = len(X_train)
            
            # Standardize X for better conditioning? 
            # The domain is [-5, 5]^5. Length scales of 1.0 might be too small or large.
            # Let's try to adapt length scales roughly based on data spread?
            # Or just use fixed reasonable values.
            
            # RBF kernel: K(x, x') = signal_var * exp(-0.5 * sum((x_i - x'_i)^2 / l_i^2)) + noise_var * I
            
            # For n_train up to 500, computing K is O(n^2 * d) = 500^2 * 5 = 1.25M ops, fast.
            # Solving linear system is O(n^3) = 500^3 = 125M ops, might be slow if done every iteration.
            # But we only do this once per BO iteration (which adds 1 point).
            # 500 iterations * 125M ops = 62.5B ops. Too slow?
            # Let's check: 125M flops in numpy is ~0.1s? 
            # 500 * 0.1s = 50s. That's fine within 285s budget.
            
            # However, we can optimize by using Cholesky factorization and updating incrementally?
            # Too complex. Let's just re-solve each time but limit n_train or use a simpler model.
            
            # Alternative: Use a simpler surrogate like a local quadratic model?
            # Or use a fixed number of training points (e.g., last 100)?
            
            # Let's try a simpler approach: 
            # Instead of a full GP, use a local search with a surrogate model that is cheaper.
            # Or, use a GP but with a smaller effective dimension or simpler kernel.
            
            # Let's stick with GP but be efficient.
            # We'll use a fixed kernel with length scales = 2.0 (to cover the domain)
            
            ls = 2.0  # Single length scale for simplicity (isotropic kernel)
            sv = 100.0
            nv = 1e-4
            
            # Compute K
            # X_train is (n, d)
            # K_ij = sv * exp(-0.5 * ||x_i - x_j||^2 / ls^2)
            
            # Efficient distance computation
            # ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2 * x_i . x_j
            X_norms = np.sum(X_train**2, axis=1)[:, None]
            K = X_norms + X_norms.T - 2 * X_train @ X_train.T
            K = np.maximum(K, 0)  # Numerical stability
            K = sv * np.exp(-K / (2 * ls**2)) + nv * np.eye(n_train)
            
            # Solve for alpha
            # K @ alpha = y_train
            try:
                alpha = np.linalg.solve(K, y_train)
            except np.linalg.LinAlgError:
                # Fallback to least squares
                alpha = np.linalg.lstsq(K, y_train, rcond=None)[0]
            
            # Now, find the next point using EI acquisition
            # EI(x) = (mu(x) - f_best) * Phi((mu(x) - f_best)/sigma(x)) + sigma(x) * phi((mu(x) - f_best)/sigma(x))
            # where mu and sigma are the GP mean and std dev at x
            
            # We need to optimize EI(x). Since dim=5, we can use a local search or grid.
            # Let's use a random search with local refinement.
            
            # Generate candidate points
            n_candidates = 200
            # Mix of random points in the domain and points near the current best
            cand_pop = []
            # 50% random in domain
            cand_pop.append(rng.uniform(-5, 5, size=(n_candidates // 2, dim)))
            # 50% near global best (sigma=1.0)
            cand_pop.append(rng.normal(global_best_x, 1.0, size=(n_candidates - n_candidates // 2, dim)))
            cand_pop = np.vstack(cand_pop)
            cand_pop = np.clip(cand_pop, -5, 5)
            
            # Compute GP mean and std for candidates
            # K_fc = K(cand, train) = sv * exp(-||cand - train||^2 / (2*ls^2))
            Xc_norms = np.sum(cand_pop**2, axis=1)[:, None]
            K_fc = Xc_norms + X_norms.T - 2 * cand_pop @ X_train.T
            K_fc = np.maximum(K_fc, 0)
            K_fc = sv * np.exp(-K_fc / (2 * ls**2))
            
            # Mean: mu = K_fc @ alpha
            mu = K_fc @ alpha
            
            # Variance: sigma^2 = sv - K_fc @ K^{-1} @ K_fc.T
            # We already have K^{-1} implicitly via alpha = K^{-1} y
            # But we need K^{-1} @ K_fc.T for each candidate.
            # Let's compute K_inv_Kfc = solve(K, K_fc.T)
            try:
                K_inv_Kfc = np.linalg.solve(K, K_fc.T)
            except np.linalg.LinAlgError:
                K_inv_Kfc = np.linalg.lstsq(K, K_fc.T, rcond=None)[0]
            
            var = sv - np.sum(K_fc * K_inv_Kfc, axis=1)
            var = np.maximum(var, 1e-8)  # Numerical stability
            sigma = np.sqrt(var)
            
            # Compute EI
            # f_best is the current global best
            imp = mu - global_best_f
            Z = imp / sigma
            from scipy.stats import norm
            EI = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            
            # Select the best candidate
            best_cand_idx = np.argmax(EI)
            x_new = cand_pop[best_cand_idx].copy()
            
            # Evaluate
            f_new = func(x_new)
            calls_made += 1
            
            # Update global best
            if f_new < global_best_f:
                global_best_f = f_new
                global_best_x = x_new.copy()
            
            # Update training set
            X_train = np.vstack([X_train, x_new])
            y_train = np.append(y_train, f_new)
            
            # Optional: If EI is very small, we might be converging. 
            # But we continue until budget is exhausted.
            
            # To speed up, we could stop if EI is below a threshold, but let's use full budget.
            
        return global_best_x, global_best_f

# EVOLVE-BLOCK-END
