"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Global-Local Hybrid with RBF surrogate and EI acquisition for BBOB f18."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # safety margin before 330s kill
        n = self.dim
        budget = self.budget

        # State tracking
        X = []  # list of points evaluated
        F = []  # corresponding function values
        best_x = None
        best_f = np.inf
        total_calls = 0

        def record(x, f):
            nonlocal best_x, best_f, total_calls
            X.append(x.copy())
            F.append(f)
            total_calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # --- Phase 1: Initial global exploration via LHS + RBF-EI ---
        # Use Latin Hypercube Sampling for initial spread
        def lhs_sample(n_points, dim, seed=0):
            rng = np.random.default_rng(seed)
            pts = []
            for _ in range(n_points):
                p = rng.random(dim)
                # Shift to [-5, 5]
                p = -5 + 10 * p
                pts.append(p)
            return np.array(pts)

        # Number of initial global points
        n_global = min(50, budget // 5)
        
        # Generate initial points
        init_points = lhs_sample(n_global, n, seed=42)
        
        for i in range(n_global):
            if time.time() > deadline:
                break
            if total_calls >= budget:
                break
            x = init_points[i]
            x = np.clip(x, -5, 5)
            f = func(x)
            record(x, f)

        # --- Phase 2: RBF Surrogate with EI Acquisition ---
        # RBF kernel: Gaussian
        def rbf_predict(X_train, F_train, x_new, length_scale=1.0):
            """
            Predict function value and variance using RBF surrogate.
            Uses a simple weighted average approach with Gaussian kernel.
            """
            if len(X_train) == 0:
                return 0.0, 1.0
            
            X_train = np.array(X_train)
            F_train = np.array(F_train)
            
            # Compute distances
            dists = np.sum((X_train - x_new) ** 2, axis=1)
            
            # Gaussian kernel weights
            w = np.exp(-dists / (2 * length_scale ** 2))
            w_sum = w.sum()
            
            if w_sum < 1e-12:
                # Fallback to nearest neighbor
                idx = np.argmin(dists)
                return F_train[idx], 0.1
            
            # Predicted value: weighted average
            f_pred = np.sum(w * F_train) / w_sum
            
            # Uncertainty: based on distance to nearest points
            # Higher uncertainty when far from training data
            min_dist = np.min(dists)
            variance = min_dist / (2 * length_scale ** 2)
            # Cap variance
            variance = min(variance, 10.0)
            
            return f_pred, variance

        def expected_improvement(x_new, best_current):
            """
            Compute Expected Improvement using RBF surrogate.
            """
            f_pred, variance = rbf_predict(X, F, x_new, length_scale=1.5)
            sigma = np.sqrt(variance)
            if sigma < 1e-12:
                return 0.0
            
            # Improvement over current best
            imp = best_current - f_pred
            z = imp / sigma
            # Standard normal CDF and PDF
            # Use approximation
            phi = 0.5 * (1 + np.tanh(z / np.sqrt(2)))  # approx CDF
            pdf = np.exp(-0.5 * z ** 2) / np.sqrt(2 * np.pi)
            
            EI = imp * phi + sigma * pdf
            return max(EI, 0.0)

        # Adaptively choose between global exploration and local exploitation
        # Use a mix of EI-based candidate generation and local search

        # Candidate generation for EI
        def generate_candidates(n_cands, rng):
            """Generate candidate points using a mix of strategies."""
            cands = []
            
            # 1. Random points in the box
            for _ in range(n_cands // 3):
                x = rng.uniform(-5, 5, n)
                cands.append(x)
            
            # 2. Points near the current best (local exploration)
            if best_x is not None:
                for _ in range(n_cands // 3):
                    sigma_local = 0.5 + 0.5 * (best_f - min(F)) / (max(F) - min(F) + 1e-8)
                    x = best_x + sigma_local * rng.standard_normal(n)
                    x = np.clip(x, -5, 5)
                    cands.append(x)
            
            # 3. Points using RBF-guided sampling
            for _ in range(n_cands - len(cands)):
                # Sample from a distribution that favors high uncertainty
                # Use a simple rejection sampling approach
                for attempt in range(20):
                    x = rng.uniform(-5, 5, n)
                    _, var = rbf_predict(X, F, x, length_scale=1.5)
                    if var > 0.1:  # Prefer uncertain regions
                        cands.append(x)
                        break
                else:
                    cands.append(rng.uniform(-5, 5, n))
            
            return cands

        # Main optimization loop
        rng = np.random.default_rng(123)
        
        while total_calls < budget:
            if time.time() > deadline:
                break
            
            # Determine phase: early on, explore more; later, exploit
            progress = total_calls / budget
            if progress < 0.3:
                n_cands = 20
                use_ei = True
            elif progress < 0.7:
                n_cands = 15
                use_ei = True
            else:
                n_cands = 10
                use_ei = False  # Switch to pure local search
            
            if use_ei:
                # Generate candidates and select by EI
                cands = generate_candidates(n_cands, rng)
                ei_scores = []
                for x in cands:
                    ei = expected_improvement(x, best_f)
                    ei_scores.append(ei)
                
                # Select top candidates by EI
                ei_scores = np.array(ei_scores)
                top_idx = np.argsort(ei_scores)[::-1][:3]
                
                # Evaluate the best EI candidate
                x_new = cands[top_idx[0]]
                x_new = np.clip(x_new, -5, 5)
                f_new = func(x_new)
                record(x_new, f_new)
                
                # Also evaluate a couple of other candidates if budget allows
                for idx in top_idx[1:]:
                    if total_calls >= budget or time.time() > deadline:
                        break
                    x_alt = cands[idx]
                    x_alt = np.clip(x_alt, -5, 5)
                    f_alt = func(x_alt)
                    record(x_alt, f_alt)
            else:
                # Local search phase: use Nelder-Mead or similar
                # Simple local refinement around best point
                x_start = best_x.copy()
                if x_start is None:
                    x_start = rng.uniform(-5, 5, n)
                
                # Try a few local perturbations
                for _ in range(3):
                    if total_calls >= budget or time.time() > deadline:
                        break
                    sigma = 0.3 * (1 - progress) + 0.05
                    x_new = x_start + sigma * rng.standard_normal(n)
                    x_new = np.clip(x_new, -5, 5)
                    f_new = func(x_new)
                    record(x_new, f_new)
                    if f_new < best_f:
                        x_start = x_new.copy()

        # --- Phase 3: Final high-precision polish ---
        # Use scipy's minimize for a final polish if we have time
        if time.time() < deadline - 5 and best_x is not None and total_calls < budget:
            try:
                # Use Nelder-Mead for local polish
                result = minimize(
                    lambda x: func(np.clip(x, -5, 5)),
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 50, 'xatol': 1e-6, 'fatol': 1e-8}
                )
                if result.fun < best_f:
                    best_x = np.clip(result.x, -5, 5)
                    best_f = result.fun
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
