<<<<<<< SEARCH
"""CMA-ES based optimizer for shifted/rotated Rosenbrock-style f8."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """CMA-ES with adaptive restarts and boundary handling for Rosenbrock."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0

        best_x = None
        best_f = np.inf
        call_count = 0
        dim = self.dim
        budget = self.budget

        def safe_call(x):
            nonlocal best_x, best_f, call_count
            if call_count >= self.budget:
                return np.inf
            call_count += 1
            x = np.asarray(x, dtype=np.float64)
            # Clip to box to ensure valid input for func
            x_clipped = np.clip(x, -5.0, 5.0)
            try:
                fx = float(func(x_clipped))
            except Exception:
                fx = np.inf
            
            if np.isfinite(fx) and fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx

        # CMA-ES implementation without external library dependency to avoid overhead
        # and ensure strict budget control.
        # We use a simplified CMA-ES or rely on scipy's minimize with Nelder-Mead 
        # if CMA-ES library is not available, but scipy has CMA-ES in newer versions?
        # Actually, scipy.optimize has 'Nelder-Mead', 'Powell', 'L-BFGS-B'. 
        # SciPy does NOT have CMA-ES built-in in standard distributions usually.
        # However, the prompt says "Python standard library, NumPy and SciPy are available".
        # It does NOT guarantee 'cma' package.
        
        # Let's implement a robust Multi-Start L-BFGS-B strategy as suggested by the 
        # "knowledge from this run" which measured an improvement to 2.626 with 
        # "Multi-start L-BFGS-B with adaptive budget allocation". 
        # The current program uses Powell/NM. L-BFGS-B is gradient-based (quasi-Newton) 
        # and often faster for smooth Rosenbrock if it can find the basin.
        # But Rosenbrock has a narrow valley. L-BFGS-B can struggle if not close.
        # CMA-ES is better for global. 
        # Since 'cma' is not guaranteed, let's implement a high-quality 
        # Multi-Start Nelder-Mead with adaptive budget allocation and better initial points,
        # OR use a simple CMA-ES implementation if feasible within time.
        
        # Given the constraint and the specific knowledge that "Multi-start L-BFGS-B ... improved island 5",
        # let's try a hybrid: 
        # 1. Coarse global search to find a good basin.
        # 2. L-BFGS-B for fast local convergence.
        
        # Strategy:
        # 1. Evaluate a set of diverse starting points (grid + random).
        # 2. Run L-BFGS-B from the best few points.
        # 3. If L-BFGS-B fails or stalls, fallback to Nelder-Mead.
        
        # Phase 1: Diverse Initial Points
        # Rosenbrock minimum is at (1,1,...,1). Shifted/rotated means it's somewhere else.
        # We need to find the basin.
        
        rng = np.random.default_rng(42)
        
        # Generate candidate starting points
        candidates = []
        
        # Uniform random points
        for _ in range(20):
            candidates.append(rng.uniform(-4, 4, dim))
            
        # Strategic points
        candidates.append(np.zeros(dim))
        candidates.append(np.ones(dim))
        candidates.append(np.full(dim, -1))
        candidates.append(np.full(dim, 2))
        
        # Evaluate all candidates
        scored_candidates = []
        for cand in candidates:
            if call_count >= budget:
                break
            if time.time() > deadline - 50: # Leave time for optimization
                break
            fx = safe_call(cand)
            if np.isfinite(fx):
                scored_candidates.append((fx, cand))
        
        # Sort by function value
        scored_candidates.sort(key=lambda x: x[0])
        
        # Phase 2: Local Optimization from Best Candidates
        # Use L-BFGS-B as it's efficient for smooth functions.
        # If L-BFGS-B is not available or fails, use Nelder-Mead.
        
        max_local_starts = min(5, len(scored_candidates))
        
        for i in range(max_local_starts):
            if call_count >= budget:
                break
            if time.time() > deadline - 30:
                break
            
            fx0, x0 = scored_candidates[i]
            
            # Try L-BFGS-B first
            try:
                # L-BFGS-B requires bounds
                bounds = [(-5.0, 5.0)] * dim
                result = minimize(
                    lambda x: safe_call(x),
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxiter': 100,
                        'maxfun': max(1, budget - call_count - 10),
                        'ftol': 1e-10,
                        'gtol': 1e-8,
                    }
                )
            except Exception:
                # Fallback to Nelder-Mead if L-BFGS-B fails
                try:
                    result = minimize(
                        lambda x: safe_call(x),
                        x0,
                        method='Nelder-Mead',
                        options={
                            'maxiter': 100,
                            'maxfev': max(1, budget - call_count - 10),
                            'xatol': 1e-8,
                            'fatol': 1e-10,
                            'adaptive': True,
                        }
                    )
                except Exception:
                    pass

        # Phase 3: Final Polish with Nelder-Mead if budget remains
        if best_x is not None and call_count < budget and time.time() < deadline - 10:
            try:
                remaining = max(1, budget - call_count - 5)
                result = minimize(
                    lambda x: safe_call(x),
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 100,
                        'maxfev': remaining,
                        'xatol': 1e-10,
                        'fatol': 1e-12,
                        'adaptive': True,
                    }
                )
            except Exception:
                pass

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = float(func(best_x))
        
        return best_x, best_f
# EVOLVE-BLOCK-END
=======
"""CMA-ES based optimizer for shifted/rotated Rosenbrock-style f8."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """CMA-ES with adaptive restarts and boundary handling for Rosenbrock."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 270.0

        best_x = None
        best_f = np.inf
        call_count = 0
        dim = self.dim
        budget = self.budget

        def safe_call(x):
            nonlocal best_x, best_f, call_count
            if call_count >= self.budget:
                return np.inf
            call_count += 1
            x = np.asarray(x, dtype=np.float64)
            x_clipped = np.clip(x, -5.0, 5.0)
            try:
                fx = float(func(x_clipped))
            except Exception:
                fx = np.inf
            
            if np.isfinite(fx) and fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx

        rng = np.random.default_rng(42)

        # Phase 1: Diverse initial sampling to find basin
        n_init = min(40, budget // 5)
        candidates = []
        for _ in range(n_init):
            candidates.append(rng.uniform(-4.5, 4.5, dim))
        
        # Add strategic points
        strategic = [np.zeros(dim), np.ones(dim), np.full(dim, -1.0), np.full(dim, 2.0),
                     np.full(dim, 0.5), np.full(dim, -0.5), np.array([1.0]*dim),
                     np.array([-1.0]*dim), np.array([2.0]*dim), np.array([0.0]*dim)]
        for s in strategic:
            if len(candidates) < n_init + 10:
                candidates.append(s)

        scored_candidates = []
        for cand in candidates:
            if call_count >= budget:
                break
            if time.time() > deadline - 60:
                break
            fx = safe_call(cand)
            if np.isfinite(fx):
                scored_candidates.append((fx, cand.copy()))
        
        scored_candidates.sort(key=lambda x: x[0])

        # Phase 2: CMA-ES global optimization from best starting point
        if scored_candidates and call_count < budget and time.time() < deadline - 30:
            x0 = scored_candidates[0][1]
            self._run_cmaes(func, x0, budget, deadline, safe_call)

        # Phase 3: Additional CMA-ES restarts from other good points if budget remains
        for i in range(1, min(3, len(scored_candidates))):
            if call_count >= budget:
                break
            if time.time() > deadline - 20:
                break
            x0 = scored_candidates[i][1]
            self._run_cmaes(func, x0, budget, deadline, safe_call, max_calls=50)

        # Phase 4: Final high-precision Nelder-Mead polish
        if best_x is not None and call_count < budget and time.time() < deadline - 5:
            try:
                remaining = max(1, budget - call_count)
                minimize(
                    lambda x: safe_call(x),
                    best_x.copy(),
                    method='Nelder-Mead',
                    options={
                        'maxiter': min(200, remaining),
                        'maxfev': remaining,
                        'xatol': 1e-12,
                        'fatol': 1e-14,
                        'adaptive': True,
                    }
                )
            except Exception:
                pass

        if best_x is None:
            best_x = np.zeros(dim)
            if call_count < budget:
                best_f = float(func(best_x))
                call_count += 1
        
        return best_x, best_f

    def _run_cmaes(self, func, x0, budget, deadline, safe_call, max_calls=None):
        """Lightweight CMA-ES implementation with strict budget control."""
        if max_calls is None:
            max_calls = max(1, budget - self._call_count())
        
        dim = self.dim
        pop_size = max(4, int(4 + 3 * np.log(dim)))
        
        # CMA-ES parameters
        sigma = 1.0
        mean = x0.astype(np.float64).copy()
        
        # Covariance matrix (identity initially)
        C = np.eye(dim)
        C_eigvals = np.ones(dim)
        C_eigvecs = np.eye(dim)
        
        # Step size control
        cm = (pop_size + 2) ** 0.5 + 1
        cs = (pop_size + 1.5 + 2 / (dim + 1)) / (dim + pop_size + 3)
        ds = 1 + 2 * max(0, (cs - 1) / (cs + 1)) + cs
        c1 = 2 / ((dim + 1) ** 2 + 10)
        cmu = min(c1, 2 * (1 - c1) / ((dim + 2) ** 2 - 2))
        
        # Path accumulators
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Inverse square root of C
        C_sqrt_inv = np.eye(dim)
        
        # Precompute
        self._update_C_sqrt_inv(C, C_eigvals, C_eigvecs)
        
        calls_in_cmaes = 0
        max_iter = max(1, min(50, max_calls // pop_size))
        
        for gen in range(max_iter):
            if time.time() > deadline - 10:
                break
            if calls_in_cmaes >= max_calls:
                break
            
            # Generate population
            offspring = []
            offspring_f = []
            
            for i in range(pop_size):
                if calls_in_cmaes >= max_calls:
                    break
                if time.time() > deadline - 10:
                    break
                
                z = np.random.randn(dim)
                x_new = mean + sigma * C_sqrt_inv @ z
                x_new = np.clip(x_new, -5.0, 5.0)
                
                fx = safe_call(x_new)
                calls_in_cmaes += 1
                
                if np.isfinite(fx):
                    offspring.append(x_new)
                    offspring_f.append(fx)
            
            if len(offspring) < pop_size:
                break
            
            # Sort by fitness
            sorted_idx = np.argsort(offspring_f)
            offspring = [offspring[i] for i in sorted_idx]
            offspring_f = [offspring_f[i] for i in sorted_idx]
            
            # Update mean
            weights = np.zeros(pop_size)
            weights[:pop_size // 2] = np.log(pop_size / 2 + 0.5) - np.log(np.arange(1, pop_size // 2 + 1))
            weights /= weights.sum()
            
            old_mean = mean.copy()
            mean = np.sum([weights[i] * offspring[i] for i in range(len(weights))], axis=0)
            
            # Update sigma (step size)
            delta_mean = (mean - old_mean) / (sigma * np.linalg.norm(C_sqrt_inv @ np.ones(dim)) + 1e-30)
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mu_eff) * delta_mean
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1))) / np.sqrt(1 - (1 - cs) ** 2) < (1.4 + 2 / (dim + 1)) * np.sqrt((dim + 1) / (dim - 1)))
            sigma *= np.exp(cs / ds * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
            sigma = max(1e-12, min(sigma, 10.0))
            
            # Update covariance
            p_c = (1 - c1) * p_c + h_sigma * np.sqrt(c1 * (2 - c1) * mu_eff) * (mean - old_mean) / sigma
            
            rank_1 = np.outer(p_c, p_c)
            rank_mu = np.sum([weights[i] * np.outer(offspring[i] - old_mean, offspring[i] - old_mean) for i in range(len(weights))], axis=0)
            
            C = (1 + c1 * (2 - c1) - cmu) * C + c1 * rank_1 + cmu * rank_mu
            
            # Ensure positive semi-definite
            C = (C + C.T) / 2
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-20)
            C_eigvals = eigvals
            C_eigvecs = eigvecs
            self._update_C_sqrt_inv(C, C_eigvals, C_eigvecs)
            
            # Convergence check
            if np.max(eigvals) / np.min(eigvals) > 1e14:
                break
            if sigma < 1e-14:
                break
            if np.linalg.norm(mean - old_mean) / (sigma + 1e-30) < 1e-12:
                break

    def _update_C_sqrt_inv(self, C, eigvals, eigvecs):
        """Compute inverse square root of C via eigendecomposition."""
        self.C_eigvals = eigvals
        self.C_eigvecs = eigvecs
        eigvals_safe = np.maximum(eigvals, 1e-20)
        C_sqrt_inv = eigvecs @ np.diag(1.0 / np.sqrt(eigvals_safe)) @ eigvecs.T
        self.C_sqrt_inv = C_sqrt_inv

    def _call_count(self):
        return getattr(self, '_cc', 0)
# EVOLVE-BLOCK-END
>>>>>>> REPLACE

Wait, I need to fix the `_call_count` issue - I'm using `safe_call` which has its own `call_count` via closure, but `_run_cmaes` needs access to it. Let me restructure to pass the state properly.

