"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Structure-Exploiting Optimizer for Büche-Rastrigin with multi-start L-BFGS-B."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # safety margin before 330s kill

        call_count = 0
        max_calls = self.budget
        best_f = float('inf')
        best_x = np.zeros(self.dim)
        dim = self.dim

        def counted_func(x):
            nonlocal call_count, best_f, best_x
            if call_count >= max_calls:
                raise RuntimeError("Budget exceeded")
            if time.time() > deadline:
                raise RuntimeError("Time limit exceeded")
            call_count += 1
            x = np.clip(np.asarray(x, dtype=float), -5, 5)
            f = func(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        rng = np.random.default_rng(42)

        # Phase 1: Structure-aware sampling near integer coordinates
        # Büche-Rastrigin has minima at integer coordinates (shifted by hidden offset)
        # Sample points near integers to find good basins
        integers = list(range(-5, 6))
        candidates = []
        
        # Sample points near each integer coordinate combination (subset)
        n_samples = min(80, max_calls // 5)
        for i in range(n_samples):
            if call_count >= max_calls or time.time() > deadline:
                break
            # Each coordinate: pick a random integer and add small noise
            x = np.array([
                rng.choice(integers) + rng.normal(0, 0.3)
                for _ in range(dim)
            ])
            x = np.clip(x, -5, 5)
            f = counted_func(x)
            candidates.append((f, x.copy()))
        
        # Also add some uniform random samples
        n_uniform = min(20, max_calls - call_count)
        for i in range(n_uniform):
            if call_count >= max_calls or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            f = counted_func(x)
            candidates.append((f, x.copy()))
        
        # Sort candidates by fitness
        candidates.sort(key=lambda t: t[0])
        
        # Keep top unique candidates
        unique_candidates = []
        min_dist = 0.5
        for f, x in candidates:
            if call_count >= max_calls or time.time() > deadline:
                break
            is_unique = True
            for _, ux in unique_candidates:
                if np.linalg.norm(x - ux) < min_dist:
                    is_unique = False
                    break
            if is_unique:
                unique_candidates.append((f, x))
            if len(unique_candidates) >= 10:
                break
        
        # Phase 2: Multi-start L-BFGS-B from top candidates
        # L-BFGS-B is effective for smooth functions like Büche-Rastrigin
        for f_init, x_init in unique_candidates:
            if call_count >= max_calls or time.time() > deadline:
                break
            
            try:
                res = minimize(
                    counted_func,
                    x_init,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass
        
        # Phase 3: Coordinate-wise refinement with ridge-jumping
        x = best_x.copy()
        f_curr = best_f
        
        for iter_local in range(30):
            if call_count >= max_calls or time.time() > deadline:
                break
            
            improved = False
            for d in range(dim):
                if call_count >= max_calls or time.time() > deadline:
                    break
                
                # Try various step sizes including ridge-jumping (near 2π ≈ 6.28)
                step_sizes = [0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 6.0, 6.28, 8.0]
                best_step_f = f_curr
                best_step_x = x.copy()
                
                for step in step_sizes:
                    for direction in [-1, 1]:
                        if call_count >= max_calls or time.time() > deadline:
                            break
                        x_trial = x.copy()
                        x_trial[d] += direction * step
                        if x_trial[d] < -5 or x_trial[d] > 5:
                            continue
                        f_trial = counted_func(x_trial)
                        if f_trial < best_step_f:
                            best_step_f = f_trial
                            best_step_x = x_trial.copy()
                
                if best_step_f < f_curr:
                    x = best_step_x
                    f_curr = best_step_f
                    improved = True
                    if f_curr < best_f:
                        best_f = f_curr
                        best_x = x.copy()
            
            if not improved:
                break
        
        # Phase 4: Final fine-tuning with Nelder-Mead if budget remains
        if call_count < max_calls - 20 and time.time() < deadline:
            try:
                nm_res = minimize(
                    counted_func,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 50, 'xtol': 1e-10, 'ftol': 1e-10}
                )
                if nm_res.fun < best_f:
                    best_f = nm_res.fun
                    best_x = nm_res.x.copy()
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
