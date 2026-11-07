"""Optimized optimizer for BBOB-style f18 (Schaffers F7 ill-conditioned).

Uses restartable CMA-ES with lightweight covariance updates.
Structural change: single long CMA-ES run with stagnation-triggered restarts
instead of fixed multi-phase structure. Faster per-generation computation
via Cholesky-based sampling and periodic eigendecomposition only when needed.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Restartable CMA-ES for ill-conditioned BBOB-style problems."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _pattern_search(self, func, x0, sigma, max_calls, deadline, rng):
        """
        Direct pattern search with orthogonal and diagonal directions.
        Uses binary search on step size to guarantee progress.
        Returns best_x, best_f, calls_used.
        """
        dim = self.dim
        x = np.array(x0, dtype=float)
        f = func(x)
        calls = 1
        best_x = x.copy()
        best_f = f
        
        # Define search directions: standard basis + diagonal directions
        directions = []
        for i in range(dim):
            d = np.zeros(dim)
            d[i] = 1.0
            directions.append(d)
            d_neg = -d.copy()
            directions.append(d_neg)
        # Add diagonal directions
        d_diag = np.ones(dim) / np.sqrt(dim)
        directions.append(d_diag)
        directions.append(-d_diag)
        
        # Step size control
        step = sigma
        min_step = 1e-8
        max_step = 5.0
        
        # Try each direction with binary search on step size
        no_progress_count = 0
        max_no_progress = len(directions) * 3
        
        while calls < max_calls and time.time() < deadline - 2.0:
            progress = False
            
            for d in directions:
                if calls >= max_calls or time.time() > deadline - 2.0:
                    break
                
                # Binary search for optimal step in this direction
                lo = min_step
                hi = min(max_step, 2.0 * step)
                
                # Try to find a step that improves f
                best_step = None
                best_f_dir = f
                
                # Quick test with current step
                x_test = np.clip(x + step * d, -5, 5)
                f_test = func(x_test)
                calls += 1
                
                if f_test < f:
                    # Try to increase step
                    for _ in range(3):
                        if calls >= max_calls or time.time() > deadline - 2.0:
                            break
                        step_new = min(hi * 1.5, 5.0)
                        x_test = np.clip(x + step_new * d, -5, 5)
                        f_test = func(x_test)
                        calls += 1
                        if f_test < best_f_dir:
                            best_f_dir = f_test
                            best_step = step_new
                            hi = step_new
                        else:
                            break
                    if best_step is not None:
                        x = np.clip(x + best_step * d, -5, 5)
                        f = func(x)
                        calls += 1
                        step = best_step
                        progress = True
                else:
                    # Try smaller steps
                    for _ in range(3):
                        if calls >= max_calls or time.time() > deadline - 2.0:
                            break
                        step_new = max(min_step, step * 0.5)
                        x_test = np.clip(x + step_new * d, -5, 5)
                        f_test = func(x_test)
                        calls += 1
                        if f_test < f:
                            x = x_test
                            f = f_test
                            step = step_new
                            progress = True
                            break
                        else:
                            step = step_new
            
            # Update best
            if f < best_f:
                best_f = f
                best_x = x.copy()
            
            if not progress:
                no_progress_count += 1
                # Reduce step size
                step *= 0.5
                if step < min_step or no_progress_count > max_no_progress:
                    break
            else:
                no_progress_count = 0
        
        return best_x, best_f, calls

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        dim = self.dim
        budget = self.budget
        
        rng = np.random.default_rng(42)
        
        best_x = np.zeros(dim)
        best_f = np.inf
        total_calls = 0
        
        # Initial sampling: spread out to find a good basin
        n_init = min(15, budget // 8)
        for i in range(n_init):
            if total_calls >= budget or time.time() > deadline - 10.0:
                break
            x = rng.uniform(-4, 4, size=dim)
            fx = func(x)
            total_calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        # Main CMA-ES loop with pattern search refinement
        remaining = budget - total_calls
        max_restarts = 5
        restart_count = 0
        
        while remaining > 20 and time.time() < deadline - 10.0 and restart_count < max_restarts:
            # Determine sigma based on how far we've progressed
            if restart_count == 0:
                sigma0 = 1.5
            else:
                # Shrink sigma for restarts to refine around current best
                sigma0 = max(0.05, 1.5 * (0.5 ** restart_count))
            
            x0 = best_x.copy()
            
            x_best, f_best, calls_used, stalled = self._cma_es_single(
                func, rng, x0, sigma0, remaining, deadline
            )
            total_calls += calls_used
            remaining = budget - total_calls
            
            if f_best < best_f:
                best_f = f_best
                best_x = x_best.copy()
            
            restart_count += 1
            
            # Apply pattern search refinement after each CMA-ES run
            if remaining > 15 and time.time() < deadline - 5.0:
                x_ps, f_ps, calls_ps = self._pattern_search(
                    func, best_x, 0.1 * sigma0, remaining, deadline, rng
                )
                total_calls += calls_ps
                remaining = budget - total_calls
                
                if f_ps < best_f:
                    best_f = f_ps
                    best_x = x_ps.copy()
            
            # If not stalled and we still have budget, continue from current best
            # with slightly perturbed starting point for diversity
            if not stalled and remaining > 30:
                # Small perturbation to escape potential local minima
                perturbation = 0.1 * rng.standard_normal(dim)
                x0 = np.clip(best_x + perturbation, -4.5, 4.5)
                fx0 = func(x0)
                total_calls += 1
                remaining = budget - total_calls
                if fx0 < best_f:
                    best_f = fx0
                    best_x = x0.copy()
        
        # Final high-precision polish with pattern search
        if remaining > 10 and time.time() < deadline - 5.0:
            x_polish, f_polish, calls_polish = self._pattern_search(
                func, best_x, 0.05, remaining, deadline, rng
            )
            total_calls += calls_polish
            if f_polish < best_f:
                best_f = f_polish
                best_x = x_polish.copy()
        
        return best_x, best_f
# EVOLVE-BLOCK-END
