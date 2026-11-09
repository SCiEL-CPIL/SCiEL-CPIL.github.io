"""Rewritten optimizer for BBOB-style f13 (Sharp ridge)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Sobol + adaptive coordinate descent for BBOB f13 (Sharp ridge)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 275.0
        n = self.dim
        budget = self.budget

        def elapsed():
            return time.time() - t0

        # Sobol sequence for low-discrepancy initialization
        try:
            from scipy.stats import qmc
            sobol = qmc.Sobol(d=n, scramble=True, seed=42)
        except ImportError:
            class _SobolFallback:
                def __init__(self, d, seed=0):
                    self.d = d
                    self._rng = np.random.default_rng(seed)
                def random(self, m):
                    return self._rng.random((m, self.d))
            sobol = _SobolFallback(n)

        best_x = None
        best_f = np.inf
        calls = 0
        all_points = []  # (f, x) pairs for tracking

        def evaluate(x):
            nonlocal calls, best_f, best_x
            if calls >= budget or elapsed() > deadline:
                return np.inf
            f = func(x)
            calls += 1
            all_points.append((f, x.copy()))
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Broad Sobol exploration
        n_init = min(60, budget // 5)
        sobol.random(1)  # skip origin
        points = sobol.random(n_init) * 10.0 - 5.0

        for i in range(n_init):
            if calls >= budget or elapsed() > deadline:
                break
            evaluate(points[i])

        if best_x is None:
            return None, np.inf

        # Phase 2: Adaptive coordinate descent with per-dimension step tracking
        # For Sharp Ridge f13: one direction is very flat (the ridge), others are steep.
        # Coordinate descent naturally exploits this separable structure.
        # We track per-dimension step sizes and success rates to adapt.

        # Per-dimension state
        step_sizes = np.ones(n) * 1.0  # initial step size per dimension
        success_count = np.zeros(n)   # consecutive successes per dimension
        fail_count = np.zeros(n)      # consecutive failures per dimension
        step_history = [1.0] * n      # moving average of successful step magnitudes

        def coordinate_descent(x_start, f_start, max_iters, base_scale):
            """Adaptive coordinate descent with per-dimension step sizes."""
            nonlocal calls, best_f, best_x, step_sizes, success_count, fail_count, step_history
            x = x_start.copy()
            f = f_start
            local_best_f = f
            local_best_x = x.copy()
            
            for it in range(max_iters):
                if calls >= budget or elapsed() > deadline:
                    break
                
                improved = False
                for d in range(n):
                    if calls >= budget or elapsed() > deadline:
                        break
                    
                    # Adaptive step: use moving average of successful steps
                    # Scale by base_scale and adapt based on success/failure history
                    eff_step = step_sizes[d] * base_scale
                    if eff_step < 1e-10:
                        eff_step = 1e-10
                    if eff_step > 5.0:
                        eff_step = 5.0
                    
                    # Try positive and negative steps
                    best_local_f = f
                    best_local_x = x.copy()
                    best_local_d = 0
                    
                    for sign in [1, -1]:
                        x_try = x.copy()
                        x_try[d] += sign * eff_step
                        x_try[d] = np.clip(x_try[d], -5.0, 5.0)
                        
                        f_try = evaluate(x_try)
                        if f_try < best_local_f:
                            best_local_f = f_try
                            best_local_x = x_try.copy()
                            best_local_d = sign * eff_step
                    
                    # If improvement found, update and adjust step size
                    if best_local_d != 0:
                        x = best_local_x.copy()
                        f = best_local_f
                        improved = True
                        
                        # Update per-dimension state
                        success_count[d] += 1
                        fail_count[d] = 0
                        step_history[d] = 0.7 * abs(best_local_d) + 0.3 * step_history[d]
                        
                        # Grow step if consistently succeeding (up to 3x)
                        if success_count[d] >= 3:
                            step_sizes[d] = min(step_sizes[d] * 1.3, 5.0)
                        elif success_count[d] == 1:
                            step_sizes[d] = min(step_sizes[d] * 1.1, 5.0)
                        
                        if f < local_best_f:
                            local_best_f = f
                            local_best_x = x.copy()
                    else:
                        # No improvement in this dimension
                        fail_count[d] += 1
                        success_count[d] = 0
                        # Shrink step if failing
                        if fail_count[d] >= 2:
                            step_sizes[d] = max(step_sizes[d] * 0.5, 1e-8)
                        elif fail_count[d] == 1:
                            step_sizes[d] = max(step_sizes[d] * 0.8, 1e-8)
                
                if not improved:
                    # Full cycle with no improvement: shrink all steps and break if too small
                    step_sizes *= 0.7
                    if np.max(step_sizes) < 1e-7:
                        break
                    # Reset failure counters to avoid over-shrinking
                    fail_count *= 0.5
                    success_count = np.zeros(n)
            
            # Update global best
            if local_best_f < best_f:
                best_f = local_best_f
                best_x = local_best_x.copy()
            
            return x, f

        # Sort all evaluated points by function value
        all_points.sort(key=lambda p: p[0])
        remaining_budget = budget - calls
        
        # Allocate budget across multiple CD restarts from top candidates
        top_k = min(4, len(all_points))
        
        for k in range(top_k):
            if calls >= budget or elapsed() > deadline:
                break
            
            x_start = all_points[k][1].copy()
            f_start = all_points[k][0]
            
            # Adaptive base scale: larger for worse starts
            improvement_ratio = (f_start - best_f) / (abs(best_f) + 1e-8)
            base_scale = 1.0 * max(0.2, min(2.0, 1.0 / (1.0 + abs(improvement_ratio))))
            
            # Allocate iterations: more for better starts
            iters = max(10, (budget - calls) // (top_k - k + 1))
            
            # Reset per-dimension state for each restart
            step_sizes = np.ones(n) * 0.5
            success_count = np.zeros(n)
            fail_count = np.zeros(n)
            step_history = [0.5] * n
            
            coordinate_descent(x_start, f_start, iters, base_scale)
        
        # Phase 3: Fine-grained coordinate descent polish on best point
        if calls < budget - 5 and elapsed() < deadline and best_x is not None:
            # Use very small steps for final precision
            step_sizes = np.ones(n) * 0.01
            success_count = np.zeros(n)
            fail_count = np.zeros(n)
            step_history = [0.01] * n
            
            remaining = budget - calls
            iters = max(5, remaining // 2)
            coordinate_descent(best_x.copy(), best_f, iters, 0.1)
        
        # Phase 4: Final micro-polish
        if calls < budget - 3 and elapsed() < deadline and best_x is not None:
            step_sizes = np.ones(n) * 0.001
            success_count = np.zeros(n)
            fail_count = np.zeros(n)
            step_history = [0.001] * n
            
            remaining = budget - calls
            iters = max(3, remaining)
            coordinate_descent(best_x.copy(), best_f, iters, 0.05)

        return best_x, best_f
# EVOLVE-BLOCK-END
