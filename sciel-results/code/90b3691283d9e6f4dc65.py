"""Rewritten optimizer for BBOB-style f13 (Sharp ridge)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start Sobol + adaptive Nelder-Mead for BBOB f13 (Sharp ridge)."""

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
        n_init = min(50, budget // 5)
        sobol.random(1)  # skip origin
        points = sobol.random(n_init) * 10.0 - 5.0

        for i in range(n_init):
            if calls >= budget or elapsed() > deadline:
                break
            evaluate(points[i])

        # Phase 2: Adaptive multi-scale Nelder-Mead refinement
        # Sort by function value, take top candidates
        if best_x is not None and calls < budget and elapsed() < deadline:
            # Sort all evaluated points
            all_points.sort(key=lambda p: p[0])
            
            # Adaptive scales: start wide, narrow as we improve
            # For sharp ridge, we want to explore along the ridge direction
            rng = np.random.default_rng(123)
            
            # Get top-K candidates for refinement
            top_k = min(5, len(all_points))
            remaining_budget = budget - calls
            
            # Allocate budget: ~60% for NM, ~30% for extra exploration, ~10% reserve
            nm_budget = int(remaining_budget * 0.6)
            explore_budget = int(remaining_budget * 0.3)
            
            # Nelder-Mead from top candidates with adaptive initial simplex size
            for k in range(top_k):
                if calls >= budget or elapsed() > deadline:
                    break
                
                x_start = all_points[k][1].copy()
                f_start = all_points[k][0]
                
                # Adaptive simplex size: larger for worse starts, smaller for better
                # The improvement ratio tells us how much we've found
                improvement_ratio = (f_start - best_f) / (abs(best_f) + 1e-8)
                base_scale = 1.5 * max(0.1, min(3.0, 1.0 / (1.0 + abs(improvement_ratio))))
                
                # For sharp ridge, use anisotropic perturbation
                # Perturb more in directions where the function is flat
                simplex_size = base_scale
                
                def nm_objective(x):
                    nonlocal calls
                    if calls >= budget:
                        return best_f
                    f = func(x)
                    calls += 1
                    return f
                
                try:
                    result = minimize(
                        nm_objective,
                        x_start,
                        method='Nelder-Mead',
                        options={
                            'maxiter': min(100, max(20, nm_budget // (k + 1))),
                            'xatol': 1e-7,
                            'fatol': 1e-9,
                            'initial_simplex_size': simplex_size
                        }
                    )
                    if result.fun < best_f:
                        best_f = result.fun
                        best_x = result.x.copy()
                        all_points.append((best_f, best_x.copy()))
                except Exception:
                    pass
        
        # Phase 3: Additional exploration if budget remains
        if calls < budget - 10 and elapsed() < deadline:
            # Generate more points in a ring around the best
            n_extra = min(20, (budget - calls) // 3)
            rng2 = np.random.default_rng(456)
            
            for i in range(n_extra):
                if calls >= budget or elapsed() > deadline:
                    break
                
                # Adaptive scale: shrink as we get closer to optimum
                # Use log-scale: from 2.0 down to 0.01
                progress = min(1.0, calls / budget)
                scale = 2.0 * (1.0 - progress) ** 2 + 0.01
                
                # Anisotropic: perturb more in random directions
                direction = rng2.standard_normal(n)
                direction /= (np.linalg.norm(direction) + 1e-10)
                x_try = best_x + direction * scale
                x_try = np.clip(x_try, -5.0, 5.0)
                
                f_try = evaluate(x_try)
                
                # If we found improvement, do a quick NM polish
                if f_try < best_f and calls < budget - 5:
                    def polish_objective(x):
                        nonlocal calls
                        if calls >= budget:
                            return best_f
                        f = func(x)
                        calls += 1
                        return f
                    
                    try:
                        result = minimize(
                            polish_objective,
                            x_try,
                            method='Nelder-Mead',
                            options={
                                'maxiter': 30,
                                'xatol': 1e-8,
                                'fatol': 1e-10
                            }
                        )
                        if result.fun < best_f:
                            best_f = result.fun
                            best_x = result.x.copy()
                    except Exception:
                        pass
        
        # Phase 4: Final high-precision polish
        if best_x is not None and calls < budget - 5 and elapsed() < deadline:
            def final_objective(x):
                nonlocal calls
                if calls >= budget:
                    return best_f
                f = func(x)
                calls += 1
                return f
            
            try:
                result = minimize(
                    final_objective,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 50,
                        'xatol': 1e-9,
                        'fatol': 1e-12
                    }
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
