"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start L-BFGS-B with CMA-ES global search and NM polishing for BBOB f8."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin before 330s kill

        dim = self.dim
        budget = self.budget

        call_count = [0]
        best_x = None
        best_f = np.inf

        def safe_func(x):
            nonlocal best_x, best_f
            if call_count[0] >= budget:
                return 1e18
            if time.time() > deadline:
                return 1e18
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            call_count[0] += 1
            f_val = float(func(x))
            if np.isnan(f_val) or np.isinf(f_val):
                f_val = 1e10
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        def time_left():
            return (deadline - time.time()) > 3.0

        # --- Phase 1: Quick multi-start L-BFGS-B to find a good basin ---
        rng = np.random.default_rng(42)
        start_points = []

        # Center
        start_points.append(np.zeros(dim))

        # Structured points
        for v in [-4, -3, -2, 2, 3, 4]:
            start_points.append(np.full(dim, v))

        # Random points
        for _ in range(20):
            pt = rng.uniform(-4.5, 4.5, size=dim)
            start_points.append(pt)

        # Points along axes
        for i in range(dim):
            for v in [-3, -1, 1, 3]:
                pt = np.zeros(dim)
                pt[i] = v
                start_points.append(pt)

        # Remove duplicates
        seen = set()
        unique_points = []
        for p in start_points:
            key = tuple(np.round(p, 6))
            if key not in seen:
                seen.add(key)
                unique_points.append(p)

        # Run L-BFGS-B from each start point, limited budget per run
        bounds = [(-5.0, 5.0)] * dim
        max_calls_per_lbfgs = 40

        for p in unique_points:
            if call_count[0] >= budget - 100:
                break
            if not time_left():
                break
            if call_count[0] >= budget:
                break

            # Evaluate the start point first
            f_start = safe_func(p)

            # Try L-BFGS-B from this point
            if call_count[0] < budget - max_calls_per_lbfgs and time_left():
                try:
                    res = minimize(
                        safe_func,
                        p,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 30, 'maxfun': max_calls_per_lbfgs, 'gtol': 1e-6}
                    )
                except Exception:
                    pass

            # If we've found a very good point, break early
            if best_f < 1.0:
                break

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = safe_func(best_x)

        # --- Phase 2: CMA-ES global search centered on best point ---
        if time_left() and call_count[0] < budget - 50:
            try:
                # CMA-ES with population size based on dim
                pop_size = max(4 + int(3 * np.log(dim)), 10)
                remaining_budget = budget - call_count[0]
                
                # Limit CMA-ES to use at most 60% of remaining budget
                max_cma_calls = min(remaining_budget - 30, int(0.6 * remaining_budget))
                
                if max_cma_calls > 20:
                    # Start sigma based on problem scale
                    sigma0 = 2.0
                    
                    res_cma = minimize(
                        safe_func,
                        best_x,
                        method='CmaES',
                        options={
                            'maxiter': 50,
                            'popsize': pop_size,
                            'maxfev': max_cma_calls,
                            'sigma0': sigma0,
                            'bounds': bounds,
                            'seed': 42,
                            'verbose': False,
                            'tolfun': 1e-10,
                            'tolfun' : 1e-10,
                        }
                    )
                    if res_cma.fun < best_f:
                        best_x = np.clip(res_cma.x, -5.0, 5.0)
                        best_f = float(res_cma.fun)
            except Exception:
                pass

        # --- Phase 3: L-BFGS-B refinement from best point ---
        if time_left() and call_count[0] < budget - 20:
            try:
                remaining = budget - call_count[0]
                res_lbfgs = minimize(
                    safe_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100, 'maxfun': min(remaining, 80), 'gtol': 1e-10, 'ftol': 1e-12}
                )
                if res_lbfgs.fun < best_f:
                    best_x = np.clip(res_lbfgs.x, -5.0, 5.0)
                    best_f = float(res_lbfgs.fun)
            except Exception:
                pass

        # --- Phase 4: Nelder-Mead polishing ---
        if time_left() and call_count[0] < budget - 10:
            try:
                remaining = budget - call_count[0]
                res_nm = minimize(
                    safe_func,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxfev': min(remaining, 60),
                        'xatol': 1e-12,
                        'fatol': 1e-14,
                        'adaptive': True,
                    }
                )
                if res_nm.fun < best_f:
                    best_x = np.clip(res_nm.x, -5.0, 5.0)
                    best_f = float(res_nm.fun)
            except Exception:
                pass

        # --- Phase 5: Final tight L-BFGS-B + NM if budget remains ---
        if time_left() and call_count[0] < budget - 5:
            try:
                remaining = budget - call_count[0]
                if remaining > 20:
                    res_final = minimize(
                        safe_func,
                        best_x,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 50, 'maxfun': min(remaining // 2, 30), 'gtol': 1e-12, 'ftol': 1e-14}
                    )
                    if res_final.fun < best_f:
                        best_x = np.clip(res_final.x, -5.0, 5.0)
                        best_f = float(res_final.fun)
            except Exception:
                pass

        if time_left() and call_count[0] < budget - 5:
            try:
                remaining = budget - call_count[0]
                if remaining > 10:
                    res_nm2 = minimize(
                        safe_func,
                        best_x,
                        method='Nelder-Mead',
                        options={
                            'maxfev': min(remaining, 20),
                            'xatol': 1e-14,
                            'fatol': 1e-16,
                            'adaptive': True,
                        }
                    )
                    if res_nm2.fun < best_f:
                        best_x = np.clip(res_nm2.x, -5.0, 5.0)
                        best_f = float(res_nm2.fun)
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
