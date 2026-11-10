"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """
    Optimizer for Bent Cigar function f12.
    Strategy:
    1. Brief global exploration using a small set of Sobol points to find a good starting basin.
    2. Local refinement using Powell's method (derivative-free, adaptive step sizes per dimension)
       which is effective for anisotropic functions like Bent Cigar.
    3. Strict budget and deadline management.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.perf_counter()
        deadline = t0 + 280.0  # 20s safety margin before 300s limit (killed at 330s)
        d = self.dim
        budget = self.budget

        state = {"count": 0, "ok": True, "best_x": None, "best_f": np.inf}

        def wf(x):
            if not state["ok"]:
                return 1e300
            if time.perf_counter() > deadline:
                state["ok"] = False
                return 1e300
            
            x = np.asarray(x, dtype=float)
            if not np.all(np.isfinite(x)):
                return 1e300
                
            try:
                v = float(func(x))
            except Exception:
                return 1e300
                
            state["count"] += 1
            if not np.isfinite(v):
                return 1e300
                
            if v < state["best_f"]:
                state["best_f"] = v
                state["best_x"] = x.copy()
                
            return v

        # --- Phase 1: Global Exploration (Sobol) ---
        # Use a larger number of deterministic Sobol points to cover the space better.
        # 50 points provides a denser grid, increasing the chance of finding a point
        # closer to the optimum in the anisotropic valley.
        n_sobol = min(50, budget // 10)
        
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=d, scramble=True, seed=42)
            sobol_points = sampler.random(n_sobol)
            sobol_points = -5.0 + 10.0 * sobol_points
            
            best_idx = 0
            best_val = np.inf
            
            for i in range(n_sobol):
                if not state["ok"]:
                    break
                x = sobol_points[i]
                v = wf(x)
                if v < best_val:
                    best_val = v
                    best_idx = i
            
            x0 = sobol_points[best_idx].copy()
        except Exception:
            # Fallback to random if Sobol fails
            rng = np.random.default_rng(0)
            x0 = rng.uniform(-5, 5, d)
            v = wf(x0)
            if v < state["best_f"]:
                state["best_f"] = v
                state["best_x"] = x0.copy()

        # --- Phase 2: Local Refinement (Powell) ---
        # Powell is robust for anisotropic functions because it adapts step sizes 
        # along the principal axes of the function's contours.
        remaining_budget = budget - state["count"]
        
        if remaining_budget > 0 and state["ok"]:
            try:
                res = minimize(
                    wf,
                    x0,
                    method="Powell",
                    options={
                        "maxiter": 1000,
                        "maxfev": remaining_budget,
                        "xrtol": 1e-12,
                        "ftol": 1e-12,
                    },
                )
                # Update best if Powell found something better
                xf = np.asarray(res.x, dtype=float)
                ff = res.fun
                if np.all(np.isfinite(xf)) and np.isfinite(ff):
                    if ff < state["best_f"]:
                        state["best_f"] = ff
                        state["best_x"] = np.clip(xf, -5.0, 5.0)
            except Exception:
                pass
        
        # --- Phase 3: Final Polish (Nelder-Mead) ---
        # If budget remains, use Nelder-Mead with a very small initial step size
        # to polish the result. NM is good at final convergence if the scale is small.
        remaining_budget = budget - state["count"]
        if remaining_budget > 5 and state["ok"] and state["best_x"] is not None:
            try:
                # Initialize simplex around the best point with a small step size
                # The step size should be small enough to stay in the valley
                # but large enough to move.
                init_step = 0.01 
                # Create a simplex
                simplex = []
                simplex.append(state["best_x"].copy())
                for i in range(d):
                    x_i = state["best_x"].copy()
                    x_i[i] += init_step
                    simplex.append(x_i)
                
                res_nm = minimize(
                    wf,
                    state["best_x"],
                    method="Nelder-Mead",
                    options={
                        "maxiter": 100,
                        "maxfev": remaining_budget,
                        "xatol": 1e-12,
                        "fatol": 1e-12,
                        "initial_simplex": simplex,
                    },
                )
                xf = np.asarray(res_nm.x, dtype=float)
                ff = res_nm.fun
                if np.all(np.isfinite(xf)) and np.isfinite(ff):
                    if ff < state["best_f"]:
                        state["best_f"] = ff
                        state["best_x"] = np.clip(xf, -5.0, 5.0)
            except Exception:
                pass

        return state["best_x"], state["best_f"]
# EVOLVE-BLOCK-END
