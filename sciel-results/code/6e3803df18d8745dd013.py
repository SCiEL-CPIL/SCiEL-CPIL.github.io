"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """
    Optimizer for Bent Cigar function f12.
    Strategy:
    1. Brief global exploration using Sobol points to find a good starting basin.
    2. Local refinement using Powell's method (derivative-free, adaptive step sizes per dimension)
       which is effective for anisotropic functions like Bent Cigar.
    3. Final high-precision polish pass with tighter tolerances.
    4. Strict budget and deadline management.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.perf_counter()
        deadline = t0 + 280.0  # 20s safety margin before 300s limit
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
        n_sobol = min(20, budget // 10)

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
            rng = np.random.default_rng(0)
            x0 = rng.uniform(-5, 5, d)
            v = wf(x0)
            if v < state["best_f"]:
                state["best_f"] = v
                state["best_x"] = x0.copy()

        # --- Phase 2: Local Refinement (Powell) ---
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
                        "xrtol": 1e-10,
                        "ftol": 1e-10,
                    },
                )
                xf = np.asarray(res.x, dtype=float)
                ff = res.fun
                if np.all(np.isfinite(xf)) and np.isfinite(ff):
                    if ff < state["best_f"]:
                        state["best_f"] = ff
                        state["best_x"] = np.clip(xf, -5.0, 5.0)
            except Exception:
                pass

        # --- Phase 3: High-precision polish pass ---
        # Use a very tight tolerance and a small neighborhood around the best point
        # to squeeze out the last bits of precision.
        remaining_budget = budget - state["count"]
        if remaining_budget > 5 and state["ok"] and state["best_x"] is not None:
            try:
                # Start from the current best point with a very small step
                x_start = state["best_x"].copy()

                # Use Nelder-Mead with very tight tolerances for final polish
                res2 = minimize(
                    wf,
                    x_start,
                    method="Nelder-Mead",
                    options={
                        "maxiter": 200,
                        "maxfev": remaining_budget,
                        "xatol": 1e-12,
                        "fatol": 1e-12,
                        "initial_simplex": x_start + 1e-4 * np.eye(d),
                    },
                )
                xf2 = np.asarray(res2.x, dtype=float)
                ff2 = res2.fun
                if np.all(np.isfinite(xf2)) and np.isfinite(ff2):
                    if ff2 < state["best_f"]:
                        state["best_f"] = ff2
                        state["best_x"] = np.clip(xf2, -5.0, 5.0)
            except Exception:
                pass

        # --- Phase 4: Coordinate-wise line search polish ---
        # A final manual coordinate-wise refinement to handle any remaining
        # anisotropy that Nelder-Mead might have missed.
        remaining_budget = budget - state["count"]
        if remaining_budget > 2 and state["ok"] and state["best_x"] is not None:
            try:
                x_best = state["best_x"].copy()
                f_best = state["best_f"]

                # Try small perturbations along each coordinate
                for _ in range(min(remaining_budget // d, 10)):
                    if not state["ok"] or time.perf_counter() > deadline:
                        break
                    improved = False
                    for j in range(d):
                        if not state["ok"] or time.perf_counter() > deadline:
                            break
                        for delta in [1e-3, 1e-4, 1e-5, -1e-3, -1e-4, -1e-5]:
                            if not state["ok"] or time.perf_counter() > deadline:
                                break
                            x_try = x_best.copy()
                            x_try[j] += delta
                            x_try = np.clip(x_try, -5.0, 5.0)
                            v_try = wf(x_try)
                            if v_try < f_best - 1e-15:
                                x_best = x_try
                                f_best = v_try
                                improved = True
                                break
                    if not improved:
                        break

                if f_best < state["best_f"]:
                    state["best_f"] = f_best
                    state["best_x"] = x_best
            except Exception:
                pass

        return state["best_x"], state["best_f"]
# EVOLVE-BLOCK-END
