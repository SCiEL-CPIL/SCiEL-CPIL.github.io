"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


class Optimizer:
    """Multi-start L-BFGS-B for Discus (f11) with budget and time guards."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 270.0  # safety margin before 330s kill

        state = {"count": 0, "best_x": None, "best_f": np.inf}

        def wrapped(x):
            if state["count"] >= budget:
                return state["best_f"]
            state["count"] += 1
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            fx = float(func(x))
            if fx < state["best_f"]:
                state["best_f"] = fx
                state["best_x"] = x.copy()
            return fx

        # Deterministic grid of 10 starting points spread over the box
        starts = []
        # Center
        starts.append(np.full(dim, 0.0))
        # 8 corners of [-4,4]^dim
        for sign in np.array(np.meshgrid([-4.0, 4.0] * dim)).reshape(-1, dim):
            starts.append(sign)
        # One more: random-ish but deterministic offset
        starts.append(np.array([3.0, -2.0, 1.0, -3.0, 2.0])[:dim])

        # Shuffle deterministically to avoid bias toward center
        rng = np.random.default_rng(42)
        perm = rng.permutation(len(starts))
        starts = [starts[i] for i in perm]

        for s in starts:
            if state["count"] >= budget:
                break
            if time.time() > deadline:
                break
            try:
                res = minimize(
                    wrapped,
                    s,
                    method="L-BFGS-B",
                    bounds=[(-5.0, 5.0)] * dim,
                    options={"maxiter": 200, "maxls": 20, "gtol": 1e-6},
                )
            except Exception:
                continue

        # Use remaining budget for a local refinement around best point
        if state["best_x"] is not None and state["count"] < budget:
            try:
                res2 = minimize(
                    wrapped,
                    state["best_x"],
                    method="L-BFGS-B",
                    bounds=[(-5.0, 5.0)] * dim,
                    options={"maxiter": 100, "maxls": 20, "gtol": 1e-8},
                )
            except Exception:
                pass

        if state["best_x"] is None:
            # Fallback: must have made at least one call
            x0 = np.zeros(dim)
            state["count"] = 1
            state["best_x"] = x0.copy()
            state["best_f"] = float(func(x0))

        return state["best_x"], state["best_f"]
# EVOLVE-BLOCK-END
