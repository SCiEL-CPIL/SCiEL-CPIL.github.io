import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Evaluator kills after 330s; keep a conservative deadline.
        deadline = start_time + 285.0

        budget = self.budget
        dim = self.dim

        calls = 0
        best_f = float('inf')
        best_x = np.zeros(dim)

        def evaluate(x):
            nonlocal calls, best_f, best_x
            if calls >= budget:
                return None
            if time.time() > deadline:
                return None
            try:
                val = float(func(x))
            except Exception:
                return None
            calls += 1
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        # ---------- Pure Nelder-Mead simplex method ----------
        # A single simplex that maintains a diverse set of points.
        # This tracks the sharp ridge geometry of f13 without the scalar
        # step-size adaptation lag of a (1+1)-ES, which helps early AOCC.

        # Initialize the simplex at the box center (0, ..., 0) with
        # axis-aligned vertices spread by the initial delta.
        center = np.zeros(dim)
        delta = 2.0  # initial spread; large enough to explore, small enough to converge
        f0 = evaluate(center)
        if f0 is None:
            return best_x, best_f

        simplex = np.zeros((dim + 1, dim))
        fvals = np.zeros(dim + 1)
        simplex[0] = center
        fvals[0] = f0

        for i in range(1, dim + 1):
            simplex[i] = center.copy()
            simplex[i][i - 1] += delta
            simplex[i] = np.clip(simplex[i], -5.0, 5.0)
            f_i = evaluate(simplex[i])
            if f_i is None:
                return best_x, best_f
            fvals[i] = f_i

        # Nelder-Mead parameters
        alpha = 1.0   # reflection
        gamma = 2.0   # expansion
        rho = 0.5     # contraction
        sigma_nm = 0.5  # shrink

        while calls < budget and time.time() < deadline:
            # Sort by function value (ascending)
            order = np.argsort(fvals)
            simplex = simplex[order]
            fvals = fvals[order]

            # Convergence check: if the spread is tiny, stop.
            if fvals[-1] - fvals[0] < 1e-10:
                break

            # Centroid of all vertices except the worst.
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            x_reflect = centroid + alpha * (centroid - simplex[-1])
            x_reflect = np.clip(x_reflect, -5.0, 5.0)
            f_reflect = evaluate(x_reflect)
            if f_reflect is None:
                break

            if f_reflect < fvals[0]:
                # Try expansion
                x_expand = centroid + gamma * (x_reflect - centroid)
                x_expand = np.clip(x_expand, -5.0, 5.0)
                f_expand = evaluate(x_expand)
                if f_expand is None:
                    break

                if f_expand < f_reflect:
                    simplex[-1] = x_expand
                    fvals[-1] = f_expand
                else:
                    simplex[-1] = x_reflect
                    fvals[-1] = f_reflect
            elif f_reflect < fvals[-2]:
                # Accept reflection as is
                simplex[-1] = x_reflect
                fvals[-1] = f_reflect
            else:
                # Contraction
                if f_reflect < fvals[-1]:
                    # Outside contraction
                    x_contract = centroid + rho * (x_reflect - centroid)
                else:
                    # Inside contraction
                    x_contract = centroid - rho * (x_reflect - centroid)

                x_contract = np.clip(x_contract, -5.0, 5.0)
                f_contract = evaluate(x_contract)
                if f_contract is None:
                    break

                if f_contract < min(f_reflect, fvals[-1]):
                    simplex[-1] = x_contract
                    fvals[-1] = f_contract
                else:
                    # Shrink: move every non-best vertex toward the best.
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                        f_new = evaluate(simplex[i])
                        if f_new is None:
                            return best_x, best_f
                        fvals[i] = f_new

            # Periodic deadline guard inside the loop
            if time.time() > deadline:
                break

        return best_x, best_f
