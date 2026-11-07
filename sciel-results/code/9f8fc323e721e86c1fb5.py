"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead with time guard for BBOB f11 (Discus)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s kill
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_made = 0

        def safe_call(x):
            nonlocal calls_made
            if calls_made >= self.budget:
                return None
            x_clipped = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            fx = func(x_clipped)
            calls_made += 1
            return fx

        def nelder_mead(x0, fx0, max_iter=200):
            nonlocal best_x, best_f, calls_made
            dim = self.dim
            # Initialize simplex
            simplex = [x0.copy()]
            fvals = [fx0]
            for i in range(dim):
                xi = x0.copy()
                xi[i] += 0.1 if x0[i] < 4.9 else -0.1
                xi = np.clip(xi, -5.0, 5.0)
                fi = safe_call(xi)
                if fi is None:
                    break
                simplex.append(xi)
                fvals.append(fi)
                if fi < best_f:
                    best_f = fi
                    best_x = xi.copy()

            alpha, gamma, rho = 1.0, 2.0, 0.5
            for _ in range(max_iter):
                if time.time() > deadline - 5.0 or calls_made >= self.budget:
                    break
                # Sort by function value
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

                x_centroid = np.mean(simplex[:-1], axis=0)
                x_worst = simplex[-1]
                f_worst = fvals[-1]

                # Reflection
                x_ref = x_centroid + alpha * (x_centroid - x_worst)
                x_ref = np.clip(x_ref, -5.0, 5.0)
                f_ref = safe_call(x_ref)
                if f_ref is None:
                    break
                if f_ref < best_f:
                    best_f = f_ref
                    best_x = x_ref.copy()

                if f_ref < fvals[0]:
                    # Expansion
                    x_exp = x_centroid + gamma * (x_ref - x_centroid)
                    x_exp = np.clip(x_exp, -5.0, 5.0)
                    f_exp = safe_call(x_exp)
                    if f_exp is None:
                        break
                    if f_exp < best_f:
                        best_f = f_exp
                        best_x = x_exp.copy()
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        fvals[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        fvals[-1] = f_ref
                elif f_ref < fvals[-2]:
                    simplex[-1] = x_ref
                    fvals[-1] = f_ref
                else:
                    # Contraction
                    x_con = x_centroid + rho * (x_worst - x_centroid)
                    x_con = np.clip(x_con, -5.0, 5.0)
                    f_con = safe_call(x_con)
                    if f_con is None:
                        break
                    if f_con < best_f:
                        best_f = f_con
                        best_x = x_con.copy()
                    if f_con < f_worst:
                        simplex[-1] = x_con
                        fvals[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                            fvals[i] = safe_call(simplex[i])
                            if fvals[i] is None:
                                return
                            if fvals[i] < best_f:
                                best_f = fvals[i]
                                best_x = simplex[i].copy()

            return best_f

        # Multi-start strategy: start from several diverse points
        # Use a deterministic set of starting points based on the problem structure
        n_starts = 8
        starts = []
        # Center
        starts.append(np.zeros(self.dim))
        # Axis-aligned points
        for i in range(self.dim):
            pt = np.zeros(self.dim)
            pt[i] = 3.0
            starts.append(pt)
        # Random points
        while len(starts) < n_starts:
            starts.append(rng.uniform(-4, 4, self.dim))

        for i, x_start in enumerate(starts):
            if time.time() > deadline - 30.0 or calls_made >= self.budget:
                break
            fx_start = safe_call(x_start)
            if fx_start is None:
                break
            if fx_start < best_f:
                best_f = fx_start
                best_x = x_start.copy()
            nelder_mead(x_start, fx_start, max_iter=150)

        # If we have remaining budget, do a final (1+1)-ES polish
        if best_x is not None and calls_made < self.budget and time.time() < deadline - 10.0:
            x = best_x.copy()
            fx = best_f
            sigma = 0.5
            while calls_made < self.budget and time.time() < deadline - 5.0:
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5.0, 5.0)
                fy = safe_call(y)
                if fy is None:
                    break
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                else:
                    sigma *= 0.85

        return best_x, best_f
# EVOLVE-BLOCK-END
