"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Stochastic multi-start Nelder-Mead on Schaffers F7."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # 10 independent starts, each gets 50 calls (including initial simplex evals)
        # Total: 10 * (dim + 1 + iterations) <= 500
        # We'll do 10 starts. For each start, we evaluate the simplex (dim+1 points)
        # then do up to (50 - (dim+1)) NM iterations.
        # dim=5, so simplex = 6 points. 50 - 6 = 44 iterations per start.
        # Total calls: 10 * (6 + 44) = 500. Perfect.

        # Deterministic distinct starting points within [-4, 4]^5
        # Use a grid-like or LHS-inspired deterministic set
        rng = np.random.default_rng(42)

        # Generate 10 deterministic distinct points
        starts = []
        # Use stratified points: divide [-4,4] into intervals and pick centers
        # For 10 points in 5D, we can use a simple deterministic pattern
        # Let's use a combination: some near origin, some spread out
        base_starts = np.array([
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [2.0, -2.0, 2.0, -2.0, 2.0],
            [-2.0, 2.0, -2.0, 2.0, -2.0],
            [3.0, 3.0, -3.0, -3.0, 3.0],
            [-3.0, -3.0, 3.0, 3.0, -3.0],
            [1.5, -1.5, 0.0, 1.5, -1.5],
            [-1.5, 1.5, 0.0, -1.5, 1.5],
            [4.0, -4.0, 4.0, -4.0, 4.0],
            [-4.0, 4.0, -4.0, 4.0, -4.0],
            [0.5, 0.5, 0.5, 0.5, 0.5],
        ])

        # Shuffle deterministically to avoid bias in ordering
        idx = rng.permutation(len(base_starts))
        starts = base_starts[idx]

        best_x = None
        best_f = np.inf
        calls_made = 0

        for start in starts:
            if calls_made >= budget:
                break

            # Nelder-Mead setup
            # Initial simplex: start point + dim perturbations
            alpha = 1.0  # expansion
            gamma = 2.0  # reflection
            rho = 0.5    # contraction
            sigma = 0.5  # shrink
            step = 0.5   # initial simplex step size

            # Build simplex
            simplex = np.zeros((dim + 1, dim))
            simplex[0] = np.clip(start, -5, 5)
            for i in range(dim):
                simplex[i + 1] = simplex[0].copy()
                simplex[i + 1][i] += step

            # Evaluate simplex
            fvals = np.zeros(dim + 1)
            for i in range(dim + 1):
                if calls_made >= budget:
                    break
                x_i = np.clip(simplex[i], -5, 5)
                fvals[i] = func(x_i)
                calls_made += 1
                if fvals[i] < best_f:
                    best_f = fvals[i]
                    best_x = x_i.copy()

            if calls_made >= budget:
                break

            # Nelder-Mead iterations
            max_iters = 50 - (dim + 1)  # 44 iterations
            for _ in range(max_iters):
                if calls_made >= budget:
                    break

                # Sort simplex by function value
                order = np.argsort(fvals)
                simplex = simplex[order]
                fvals = fvals[order]

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                worst = simplex[-1]
                x_ref = centroid + gamma * (centroid - worst)
                x_ref = np.clip(x_ref, -5, 5)
                f_ref = func(x_ref)
                calls_made += 1
                if f_ref < best_f:
                    best_f = f_ref
                    best_x = x_ref.copy()

                if f_ref < fvals[0]:
                    # Expansion
                    x_exp = centroid + alpha * (x_ref - centroid)
                    x_exp = np.clip(x_exp, -5, 5)
                    f_exp = func(x_exp)
                    calls_made += 1
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
                    if f_ref < fvals[-1]:
                        x_con = centroid + rho * (x_ref - centroid)
                    else:
                        x_con = centroid - rho * (worst - centroid)
                    x_con = np.clip(x_con, -5, 5)
                    f_con = func(x_con)
                    calls_made += 1
                    if f_con < best_f:
                        best_f = f_con
                        best_x = x_con.copy()
                    if f_con < min(f_ref, fvals[-1]):
                        simplex[-1] = x_con
                        fvals[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fvals[i] = func(simplex[i])
                            calls_made += 1
                            if fvals[i] < best_f:
                                best_f = fvals[i]
                                best_x = simplex[i].copy()
                        if calls_made >= budget:
                            break

                if calls_made >= budget:
                    break

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls_made += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
