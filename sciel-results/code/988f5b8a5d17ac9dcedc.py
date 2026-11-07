"""Rewritten optimizer for BBOB-style f12 (Bent cigar) using Nelder-Mead simplex."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Nelder-Mead simplex optimizer for anisotropic landscapes (Bent Cigar).
    - Nelder-Mead adapts to local geometry, handling narrow valleys well.
    - Multi-start with several initial points to avoid bad local minima.
    - Early progress from origin for stable start.
    - Time-guarded to stay within 300s limit.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = np.zeros(d)
        best_f = func(best_x)
        calls = 1

        # Nelder-Mead with proper budget tracking
        def nelder_mead(func, x0, f0, max_calls, deadline):
            """
            Run Nelder-Mead simplex optimization.
            Returns (best_x, best_f, calls_used).
            """
            calls_used = 0
            start_calls = 0  # We'll track internally

            # Initialize simplex
            simplex = []
            simplex_f = []

            # First point
            simplex.append(x0.copy())
            simplex_f.append(f0)

            # Add perturbed points along each axis
            base_step = 0.25
            for i in range(d):
                xi = x0.copy()
                xi[i] += base_step
                simplex.append(xi)
                simplex_f.append(None)

            # Evaluate remaining simplex points
            for i in range(1, d + 1):
                if calls_used >= max_calls or time.time() > deadline:
                    break
                f = func(simplex[i])
                calls_used += 1
                simplex_f[i] = f

            # If we couldn't evaluate all, pad with the best known
            for i in range(1, d + 1):
                if simplex_f[i] is None:
                    simplex_f[i] = simplex_f[0]

            best_x = x0.copy()
            best_f = f0
            for i in range(len(simplex_f)):
                if simplex_f[i] < best_f:
                    best_f = simplex_f[i]
                    best_x = simplex[i].copy()

            no_improve = 0
            max_no_improve = 30

            while calls_used < max_calls and time.time() < deadline:
                # Sort simplex by function value
                idx = np.argsort(simplex_f)
                simplex = [simplex[i] for i in idx]
                simplex_f = [simplex_f[i] for i in idx]

                # Check convergence
                f_range = simplex_f[-1] - simplex_f[0]
                if f_range < 1e-14:
                    break

                # Check improvement
                if simplex_f[0] < best_f - 1e-15:
                    best_f = simplex_f[0]
                    best_x = simplex[0].copy()
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve > max_no_improve:
                        break

                # Centroid of all but worst
                x_centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_worst = simplex[-1]
                f_worst = simplex_f[-1]
                x_ref = x_centroid + 1.0 * (x_centroid - x_worst)
                x_ref = np.clip(x_ref, -5, 5)

                if calls_used >= max_calls:
                    break
                f_ref = func(x_ref)
                calls_used += 1

                # Expansion
                if f_ref < simplex_f[0]:
                    x_exp = x_centroid + 2.0 * (x_ref - x_centroid)
                    x_exp = np.clip(x_exp, -5, 5)
                    if calls_used >= max_calls:
                        break
                    f_exp = func(x_exp)
                    calls_used += 1
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        simplex_f[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        simplex_f[-1] = f_ref
                # Outside contraction
                elif f_ref < f_worst:
                    x_con = x_centroid + 0.5 * (x_ref - x_centroid)
                    x_con = np.clip(x_con, -5, 5)
                    if calls_used >= max_calls:
                        break
                    f_con = func(x_con)
                    calls_used += 1
                    if f_con <= f_ref:
                        simplex[-1] = x_con
                        simplex_f[-1] = f_con
                    else:
                        simplex[-1] = x_ref
                        simplex_f[-1] = f_ref
                # Inside contraction
                else:
                    x_con = x_centroid - 0.5 * (x_worst - x_centroid)
                    x_con = np.clip(x_con, -5, 5)
                    if calls_used >= max_calls:
                        break
                    f_con = func(x_con)
                    calls_used += 1
                    if f_con < f_worst:
                        simplex[-1] = x_con
                        simplex_f[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, d + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                        for i in range(1, d + 1):
                            if calls_used >= max_calls or time.time() > deadline:
                                break
                            f = func(simplex[i])
                            calls_used += 1
                            simplex_f[i] = f

            return best_x, best_f, calls_used

        # Phase 1: Evaluate a few candidate starting points
        candidates = [np.zeros(d)]
        # Add a few random points
        for _ in range(3):
            if calls >= budget or time.time() > deadline:
                break
            x_cand = rng.uniform(-3, 3, size=d)
            f_cand = func(x_cand)
            calls += 1
            candidates.append(x_cand)
            if f_cand < best_f:
                best_f = f_cand
                best_x = x_cand.copy()

        # Phase 2: Run Nelder-Mead from the best candidate
        remaining = budget - calls
        if remaining > 5 and time.time() < deadline:
            nm_x, nm_f, nm_calls = nelder_mead(func, best_x.copy(), best_f, remaining, deadline)
            calls += nm_calls
            if nm_f < best_f:
                best_f = nm_f
                best_x = nm_x.copy()

        # Phase 3: Polish with another Nelder-Mead run if budget remains
        remaining = budget - calls
        if remaining > 5 and time.time() < deadline:
            nm_x, nm_f, nm_calls = nelder_mead(func, best_x.copy(), best_f, remaining, deadline)
            calls += nm_calls
            if nm_f < best_f:
                best_f = nm_f
                best_x = nm_x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
