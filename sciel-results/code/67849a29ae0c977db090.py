"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Sobol global exploration followed by Nelder-Mead local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 270  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        # Sobol sequence generator (use scipy if available, else fallback)
        try:
            from scipy.stats import qmc
            sobol = qmc.Sobol(d=dim, scramble=True, seed=42)
            has_sobol = True
        except ImportError:
            has_sobol = False

        # Global exploration phase: sample diverse points
        # Use ~40% of budget for global exploration
        n_global = max(50, budget // 3)
        n_local = budget - n_global

        best_x = None
        best_f = np.inf

        # Generate global samples
        if has_sobol:
            # Sobol points in [0,1]^d, map to [-5,5]^d
            # Use more points than needed to pick top candidates
            n_sobol = min(200, n_global)
            sobol_points = sobol.random(n_sobol)
            # Map [0,1] -> [-5,5]
            global_candidates = sobol_points * 10.0 - 5.0
        else:
            rng = np.random.default_rng(42)
            global_candidates = rng.uniform(-5, 5, (n_global, dim))

        # Evaluate global candidates
        call_count = 0
        for i in range(len(global_candidates)):
            if time.time() > deadline:
                break
            x = global_candidates[i]
            fx = func(x)
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Collect top candidates for local refinement
        # Re-evaluate top few with local search
        # Keep track of evaluated points to avoid re-evaluation
        evaluated = []
        evaluated.append((best_x, best_f))

        # Select top candidates from global phase for local refinement
        # We'll pick the best few and do Nelder-Mead on them
        n_local_starts = min(3, call_count)

        # Sort global candidates by function value (we only tracked best, so re-evaluate a few)
        # Actually, let's just do local refinement on the best point found
        # and also try a few other promising points

        # Nelder-Mead implementation
        def nelder_mead(func, x0, f0, max_iter, dim, deadline, call_counter):
            """Simplex-based local optimization."""
            # Initial simplex
            simplex = [x0.copy()]
            fvals = [f0]

            # Add perturbed points
            for j in range(dim):
                x_new = x0.copy()
                x_new[j] += 0.1 if x0[j] < 4.9 else -0.1
                x_new = np.clip(x_new, -5, 5)
                if call_counter[0] >= budget or time.time() > deadline:
                    break
                f_new = func(x_new)
                call_counter[0] += 1
                simplex.append(x_new)
                fvals.append(f_new)

            # Sort simplex
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]

            alpha, gamma, rho = 1.0, 2.0, 0.5

            for iteration in range(max_iter):
                if time.time() > deadline or call_counter[0] >= budget:
                    break

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5, 5)
                if call_counter[0] >= budget or time.time() > deadline:
                    break
                f_r = func(x_r)
                call_counter[0] += 1

                if f_r < fvals[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5, 5)
                    if call_counter[0] >= budget or time.time() > deadline:
                        simplex[-1] = x_r
                        fvals[-1] = f_r
                        break
                    f_e = func(x_e)
                    call_counter[0] += 1
                    if f_e < f_r:
                        simplex[-1] = x_e
                        fvals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        fvals[-1] = f_r
                elif f_r < fvals[-2]:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
                else:
                    # Contraction
                    if f_r < fvals[-1]:
                        # Outside contraction
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        # Inside contraction
                        x_c = centroid - rho * (x_r - centroid)
                    x_c = np.clip(x_c, -5, 5)
                    if call_counter[0] >= budget or time.time() > deadline:
                        break
                    f_c = func(x_c)
                    call_counter[0] += 1
                    if f_c < min(f_r, fvals[-1]):
                        simplex[-1] = x_c
                        fvals[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            if call_counter[0] >= budget or time.time() > deadline:
                                break
                            fvals[i] = func(simplex[i])
                            call_counter[0] += 1
                        if time.time() > deadline or call_counter[0] >= budget:
                            break

                # Sort
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

            return simplex[0], fvals[0]

        # Do local refinement on the best global point
        call_counter = [call_count]
        max_nm_iter = max(20, n_local // (dim + 1))

        if best_x is not None:
            x_opt, f_opt = nelder_mead(func, best_x, best_f, max_nm_iter, dim, deadline, call_counter)
            if f_opt < best_f:
                best_f = f_opt
                best_x = x_opt

        # Try local refinement on a few other points if budget remains
        if call_counter[0] < budget and time.time() < deadline:
            # Pick a few other points from the Sobol set
            for i in range(1, min(3, len(global_candidates))):
                if time.time() > deadline or call_counter[0] >= budget:
                    break
                x_start = global_candidates[i]
                # Quick check: evaluate and see if promising
                f_start = func(x_start)
                call_counter[0] += 1
                if f_start < best_f * 1.5:  # reasonably close
                    x_opt, f_opt = nelder_mead(func, x_start, f_start, max_nm_iter // 2, dim, deadline, call_counter)
                    if f_opt < best_f:
                        best_f = f_opt
                        best_x = x_opt

        return best_x, best_f
# EVOLVE-BLOCK-END
