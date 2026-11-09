"""Rewritten optimizer: multi-start Nelder-Mead simplex ensemble for Rastrigin f3."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead simplex ensemble."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin

        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        call_count = 0

        def evaluate(x):
            nonlocal best_x, best_f, call_count
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            f = float(func(x))
            call_count += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Number of independent Nelder-Mead runs
        num_runs = 5

        # Generate diverse starting points
        starts = []
        # Center
        starts.append(np.zeros(dim))
        # Some corner-like points
        for i in range(1, num_runs):
            p = rng.uniform(-4.0, 4.0, dim)
            starts.append(p)

        # Initialize simplexes: each has dim+1 points
        simplex_size = dim + 1
        simplex_step = 0.5

        # Store each simplex as list of (point, f)
        simplexes = []
        for s in starts:
            simplex = [(s.copy(), np.inf)]
            for j in range(dim):
                point = s.copy()
                point[j] += simplex_step
                simplex.append((point, np.inf))
            simplexes.append(simplex)

        # Evaluate all initial points
        for simplex in simplexes:
            for idx, (pt, _) in enumerate(simplex):
                if call_count >= budget:
                    break
                f = evaluate(pt)
                simplex[idx] = (pt, f)
            if call_count >= budget:
                break

        # Nelder-Mead parameters
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma_shrink = 0.5

        # Interleave iterations across simplexes
        iteration = 0
        active_simplexes = list(range(num_runs))

        while call_count < budget and active_simplexes:
            if time.time() > deadline:
                break

            # Round-robin select a simplex
            si = active_simplexes[iteration % len(active_simplexes)]
            simplex = simplexes[si]

            if len(simplex) < 2:
                continue

            # Sort by function value
            simplex.sort(key=lambda pf: pf[1])

            # Check if simplex has converged (all points nearly equal)
            x_best = simplex[0][0]
            x_worst = simplex[-1][0]
            if np.max(np.abs(x_best - x_worst)) < 1e-6:
                # Converged, remove from active
                active_simplexes.remove(si)
                continue

            # Get worst, second worst, and centroid of best points
            x_worst = simplex[-1][0]
            f_worst = simplex[-1][1]
            x_second = simplex[-2][0]

            # Centroid of all points except worst
            centroid = np.zeros(dim)
            for i in range(len(simplex) - 1):
                centroid += simplex[i][0]
            centroid /= (len(simplex) - 1)

            # Reflection
            x_ref = centroid + alpha * (centroid - x_worst)
            x_ref = np.clip(x_ref, -5.0, 5.0)
            f_ref = evaluate(x_ref)

            if call_count >= budget:
                break

            if f_ref < simplex[0][1]:
                # Try expansion
                x_exp = centroid + gamma * (x_ref - centroid)
                x_exp = np.clip(x_exp, -5.0, 5.0)
                f_exp = evaluate(x_exp)
                if call_count >= budget:
                    break
                if f_exp < f_ref:
                    simplex[-1] = (x_exp, f_exp)
                else:
                    simplex[-1] = (x_ref, f_ref)
            elif f_ref < simplex[-2][1]:
                # Accept reflection
                simplex[-1] = (x_ref, f_ref)
            else:
                # Contraction
                if f_ref < f_worst:
                    # Outside contraction
                    x_con = centroid + rho * (x_ref - centroid)
                else:
                    # Inside contraction
                    x_con = centroid - rho * (x_worst - centroid)
                x_con = np.clip(x_con, -5.0, 5.0)
                f_con = evaluate(x_con)
                if call_count >= budget:
                    break
                if f_con <= min(f_ref, f_worst):
                    simplex[-1] = (x_con, f_con)
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        pt, fv = simplex[i]
                        new_pt = x_best + sigma_shrink * (pt - x_best)
                        new_pt = np.clip(new_pt, -5.0, 5.0)
                        new_f = evaluate(new_pt)
                        if call_count >= budget:
                            break
                        simplex[i] = (new_pt, new_f)
                    if call_count >= budget:
                        break

            iteration += 1

        # Final high-precision polish pass using coordinate descent / local search
        if best_x is not None and call_count < budget:
            # Use a finer local search around the best point found
            polish_x = best_x.copy()
            polish_f = best_f
            step_sizes = [0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]
            
            for step in step_sizes:
                if call_count >= budget:
                    break
                if time.time() > deadline:
                    break
                improved = True
                while improved and call_count < budget:
                    improved = False
                    for d in range(dim):
                        if call_count >= budget:
                            break
                        for direction in [-1, 1]:
                            if call_count >= budget:
                                break
                            trial = polish_x.copy()
                            trial[d] += direction * step
                            trial = np.clip(trial, -5.0, 5.0)
                            f_trial = evaluate(trial)
                            if f_trial < polish_f:
                                polish_x = trial
                                polish_f = f_trial
                                improved = True
                                best_x = polish_x.copy()
                                best_f = polish_f
                            elif time.time() > deadline:
                                break

        return best_x, best_f
# EVOLVE-BLOCK-END
