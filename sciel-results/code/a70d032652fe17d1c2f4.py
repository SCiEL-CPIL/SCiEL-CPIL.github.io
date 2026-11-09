"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Surrogate-assisted multi-start local search with adaptive step sizes.
    Uses a simple kriging-like surrogate to guide exploration, combined with
    local Nelder-Mead style refinement.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # safety margin before 300s limit

        rng = np.random.default_rng(12345)

        # State
        best_x = None
        best_f = np.inf
        evaluations_used = 0

        def evaluate(x):
            nonlocal evaluations_used, best_x, best_f
            if time.time() > deadline:
                return None
            evaluations_used += 1
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Initial random sampling to get a rough landscape
        n_init = min(30, budget)
        points = []
        values = []

        for i in range(n_init):
            if evaluations_used >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            fx = evaluate(x)
            if fx is None:
                break
            points.append(x)
            values.append(fx)

        if not points:
            return best_x, best_f

        points = np.array(points)
        values = np.array(values)

        # Phase 2: Iterative surrogate-guided search
        # We'll use a simple approach: find promising regions via local search
        # and use the remaining budget for refined exploration

        # Sort by function value
        order = np.argsort(values)
        points = points[order]
        values = values[order]

        # Local search from the best few points
        def local_search(start_x, max_iter=20):
            """Simple adaptive step size local search."""
            nonlocal evaluations_used
            if best_x is None:
                return
            x = start_x.copy()
            fx = evaluate(x)
            if fx is None:
                return

            # Adaptive step size
            step = 0.5
            for _ in range(max_iter):
                if evaluations_used >= budget or time.time() > deadline:
                    break
                improved = False
                for d in range(dim):
                    for direction in [-1, 1]:
                        if evaluations_used >= budget or time.time() > deadline:
                            break
                        trial = x.copy()
                        trial[d] += direction * step
                        trial = np.clip(trial, -5, 5)
                        if np.allclose(trial, x):
                            continue
                        f_trial = evaluate(trial)
                        if f_trial is None:
                            return
                        if f_trial < fx - 1e-10:
                            x = trial
                            fx = f_trial
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    step = max(0.1, step * 1.2)
                else:
                    step = step * 0.7
                    if step < 0.01:
                        break
            return x, fx

        # Do local search from top candidates
        n_local = min(5, len(points))
        for i in range(n_local):
            if evaluations_used >= budget or time.time() > deadline:
                break
            result = local_search(points[i], max_iter=15)
            if result is None:
                continue
            # Add the result to our knowledge
            if evaluations_used < budget and time.time() < deadline:
                points = np.vstack([points, result[0]])
                values = np.append(values, result[1])

        # Phase 3: Use remaining budget for global exploration with
        # a simple kriging-like model to guide new point selection
        if evaluations_used < budget and time.time() < deadline:
            # Simple surrogate: use nearest neighbors for prediction
            # We'll do a few rounds of selecting points based on uncertainty
            n_surrogate_rounds = min(20, (budget - evaluations_used) // 5)

            for _ in range(n_surrogate_rounds):
                if evaluations_used >= budget or time.time() > deadline:
                    break

                # Generate candidate points
                n_candidates = 100
                candidates = rng.uniform(-5, 5, (n_candidates, dim))

                # Score candidates: balance between predicted value and uncertainty
                # Simple approach: use distance to existing points as uncertainty proxy
                # and interpolate for value prediction

                # Predicted value: weighted average of nearest neighbors
                # Uncertainty: distance to nearest neighbor

                scores = np.zeros(n_candidates)
                for j in range(n_candidates):
                    # Distance to all existing points
                    dists = np.linalg.norm(points - candidates[j], axis=1)
                    # Nearest 3 neighbors
                    k = min(3, len(points))
                    nearest_idx = np.argpartition(dists, k)[:k]
                    nearest_dists = dists[nearest_idx]
                    nearest_vals = values[nearest_idx]

                    # Weighted prediction
                    weights = 1.0 / (nearest_dists + 1e-8)
                    weights /= np.sum(weights)
                    pred_val = np.sum(weights * nearest_vals)

                    # Uncertainty: average distance to nearest neighbors
                    uncertainty = np.mean(nearest_dists)

                    # Acquisition: minimize predicted value + exploration bonus
                    # Scale uncertainty to be comparable to function values
                    val_range = np.max(values) - np.min(values) + 1e-8
                    scores[j] = pred_val + 0.5 * uncertainty * val_range / (np.max(nearest_dists) + 1e-8)

                # Select best candidate
                best_idx = np.argmin(scores)
                x_new = candidates[best_idx]
                fx_new = evaluate(x_new)
                if fx_new is None:
                    break

                # Add to our knowledge base
                points = np.vstack([points, x_new])
                values = np.append(values, fx_new)

                # If this was a good improvement, do local search from it
                if fx_new < best_f * 0.9 or (best_f > 0 and fx_new < best_f - 1):
                    if evaluations_used < budget and time.time() < deadline:
                        result = local_search(x_new, max_iter=10)
                        if result is not None:
                            if evaluations_used < budget and time.time() < deadline:
                                points = np.vstack([points, result[0]])
                                values = np.append(values, result[1])

        return best_x, best_f
# EVOLVE-BLOCK-END
