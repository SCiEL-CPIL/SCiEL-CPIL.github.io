"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for RE37 using weighted-sum + local refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s limit

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Track all evaluated points and their objective values
        all_X = []
        all_F = []
        used = 0

        def evaluate(x):
            nonlocal used
            x = np.clip(x, lo, hi)
            f = np.array(problem(x))
            all_X.append(x.copy())
            all_F.append(f.copy())
            used += 1
            return f

        # Phase 1: Initial random sampling
        n_init = 40
        for _ in range(n_init):
            x = lo + (hi - lo) * rng.random(dim)
            evaluate(x)

        # Helper: check if a point is non-dominated among a set
        def is_non_dominated(f, F_set):
            for g in F_set:
                if np.all(g <= f) and np.any(g < f):
                    return False
            return True

        def get_non_dominated_indices(F):
            n = len(F)
            nd_idx = []
            for i in range(n):
                dominated = False
                for j in range(n):
                    if i != j and np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated = True
                        break
                if not dominated:
                    nd_idx.append(i)
            return nd_idx

        # Phase 2: Weighted-sum exploration to cover the Pareto front
        # Generate diverse weight vectors
        n_weights = 30
        weights = []
        # Include extreme weights
        for i in range(n_obj):
            w = np.zeros(n_obj)
            w[i] = 1.0
            weights.append(w)
        # Add random weights
        while len(weights) < n_weights:
            w = rng.random(n_obj)
            w = w / np.sum(w)
            weights.append(w)

        for w in weights:
            if used >= budget or time.time() > deadline:
                break
            # Find point that minimizes weighted sum
            # Use a few random starts and pick the best
            best_f = None
            best_x = None
            n_starts = 5
            for _ in range(n_starts):
                x0 = lo + (hi - lo) * rng.random(dim)
                # Simple local search: coordinate descent with random restarts
                x = x0.copy()
                f = evaluate(x)
                if best_f is None or np.dot(w, f) < np.dot(w, best_f):
                    best_f = f.copy()
                    best_x = x.copy()
                # Coordinate descent refinement
                for _iter in range(3):
                    improved = False
                    for d in range(dim):
                        for delta in [0.05, 0.1, 0.2, 0.05, -0.05, -0.1, -0.2, -0.05]:
                            x_trial = x.copy()
                            x_trial[d] = np.clip(x_trial[d] + delta * (hi[d] - lo[d]), lo[d], hi[d])
                            f_trial = evaluate(x_trial)
                            if np.dot(w, f_trial) < np.dot(w, f):
                                x = x_trial.copy()
                                f = f_trial.copy()
                                improved = True
                                if best_f is None or np.dot(w, f) < np.dot(w, best_f):
                                    best_f = f.copy()
                                    best_x = x.copy()
                    if not improved:
                        break

        # Phase 3: Local refinement around non-dominated points
        # Get non-dominated points from all evaluated
        if len(all_F) > 0:
            F_arr = np.array(all_F)
            nd_idx = get_non_dominated_indices(F_arr)
            
            # Sort by hypervolume contribution (approximate by sum of objectives)
            nd_points = [(i, all_X[i], all_F[i]) for i in nd_idx]
            
            # Refine the top non-dominated points
            n_refine = min(len(nd_points), 15)
            # Prioritize points with lower sum of objectives
            nd_points.sort(key=lambda t: np.sum(t[2]))
            
            for idx, x0, f0 in nd_points[:n_refine]:
                if used >= budget or time.time() > deadline:
                    break
                
                # Nelder-Mead style local search minimizing a weighted sum
                # Use different weight vectors for different points to explore different directions
                w = np.ones(n_obj) / n_obj
                # Add some randomness to the weight
                w += 0.3 * rng.random(n_obj)
                w = w / np.sum(w)
                
                x = x0.copy()
                f = f0.copy()
                
                # Simple local search with shrinking step size
                step_sizes = [0.1, 0.05, 0.02, 0.01]
                for step in step_sizes:
                    improved = False
                    for _iter in range(5):
                        for d in range(dim):
                            for sign in [1, -1]:
                                x_trial = x.copy()
                                x_trial[d] = np.clip(x_trial[d] + sign * step * (hi[d] - lo[d]), lo[d], hi[d])
                                f_trial = evaluate(x_trial)
                                if np.dot(w, f_trial) < np.dot(w, f):
                                    x = x_trial.copy()
                                    f = f_trial.copy()
                                    improved = True
                    if not improved:
                        break

        # Phase 4: Final random sampling if budget remains
        while used < budget and time.time() < deadline:
            x = lo + (hi - lo) * rng.random(dim)
            evaluate(x)

        return all_X
# EVOLVE-BLOCK-END
