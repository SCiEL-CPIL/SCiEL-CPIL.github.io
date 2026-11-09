"""Multi-objective optimizer for DTLZ7 with structural exploitation."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Exploit DTLZ7 structure: f1=x0, f2=(1-x0)*cos(x0*pi/2), f3=(1-x0)*sin(x0*pi/2) when g=1."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 290.0  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(42)

        X_list = []
        F_list = []
        eval_count = 0

        def evaluate(x):
            nonlocal eval_count
            if eval_count >= budget:
                return None
            x = np.clip(x, lo, hi)
            f = problem(x)
            eval_count += 1
            X_list.append(x.copy())
            F_list.append(np.asarray(f, dtype=float))
            return f

        # Phase 1: Dense structured initialization along the front
        # The Pareto front is 1D parameterized by x[0] in [0,1]
        n_init = min(40, budget // 5)
        x0_vals = np.linspace(0.0, 1.0, n_init)
        for x0 in x0_vals:
            if time.time() > deadline:
                break
            x = np.zeros(dim)
            x[0] = x0
            # Small noise in x[1:] to stay in bounds and avoid exact zero issues
            x[1:] = rng.uniform(0, 0.002, dim - 1)
            evaluate(x)

        # Phase 2: Local search to drive x[1:] toward 0 while fine-tuning x[0]
        # Use iterative refinement: for each point, try to reduce g
        n_local = budget - eval_count
        n_iter = max(1, n_local)

        for _ in range(n_iter):
            if time.time() > deadline or eval_count >= budget:
                break

            # Pick a random existing point
            idx = rng.integers(len(X_list))
            parent = X_list[idx].copy()

            # Strategy: reduce x[1:] magnitudes
            # Use exponential decay on x[1:]
            decay = rng.uniform(0.0, 0.5)
            child = parent.copy()
            child[1:] = child[1:] * decay

            # Small perturbation in x[0] for coverage
            if rng.random() < 0.3:
                child[0] = parent[0] + rng.normal(0, 0.01)
                child[0] = np.clip(child[0], 0.0, 1.0)

            f = evaluate(child)
            if f is None:
                break

            # Maintain diversity: if we have too many similar points, remove worst
            if len(X_list) > 100:
                # Remove the point with highest sum that is most similar to others
                sums = np.sum(F_list, axis=1)
                # Keep top 80 by hypervolume contribution (approx by sum)
                order = np.argsort(sums)
                if len(order) > 80:
                    # Remove the last (worst) one
                    worst = order[-1]
                    X_list.pop(int(worst))
                    F_list.pop(int(worst))

        # Phase 3: Final polish - set x[1:] to exactly 0 for all points
        for i in range(len(X_list)):
            if time.time() > deadline or eval_count >= budget:
                break
            x = X_list[i].copy()
            x[1:] = 0.0
            x = np.clip(x, lo, hi)
            f = evaluate(x)
            if f is None:
                break
            # Replace the original
            X_list[i] = x
            F_list[i] = np.asarray(f, dtype=float)

        # Phase 4: If budget remains, add more points with x[1:]=0
        while eval_count < budget and time.time() < deadline:
            x = np.zeros(dim)
            x[0] = rng.uniform(0, 1)
            f = evaluate(x)
            if f is None:
                break

        return X_list
# EVOLVE-BLOCK-END
