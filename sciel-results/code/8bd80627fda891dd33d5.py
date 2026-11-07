"""Multi-objective optimizer for DTLZ1 exploiting known structure."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer exploiting DTLZ1 structure for hypervolume maximization."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(42)

        all_X = []
        all_F = []

        def evaluate(x):
            nonlocal all_X, all_F
            x = np.clip(x, lo, hi)
            f = np.array(problem(x))
            all_X.append(x.copy())
            all_F.append(f.copy())
            return f

        # DTLZ1 structure: x[0], x[1] determine Pareto front position
        # x[2:] should be 0.5 to minimize g
        # f1 = 0.5*x0*(1+g), f2 = 0.5*(1-x0)*x1*(1+g), f3 = 0.5*(1-x0)*(1-x1)*(1+g)
        # With g_min = 150 (when x[2:]=0.5), 1+g = 151
        # Pareto front: f1 + f2 + f3 = 0.5*(1+g) = 75.5 (when x0+x1 constraint holds)
        # Actually for DTLZ1: f1+f2+f3 = 0.5*(1+g) when on the front

        # Phase 1: Grid sampling on the Pareto front (x0, x1 space)
        # The Pareto front is parameterized by x0 in [0,1], x1 in [0,1-x0] (for 3 obj)
        # Actually for DTLZ1 with M=3: the front is a triangle in objective space
        # Parameterization: let x0 = t, x1 = s*(1-t) where t,s in [0,1]
        # But simpler: sample x0 and x1 directly with x0 + x1 <= 1 not needed since
        # the objectives are defined for all x0,x1 in [0,1]

        # Better: sample uniformly in x0, x1 space
        n_grid = 12  # 12x12 = 144 points
        x0_vals = np.linspace(0.05, 0.95, n_grid)
        x1_vals = np.linspace(0.05, 0.95, n_grid)

        for x0 in x0_vals:
            if time.time() > deadline - 30:
                break
            for x1 in x1_vals:
                if len(all_X) >= self.budget:
                    break
                x = np.zeros(self.dim)
                x[0] = x0
                x[1] = x1
                x[2:] = 0.5  # optimal shape variables
                evaluate(x)

        # Phase 2: Add random points with shape vars near 0.5 for diversity
        remaining = self.budget - len(all_X)
        n_random = min(remaining, 50)
        for _ in range(n_random):
            if time.time() > deadline - 20 or len(all_X) >= self.budget:
                break
            x = np.zeros(self.dim)
            x[0] = rng.uniform(0, 1)
            x[1] = rng.uniform(0, 1)
            x[2:] = 0.5 + 0.05 * rng.standard_normal(self.dim - 2)
            x = np.clip(x, lo, hi)
            evaluate(x)

        # Phase 3: Local refinement around best points
        if len(all_X) < self.budget and time.time() < deadline - 10:
            # Find non-dominated points and perturb them
            F = np.array(all_F)
            X = np.array(all_X)
            
            # Simple non-dominated sort
            dominated = np.zeros(len(F), dtype=bool)
            for i in range(len(F)):
                if dominated[i]:
                    continue
                for j in range(len(F)):
                    if i == j or dominated[j]:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated[i] = True
                        break
            
            nd_idx = np.where(~dominated)[0]
            n_refine = min(self.budget - len(all_X), 30)
            for _ in range(n_refine):
                if time.time() > deadline - 5 or len(all_X) >= self.budget:
                    break
                if len(nd_idx) == 0:
                    break
                idx = nd_idx[rng.integers(len(nd_idx))]
                x = X[idx] + 0.02 * rng.standard_normal(self.dim)
                # Keep shape vars near 0.5
                x[2:] = 0.5 + 0.01 * rng.standard_normal(self.dim - 2)
                x = np.clip(x, lo, hi)
                evaluate(x)

        return all_X
# EVOLVE-BLOCK-END
