"""Multi-objective optimizer for DTLZ7 exploiting known front structure."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Exploit DTLZ7 structure: drive g->0, sample front variables effectively."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 320.0  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        # DTLZ7: x[0]=x1, x[1]=x2, x[2]=x3 are front vars; x[3:] are g-vars
        # g = 1 + 9/n_g * sum(x_g) where x_g = x[3:]
        # f1 = x[0]*(1+g), f2 = x[1]*(1+g), f3 = (1+g)*(3 - x[0]*(1+x[1]) - x[1]*(1+x[2]))

        used = 0
        all_X = []
        all_F = []

        def make_point(x1, x2, x3, g_vars):
            x = np.zeros(self.dim)
            x[0] = x1
            x[1] = x2
            x[2] = x3
            x[3:] = g_vars
            return np.clip(x, lo, hi)

        def evaluate(x):
            nonlocal used
            f = np.array(problem(x))
            used += 1
            all_X.append(x.copy())
            all_F.append(f)
            return f

        # Phase 1: Structured sampling with g-vars biased toward 0
        # Use power-law: g_vars ~ U(0,1)^alpha to bias toward 0
        # Front vars: sample on a structured grid/Latin hypercube

        n_phase1 = min(120, self.budget)
        n_phase2 = self.budget - n_phase1

        # Generate front variable samples using stratified sampling
        # We want good coverage of the Pareto front in (x1, x2, x3) space
        # The front: f1=x1, f2=x2, f3=3-x1(1+x2)-x2(1+x3)
        # For hypervolume with ref [1.1, 1.1, 25], we want small f1, f2, f3

        # Use a combination: some uniform, some biased toward corners
        rng = np.random.default_rng(42)

        # Generate diverse front variable combinations
        # Key regions: small x1, small x2 help; x3 helps with f3
        # Sample in a way that covers the front well

        front_samples = []
        # Uniform sampling in [0,1]^3 for front vars
        n_uniform = n_phase1 // 3
        for _ in range(n_uniform):
            x1 = rng.random()
            x2 = rng.random()
            x3 = rng.random()
            front_samples.append((x1, x2, x3))

        # Biased: small x1, x2 (important for f1, f2)
        n_biased = n_phase1 // 3
        for _ in range(n_biased):
            x1 = rng.random() ** 2  # bias toward 0
            x2 = rng.random() ** 2  # bias toward 0
            x3 = rng.random() ** 1.5  # slight bias toward 0
            front_samples.append((x1, x2, x3))

        # Fill remaining with more aggressive bias
        n_remaining = n_phase1 - n_uniform - n_biased
        for _ in range(n_remaining):
            x1 = rng.random() ** 3
            x2 = rng.random() ** 3
            x3 = rng.random() ** 2
            front_samples.append((x1, x2, x3))

        # For g-vars: use power-law to bias toward 0
        # g_vars ~ U(0,1)^alpha, with alpha increasing over phases
        for i, (x1, x2, x3) in enumerate(front_samples):
            if time.time() > deadline:
                break
            alpha = 2.0 + 0.5 * (i / max(1, n_phase1))  # increase bias over time
            g_vars = (rng.random(self.dim - 3)) ** alpha
            x = make_point(x1, x2, x3, g_vars)
            evaluate(x)

        # Phase 2: Refine best solutions
        if used < self.budget and time.time() < deadline:
            F_arr = np.array(all_F)
            X_arr = np.array(all_X)

            # Find non-dominated set
            def dominates(a, b):
                return np.all(a <= b) and np.any(a < b)

            # Simple non-dominated sort
            nondom_mask = np.ones(len(F_arr), dtype=bool)
            for i in range(len(F_arr)):
                if not nondom_mask[i]:
                    continue
                for j in range(len(F_arr)):
                    if i == j or not nondom_mask[j]:
                        continue
                    if dominates(F_arr[j], F_arr[i]):
                        nondom_mask[i] = False
                        break

            nondom_idx = np.where(nondom_mask)[0]
            if len(nondom_idx) == 0:
                nondom_idx = np.array([np.argmin(np.sum(F_arr, axis=1))])

            # Take top non-dominated points for refinement
            n_refine = min(len(nondom_idx), 10)
            # Sort by sum of objectives (proxy for quality)
            sums = np.sum(F_arr[nondom_idx], axis=1)
            top_idx = nondom_idx[np.argsort(sums)[:n_refine]]

            n_refine_budget = self.budget - used
            per_point = max(1, n_refine_budget // len(top_idx))

            for idx in top_idx:
                if time.time() > deadline:
                    break
                base_x = X_arr[idx].copy()
                for _ in range(per_point):
                    if used >= self.budget or time.time() > deadline:
                        break
                    # Perturb: reduce g-vars further, tweak front vars
                    child = base_x.copy()
                    # Reduce g-vars (indices 3+)
                    g_scale = rng.uniform(0.3, 0.8)
                    child[3:] *= g_scale
                    # Tweak front vars
                    child[0] += rng.normal(0, 0.05)
                    child[1] += rng.normal(0, 0.05)
                    child[2] += rng.normal(0, 0.05)
                    child = np.clip(child, lo, hi)
                    evaluate(child)
                    # Update base if improved
                    if np.sum(all_F[-1]) < np.sum(F_arr[idx]):
                        base_x = child.copy()

        return all_X
# EVOLVE-BLOCK-END
