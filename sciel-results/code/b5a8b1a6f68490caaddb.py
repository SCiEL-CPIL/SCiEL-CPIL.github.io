"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ7 using structured initialization and local refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280  # safety margin before 330s kill

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        n = self.dim
        budget = self.budget

        # DTLZ7 structure:
        # g = 100 * (1 + (|X[1:11]| - 1)^2)  (10 decision vars)
        # h = (n - 10) = 12
        # g(x) = 1 + g * (1 - prod(1 - x_i/h) for i in 1..h)
        # f1 = x0 * g
        # f2 = (1 - x0) * g
        # f3 = 1 - sqrt(x0 * (1 - x0))
        #
        # To get good hypervolume, we want:
        # 1. g as close to 1 as possible (i.e., X[1:11] close to 1)
        # 2. Good spread in x0 and the product term
        #
        # Strategy:
        # - Initialize with X[1:11] near 1 (to get g≈1)
        # - Vary x0 and the 12 product vars for spread
        # - Use a local search to refine

        def dtlz7(x):
            """Evaluate DTLZ7 objectives."""
            x = np.asarray(x)
            g_vars = x[1:11]
            g = 100.0 * np.sum((np.abs(g_vars) - 1.0) ** 2)
            g_total = 1.0 + g
            h = 12
            prod_terms = 1.0
            for i in range(h):
                prod_terms *= (1.0 - x[11 + i] / h)
            g_obj = g_total * (1.0 - prod_terms)
            x0 = x[0]
            f1 = x0 * g_obj
            f2 = (1.0 - x0) * g_obj
            f3 = 1.0 - np.sqrt(x0 * (1.0 - x0))
            return np.array([f1, f2, f3])

        # Phase 1: Structured initialization
        # Set g_vars to 1.0 (or near 1) to minimize g
        # Vary x0 and product vars
        pop_size = min(60, budget // 5)
        X = np.zeros((pop_size, n))
        # g_vars = 1.0
        X[:, 1:11] = 1.0
        # x0: spread across [0, 1]
        X[:, 0] = rng.random(pop_size)
        # Product vars: spread across [0, h] = [0, 12] but clipped to [0,1]
        # Actually bounds are [0,1], so x[11:22] in [0,1]
        # prod term: 1 - x_i/12, so x_i in [0,1] gives prod in [1-1/12, 1]
        # For better spread, we want to vary the product
        X[:, 11:22] = rng.random((pop_size, 12))

        F = np.array([problem(x) for x in X])
        used = pop_size

        # Phase 2: Local search with mutation
        # Focus on improving g (push g_vars toward 1) and refining spread
        max_iter = budget - used
        step_size = 0.05

        for _ in range(max_iter):
            if time.time() > deadline:
                break
            if used >= budget:
                break

            # Select parent: non-dominated or random
            # Find non-dominated set
            dominated = np.zeros(pop_size, dtype=bool)
            for i in range(pop_size):
                for j in range(pop_size):
                    if i != j:
                        if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                            dominated[i] = True
                            break
            non_dom_idx = np.where(~dominated)[0]
            if len(non_dom_idx) > 0:
                parent_idx = non_dom_idx[rng.integers(len(non_dom_idx))]
            else:
                parent_idx = rng.integers(pop_size)

            parent = X[parent_idx].copy()

            # Mutation strategy:
            # 1. Push g_vars toward 1.0 (strongly)
            # 2. Vary x0 and product vars for diversity
            child = parent.copy()

            # Push g_vars toward 1
            for i in range(1, 11):
                if rng.random() < 0.7:
                    child[i] += 0.3 * (1.0 - child[i]) + 0.02 * rng.standard_normal()

            # Vary x0
            if rng.random() < 0.5:
                child[0] += step_size * rng.standard_normal()

            # Vary product vars
            for i in range(11, 22):
                if rng.random() < 0.3:
                    child[i] += step_size * rng.standard_normal()

            child = np.clip(child, lo, hi)
            fc = problem(child)
            used += 1

            # Replace worst point
            # Use a simple dominance-based replacement
            # Find the worst point (most dominated)
            # For simplicity, replace the point with highest sum of objectives
            worst_idx = int(np.argmax(np.sum(F, axis=1)))
            X[worst_idx] = child
            F[worst_idx] = fc

        return X
# EVOLVE-BLOCK-END
