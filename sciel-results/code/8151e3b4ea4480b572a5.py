"""Direct search pattern search with sparse random restarts for BBOB f18."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Pattern search with binary step-size refinement and sparse restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Build search directions: orthogonal + diagonal
        # Orthogonal: e_i for i in range(dim)
        # Diagonal: all ones, and pairs (e_i + e_j)/sqrt(2)
        directions = []
        for i in range(dim):
            e = np.zeros(dim)
            e[i] = 1.0
            directions.append(e)
        # Full diagonal
        directions.append(np.ones(dim) / np.sqrt(dim))
        # Pair diagonals
        for i in range(dim):
            for j in range(i + 1, dim):
                d = np.zeros(dim)
                d[i] = 1.0 / np.sqrt(2)
                d[j] = 1.0 / np.sqrt(2)
                directions.append(d)
        # Negative versions
        directions = directions + [-d for d in directions]
        directions = np.array(directions)
        n_dirs = len(directions)

        # State
        calls = [0]

        def call_func(x):
            calls[0] += 1
            return func(x)

        # Initial point: center of box
        x = np.zeros(dim)
        fx = call_func(x)
        best_x = x.copy()
        best_fx = fx

        # Step size parameters
        sigma_min = 1e-8
        sigma_max = 10.0
        # We'll use a single global step size that adapts

        # Pattern search with binary refinement per direction
        # For each direction, try expanding step size via binary search
        # to find the best step along that direction

        # Phase 1: Global exploration with random restarts
        # Phase 2: Local pattern search refinement

        # Let's do: initial few random points to find a good starting region,
        # then pattern search from the best.

        # Reserve budget: ~20% for exploration, 80% for pattern search
        n_explore = min(50, budget // 5)
        n_search = budget - n_explore - 1  # -1 for initial center call

        # Exploration: sample random points
        for _ in range(n_explore):
            if calls[0] >= budget:
                break
            x_try = rng.uniform(-5, 5, dim)
            fx_try = call_func(x_try)
            if fx_try < best_fx:
                best_fx = fx_try
                best_x = x_try.copy()

        # Pattern search phase
        # Current best point
        x = best_x.copy()
        fx = best_fx
        sigma = 1.0  # initial step size

        # For pattern search, we iterate over directions
        # For each direction, we do a binary search on step size in [sigma_min, sigma_max]
        # to find the step that minimizes f along that direction
        # Then we move to the best point found

        # To be efficient, we limit the number of binary search iterations
        # and the number of full passes over directions

        # Let's estimate: n_dirs * (binary search iters) calls per pass
        # n_dirs is about 2*(dim + 1 + dim*(dim-1)/2) = 2*(5+1+10)=32
        # Binary search: ~10 iterations per direction
        # So ~320 calls per pass. With ~450 calls left, we can do ~1 pass.
        # Let's be more careful.

        # Actually, let's use a simpler approach: for each direction, try a few
        # step sizes via binary search, but limit total calls.

        # Alternative: use a fixed set of step sizes and try them.
        # Or: do a line search with a few evaluations.

        # Let me reconsider: pattern search with adaptive step size.
        # For each direction d:
        #   Try x + sigma*d and x - sigma*d
        #   If improvement, increase sigma and continue in that direction
        #   If no improvement, decrease sigma
        # This is classic pattern search.

        # Let's implement classic pattern search:
        # sigma starts at some value
        # For each direction d in directions:
        #   y1 = x + sigma * d
        #   y2 = x - sigma * d
        #   Evaluate both (2 calls)
        #   If either improves, move to the best, increase sigma
        #   If neither improves, decrease sigma
        # Repeat until budget exhausted

        # This uses 2 calls per direction per iteration
        # n_dirs ~ 32, so 64 calls per full cycle
        # With 450 calls, we can do ~7 cycles

        sigma = 1.0
        sigma_increase = 1.5
        sigma_decrease = 0.5

        while calls[0] < budget:
            improved = False
            for d in directions:
                if calls[0] >= budget:
                    break
                # Try positive direction
                y1 = np.clip(x + sigma * d, -5, 5)
                fx1 = call_func(y1)
                if fx1 < fx:
                    x, fx = y1, fx1
                    improved = True
                    if fx < best_fx:
                        best_fx = fx
                        best_x = x.copy()
                # Try negative direction
                if calls[0] >= budget:
                    break
                y2 = np.clip(x - sigma * d, -5, 5)
                fx2 = call_func(y2)
                if fx2 < fx:
                    x, fx = y2, fx2
                    improved = True
                    if fx < best_fx:
                        best_fx = fx
                        best_x = x.copy()

            if improved:
                sigma = min(sigma * sigma_increase, sigma_max)
            else:
                sigma *= sigma_decrease
                if sigma < sigma_min:
                    # Try a random restart
                    if calls[0] < budget:
                        x_try = rng.uniform(-5, 5, dim)
                        fx_try = call_func(x_try)
                        if fx_try < fx:
                            x, fx = x_try, fx_try
                        sigma = 1.0

        return best_x, best_fx
# EVOLVE-BLOCK-END
