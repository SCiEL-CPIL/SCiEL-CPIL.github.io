"""ZDT6 multi-objective optimizer: structured Pareto-front sampling."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Structured sampling of ZDT6 Pareto front for maximum hypervolume."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 300.0

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # For ZDT6: f1 = 1 + 10*n where n = #{i: f2_i(x_i) > 0.5}
        # f2_i(x_i) = sin(10*x_i*sin(pi*x_i/2))
        # To make f2_i > 0.5: choose x_i such that sin(10*x_i*sin(pi*x_i/2)) > 0.5
        # To make f2_i <= 0.5: choose x_i = 0 (gives f2_i = 0)
        #
        # Strategy: For each possible n (0 to 10), set exactly n variables to
        # a value where f2_i > 0.5, and set the rest to 0.
        # Then vary one "free" variable to get different f2 values.
        #
        # Simpler: just sweep x_0 in [0,1] with x_{1:}=0.
        # This gives f1=1 or 2, with varying f2. Not ideal but works.
        #
        # Better: allocate evaluations to cover n=0,1,...,10.

        # For a given n, set first n variables to x_val where f2_i > 0.5
        # and remaining to 0. Then f1 = 1+10n.
        # f2 = (1+sqrt(f1)) * (1 - sqrt(1 - sum(f2_i)/f1))
        # With first n vars at x_val (f2_i = v) and rest at 0:
        # sum(f2_i) = n*v
        # f2 = (1+sqrt(1+10n)) * (1 - sqrt(1 - n*v/(1+10n)))

        # Choose x_val such that f2(x_val) is in (0.5, 1.0)
        # f2_i(x) = sin(10*x*sin(pi*x/2))
        # At x=0.5: sin(pi/2)=1, so 10*0.5*1=5, sin(5)≈-0.959. Not good.
        # Let's find x where f2_i(x) = 0.75 (mid-range above 0.5)
        # Try x=0.3: sin(pi*0.3/2)=sin(0.471)≈0.454, 10*0.3*0.454=1.362, sin(1.362)≈0.978
        # Try x=0.2: sin(pi*0.1)=0.309, 10*0.2*0.309=0.618, sin(0.618)≈0.579
        # x=0.2 gives f2_i≈0.579 > 0.5. Good.

        x_active = 0.2  # gives f2_i ≈ 0.579 > 0.5
        x_inactive = 0.0  # gives f2_i = 0

        # Allocate budget: distribute across n=0,1,...,10 (11 groups)
        # For each n, vary a "free" variable to get different f2 values
        n_groups = 11
        per_group = max(1, budget // n_groups)
        remaining = budget - per_group * n_groups

        results = []
        eval_count = 0

        for n in range(n_groups):
            alloc = per_group + (1 if n < remaining else 0)
            for j in range(alloc):
                if eval_count >= budget:
                    break
                if time.time() > deadline:
                    break

                x = np.zeros(dim)
                # Set first n variables to active value
                for i in range(min(n, dim)):
                    x[i] = x_active
                # Set rest to inactive (already 0)

                # Vary one free variable to get different f2
                if dim > n:
                    # Vary variable n (the first inactive one)
                    free_val = j / max(alloc - 1, 1) * 1.0
                    x[n] = free_val
                elif dim > 0:
                    # All variables active, vary the last one slightly
                    x[dim-1] = x_active + 0.01 * (j / max(alloc-1, 1) - 0.5)
                    x[dim-1] = np.clip(x[dim-1], 0, 1)

                f = problem(x)
                results.append((x, f))
                eval_count += 1

        return results
# EVOLVE-BLOCK-END
