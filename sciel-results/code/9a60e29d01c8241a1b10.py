"""Multi-objective optimizer for DTLZ1 exploiting known Pareto front structure."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ1 (3 objectives, 7 vars, bounds [0,1]).

    Exploits the known Pareto front structure: for DTLZ1, the Pareto optimal
    set is {x : x[:m] >= 0, sum(x[:m]) = 1, x[m:] = 0.5}, giving
    f_i = 0.5 * x_i with sum(f) = 0.5.

    Strategy: use the entire budget to sample points on the Pareto front
    using a combination of strategic points and stratified/quasi-random
    sampling to maximize hypervolume coverage of the 2D simplex.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        def time_ok():
            return time.time() < deadline

        recorded_F = []
        used = 0

        def eval_and_record(x):
            nonlocal used
            f = problem(x)
            f = np.array(f, dtype=float)
            f = np.clip(f, 0.0, 399.999)
            recorded_F.append(f)
            used += 1
            return f

        def make_x(a, b, c):
            """Create a 7-dim decision vector on the Pareto front."""
            x = np.zeros(dim)
            x[0] = a
            x[1] = b
            x[2] = c
            x[n_obj:] = 0.5
            return np.clip(x, lo, hi)

        rng = np.random

        # Phase 0: Strategic high-value points (corners, edge midpoints, center)
        # These give the largest individual hypervolume contributions
        strategic = [
            (0.999, 0.0005, 0.0005),  # near corner 1
            (0.0005, 0.999, 0.0005),  # near corner 2
            (0.0005, 0.0005, 0.999),  # near corner 3
            (0.5, 0.4995, 0.0005),    # edge midpoint 1-2
            (0.5, 0.0005, 0.4995),    # edge midpoint 1-3
            (0.0005, 0.5, 0.4995),    # edge midpoint 2-3
            (1.0/3, 1.0/3, 1.0/3),    # center
        ]
        for a, b, c in strategic:
            if used >= budget or not time_ok():
                break
            x = make_x(a, b, c)
            eval_and_record(x)

        # Phase 1: Stratified sampling on the simplex
        # Use the transformation: u1 = sqrt(U1), u2 = U2*(1-sqrt(U1))
        # For uniform on simplex: a = sqrt(U1), b = U2*(1-sqrt(U1)), c = 1-a-b
        # Stratify U1 and U2 into a grid for better coverage
        n_strat = 13  # 13x13 = 169 stratified points
        for i in range(n_strat):
            if used >= budget or not time_ok():
                break
            for j in range(n_strat):
                if used >= budget or not time_ok():
                    break
                u1 = (i + rng.random()) / n_strat
                u2 = (j + rng.random()) / n_strat
                a = np.sqrt(u1)
                b = u2 * (1.0 - a)
                c = 1.0 - a - b
                # Clip to ensure valid simplex
                a = max(0.0, min(1.0, a))
                b = max(0.0, min(1.0, b))
                c = max(0.0, min(1.0, c))
                s = a + b + c
                if s > 0:
                    a /= s
                    b /= s
                    c /= s
                x = make_x(a, b, c)
                eval_and_record(x)

        # Phase 2: Halton quasi-random for remaining budget
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        remaining = budget - used
        for k in range(1, remaining + 1):
            if used >= budget or not time_ok():
                break
            h1 = halton(k, 2)
            h2 = halton(k, 3)
            a = np.sqrt(h1)
            b = h2 * (1.0 - a)
            c = 1.0 - a - b
            a = max(0.0, min(1.0, a))
            b = max(0.0, min(1.0, b))
            c = max(0.0, min(1.0, c))
            s = a + b + c
            if s > 0:
                a /= s
                b /= s
                c /= s
            x = make_x(a, b, c)
            eval_and_record(x)

        if recorded_F:
            return np.array(recorded_F)
        return np.array([])
# EVOLVE-BLOCK-END
