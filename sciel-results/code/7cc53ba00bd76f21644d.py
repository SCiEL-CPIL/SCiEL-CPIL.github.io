# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for WFG7 using a two-phase strategy:
    Phase 1: Random sampling to get initial spread
    Phase 2: Local refinement around best points using adaptive step sizes
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim
        m = self.n_obj

        # Use the seeded RNG from np.random.seed
        def rng_rand(*shape):
            return np.random.random(shape)

        def rng_randn(*shape):
            return np.random.standard_normal(shape)

        def rng_integers(low, high=None):
            if high is None:
                return np.random.randint(low)
            return np.random.randint(low, high)

        # Track all evaluated points and their objective values
        X_all = []  # list of position vectors
        F_all = []  # list of objective vectors

        def evaluate(x):
            """Evaluate a point, record it, and return objectives."""
            x = np.asarray(x, dtype=float).flatten()
            x = np.clip(x, lo, hi)
            f = np.asarray(problem(x), dtype=float).flatten()
            X_all.append(x.copy())
            F_all.append(f.copy())
            return f

        # Phase 1: Initial random sampling
        n_init = min(40, self.budget // 3)
        for i in range(n_init):
            if time.time() > deadline:
                break
            x = lo + (hi - lo) * rng_rand(n)
            evaluate(x)

        if not X_all:
            # Fallback: just return what we have
            return np.array(X_all) if X_all else np.zeros((1, n))

        F_arr = np.array(F_all)

        # Find non-dominated points
        def non_dominated_indices(F):
            """Return indices of non-dominated points."""
            npts = len(F)
            is_dom = np.zeros(npts, dtype=bool)
            for i in range(npts):
                for j in range(npts):
                    if i == j:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        is_dom[i] = True
                        break
            return np.where(~is_dom)[0]

        # Phase 2: Local refinement around non-dominated points
        # Use adaptive step sizes and direction sampling
        remaining = self.budget - len(X_all)
        if remaining <= 0:
            return np.array(X_all)

        # Get current non-dominated set
        nd_idx = non_dominated_indices(F_arr)
        if len(nd_idx) == 0:
            nd_idx = np.array([np.argmin(np.sum(F_arr, axis=1))])

        # Adaptive local search
        base_step = 0.05  # Fraction of range
        min_step = 0.001
        max_step = 0.2

        evals_this_phase = 0
        # Sort non-dominated points by sum of objectives (prefer closer to origin)
        nd_sorted = nd_idx[np.argsort(np.sum(F_arr[nd_idx], axis=1))]

        # Cycle through non-dominated points for refinement
        attempt = 0
        while evals_this_phase < remaining:
            if time.time() > deadline:
                break

            # Pick a parent from non-dominated set (cycle through)
            parent_idx = nd_sorted[attempt % len(nd_sorted)]
            parent_x = X_all[parent_idx].copy()
            parent_f = F_all[parent_idx].copy()

            # Adaptive step size: start moderate, shrink if no improvement
            step = base_step * (0.5 + 0.5 * rng_rand())

            # Generate a random direction
            direction = rng_randn(n)
            norm = np.linalg.norm(direction)
            if norm > 1e-10:
                direction /= norm

            # Try the step
            child_x = parent_x + step * (hi - lo) * direction
            child_x = np.clip(child_x, lo, hi)
            child_f = evaluate(child_x)
            evals_this_phase += 1

            # Check if child improves hypervolume contribution
            # Simple check: is child non-dominated or does it dominate parent?
            if np.all(child_f <= parent_f) and np.any(child_f < parent_f):
                # Child dominates parent - good, keep step size
                pass
            elif np.all(parent_f <= child_f) and np.any(parent_f < child_f):
                # Parent dominates child - shrink step
                base_step = max(min_step, base_step * 0.7)
            else:
                # Incomparable - moderate response
                base_step = max(min_step, base_step * 0.9)

            # Update non-dominated set periodically
            if evals_this_phase % 10 == 0:
                F_arr = np.array(F_all)
                nd_idx = non_dominated_indices(F_arr)
                if len(nd_idx) == 0:
                    nd_idx = np.array([np.argmin(np.sum(F_arr, axis=1))])
                nd_sorted = nd_idx[np.argsort(np.sum(F_arr[nd_idx], axis=1))]

            attempt += 1

        return np.array(X_all) if X_all else np.zeros((1, n))
# EVOLVE-BLOCK-END
