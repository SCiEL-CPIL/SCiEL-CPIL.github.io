# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Hypervolume-driven adaptive multi-objective optimizer for RE37."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 280  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        rng = np.random.default_rng(42)

        # Reference point for hypervolume (fixed by task)
        ref = np.array([1.1, 1.3, 1.1], dtype=np.float64)

        # Track all evaluations
        all_X = []
        all_F = []
        eval_count = 0

        def evaluate(x):
            nonlocal eval_count
            if eval_count >= self.budget:
                return None
            x = np.clip(x, lo, hi)
            f = np.asarray(problem(x), dtype=np.float64)
            if not np.all(np.isfinite(f)):
                f = np.where(np.isfinite(f), f, 10.0)
            all_X.append(x.copy())
            all_F.append(f.copy())
            eval_count += 1
            return f

        # Phase 1: Initial diverse sampling
        n_init = 12
        # Corners of the hypercube
        corners = np.array(list(np.ndindex(2**self.dim)))
        corners = corners * (hi - lo) + lo
        # Add center
        center = (lo + hi) / 2.0
        # Add a few random points
        random_pts = lo + (hi - lo) * rng.random((n_init - len(corners) - 1, self.dim))
        
        init_pts = np.vstack([corners, center, random_pts])
        
        for x in init_pts:
            if eval_count >= self.budget:
                break
            evaluate(x)

        # Helper: compute non-dominated set indices
        def non_dominated_indices(F):
            if len(F) == 0:
                return np.array([])
            mask = np.ones(len(F), dtype=bool)
            for i in range(len(F)):
                if not mask[i]:
                    continue
                for j in range(len(F)):
                    if i == j or not mask[j]:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        mask[i] = False
                        break
            return np.where(mask)[0]

        # Phase 2: Local refinement using Nelder-Mead with diverse weights
        # Generate weight vectors on the simplex
        weights = []
        # Use a set of well-distributed weights
        # For 3 objectives, we can use a grid
        # w1, w2, w3 > 0, sum = 1
        # Let's use 10 specific weights
        weight_list = [
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.6, 0.2, 0.2],
            [0.2, 0.6, 0.2],
            [0.2, 0.2, 0.6],
            [0.5, 0.25, 0.25],
            [0.25, 0.5, 0.25],
            [0.25, 0.25, 0.5],
            [0.33, 0.33, 0.34],
        ]
        weights = np.array(weight_list)
        
        # Shuffle to diversify
        rng.shuffle(weights)

        for w in weights:
            if eval_count >= self.budget:
                break
            if time.time() > deadline - 20:
                break

            # Find the best point in current archive for this weight
            if not all_F:
                x0 = (lo + hi) / 2.0
            else:
                F_arr = np.array(all_F)
                scores = F_arr @ w
                start_idx = np.argmin(scores)
                x0 = all_X[start_idx].copy()

            # Define the scalarized objective function using our recording wrapper
            def objective(x, w=w): # Default arg to capture current w
                x_clipped = np.clip(x, lo, hi)
                f = evaluate(x_clipped)
                if f is None:
                    return 1e10
                return np.dot(f, w)

            # Run Nelder-Mead
            maxiter = 15
            try:
                res = minimize(objective, x0, method='Nelder-Mead', 
                               options={'maxiter': maxiter, 'xatol': 1e-4, 'fatol': 1e-6})
            except Exception:
                pass

        # Phase 3: Gap filling with random sampling if budget remains
        if eval_count < self.budget:
            while eval_count < self.budget:
                if time.time() > deadline - 5:
                    break
                x_rand = lo + (hi - lo) * rng.random(self.dim)
                evaluate(x_rand)

        return all_X[-1] if all_X else np.zeros(self.dim)
# EVOLVE-BLOCK-END
