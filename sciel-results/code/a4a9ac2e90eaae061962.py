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
        deadline = t_start + 280  # Safety margin before 330s kill

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
            if time.time() > deadline:
                return None
            x = np.clip(x, lo, hi)
            try:
                f = np.asarray(problem(x), dtype=np.float64)
                if not np.all(np.isfinite(f)):
                    f = np.where(np.isfinite(f), f, 10.0)
            except Exception:
                return None
            all_X.append(x.copy())
            all_F.append(f.copy())
            eval_count += 1
            return f

        # Phase 1: Coarse Grid Sampling
        # For dim=4, a grid of size 2^4=16 is small. Let's use a slightly finer grid or just corners + center + random.
        # The directive suggests "partition the 4D domain into a coarse hypercube grid".
        # Let's use a 2x2x2x2 grid (16 points) + center + 10 random points = 27 points.
        # This fits well within budget 200.
        
        # Grid points
        grid_res = 2
        grid_points = []
        for i in range(grid_res ** self.dim):
            # Convert index i to multi-dimensional index
            idx = np.array([i // (grid_res ** (self.dim - 1 - d)) % grid_res for d in range(self.dim)])
            # Map to bounds: 0 -> lo, 1 -> hi (for grid_res=2)
            # General mapping: (idx + 0.5) / grid_res
            x_grid = lo + (hi - lo) * ((idx + 0.5) / grid_res)
            grid_points.append(x_grid)
        
        # Add center
        center = (lo + hi) / 2.0
        grid_points.append(center)
        
        # Add some random points to ensure coverage
        n_rand_init = 10
        random_pts = lo + (hi - lo) * rng.random((n_rand_init, self.dim))
        for x in random_pts:
            grid_points.append(x)

        for x in grid_points:
            if eval_count >= self.budget:
                break
            evaluate(x)

        # Helper: compute non-dominated set indices
        def non_dominated_indices(F):
            if len(F) == 0:
                return np.array([])
            F = np.asarray(F)
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

        # Phase 2: Targeted Local Refinement
        # Identify promising starting points from the initial set.
        # We will pick points that are non-dominated or have low scalarized values for a set of weight vectors.
        
        # Define a set of weight vectors for decomposition
        weights = []
        # Simple uniform weights for 3 objectives
        weights.append(np.array([1.0, 0.0, 0.0]))
        weights.append(np.array([0.0, 1.0, 0.0]))
        weights.append(np.array([0.0, 0.0, 1.0]))
        weights.append(np.array([0.5, 0.5, 0.0]))
        weights.append(np.array([0.5, 0.0, 0.5]))
        weights.append(np.array([0.0, 0.5, 0.5]))
        weights.append(np.array([1.0/3, 1.0/3, 1.0/3]))
        
        # Shuffle weights
        rng.shuffle(weights)

        # For each weight, find the best point in the current archive and refine it.
        # We limit the number of refinements to stay within budget and time.
        max_refinements = 15
        current_refinements = 0

        # Sort weights by priority? Or just iterate.
        for w in weights:
            if current_refinements >= max_refinements:
                break
            if eval_count >= self.budget - 20: # Leave some budget for final randoms if needed
                break
            if time.time() > deadline - 30:
                break

            if not all_F:
                continue

            F_arr = np.array(all_F)
            scores = F_arr @ w
            # Find the best point (lowest score)
            start_idx = np.argmin(scores)
            x0 = all_X[start_idx].copy()

            # Define the scalarized objective function using our recording wrapper
            def objective(x):
                x_clipped = np.clip(x, lo, hi)
                f = evaluate(x_clipped)
                if f is None:
                    return 1e10
                return np.dot(f, w)

            # Run Nelder-Mead
            # We use a smaller maxiter to allow multiple restarts
            maxiter = 20
            try:
                res = minimize(objective, x0, method='Nelder-Mead', 
                               options={'maxiter': maxiter, 'xatol': 1e-4, 'fatol': 1e-6})
                current_refinements += 1
            except Exception:
                pass

        # Phase 3: Fill remaining budget with random sampling if time allows
        while eval_count < self.budget:
            if time.time() > deadline - 5:
                break
            x_rand = lo + (hi - lo) * rng.random(self.dim)
            evaluate(x_rand)

        return all_X[-1] if all_X else np.zeros(self.dim)
# EVOLVE-BLOCK-END
