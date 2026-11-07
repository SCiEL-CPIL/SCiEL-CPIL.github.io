"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-objective optimizer for RE21 using Multi-Start Nelder-Mead."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj

        # Track all evaluated points
        all_X = []
        all_F = []
        used = 0

        def evaluate(x):
            nonlocal used
            if used >= self.budget:
                return None
            if time.time() > deadline - 2.0:
                return None
            x = np.clip(x, lo, hi)
            f = problem(x)
            all_X.append(x.copy())
            all_F.append(np.array(f, dtype=np.float64))
            used += 1
            return f

        # Reference point for hypervolume
        ref = np.array([3000.0, 0.06])

        # Hypervolume calculation for 2 objectives
        def hypervolume(points, ref_point):
            """Calculate exact hypervolume for 2D case."""
            if len(points) == 0:
                return 0.0
            # Filter points strictly better than reference
            valid = np.all(points < ref_point, axis=1)
            if not np.any(valid):
                return 0.0
            points = points[valid]
            if len(points) == 0:
                return 0.0
            # Sort by first objective
            idx = np.argsort(points[:, 0])
            points = points[idx]
            hv = 0.0
            for i in range(len(points)):
                # Width in obj1
                if i == 0:
                    w1 = ref_point[0] - points[i, 0]
                else:
                    w1 = points[i, 0] - points[i-1, 0]
                # Height in obj2: min of current and all subsequent
                h2 = ref_point[1] - np.min(points[i:, 1])
                if h2 > 0:
                    hv += w1 * h2
            return hv

        # Initial grid: reference-point-aligned deterministic sampling
        # Focus on the asymmetric space where obj1 dominates hypervolume
        n_grid = 16
        # Grid in normalized space, biased toward low obj1 values
        grid_X = np.zeros((n_grid, dim))
        # 4x4 grid with bias toward lower values
        for i in range(4):
            for j in range(4):
                idx = i * 4 + j
                # Non-uniform spacing: denser near low end
                t1 = (i + 0.5) / 4.0
                t2 = (j + 0.5) / 4.0
                # Apply bias: compress toward low end
                t1 = t1 ** 1.5  # More points near 0
                t2 = t2 ** 1.2
                grid_X[idx] = lo + span * np.array([t1, t2, 0.5, 0.5])
        
        # Add some varied structured points
        grid_X[12] = lo + span * np.array([0.1, 0.1, 0.3, 0.7])
        grid_X[13] = lo + span * np.array([0.2, 0.3, 0.5, 0.5])
        grid_X[14] = lo + span * np.array([0.05, 0.5, 0.5, 0.5])
        grid_X[15] = lo + span * np.array([0.3, 0.05, 0.5, 0.5])

        # Evaluate grid
        for x in grid_X:
            f = evaluate(x)
            if f is None:
                break

        # Convert to arrays for vectorized operations
        all_X_arr = np.array(all_X) if all_X else np.zeros((0, dim))
        all_F_arr = np.array(all_F) if all_F else np.zeros((0, n_obj))

        # Multi-start Nelder-Mead local search
        # Select diverse starting points from evaluated points
        # Use non-dominated front + some random points
        def non_dominated_indices(F_mat):
            n = len(F_mat)
            if n == 0:
                return np.array([], dtype=int)
            le = np.all(F_mat[:, None, :] <= F_mat[None, :, :], axis=2)
            lt = np.any(F_mat[:, None, :] < F_mat[None, :, :], axis=2)
            dominated = np.any(le & lt, axis=0)
            return np.where(~dominated)[0]

        # Select starting points for local search
        front_idx = non_dominated_indices(all_F_arr)
        if len(front_idx) > 0:
            # Take up to 8 diverse points from front
            start_indices = front_idx[:8]
        else:
            start_indices = np.arange(min(8, len(all_X_arr)))

        # Add a few random points for diversity
        n_random_starts = 4
        for _ in range(n_random_starts):
            if used >= self.budget or time.time() > deadline - 10.0:
                break
            rand_idx = np.random.randint(len(all_X_arr))
            if rand_idx not in start_indices:
                start_indices = np.append(start_indices, rand_idx)

        # Nelder-Mead for each starting point
        # Use a weighted objective that emphasizes hypervolume contribution
        # Since obj1 dominates hypervolume (ref[0]=3000 >> ref[1]=0.06),
        # we weight obj1 more heavily in the scalarization
        
        n_lm_per_start = 3  # Number of local search iterations per start
        
        for start_idx in start_indices:
            if used >= self.budget or time.time() > deadline - 15.0:
                break
            
            x0 = all_X_arr[start_idx].copy()
            
            # Run Nelder-Mead with different scalarizations
            for lm_iter in range(n_lm_per_start):
                if used >= self.budget or time.time() > deadline - 5.0:
                    break
                
                # Different weightings for each iteration
                if lm_iter == 0:
                    # Emphasize obj1 (dominates hypervolume)
                    w1, w2 = 0.8, 0.2
                elif lm_iter == 1:
                    # Balanced
                    w1, w2 = 0.5, 0.5
                else:
                    # Emphasize obj2 (to find diverse solutions)
                    w1, w2 = 0.3, 0.7
                
                def objective(x):
                    if used >= self.budget:
                        return np.inf
                    x_clipped = np.clip(x, lo, hi)
                    f = problem(x_clipped)
                    all_X.append(x_clipped.copy())
                    all_F.append(np.array(f, dtype=np.float64))
                    used += 1
                    # Scalarized objective
                    return w1 * f[0] + w2 * f[1]
                
                try:
                    result = minimize(objective, x0, method='Nelder-Mead',
                                     options={'maxiter': 50, 'xatol': 1e-6, 'fatol': 1e-8})
                    if result.success and result.fun < np.inf:
                        x_best = np.clip(result.x, lo, hi)
                        # Evaluate the best point found
                        f_best = problem(x_best)
                        all_X.append(x_best.copy())
                        all_F.append(np.array(f_best, dtype=np.float64))
                        used += 1
                        x0 = x_best  # Use best as next start
                except Exception:
                    pass

        # Final: use remaining budget for random exploration if time allows
        while used < self.budget and time.time() < deadline - 3.0:
            x_rand = lo + span * np.random.random(dim)
            f = evaluate(x_rand)
            if f is None:
                break

        return np.array(all_X) if all_X else np.zeros((0, dim))
# EVOLVE-BLOCK-END
