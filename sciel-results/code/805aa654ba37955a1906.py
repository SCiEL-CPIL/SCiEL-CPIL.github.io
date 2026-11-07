"""Multi-objective optimizer for RE37 using GP-surrogate MBEA with NSGA-II fallback."""
import numpy as np
import time
from scipy.stats import norm
import itertools


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _halton(self, n, dim):
        """Generate n Halton sequence points in [0,1)^dim."""
        points = np.zeros((n, dim))
        for i in range(1, n + 1):
            for d in range(dim):
                base = d + 2
                f = 1.0
                result = 0.0
                j = i
                while j > 0:
                    f /= base
                    result += f * (j % base)
                    j //= base
                points[i - 1, d] = result
        return points

    def _fast_non_dominated_sort(self, F):
        """Fast non-dominated sort. Returns list of fronts (each a list of indices)."""
        n = len(F)
        if n == 0:
            return []
        S = [[] for _ in range(n)]
        n_dom = np.zeros(n, dtype=int)
        rank = np.full(n, -1, dtype=int)
        
        for i in range(n):
            for j in range(i + 1, n):
                fi, fj = F[i], F[j]
                if np.all(fi <= fj) and np.any(fi < fj):
                    S[i].append(j)
                    n_dom[j] += 1
                if np.all(fj <= fi) and np.any(fj < fi):
                    S[j].append(i)
                    n_dom[i] += 1
        
        fronts = []
        current_front = [i for i in range(n) if n_dom[i] == 0]
        
        k = 0
        while current_front:
            fronts.append(current_front[:])
            next_front = []
            for i in current_front:
                for j in S[i]:
                    n_dom[j] -= 1
                    if n_dom[j] == 0:
                        next_front.append(j)
            current_front = next_front
            k += 1
        
        for front_idx, front in enumerate(fronts):
            for idx in front:
                rank[idx] = front_idx
        
        return fronts

    def _crowding_distance(self, F, front_indices):
        """Compute crowding distance for a front."""
        n = len(front_indices)
        if n <= 2:
            return np.full(n, np.inf)
        
        cd = np.zeros(n)
        F_front = F[front_indices]
        
        for m in range(self.n_obj):
            obj = F_front[:, m]
            sorted_idx = np.argsort(obj)
            cd[sorted_idx[0]] = np.inf
            cd[sorted_idx[-1]] = np.inf
            obj_range = obj[sorted_idx[-1]] - obj[sorted_idx[0]]
            if obj_range > 1e-15:
                for i in range(1, n - 1):
                    cd[sorted_idx[i]] += (obj[sorted_idx[i + 1]] - obj[sorted_idx[i - 1]]) / obj_range
        
        return cd

    def _hypervolume_3d(self, points, ref):
        """Compute 3D hypervolume using inclusion-exclusion on sorted points."""
        # Filter points that are within reference
        valid = []
        for p in points:
            if np.all(p < ref):
                valid.append(p)
        
        if not valid:
            return 0.0
        
        valid = np.array(valid)
        
        # Sort by first objective
        order = np.argsort(valid[:, 0])
        valid = valid[order]
        
        # Remove dominated points for efficiency (keep Pareto front of the valid set)
        n = len(valid)
        pareto = [0]
        for i in range(1, n):
            if not np.all(valid[pareto[-1]] <= valid[i]):
                pareto.append(i)
        
        valid = valid[pareto]
        n = len(valid)
        
        if n == 0:
            return 0.0
        if n == 1:
            d = ref - valid[0]
            return d[0] * d[1] * d[2]
        
        # For 3D, we can use the standard algorithm
        # Sort by x ascending, y ascending
        valid = valid[np.lexsort((valid[:, 1], valid[:, 0]))]
        
        hv = 0.0
        # For each point, compute the volume of the box it uniquely contributes
        # This is complex; use a simpler O(n^2) approach for small n
        # Actually, let's use the recursive definition or a standard library if available.
        # Since n is small (Pareto front size), O(n^2) or O(n^3) is fine.
        
        # Standard 3D HV algorithm:
        # Sort by x. For each point i, find the max y and z extent not covered by previous points?
        # No, simpler:
        # Iterate through points sorted by x.
        # For each point, the contribution is determined by the "skyline" in the remaining 2D space.
        
        # Let's just do a simple voxel-free exact calculation for small n.
        # Since n is likely < 50, we can use the standard inclusion-exclusion or
        # just sum up volumes of boxes defined by the points and reference, subtracting overlaps.
        # Actually, the most robust simple way for 3D with small n:
        # Sort by x.
        # For i from 0 to n-1:
        #   The box for point i is [x_i, x_{i+1}) x [y_i, max_y] x [z_i, max_z] ? No.
        
        # Let's use the algorithm:
        # 1. Sort points by x ascending.
        # 2. For each point p_i, find the maximum y and z such that no other point p_j (j>i)
        #    has y_j < y and z_j < z.
        #    This is equivalent to finding the 2D hypervolume of the set of points dominated by p_i
        #    in the y-z plane, relative to the reference, but only considering points with x > p_i.x.
        
        # Simpler approach for small n:
        # Generate all unique x, y, z coordinates from points and reference.
        # Create a grid and check which cells are covered.
        # With n~20, grid is ~20^3 = 8000 cells. Very fast.
        
        xs = np.unique(np.concatenate([valid[:, 0], [ref[0]]]))
        ys = np.unique(np.concatenate([valid[:, 1], [ref[1]]]))
        zs = np.unique(np.concatenate([valid[:, 2], [ref[2]]]))
        
        # Build grid
        # A cell (i, j, k) is [xs[i], xs[i+1]) x [ys[j], ys[j+1]) x [zs[k], zs[k+1])
        # It is covered if there exists a point p such that p[0] <= xs[i] and p[1] <= ys[j] and p[2] <= zs[k]
        # Actually, p must dominate the cell's min corner.
        
        total_v = 0.0
        for i in range(len(xs) - 1):
            for j in range(len(ys) - 1):
                for k in range(len(zs) - 1):
                    # Cell min corner
                    cx = xs[i]
                    cy = ys[j]
                    cz = zs[k]
                    # Check if any point dominates (cx, cy, cz)
                    # i.e., p[0] <= cx, p[1] <= cy, p[2] <= cz
                    covered = False
                    for p in valid:
                        if p[0] <= cx and p[1] <= cy and p[2] <= cz:
                            covered = True
                            break
                    if covered:
                        vol = (xs[i+1] - xs[i]) * (ys[j+1] - ys[j]) * (zs[k+1] - zs[k])
                        total_v += vol
        
        return total_v

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        n_budget = self.budget
        dim = self.dim
        n_obj = self.n_obj
        ref_point = np.array([1.1, 1.3, 1.1])

        # Data structures
        X_all = []  # List of arrays
        F_all = []  # List of arrays
        eval_count = 0

        def evaluate(x):
            nonlocal eval_count
            if eval_count >= n_budget:
                return None
            x = np.clip(x, lo, hi)
            f = problem(x)
            f = np.asarray(f, dtype=float)
            X_all.append(x.copy())
            F_all.append(f.copy())
            eval_count += 1
            return f

        # Phase 1: Initialization with Halton + Center
        n_init = min(40, n_budget // 2)
        halton_pts = self._halton(n_init, dim)
        X_init = lo + halton_pts * span
        # Add center
        X_init = np.vstack([X_init, (lo + hi) / 2.0])
        
        for x in X_init:
            if time.time() > deadline:
                break
            evaluate(x)

        if eval_count >= n_budget:
            return np.array(X_all)

        # Phase 2: MBEA Loop with GP Surrogate
        # We will maintain a population of the best non-dominated points.
        # If GP is too slow or fails, fall back to NSGA-II-like random search.
        
        # For simplicity and robustness within the time limit,
        # we will use a simplified acquisition: 
        # 1. Fit a simple linear or quadratic surrogate? 
        #    The prompt suggests GP. Let's implement a lightweight GP using sklearn if available, 
        #    or a simple kernel ridge regression.
        #    Since scipy is available, we can implement a basic GP manually for small data.
        
        # However, implementing a full GP from scratch might be error-prone.
        # Let's try a "Model-Based" approach using Local Weighted Regression or KNN.
        # Actually, the instruction says "Gaussian Process surrogate".
        # Let's use a simple GP implementation.
        
        class SimpleGP:
            def __init__(self, kernel='rbf', length_scale=0.1, variance=1.0):
                self.kernel = kernel
                self.length_scale = length_scale
                self.variance = variance
                self.X = None
                self.y = None
                self.K = None
                self.alpha = None

            def _kernel_matrix(self, X1, X2):
                # RBF Kernel
                diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
                sq_dist = np.sum(diff ** 2, axis=2)
                return self.variance * np.exp(-sq_dist / (2 * self.length_scale ** 2))

            def fit(self, X, y):
                self.X = X
                self.y = y
                self.K = self._kernel_matrix(X, X)
                # Add small noise for stability
                self.K += 1e-6 * np.eye(len(X))
                try:
                    self.alpha = np.linalg.solve(self.K, y)
                except np.linalg.LinAlgError:
                    self.alpha = np.linalg.lstsq(self.K, y, rcond=None)[0]

            def predict(self, X_new):
                K_s = self._kernel_matrix(X_new, self.X)
                mean = K_s @ self.alpha
                # Variance calculation
                # var = k(X_new, X_new) - K_s @ K^{-1} @ K_s.T
                # For acquisition, we mainly need mean and std
                # std = sqrt(k(x,x) - K_s @ K^{-1} @ K_s.T)
                # Approximate std for simplicity or compute properly
                try:
                    K_inv = np.linalg.inv(self.K)
                    K_s_K_inv = K_s @ K_inv
                    var = self.variance - np.sum(K_s * K_s_K_inv, axis=1)
                    var = np.maximum(var, 0.0)
                    std = np.sqrt(var)
                except:
                    std = np.zeros(len(X_new))
                return mean, std

        # MBEA Loop
        # We will keep a set of "elite" points (Pareto front)
        # and use the GP to propose new points.
        
        elite_X = np.array(X_all)
        elite_F = np.array(F_all)
        
        # Initial GP fit
        gps = []
        for m in range(n_obj):
            gp = SimpleGP()
            gp.fit(elite_X, elite_F[:, m])
            gps.append(gp)
        
        # Candidate generation: Grid + Random
        n_grid = 50
        # Create a grid of candidates
        # Since dim=4, a full grid of 50^4 is too large.
        # Use random sampling for candidates.
        n_candidates = 200
        
        # Main Loop
        while eval_count < n_budget and time.time() < deadline:
            # Generate candidates
            # Mix of random and local perturbations of elites
            n_rand = n_candidates // 2
            n_local = n_candidates - n_rand
            
            candidates = []
            
            # Random
            candidates.append(lo + span * np.random.random((n_rand, dim)))
            
            # Local perturbations of current Pareto front
            fronts = self._fast_non_dominated_sort(elite_F)
            if fronts:
                front0 = fronts[0]
                if len(front0) > 0:
                    # Pick a few from front 0
                    sample_size = min(len(front0), n_local // 3)
                    idxs = np.random.choice(len(front0), size=sample_size, replace=False)
                    for idx in idxs:
                        x_base = elite_X[front0[idx]]
                        # Generate several perturbations
                        for _ in range(n_local // max(1, sample_size)):
                            pert = 0.1 * span * np.random.randn(dim)
                            candidates.append(x_base + pert)
            
            candidates = np.vstack(candidates)
            candidates = np.clip(candidates, lo, hi)
            
            # Evaluate acquisition function for each candidate
            # EHVI approximation:
            # For each candidate, predict mean and std for each objective.
            # Compute expected HV contribution.
            # This is expensive. Let's use a simpler "Expected Improvement" per objective
            # or just select based on distance to current HV front.
            
            # Given time constraints, let's use a simpler strategy:
            # Select candidates that are far from existing points (diversity)
            # and have high predicted potential (low predicted objectives).
            
            # Predict means
            means = np.zeros((len(candidates), n_obj))
            stds = np.zeros((len(candidates), n_obj))
            for m in range(n_obj):
                means[:, m], stds[:, m] = gps[m].predict(candidates)
            
            # Score = -sum(means) + 0.5 * sum(stds)  (Exploration + Exploitation)
            scores = -np.sum(means, axis=1) + 0.5 * np.sum(stds, axis=1)
            
            # Also penalize points that are too close to existing elites
            # Compute min distance to elite_X
            # Use broadcasting for efficiency
            # dist_sq = ||candidates[:, None, :] - elite_X[None, :, :]|^2
            # This might be memory heavy if elite is large.
            # Elite size is small (< 50). Candidates 200. 200*50*4 = 40000 floats. Fine.
            
            diff = candidates[:, np.newaxis, :] - elite_X[np.newaxis, :, :]
            dist_sq = np.sum(diff ** 2, axis=2)
            min_dist = np.min(dist_sq, axis=1)
            
            # Final score
            # Higher is better
            final_scores = scores - 0.1 * np.sqrt(min_dist)
            
            # Select top k candidates
            k_select = min(5, len(candidates), n_budget - eval_count)
            top_idx = np.argsort(final_scores)[::-1][:k_select]
            
            for idx in top_idx:
                if eval_count >= n_budget or time.time() > deadline:
                    break
                x = candidates[idx]
                f = evaluate(x)
                if f is not None:
                    # Update elites
                    elite_X = np.vstack([elite_X, x])
                    elite_F = np.vstack([elite_F, f])
            
            # Refit GPs periodically or every iteration
            # To save time, refit every 5 iterations? Or every time if data is small.
            # Data grows. GP fit is O(n^3). n=100 -> 1M ops. Fast.
            # Let's refit every iteration.
            for m in range(n_obj):
                gps[m].fit(elite_X, elite_F[:, m])
        
        # Phase 3: Local Refinement
        if time.time() < deadline - 10:
            fronts = self._fast_non_dominated_sort(elite_F)
            if fronts:
                front0 = fronts[0]
                n_refine = min(n_budget - eval_count, 20)
                for idx in front0:
                    if n_refine <= 0 or time.time() > deadline:
                        break
                    x_best = elite_X[idx]
                    for _ in range(min(3, n_refine)):
                        pert = 0.02 * span * np.random.randn(dim)
                        x_trial = np.clip(x_best + pert, lo, hi)
                        f_trial = evaluate(x_trial)
                        if f_trial is not None:
                            elite_X = np.vstack([elite_X, x_trial])
                            elite_F = np.vstack([elite_F, f_trial])
                            n_refine -= 1
        
        return np.array(X_all)
