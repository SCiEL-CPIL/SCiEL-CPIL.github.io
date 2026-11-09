"""Multi-objective optimizer for RE34: Bayesian Optimization with EHVI."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm
from scipy.spatial import Delaunay

class Optimizer:
    """
    Bayesian Optimizer using Gaussian Processes and Expected Hypervolume Improvement (EHVI).
    Designed for RE34 (5 variables, 3 objectives).
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        # Safety margin: Evaluator kills at 330s. We aim to finish well before.
        # GP fitting is O(N^3). With N=200, it's fast, but we need to be careful.
        deadline = start_time + 300 

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        ref = np.array([1705.0, 12.0, 0.3])
        
        # 1. Initialization: Sobol/Halton sequence for diversity
        # We use a small initial population to train the GP.
        # 5D space, 3 objectives. 
        # Let's use 20 initial points.
        n_init = 20
        X_init = self._generate_initial_points(n_init, lo, hi)
        F_init = np.array([problem(x) for x in X_init])
        
        X = X_init.copy()
        F = F_init.copy()
        n_evals = n_init
        
        # 2. Bayesian Optimization Loop
        # We will use a GP for each objective.
        # To speed up EHVI, we will sample candidates from a grid + random points.
        
        # Pre-compute constants for EHVI
        # The EHVI calculation requires integrating over the Pareto front of the GP predictions.
        # This is complex. A common approximation is to use a set of candidate points 
        # and estimate the HV improvement for each, then pick the best.
        
        # Candidate generation strategy:
        # - A grid of points (e.g., 5^5 = 3125 points) is too large for exact EHVI per step if done naively.
        # - We will use a "batch" of candidates generated via LHS (Latin Hypercube) + local perturbations of current front.
        
        while n_evals < self.budget:
            if time.time() > deadline:
                break
            
            # 1. Update GPs
            # We fit a separate GP for each objective.
            # To make this robust and fast, we use a simple RBF kernel with length-scale optimization 
            # or fixed length-scales. Given the small N, we can try to optimize hyperparameters 
            # but it might be slow. Let's start with fixed reasonable hyperparameters 
            # or a very fast optimization.
            
            # For speed, we will use a fixed kernel structure but normalize the inputs and outputs.
            # Input normalization: map [lo, hi] to [0, 1]
            # Output normalization: map observed min-max to [0, 1] or standardize.
            
            # We will implement a lightweight GP class inside the loop or use a helper.
            # Since we can't define classes easily inside __call__ without overhead, 
            # we will use helper functions.
            
            # Fit GPs
            gps = []
            for j in range(self.n_obj):
                gp = self._fit_gp(X, F[:, j], lo, hi)
                gps.append(gp)
            
            # 2. Generate Candidates
            # We want to explore the space efficiently.
            # Generate a set of candidates:
            # - 500 random points
            # - 100 points perturbed from the current Pareto front
            # - 100 points from a coarse grid
            
            n_candidates = 500
            candidates = np.random.uniform(lo, hi, size=(n_candidates, self.dim))
            
            # Add perturbations of current front
            front_idx = self._get_pareto_front_indices(F)
            if len(front_idx) > 0:
                front_points = X[front_idx]
                # Perturb each front point
                n_perturb_per_point = 5
                perturbations = []
                for p in front_points:
                    for _ in range(n_perturb_per_point):
                        noise = np.random.randn(self.dim) * 0.1 * (hi - lo)
                        new_p = p + noise
                        new_p = np.clip(new_p, lo, hi)
                        perturbations.append(new_p)
                candidates = np.vstack([candidates, np.array(perturbations)])
            
            # 3. Evaluate Acquisition Function (EHVI) for each candidate
            # Exact EHVI is expensive. We use a Monte Carlo approximation of EHVI.
            # EHVI(x) = E[ HV( (Y_front U {y(x)}) ) - HV(Y_front) ]
            
            # To speed this up, we pre-compute the current HV and the "envelope" of the front.
            # However, with 3 objectives, we can compute HV of a set of points quickly.
            
            # We will approximate the expectation by sampling from the GP posteriors.
            # n_samples = 50 for each candidate? That's 50 * 500 = 25,000 HV calculations.
            # HV calculation for 3D with ~20-50 points is O(N^2) or O(N log N).
            # 25,000 * 1000 ops = 25M ops. Might be too slow in Python.
            
            # Optimization: 
            # Instead of full MC for each candidate, we can use a cheaper acquisition function 
            # like Expected Improvement (EI) for a weighted sum, or a simplified EHVI.
            # Or, reduce candidates to 100 and samples to 20.
            
            # Let's try: 100 candidates, 20 MC samples.
            # 100 * 20 = 2000 HV calcs. Feasible.
            
            # Refine candidate set
            if len(candidates) > 100:
                # Randomly subsample to 100
                idx = np.random.choice(len(candidates), 100, replace=False)
                candidates = candidates[idx]
            
            acq_vals = np.zeros(len(candidates))
            n_mc_samples = 20
            
            # Current Pareto Front F
            F_front = F[front_idx]
            current_hv = self._hypervolume_3d(F_front, ref)
            
            for i, x_cand in enumerate(candidates):
                # Sample from GPs at x_cand
                # y_samples: (n_mc_samples, n_obj)
                y_samples = np.zeros((n_mc_samples, self.n_obj))
                for j in range(self.n_obj):
                    # GP predict mean and variance
                    mu, var = gps[j].predict(x_cand)
                    y_samples[:, j] = np.random.normal(mu, np.sqrt(var))
                
                # Calculate EHVI
                # For each sample, compute HV of (F_front U {y_sample}) - HV(F_front)
                # This is still slow if we do it for every sample.
                # We can approximate the improvement by checking if the sample is non-dominated 
                # and estimating the box volume, or just doing the HV calc.
                
                # Since n_front is small (~10-30), HV calc is fast.
                total_improvement = 0.0
                for k in range(n_mc_samples):
                    y_new = y_samples[k]
                    # Check if y_new is dominated by any point in F_front
                    # If dominated, improvement is 0.
                    is_dominated = False
                    for p in F_front:
                        if np.all(p <= y_new) and np.any(p < y_new):
                            is_dominated = True
                            break
                    if is_dominated:
                        continue
                    
                    # New front is F_front + y_new, but we must remove points dominated by y_new
                    # To be precise, we should compute HV of the union and subtract.
                    # But a faster approximation:
                    # Just compute HV of the set [F_front, y_new]
                    # This is O(N^2) where N is size of front.
                    combined = np.vstack([F_front, y_new])
                    hv_new = self._hypervolume_3d(combined, ref)
                    total_improvement += max(0, hv_new - current_hv)
                
                acq_vals[i] = total_improvement / n_mc_samples
            
            # Select best candidate
            best_idx = np.argmax(acq_vals)
            x_next = candidates[best_idx]
            
            # Evaluate
            f_next = problem(x_next)
            n_evals += 1
            
            # Update X and F
            X = np.vstack([X, x_next])
            F = np.vstack([F, f_next])
            
            # Optional: Prune dominated points to keep GP training set small?
            # GPs can handle dominated points, but they don't help the front.
            # However, they provide information about the landscape.
            # With budget 200, keeping all points is fine.
            
            # Check time
            if time.time() > deadline - 5:
                break

        return X

    def _generate_initial_points(self, n, lo, hi):
        # Use Halton sequence
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        bases = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        points = []
        for i in range(1, n + 1):
            p = [halton(i, b) for b in bases[:self.dim]]
            points.append(p)
        return lo + (hi - lo) * np.array(points)

    def _fit_gp(self, X, y, lo, hi):
        # Simple GP with RBF kernel
        # Normalize X to [0, 1]
        X_norm = (X - lo) / (hi - lo)
        # Normalize y to mean 0, std 1
        y_mean = np.mean(y)
        y_std = np.std(y)
        if y_std == 0:
            y_std = 1.0
        y_norm = (y - y_mean) / y_std
        
        # Kernel
        # We use a fixed length-scale for speed, or optimize it quickly.
        # Let's use a fixed length-scale of 0.1 in normalized space.
        l = 0.1
        sigma_f = 1.0
        sigma_n = 1e-4
        
        # Compute Kernel Matrix
        # K_ij = sigma_f^2 * exp(-0.5 * ||x_i - x_j||^2 / l^2) + sigma_n^2 * I
        X_diff = X_norm[:, np.newaxis, :] - X_norm[np.newaxis, :, :]
        sq_dist = np.sum(X_diff ** 2, axis=-1)
        K = sigma_f ** 2 * np.exp(-0.5 * sq_dist / l ** 2)
        K += sigma_n ** 2 * np.eye(len(X))
        
        # Cholesky decomposition
        L = np.linalg.cholesky(K)
        # Solve for alpha
        # alpha = L^-T L^-1 y
        v = np.linalg.solve(L, y_norm)
        alpha = np.linalg.solve(L.T, v)
        
        class SimpleGP:
            def __init__(self, X_norm, alpha, L, l, sigma_f, sigma_n, y_mean, y_std, lo, hi):
                self.X_norm = X_norm
                self.alpha = alpha
                self.L = L
                self.l = l
                self.sigma_f = sigma_f
                self.sigma_n = sigma_n
                self.y_mean = y_mean
                self.y_std = y_std
                self.lo = lo
                self.hi = hi
                
            def predict(self, x):
                # x is original scale
                x_norm = (x - self.lo) / (self.hi - self.lo)
                x_norm = x_norm.reshape(1, -1)
                
                # k(x, X)
                X_diff = x_norm - self.X_norm
                sq_dist = np.sum(X_diff ** 2, axis=-1)
                k = self.sigma_f ** 2 * np.exp(-0.5 * sq_dist / self.l ** 2)
                
                # Mean
                mu_norm = k @ self.alpha
                mu = mu_norm * self.y_std + self.y_mean
                
                # Variance
                # var = k(x,x) - k(x,X) @ K^-1 @ k(X,x)
                # K^-1 = L^-T L^-1
                # Let u = L^-1 k
                u = np.linalg.solve(self.L, k)
                var_norm = self.sigma_f ** 2 - np.sum(k * u)
                var = max(var_norm * self.y_std ** 2, 1e-8)
                
                return mu[0], var[0]
        
        return SimpleGP(X_norm, alpha, L, l, sigma_f, sigma_n, y_mean, y_std, lo, hi)

    def _get_pareto_front_indices(self, F):
        # Return indices of non-dominated points
        n = len(F)
        is_dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            if is_dominated[i]:
                continue
            for j in range(n):
                if i == j or is_dominated[j]:
                    continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    is_dominated[i] = True
                    break
        return np.where(~is_dominated)[0]

    def _hypervolume_3d(self, points, ref):
        # Exact 3D Hypervolume
        # Filter points dominated by ref
        valid = np.all(points < ref, axis=1)
        points = points[valid]
        if len(points) == 0:
            return 0.0
        
        # Remove dominated points
        front_idx = self._get_pareto_front_indices(points)
        points = points[front_idx]
        
        if len(points) == 0:
            return 0.0
        
        # Sort by first objective
        points = points[np.argsort(points[:, 0])]
        
        # Sweep algorithm
        # Volume = sum over i of (x_{i+1} - x_i) * HV_2D(points 1..i in y-z plane)
        # This is O(N^3) if we recompute HV_2D from scratch.
        # For N=200, N^3 is 8M, which is okay in C but slow in Python loops.
        # Let's use a simpler O(N^2) approximation or a more efficient sweep.
        
        # Efficient 3D HV:
        # Sort by x.
        # Maintain a list of "active" segments in y-z.
        # Actually, for small N, a simple inclusion-exclusion or grid might be faster in Python.
        # Let's stick to a simple O(N^2) loop for now, as N is small (< 50 in front usually).
        
        # If N is large, this might be slow.
        # Let's implement a fast O(N^2) for 3D:
        # HV = sum_{i} (x_{i+1} - x_i) * (y_max_i - y_i) * (z_max_i - z_i) ? No.
        
        # Correct simple method:
        # For each point i, it contributes a box [x_i, x_{i+1}] x [y_i, y_max] x [z_i, z_max]? No.
        
        # Let's use the standard "layer" method:
        # Sort by x.
        # For each slice [x_i, x_{i+1}], the dominated region in y-z is the union of rectangles 
        # corresponding to points with index <= i.
        # We can maintain the 2D HV of the y-z projections of the active points.
        
        # Given the small size of the front (usually < 30), we can just compute the 2D HV 
        # for each step.
        # 2D HV is O(N log N).
        # Total O(N^2 log N).
        
        total_hv = 0.0
        n_pts = len(points)
        
        # Add a dummy point at ref
        # x_{n+1} = ref[0]
        
        # We need the 2D HV of the set of points (y_j, z_j) for j=0..i
        # where x_j <= x_current.
        
        # To avoid recomputing from scratch, we can just recompute. N is small.
        
        active_points = []
        for i in range(n_pts):
            active_points.append(points[i, 1:3])
            # Compute 2D HV of active_points with ref (ref[1], ref[2])
            hv_2d = self._hypervolume_2d(active_points, ref[1:3])
            dx = points[i+1, 0] if i + 1 < n_pts else ref[0]
            dx = max(0, dx - points[i, 0])
            total_hv += dx * hv_2d
            
        return total_hv

    def _hypervolume_2d(self, points, ref):
        # points: Mx2 array
        # ref: 2D array
        # Filter
        valid = np.all(points < ref, axis=1)
        points = points[valid]
        if len(points) == 0:
            return 0.0
        
        # Sort by x
        points = points[np.argsort(points[:, 0])]
        
        total = 0.0
        max_y = 0.0 # Actually, we need the max y of points that are "active"
        # Wait, the standard 2D HV sweep:
        # Sort by x.
        # Keep track of the maximum y seen so far? No, that's for union of boxes from origin.
        # Here we have a reference point.
        
        # The dominated region is the union of boxes [p_i, ref].
        # Sort by x.
        # Let x_{n+1} = ref[0].
        # For each i, the slice [x_i, x_{i+1}] has height determined by the max y of points j <= i?
        # No, it's the max y of points j such that x_j <= x.
        # Since we sort by x, for slice i, the max y is the max of y_j for j=0..i.
        
        max_y_so_far = -np.inf
        for i in range(len(points)):
            if points[i, 1] > max_y_so_far:
                max_y_so_far = points[i, 1]
            
            dx = (points[i+1, 0] if i + 1 < len(points) else ref[0]) - points[i, 0]
            dy = ref[1] - max_y_so_far
            if dy > 0 and dx > 0:
                total += dx * dy
                
        return total
# EVOLVE-BLOCK-END
