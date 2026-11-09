"""Multi-objective optimizer using Bayesian Optimization with Gaussian Process and EHVI."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        ref_point = np.array([3000.0, 0.06])
        
        # Initialize archive with random points
        # Use a smaller initial set to allow more BO iterations
        n_init = 10
        X = lo + (hi - lo) * np.random.rand(n_init, self.dim)
        F = np.array([problem(x) for x in X])
        
        # Filter out points not strictly better than reference point
        mask = np.all(F < ref_point, axis=1)
        X = X[mask]
        F = F[mask]
        
        if len(X) == 0:
            # Fallback if no points are better than ref (unlikely given random init and ref being large)
            # But for safety, keep the best ones or just return
            return X

        used = n_init
        alpha = 1.0  # Exploration-exploitation trade-off for EHVI

        # GP Kernel function (Matern 5/2)
        def matern52(X1, X2, length_scales, sigma):
            dist = np.sqrt(np.sum((X1[:, np.newaxis, :] - X2[np.newaxis, :, :])**2, axis=2))
            return sigma**2 * (1 + np.sqrt(5)*dist/length_scales) * np.exp(-np.sqrt(5)*dist/length_scales)

        def hypervolume_2d(points, ref):
            """Exact hypervolume for 2D points strictly better than ref."""
            if len(points) == 0:
                return 0.0
            # Sort by first objective (ascending)
            idx = np.argsort(points[:, 0])
            sorted_pts = points[idx]
            hv = 0.0
            prev_y = ref[1]
            for i in range(len(sorted_pts)):
                curr_x = sorted_pts[i, 0]
                curr_y = sorted_pts[i, 1]
                if curr_y >= ref[1]:
                    continue
                # Width in x is from curr_x to ref[0] (since we minimize, and curr_x < ref[0])
                # Actually, we need to be careful. 
                # Points are (f1, f2). Ref is (r1, r2).
                # Area is sum over points of (r1 - f1_i) * (r2 - f2_i) ... no, that's not right for non-dominated set.
                # Standard 2D HV:
                # Sort by f1 ascending.
                # Keep track of min f2 seen so far? No, we want the union of rectangles.
                # Since points are non-dominated (or we should prune dominated ones first), 
                # if sorted by f1 asc, f2 should be desc for non-dominated.
                # Let's just compute the union of rectangles [f1_i, r1] x [f2_i, r2].
                # Since we want exact HV, and n is small (archive size < 200), we can do a simple O(N^2) or sort-based method.
                # Sort by f1 asc.
                # The contribution of point i is (r1 - f1_i) * (min(f2_{i-1}, r2) - f2_i) ?? No.
                # Let's use the standard formula:
                # Sort by f1 ascending.
                # hv = 0
                # prev_y = r2
                # for p in sorted_points:
                #   if p[1] < prev_y:
                #     hv += (r1 - p[0]) * (prev_y - p[1])
                #     prev_y = p[1]
                #   # if p[1] >= prev_y, it is dominated in the y-direction by a previous point with smaller x, so contributes 0 to the "new" area.
                
                # Wait, the standard algorithm:
                # Sort points by x ascending.
                # Initialize max_y = -inf (or ref_y)
                # Actually, let's just iterate and keep the "best" y for a given x range.
                # Since we are minimizing, a point (x,y) is useful if x < r1 and y < r2.
                # Sort by x asc.
                # prev_y = r2
                # for x, y in sorted:
                #   if y < prev_y:
                #      area += (r1 - x) * (prev_y - y)
                #      prev_y = y
                #   # else: this point is dominated by a previous point with smaller x and smaller y (since sorted by x, previous x < current x, and prev_y <= current y implies previous point dominates or is equal in y). 
                #   # Actually if prev_y <= y, then previous point (x_prev, prev_y) has x_prev < x and prev_y <= y, so it dominates.
                
                # Let's implement this simple O(N log N) exact HV.
                pass # Placeholder, will implement below

        def calc_hv(F_archive, ref):
            if len(F_archive) == 0:
                return 0.0
            # Sort by first objective
            idx = np.argsort(F_archive[:, 0])
            sorted_F = F_archive[idx]
            hv = 0.0
            prev_y = ref[1]
            for i in range(len(sorted_F)):
                x, y = sorted_F[i]
                if y < prev_y:
                    hv += (ref[0] - x) * (prev_y - y)
                    prev_y = y
            return hv

        def ehvi_candidate(X_cand, F_archive, ref, mu, sigma):
            """
            Calculate EHVI for a single candidate point given GP mean and std.
            mu: (n_obj,) mean
            sigma: (n_obj,) std
            """
            # Sample from GP
            n_samples = 50
            samples = mu + sigma * np.random.randn(n_samples, self.n_obj)
            
            # Filter samples that are better than ref
            valid_mask = np.all(samples < ref, axis=1)
            if not np.any(valid_mask):
                return 0.0
            
            valid_samples = samples[valid_mask]
            
            # Combine with current archive
            # We need to calculate HV of (Archive U {Sample}) - HV(Archive)
            # Since n_samples is small, we can do this efficiently.
            # HV_current is known.
            # For each sample, calculate HV of Archive + Sample.
            
            # Optimization: Only samples that are non-dominated w.r.t Archive contribute significantly?
            # Actually, any sample better than ref can contribute if it's not dominated by Archive.
            
            # Let's just compute HV for each sample combined with Archive.
            # To speed up, we can check if sample is dominated by any point in Archive.
            # If dominated, contribution is 0.
            
            hv_current = calc_hv(F_archive, ref)
            
            total_ehvi = 0.0
            for s in valid_samples:
                # Check if s is dominated by any point in F_archive
                is_dominated = False
                for a in F_archive:
                    if np.all(a <= s) and np.any(a < s):
                        is_dominated = True
                        break
                if is_dominated:
                    continue
                
                # Calculate HV of F_archive + s
                new_F = np.vstack([F_archive, s.reshape(1, -1)])
                hv_new = calc_hv(new_F, ref)
                total_ehvi += max(0, hv_new - hv_current)
            
            return total_ehvi / n_samples

        # Main BO Loop
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Fit GP
            # Scale X to [0,1] for better GP conditioning
            X_scaled = (X - lo) / (hi - lo)
            
            # Simple GP implementation for small N
            # Use Matern 5/2 kernel
            # Hyperparameters: length_scales (dim,), sigma (n_obj,)
            # For simplicity and speed, use fixed reasonable hyperparameters or simple optimization
            # Given dim=4, n_obj=2, and N<200, a simple grid search or fixed params might work.
            # Let's use fixed length scales based on the range, e.g., 0.1 in scaled space.
            
            length_scales = np.ones(self.dim) * 0.2
            sigmas = np.ones(self.n_obj) * 0.5 # Initial std
            
            # Compute Kernel Matrix
            K = np.zeros((len(X), len(X)))
            for i in range(len(X)):
                for j in range(len(X)):
                    for k in range(self.n_obj):
                        dist = np.sqrt(np.sum((X_scaled[i] - X_scaled[j])**2))
                        K[i, j] += sigmas[k]**2 * (1 + np.sqrt(5)*dist/length_scales) * np.exp(-np.sqrt(5)*dist/length_scales)
                        
            # Add noise
            K += 1e-6 * np.eye(len(X))
            
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                # Fallback to random if GP fails
                X_new = lo + (hi - lo) * np.random.rand(self.dim)
                F_new = problem(X_new)
                used += 1
                # Update Archive
                # Remove dominated points from archive? 
                # Keep archive small by pruning dominated points
                # Simple pruning: if new point dominates an old one, remove old.
                # If old point dominates new one, don't add.
                # For now, just add and let HV calc handle it, or simple prune.
                
                # Prune Archive
                new_archive_X = [X_new]
                new_archive_F = [F_new]
                for i in range(len(X)):
                    # Check if F[i] is dominated by F_new
                    if np.all(F_new <= F[i]) and np.any(F_new < F[i]):
                        continue
                    # Check if F_new is dominated by F[i]
                    if np.all(F[i] <= F_new) and np.any(F[i] < F_new):
                        new_archive_X = []
                        new_archive_F = []
                        break
                    new_archive_X.append(X[i])
                    new_archive_F.append(F[i])
                
                if new_archive_X:
                    X = np.array(new_archive_X)
                    F = np.array(new_archive_F)
                else:
                    X = np.array([[X_new]])
                    F = np.array([[F_new]])
                continue

            # Predict at candidate points
            # We need to find the point that maximizes EHVI.
            # Use a local search or random sampling for the acquisition function.
            # Since dim=4, we can use a small population of candidates.
            
            n_cands = 50
            # Sample candidates from a mixture of uniform and around best points
            X_cands = lo + (hi - lo) * np.random.rand(n_cands, self.dim)
            
            # Add candidates around current best points
            best_idx = np.argmin(np.sum(F, axis=1))
            for _ in range(10):
                X_cands[len(X_cands)-1] = X[best_idx] + 0.1 * (hi - lo) * np.random.randn(self.dim)
                X_cands[len(X_cands)-1] = np.clip(X_cands[len(X_cands)-1], lo, hi)
            
            # Compute GP predictions for candidates
            # K_s = K(X_cands, X)
            # mu = K_s @ L^{-1} @ F
            # sigma = sqrt(1 - K_s @ L^{-1} @ L^{-T} @ K_s^T)
            
            best_ehvi = -np.inf
            best_x = X_cands[0]
            
            for x_c in X_cands:
                x_c_scaled = (x_c - lo) / (hi - lo)
                
                # Compute K_s
                K_s = np.zeros((len(X),))
                for i in range(len(X)):
                    dist = np.sqrt(np.sum((x_c_scaled - X_scaled[i])**2))
                    for k in range(self.n_obj):
                        K_s[i] += sigmas[k]**2 * (1 + np.sqrt(5)*dist/length_scales) * np.exp(-np.sqrt(5)*dist/length_scales)
                
                # Solve L v = K_s
                v = np.linalg.solve(L, K_s)
                
                # mu
                mu = v @ F
                
                # sigma
                # var = 1 - K_s @ v  (since K is symmetric)
                var = 1.0 - np.dot(K_s, v)
                if var < 0:
                    var = 0
                sigma = np.sqrt(var)
                
                # Calculate EHVI
                # Note: sigma is scalar for the whole vector? 
                # In multi-output GP, covariance between outputs matters.
                # For simplicity, we assume independent outputs with same kernel structure?
                # Or use a simple MC approach where we sample the whole vector.
                # My `ehvi_candidate` function assumes `mu` and `sigma` are vectors.
                # If we treat outputs as independent, sigma is a vector of same variances?
                # Let's assume independent for now, so sigma_vec = np.full(self.n_obj, np.sqrt(var))
                
                sigma_vec = np.full(self.n_obj, np.sqrt(var))
                
                ehvi_val = ehvi_candidate(x_c, F, ref_point, mu, sigma_vec)
                
                if ehvi_val > best_ehvi:
                    best_ehvi = ehvi_val
                    best_x = x_c
            
            # Evaluate best candidate
            F_new = problem(best_x)
            used += 1
            
            # Update Archive
            # Prune dominated points
            new_archive_X = [best_x]
            new_archive_F = [F_new]
            dominated_new = False
            for i in range(len(X)):
                # Check if F[i] is dominated by F_new
                if np.all(F_new <= F[i]) and np.any(F_new < F[i]):
                    continue
                # Check if F_new is dominated by F[i]
                if np.all(F[i] <= F_new) and np.any(F[i] < F_new):
                    dominated_new = True
                    break
                new_archive_X.append(X[i])
                new_archive_F.append(F[i])
            
            if not dominated_new:
                X = np.array(new_archive_X)
                F = np.array(new_archive_F)
            else:
                # If new point is dominated, we still "used" the budget, 
                # but we don't add it to the archive. 
                # The archive remains the same.
                pass

        return X
# EVOLVE-BLOCK-END
