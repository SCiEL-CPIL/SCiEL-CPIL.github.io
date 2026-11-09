"""Multi-objective optimizer for RE34 using Bayesian Optimization with EHVI."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


class Optimizer:
    """Bayesian Optimization with EHVI for RE34."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 285.0  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj

        # Reference point for hypervolume (from task contract)
        ref = np.array([1705.0, 12.0, 0.3])

        # Initialize with a diverse set of points
        n_init = min(30, self.budget)
        X = np.empty((n_init, dim))
        
        # Structured points for initial coverage
        structured = []
        structured.append(lo.copy())
        structured.append(hi.copy())
        mid = (lo + hi) / 2.0
        structured.append(mid.copy())
        for i in range(dim):
            pt = mid.copy()
            pt[i] = lo[i]
            structured.append(pt.copy())
            pt = mid.copy()
            pt[i] = hi[i]
            structured.append(pt.copy())
        
        structured = np.array(structured[:min(len(structured), n_init)])
        X[:len(structured)] = structured
        
        # Latin hypercube for remaining initial points
        if n_init > len(structured):
            n_lhc = n_init - len(structured)
            lhc = np.zeros((n_lhc, dim))
            for d in range(dim):
                perm = np.random.permutation(n_lhc)
                for i in range(n_lhc):
                    lhc[i, d] = (perm[i] + np.random.random()) / n_lhc
            X[len(structured):] = lo + (hi - lo) * lhc

        # Evaluate initial population
        F = np.array([problem(x) for x in X])
        evaluated = n_init

        # GP kernel: Matern 5/2
        def matern52_kernel(X1, X2, length_scales):
            """Compute Matern 5/2 kernel matrix between X1 and X2."""
            # X1: (n1, dim), X2: (n2, dim)
            # length_scales: (dim,)
            diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
            dist_sq = np.sum(diff ** 2 / length_scales[np.newaxis, np.newaxis, :] ** 2, axis=2)
            dist = np.sqrt(dist_sq + 1e-10)
            sqrt3 = np.sqrt(3.0)
            kernel = (1.0 + sqrt3 * dist + 3.0 * dist ** 2 / 3.0) * np.exp(-sqrt3 * dist)
            return kernel

        def gp_predict(X_train, y_train, X_test, length_scales, noise_var):
            """Gaussian Process prediction for a single objective."""
            n = len(X_train)
            K = matern52_kernel(X_train, X_train, length_scales) + noise_var * np.eye(n)
            Ktest = matern52_kernel(X_train, X_test, length_scales)
            
            # Cholesky decomposition for stability
            try:
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
            except np.linalg.LinAlgError:
                K += 1e-6 * np.eye(n)
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
            
            mu = Ktest.T @ alpha
            v = np.linalg.solve(L, Ktest)
            var = 1.0 - np.sum(v ** 2, axis=0)  # Using standard deviation of 1 for simplicity
            var = np.maximum(var, 1e-10)
            
            return mu, np.sqrt(var)

        def compute_hypervolume_3d(points, ref_pt):
            """Compute exact 3D hypervolume contribution of a set of points."""
            # Filter points strictly better than reference
            valid = np.all(points < ref_pt, axis=1)
            points = points[valid]
            if len(points) == 0:
                return 0.0
            
            n = len(points)
            # Sort by first objective
            order = np.argsort(points[:, 0])
            points = points[order]
            
            # Inclusive hypervolume computation for 3D
            # Use the standard 3D HV algorithm
            hv = 0.0
            for i in range(n):
                # Volume dominated by point i but not by any point j with j < i
                # This is complex; use a simpler approximation for speed
                # Actually, let's use a more efficient approach
                pass
            
            # For efficiency, use a grid-based approximation or the standard algorithm
            # Standard 3D HV:
            # Sort by f1
            # For each point, compute the volume it uniquely dominates
            # This requires careful handling
            
            # Alternative: use the fact that for small n, we can compute exactly
            # by decomposing into boxes
            # For 3D, the exact HV can be computed in O(n^2 log n)
            
            # Let's implement the standard 3D HV algorithm
            # Sort by first objective
            idx = np.argsort(points[:, 0])
            pts = points[idx]
            
            hv = 0.0

            # For now, use a Monte Carlo estimate for speed
            n_samples = 10000
            # Sample in the region [min(points), ref_pt]
            f_min = np.min(points, axis=0)
            f_max = ref_pt
            samples = np.random.uniform(f_min, f_max, size=(n_samples, 3))
            
            # Count how many samples are dominated by at least one point
            dominated = np.zeros(n_samples, dtype=bool)
            for p in pts:
                dominated |= np.all(samples > p, axis=1)
            
            volume = np.prod(f_max - f_min)
            hv = volume * np.mean(dominated)
            
            return hv

        def ehvi_acquisition(X_cand, F_train, ref_pt, length_scales, noise_var, n_mc=500):
            """Compute Expected Hypervolume Improvement for candidate points."""
            n_cand = len(X_cand)
            n_train = len(F_train)
            
            # Predict mean and variance for each objective at candidate points
            mus = np.zeros((n_cand, n_obj))
            vars_ = np.zeros((n_cand, n_obj))
            
            for obj in range(n_obj):
                mu, std = gp_predict(F_train[:, obj].reshape(-1, 1), 
                                     F_train[:, obj], 
                                     X_cand, 
                                     length_scales, 
                                     noise_var)
                mus[:, obj] = mu.flatten()
                vars_[:, obj] = std.flatten() ** 2
            
            # Compute current HV (approximate)
            # For EHVI, we need to estimate the improvement over current HV
            # Use Monte Carlo: sample from the posterior and compute HV increment
            
            ehvi = np.zeros(n_cand)
            
            # Current Pareto front points
            # Filter to only points better than reference
            valid_train = np.all(F_train < ref_pt, axis=1)
            F_valid = F_train[valid_train]
            
            if len(F_valid) == 0:
                # If no valid points, EHVI is just the expected volume
                for i in range(n_cand):
                    # Sample from posterior
                    samples = np.random.normal(mus[i], np.sqrt(vars_[i]), size=(n_mc, n_obj))
                    samples = np.maximum(samples, 0)  # Objectives are positive
                    valid_samples = np.all(samples < ref_pt, axis=1)
                    if np.any(valid_samples):
                        # Compute HV of samples + current front
                        # Simplified: just the volume dominated by this point
                        vol = np.prod(np.maximum(ref_pt - mus[i], 0))
                        ehvi[i] = vol
                return ehvi
            
            # Monte Carlo EHVI
            for i in range(n_cand):
                total_improvement = 0.0
                for _ in range(n_mc):
                    # Sample objective values
                    sample_obj = np.random.normal(mus[i], np.sqrt(vars_[i]))
                    sample_obj = np.maximum(sample_obj, 0)
                    
                    if not np.all(sample_obj < ref_pt):
                        continue
                    
                    # New set of points
                    new_points = np.vstack([F_valid, sample_obj.reshape(1, -1)])
                    
                    # Compute HV of new set and old set
                    # This is expensive; use approximation
                    # Approximate HV improvement as the volume dominated by sample_obj 
                    # that is not dominated by any point in F_valid
                    
                    # Check if sample_obj is dominated by any existing point
                    dominated = False
                    for p in F_valid:
                        if np.all(p <= sample_obj) and np.any(p < sample_obj):
                            dominated = True
                            break
                    
                    if dominated:
                        continue
                    
                    # Compute volume uniquely dominated by sample_obj
                    # This is the volume of the box [sample_obj, ref_pt] minus
                    # the volume already covered by other points
                    # Approximate by the full box volume (overestimates)
                    vol = np.prod(np.maximum(ref_pt - sample_obj, 0))
                    total_improvement += vol
                
                ehvi[i] = total_improvement / n_mc
            
            return ehvi

        # Adaptive length scales
        length_scales = (hi - lo) * 0.5
        noise_var = 1e-4

        # Main BO loop
        batch_size = 5
        while evaluated + batch_size <= self.budget:
            if time.time() > deadline:
                break
            
            # Generate candidate points using Latin hypercube
            n_cand = 200
            X_cand = np.random.uniform(lo, hi, size=(n_cand, dim))
            
            # Also add some local perturbations around existing points
            n_local = min(100, n_cand // 2)
            local_idx = np.random.choice(len(X), size=n_local, replace=False)
            X_local = X[local_idx] + np.random.normal(0, 0.1 * (hi - lo), size=(n_local, dim))
            X_local = np.clip(X_local, lo, hi)
            X_cand = np.vstack([X_cand[:n_cand - n_local], X_local])
            
            # Compute EHVI for each candidate
            ehvi_vals = ehvi_acquisition(X_cand, F, ref, length_scales, noise_var, n_mc=100)
            
            # Select top batch_size candidates
            top_idx = np.argsort(ehvi_vals)[-batch_size:]
            X_new = X_cand[top_idx]
            
            # Evaluate new points
            F_new = np.array([problem(x) for x in X_new])
            evaluated += batch_size
            
            # Update training set
            X = np.vstack([X, X_new])
            F = np.vstack([F, F_new])
            
            # Adaptively adjust length scales based on observed variation
            # Simple heuristic: if HV is improving, keep length scales; else, increase
            # For now, keep fixed
            
            # Check time
            if time.time() > deadline - 10:
                break
        
        return X
# EVOLVE-BLOCK-END
