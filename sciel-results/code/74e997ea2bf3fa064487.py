"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Surrogate-assisted multi-objective optimizer with quadratic response surfaces and maximin sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim
        d = self.n_obj
        
        # Initial random sample to fit surrogate
        n_init = min(30, self.budget // 4)
        X_init = lo + (hi - lo) * np.random.random((n_init, n))
        F_init = np.array([problem(x) for x in X_init])
        used = n_init
        
        # Store all evaluated points
        X_all = X_init.copy()
        F_all = F_init.copy()
        
        # Helper: fit quadratic response surface for one objective
        def fit_quadratic(X, y):
            # Build design matrix: [1, x_i, x_i*x_j (i<=j)]
            m = len(X)
            n_features = 1 + n + n*(n+1)//2
            A = np.ones((m, n_features))
            A[:, 1:1+n] = X
            # Add quadratic terms
            quad_idx = 1 + n
            for i in range(n):
                for j in range(i, n):
                    A[:, quad_idx] = X[:, i] * X[:, j]
                    quad_idx += 1
            # Solve least squares
            try:
                coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
                return coeffs, A
            except:
                return None, None
        
        # Fit surrogates for each objective
        surrogates = []
        for k in range(d):
            coeffs, A_mat = fit_quadratic(X_all, F_all[:, k])
            surrogates.append((coeffs, A_mat))
        
        def predict_surrogate(x):
            """Predict all objectives using quadratic surrogates."""
            x = np.atleast_1d(x)
            pred = np.zeros(d)
            for k in range(d):
                coeffs, A_mat = surrogates[k]
                if coeffs is None:
                    pred[k] = np.median(F_all[:, k])
                else:
                    # Build design vector for this point
                    m = 1
                    n_features = 1 + n + n*(n+1)//2
                    a_vec = np.ones(n_features)
                    a_vec[1:1+n] = x
                    quad_idx = 1 + n
                    for i in range(n):
                        for j in range(i, n):
                            a_vec[quad_idx] = x[i] * x[j]
                            quad_idx += 1
                    pred[k] = a_vec @ coeffs
            return pred
        
        def normalize_objectives(F, ref):
            """Normalize objectives to [0,1] range using current min/max."""
            F_min = np.min(F, axis=0)
            F_max = np.max(F, axis=0)
            F_range = F_max - F_min
            F_range[F_range == 0] = 1.0
            return (F - F_min) / F_range
        
        def maximin_distance_select(F_norm, X_candidates, k_select=5):
            """Select k points from candidates that maximize minimum distance in normalized objective space."""
            n_cand = len(X_candidates)
            if n_cand <= k_select:
                return np.arange(n_cand)
            
            # Compute pairwise distances in normalized objective space
            dists = np.zeros((n_cand, n_cand))
            for i in range(n_cand):
                for j in range(i+1, n_cand):
                    dists[i, j] = dists[j, i] = np.linalg.norm(F_norm[i] - F_norm[j])
            
            # Greedy maximin selection
            selected = [np.argmax(np.min(dists, axis=1))]
            for _ in range(k_select - 1):
                if len(selected) >= n_cand:
                    break
                # For each unselected point, compute min distance to selected
                min_dists = np.full(n_cand, np.inf)
                for s in selected:
                    min_dists = np.minimum(min_dists, dists[s])
                min_dists[selected] = -np.inf
                new_sel = np.argmax(min_dists)
                selected.append(new_sel)
            
            return np.array(selected)
        
        # Iterative surrogate-assisted sampling
        max_iter = (self.budget - used) // 5  # Batch size of 5 per iteration
        batch_size = 5
        
        while used < self.budget:
            if time.time() > deadline - 10:
                break
            
            # Update surrogates with all data so far
            surrogates = []
            for k in range(d):
                coeffs, A_mat = fit_quadratic(X_all, F_all[:, k])
                surrogates.append((coeffs, A_mat))
            
            # Generate candidate points using different strategies
            n_cand = 50
            X_cands = np.zeros((n_cand, n))
            
            # Strategy 1: Random points
            n_rand = n_cand // 3
            X_cands[:n_rand] = lo + (hi - lo) * np.random.random((n_rand, n))
            
            # Strategy 2: Perturb existing good points (non-dominated)
            F_norm = normalize_objectives(F_all, np.array([3.0, 5.0, 7.0]))
            # Find non-dominated points
            non_dom_mask = np.ones(len(F_all), dtype=bool)
            for i in range(len(F_all)):
                for j in range(len(F_all)):
                    if i != j and np.all(F_all[j] <= F_all[i]) and np.any(F_all[j] < F_all[i]):
                        non_dom_mask[i] = False
                        break
            n_perturb = min(n_cand - n_rand, len(non_dom_mask))
            if n_perturb > 0:
                non_dom_idx = np.where(non_dom_mask)[0]
                pert_idx = np.random.choice(non_dom_idx, size=n_perturb, replace=False)
                X_cands[n_rand:n_rand+n_perturb] = X_all[pert_idx] + 0.1 * (hi - lo) * np.random.random((n_perturb, n))
                X_cands[n_rand:n_rand+n_perturb] = np.clip(X_cands[n_rand:n_rand+n_perturb], lo, hi)
            
            # Strategy 3: Use surrogate to find promising regions (minimize predicted weighted sum)
            n_surrogate = n_cand - n_rand - n_perturb
            if n_surrogate > 0:
                # Grid search in low-dimensional subspace
                n_grid = 5
                best_pred = np.inf
                best_x = None
                for _ in range(20):
                    x_try = lo + (hi - lo) * np.random.random(n)
                    pred = predict_surrogate(x_try)
                    # Use weighted sum with random weights
                    w = np.random.dirichlet(np.ones(d))
                    ws = w @ pred
                    if ws < best_pred:
                        best_pred = ws
                        best_x = x_try.copy()
                # Add multiple perturbations of the best surrogate point
                for i in range(n_surrogate):
                    X_cands[n_rand+n_perturb+i] = best_x + 0.05 * (hi - lo) * np.random.random(n)
                    X_cands[n_rand+n_perturb+i] = np.clip(X_cands[n_rand+n_perturb+i], lo, hi)
            
            # Predict objectives for candidates
            F_cand_pred = np.array([predict_surrogate(x) for x in X_cands])
            
            # Normalize for distance calculation
            F_cand_norm = normalize_objectives(F_cand_pred, np.array([3.0, 5.0, 7.0]))
            
            # Use maximin distance to select diverse candidates
            selected_idx = maximin_distance_select(F_cand_norm, X_cands, k_select=min(batch_size, len(X_cands)))
            
            # Evaluate selected candidates
            for idx in selected_idx:
                if used >= self.budget:
                    break
                x_new = X_cands[idx]
                f_new = problem(x_new)
                X_all = np.vstack([X_all, x_new])
                F_all = np.vstack([F_all, f_new])
                used += 1
        
        return X_all
# EVOLVE-BLOCK-END
