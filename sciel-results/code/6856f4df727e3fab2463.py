"""Sobol-based adaptive search with k-NN model and local refinement for BBOB f21."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc


class Optimizer:
    """Sobol-based adaptive search with k-NN model and local refinement for BBOB f21."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        calls_made = 0
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)

        best_x = None
        best_f = np.inf

        def evaluate(x):
            nonlocal best_x, best_f, calls_made
            if calls_made >= budget:
                return np.inf
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            f = float(func(x))
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Generate Sobol sequence
        # Use a large pool to allow for adaptive selection
        n_sobol = 2000
        sobol = qmc.Sobol(d=dim, scramble=True)
        # generate 2000 points in [0,1]^dim
        points_unit = sobol.random(n_sobol)
        # map to [-5, 5]^dim
        X_pool = points_unit * 10.0 - 5.0
        
        # Shuffle the pool to ensure we don't just pick the first ones if the model is bad?
        # No, the model should guide us. But we need to track which are evaluated.
        # We'll maintain a list of indices in the pool that are un-evaluated.
        
        # Initialization: Evaluate first 10 points (or random subset of pool)
        # Let's take the first 10 from the pool
        X_obs = []
        f_obs = []
        
        n_init = 10
        for i in range(n_init):
            if calls_made >= budget or time.time() > deadline:
                break
            x = X_pool[i]
            f = evaluate(x)
            X_obs.append(x)
            f_obs.append(f)
            
        X_obs = np.array(X_obs)
        f_obs = np.array(f_obs)
        
        # Track un-evaluated indices in X_pool
        # We used 0 to n_init-1
        un_evaluated_indices = list(range(n_init, n_sobol))
        
        # k-NN Model
        k_neighbors = 10
        
        def predict_knn(X_new, X_train, y_train):
            """
            Predict mean and std using k-NN with distance weighting.
            """
            if len(X_train) == 0:
                return np.zeros(X_new.shape[0]), np.ones(X_new.shape[0])
            
            # Calculate distances from each new point to all training points
            # X_new: (n_new, dim), X_train: (n_train, dim)
            # We want (n_new, n_train) distances
            # For memory efficiency, we process in batches if n_new is large, but here n_new is small (1 per iter)
            # Actually, we will call this for all un-evaluated points to pick the best one.
            # n_un_evaluated can be ~2000. n_train can be ~100.
            # 2000 * 100 * 5 ops is fine.
            
            dists = np.linalg.norm(X_new[:, None, :] - X_train[None, :, :], axis=2)
            
            # Get k nearest neighbors
            k = min(k_neighbors, len(X_train))
            # Get indices of k smallest distances for each point
            # argsort is expensive for 2000*100, but acceptable.
            # Alternatively, use partition.
            # Let's use argsort for simplicity, it's fast enough for these sizes.
            idx = np.argpartition(dists, k, axis=1)[:, :k]
            
            # Get distances for these neighbors
            # We need to gather the correct distances
            # dists[i, idx[i]]
            row_indices = np.arange(X_new.shape[0])[:, None]
            neighbor_dists = dists[row_indices, idx]
            
            # Get neighbor values
            neighbor_vals = y_train[idx]
            
            # Weight by inverse distance (with epsilon to avoid div by zero)
            weights = 1.0 / (neighbor_dists + 1e-6)
            weights /= np.sum(weights, axis=1, keepdims=True)
            
            # Weighted mean
            mean = np.sum(weights * neighbor_vals, axis=1)
            
            # Weighted std (approx)
            # Var = sum(w * (x - mean)^2)
            var = np.sum(weights * (neighbor_vals - mean[:, None]) ** 2, axis=1)
            std = np.sqrt(var)
            
            return mean, std

        # Phase 2: Adaptive selection from Sobol pool
        # Run until we have used ~60% of budget or time is low
        max_global_calls = int(budget * 0.6)
        
        while calls_made < max_global_calls and time.time() < deadline and len(un_evaluated_indices) > 0:
            
            # Get candidate points
            X_cands = X_pool[un_evaluated_indices]
            
            # Predict
            mean, std = predict_knn(X_cands, X_obs, f_obs)
            
            # Acquisition: Minimize mean - beta * std (Uncertainty reduction / Exploration)
            # High beta encourages exploration. Low beta encourages exploitation.
            # Let's start with beta=0.5 and decrease it as we find better points?
            # Or just fixed.
            beta = 0.5
            acq = mean - beta * std
            
            # Select best candidate
            best_idx_in_cands = np.argmin(acq)
            
            # Get the actual index in the pool
            pool_idx = un_evaluated_indices[best_idx_in_cands]
            
            # Remove from un-evaluated
            un_evaluated_indices.pop(best_idx_in_cands)
            
            # Evaluate
            x_next = X_pool[pool_idx]
            f_next = evaluate(x_next)
            
            # Update observation data
            X_obs = np.vstack([X_obs, x_next])
            f_obs = np.append(f_obs, f_next)
            
            # Early stopping if no improvement for a long time? 
            # The k-NN model will naturally guide us to promising areas.
            
        # Phase 3: Local Refinement
        # Use (1+1)-ES or Nelder-Mead on the best point found
        if time.time() < deadline and calls_made < budget and best_x is not None:
            x_curr = best_x.copy()
            f_curr = best_f
            sigma = 0.3
            max_es_calls = budget - calls_made
            stagnation = 0
            
            while calls_made < budget and time.time() < deadline:
                y = np.clip(x_curr + sigma * rng.standard_normal(dim), -5, 5)
                fy = evaluate(y)
                
                if fy < f_curr:
                    x_curr, f_curr = y, fy
                    sigma = min(sigma * 1.1, 3.0)
                    stagnation = 0
                else:
                    sigma = max(sigma * 0.8, 0.01)
                    stagnation += 1
                    
                if stagnation >= 30:
                    break
                    
        return best_x, best_f
# EVOLVE-BLOCK-END
