import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # 300s limit, keep 10s margin
        deadline = start_time + 290
        
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        # History
        X = []
        Y = []
        
        best_x = None
        best_f = float('inf')
        
        # Initial random sampling
        n_init = 20 # More initialization to get a better initial model
        if budget <= n_init:
             # Handle very small budget
             for _ in range(budget):
                 x = rng.uniform(-5, 5, dim)
                 f = func(x)
                 if f < best_f:
                     best_f = f
                     best_x = x
             return best_x, best_f

        for _ in range(n_init):
            x = rng.uniform(-5, 5, dim)
            f = func(x)
            X.append(x)
            Y.append(f)
            if f < best_f:
                best_f = f
                best_x = x
        
        X = np.array(X)
        Y = np.array(Y)
        
        k = 5 # neighbors for IDW
        lambda_explore = 1.0
        no_improve_count = 0
        
        # Main loop
        for i in range(budget - n_init):
            if time.time() > deadline:
                break
            
            # Generate candidates
            n_cand = 100
            candidates = rng.uniform(-5, 5, (n_cand, dim))
            
            # Compute distances
            dists = np.linalg.norm(candidates[:, np.newaxis, :] - X[np.newaxis, :, :], axis=2)
            
            # Nearest neighbor distance
            nn_dist = np.min(dists, axis=1)
            
            # IDW Prediction
            preds = np.zeros(n_cand)
            # Optimization: If X is large, this loop is slow.
            # But budget is 500, so X is at most 500.
            # 100 candidates * 500 history = 50k operations per step.
            # 500 steps * 50k = 25M operations. Python can handle this in <1s?
            # Actually, np.argpartition is fast.
            
            for j in range(n_cand):
                d = dists[j]
                # Get k smallest indices
                if len(d) <= k:
                    k_idx = np.arange(len(d))
                else:
                    k_idx = np.argpartition(d, k)[:k]
                
                k_d = d[k_idx]
                k_y = Y[k_idx]
                
                # Inverse distance weighting
                # Add small epsilon to avoid div by 0
                k_d = np.maximum(k_d, 1e-8)
                w = 1.0 / (k_d ** 2)
                
                preds[j] = np.dot(w, k_y) / np.sum(w)
            
            # Acquisition
            # Normalize nn_dist?
            # Max possible dist is sqrt(5)*10 ~ 22.
            # Function range is ~200.
            # lambda should be scaled.
            # Let's try lambda = 5.0 to make exploration significant.
            scores = preds - lambda_explore * nn_dist
            
            best_idx = np.argmin(scores)
            x_new = candidates[best_idx]
            
            f_new = func(x_new)
            
            X = np.vstack([X, x_new])
            Y = np.append(Y, f_new)
            
            if f_new < best_f:
                best_f = f_new
                best_x = x_new
                no_improve_count = 0
            else:
                no_improve_count += 1
                # Increase exploration if stuck
                if no_improve_count > 20:
                    lambda_explore *= 1.5
                    if lambda_explore > 50:
                        lambda_explore = 50
                elif no_improve_count > 5:
                    lambda_explore *= 1.1

        return best_x, best_f
