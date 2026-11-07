import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        eval_count = [0]
        all_X = []
        all_F = []
        
        def count_eval(x):
            eval_count[0] += 1
            if eval_count[0] > self.budget:
                raise Exception("Budget exceeded")
            x = np.clip(x, lo, hi)
            f = problem(x)
            all_X.append(x)
            all_F.append(f)
            return f

        # 1. Initialization: Latin Hypercube Sampling for better coverage
        n_init = 30
        rng = np.random.default_rng(0)
        
        X_init = []
        for d in range(self.dim):
            samples = (np.arange(n_init) + rng.random(n_init)) / n_init
            X_init.append(samples)
        
        X_init = np.array(X_init).T
        X_init = lo + (hi - lo) * X_init
        
        for x in X_init:
            if eval_count[0] >= self.budget:
                break
            count_eval(x)
        
        # Estimate ideal point and ranges from initial samples
        if all_F:
            f_arr = np.array(all_F)
            ideal = np.min(f_arr, axis=0)
            ranges = np.ptp(f_arr, axis=0)
            ranges[ranges == 0] = 1.0
        else:
            ideal = np.zeros(self.n_obj)
            ranges = np.ones(self.n_obj)
        
        center = (lo + hi) / 2
        
        # 2. Systematic Tchebycheff with smooth approximation (log-sum-exp)
        # Generate weight vectors: vertices, edges, face centers, and random
        key_weights = []
        
        # Vertices
        for i in range(self.n_obj):
            w = np.zeros(self.n_obj)
            w[i] = 1.0
            key_weights.append(w)
        
        # Edge midpoints
        for i in range(self.n_obj):
            for j in range(i+1, self.n_obj):
                w = np.zeros(self.n_obj)
                w[i] = 0.5
                w[j] = 0.5
                key_weights.append(w)
        
        # Face center
        w = np.ones(self.n_obj) / self.n_obj
        key_weights.append(w)
        
        # Additional structured weights for better coverage
        structured = [
            [0.7, 0.2, 0.1], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7],
            [0.7, 0.1, 0.2], [0.1, 0.7, 0.2], [0.2, 0.1, 0.7],
            [0.5, 0.3, 0.2], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5],
            [0.5, 0.2, 0.3], [0.2, 0.5, 0.3], [0.3, 0.2, 0.5],
            [0.6, 0.3, 0.1], [0.3, 0.6, 0.1], [0.1, 0.3, 0.6],
            [0.6, 0.1, 0.3], [0.1, 0.6, 0.3], [0.3, 0.1, 0.6],
        ]
        for sw in structured:
            w = np.array(sw)
            w = w / w.sum()
            key_weights.append(w)
        
        # Add random weights for diversity
        for _ in range(8):
            w = rng.random(self.n_obj)
            w /= w.sum()
            key_weights.append(w)
        
        # Shuffle
        rng.shuffle(key_weights)
        
        # Smooth Tchebycheff: log-sum-exp approximation of max
        # min log(sum_i exp(alpha * w_i * (f_i - ideal_i) / ranges_i)) / alpha
        alpha = 20.0  # Smoothness parameter
        
        def tchebycheff_smooth(x, w):
            f = count_eval(x)
            # Normalize by ranges to handle different scales
            scaled = w * (f - ideal) / ranges
            # Log-sum-exp for numerical stability
            max_val = np.max(scaled)
            if max_val > 1e6:
                return max_val
            log_sum = max_val + np.log(np.sum(np.exp(scaled - max_val)))
            return log_sum / alpha
        
        # Phase 2a: Solve for each weight vector
        for w in key_weights:
            if eval_count[0] >= self.budget - 2:
                break
            if time.time() > deadline - 15:
                break
            
            # Try multiple starting points
            starts = [center]
            # Add 2 random starts
            for _ in range(2):
                starts.append(lo + (hi - lo) * rng.random(self.dim))
            
            best_f = None
            best_x = None
            
            for x0 in starts:
                if eval_count[0] >= self.budget - 2:
                    break
                if time.time() > deadline - 10:
                    break
                
                def obj(x):
                    return tchebycheff_smooth(x, w)
                
                try:
                    res = minimize(obj, x0, method='L-BFGS-B', 
                                   bounds=list(zip(lo, hi)),
                                   options={'maxiter': 15, 'maxfev': 12, 'ftol': 1e-8, 'gtol': 1e-6})
                    if res.success and res.fun < (best_f if best_f is not None else float('inf')):
                        best_f = res.fun
                        best_x = res.x
                except Exception:
                    pass
        
        # Phase 2b: Adaptive refinement - find gaps in the front
        # Compute non-dominated set
        if all_F:
            f_arr = np.array(all_F)
            
            # Find non-dominated points
            non_dom_idx = []
            for i in range(len(f_arr)):
                dominated = False
                for j in range(len(f_arr)):
                    if i == j:
                        continue
                    if np.all(f_arr[j] <= f_arr[i]) and np.any(f_arr[j] < f_arr[i]):
                        dominated = True
                        break
                if not dominated:
                    non_dom_idx.append(i)
            
            # For each pair of adjacent non-dominated points, try to fill the gap
            if len(non_dom_idx) >= 2:
                # Sort by first objective
                non_dom_idx_sorted = sorted(non_dom_idx, key=lambda i: f_arr[i, 0])
                
                # Try to add points between adjacent pairs
                for k in range(len(non_dom_idx_sorted) - 1):
                    if eval_count[0] >= self.budget - 3:
                        break
                    if time.time() > deadline - 10:
                        break
                    
                    i1 = non_dom_idx_sorted[k]
                    i2 = non_dom_idx_sorted[k + 1]
                    
                    # Interpolate in objective space
                    f1 = f_arr[i1]
                    f2 = f_arr[i2]
                    
                    # Try midpoint in objective space
                    f_mid = (f1 + f2) / 2
                    
                    # Use weighted sum to find a point close to f_mid
                    # Weight: direction from f_mid toward the ideal
                    w_fill = np.zeros(self.n_obj)
                    for d in range(self.n_obj):
                        if ideal[d] < f_mid[d]:
                            w_fill[d] = 1.0 / (f_mid[d] - ideal[d] + 1e-6)
                    
                    if w_fill.sum() > 0:
                        w_fill = w_fill / w_fill.sum()
                    
                    # Start from midpoint of the two x points
                    x0 = (all_X[i1] + all_X[i2]) / 2
                    
                    def obj_fill(x):
                        f = count_eval(x)
                        return np.dot(w_fill, f)
                    
                    try:
                        res = minimize(obj_fill, x0, method='L-BFGS-B',
                                       bounds=list(zip(lo, hi)),
                                       options={'maxiter': 10, 'maxfev': 8, 'ftol': 1e-8})
                    except Exception:
                        pass
        
        # Phase 3: Final polish on best points
        if all_F and eval_count[0] < self.budget:
            f_arr = np.array(all_F)
            # Find non-dominated points and polish them
            non_dom_idx = []
            for i in range(len(f_arr)):
                dominated = False
                for j in range(len(f_arr)):
                    if i == j:
                        continue
                    if np.all(f_arr[j] <= f_arr[i]) and np.any(f_arr[j] < f_arr[i]):
                        dominated = True
                        break
                if not dominated:
                    non_dom_idx.append(i)
            
            # Polish up to 5 non-dominated points
            for idx in non_dom_idx[:5]:
                if time.time() > deadline - 5:
                    break
                if eval_count[0] >= self.budget - 2:
                    break
                
                x0 = all_X[idx]
                
                def sum_obj(x):
                    f = count_eval(x)
                    return np.sum(f)
                
                try:
                    res = minimize(sum_obj, x0, method='L-BFGS-B', 
                                   bounds=list(zip(lo, hi)),
                                   options={'maxiter': 20, 'maxfev': 10, 'ftol': 1e-10, 'gtol': 1e-8})
                except Exception:
                    pass

        return all_X
