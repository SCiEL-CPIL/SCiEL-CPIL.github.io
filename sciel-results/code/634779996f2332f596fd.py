import numpy as np
import time
from scipy.optimize import minimize
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
        ref_point = np.array([3.0, 5.0, 7.0])
        
        # Data storage
        X_data = []
        F_data = []
        
        def evaluate(x):
            x = np.clip(x, lo, hi)
            f = np.array(problem(x))
            X_data.append(x)
            F_data.append(f)
            return f
            
        # 1. Initial Sampling (Latin Hypercube for better coverage)
        n_init = 20
        # Simple LHS implementation
        n_points = n_init
        indices = np.random.permutation(n_points)
        X_init = np.zeros((n_points, self.dim))
        for d in range(self.dim):
            # Stratified sampling
            strata = np.linspace(0, 1, n_points + 1)
            # Random offset within strata
            offsets = np.random.uniform(0, 1, n_points)
            # Map to [lo, hi]
            X_init[:, d] = lo[d] + (hi[d] - lo[d]) * (strata[indices] + offsets) / n_points
            # Wait, standard LHS: 
            # sort random numbers? No, permute the strata assignment.
            # Let's do it properly:
            # x_i = lo + (hi-lo) * (u_i - 1) / n
            # where u_i is a permutation of 1..n
            
        # Correct LHS
        X_init = np.zeros((n_init, self.dim))
        for d in range(self.dim):
            perm = np.random.permutation(n_init)
            X_init[:, d] = lo[d] + (hi[d] - lo[d]) * (perm + 0.5) / n_init
            
        for x in X_init:
            evaluate(x)
            
        used = n_init
        
        # GP Surrogate Model
        def gp_predict(X_new, X_train, Y_train, length_scale=0.5, sigma_f=1.0):
            """
            Simple GP prediction for multi-objective.
            X_train: (N, D)
            Y_train: (N, M)
            X_new: (K, D)
            Returns: mean (K, M), var (K, M)
            """
            N = len(X_train)
            D = X_train.shape[1]
            M = Y_train.shape[1]
            
            # Standardize inputs for GP stability
            X_train_std = (X_train - X_train.mean(axis=0)) / (X_train.std(axis=0) + 1e-8)
            X_new_std = (X_new - X_train.mean(axis=0)) / (X_train.std(axis=0) + 1e-8)
            
            # RBF Kernel
            def rbf(A, B, ls):
                # A: (N, D), B: (K, D)
                # dist^2 = sum(A^2) + sum(B^2) - 2AB
                A_sq = np.sum(A**2, axis=1)[:, np.newaxis]
                B_sq = np.sum(B**2, axis=1)[np.newaxis, :]
                dist_sq = np.maximum(A_sq + B_sq - 2 * A.dot(B.T), 0)
                return np.exp(-0.5 * dist_sq / (ls**2))
                
            K = rbf(X_train_std, X_train_std, length_scale)
            K += sigma_f**2 * np.eye(N)
            
            # Inverse K
            try:
                K_inv = np.linalg.inv(K)
            except:
                K_inv = np.linalg.pinv(K)
                
            K_new = rbf(X_train_std, X_new_std, length_scale)
            
            # Mean and Variance for each objective
            means = np.zeros((len(X_new), M))
            vars_ = np.zeros((len(X_new), M))
            
            for m in range(M):
                y = Y_train[:, m]
                alpha = K_inv.dot(y)
                mu = K_new.T.dot(alpha)
                var = sigma_f**2 - np.sum(K_new**2 * K_inv, axis=1)
                var = np.maximum(var, 1e-8)
                means[:, m] = mu
                vars_[:, m] = var
                
            return means, vars_
            
        def hypervolume_2d(points, ref):
            """Calculate hypervolume for 2D points relative to ref."""
            # points: (N, 2), ref: (2,)
            # Filter dominated by ref
            mask = np.all(points < ref, axis=1)
            pts = points[mask]
            if len(pts) == 0: return 0.0
            
            # Sort by x
            pts = pts[np.argsort(pts[:, 0])]
            hv = 0.0
            max_y = ref[1]
            for i in range(len(pts)):
                if pts[i, 1] >= max_y:
                    continue
                # Width is distance to next point's x or ref x
                if i + 1 < len(pts):
                    w = pts[i+1, 0] - pts[i, 0]
                else:
                    w = ref[0] - pts[i, 0]
                h = max_y - pts[i, 1]
                hv += w * h
                max_y = min(max_y, pts[i, 1])
            return hv
            
        def hypervolume_3d_approx(points, ref):
            """
            Approximate 3D hypervolume by projecting onto 2D planes.
            This is not exact but serves as a proxy for EHVI.
            """
            if len(points) == 0: return 0.0
            # Project onto xy, xz, yz planes
            hv_xy = hypervolume_2d(points[:, :2], ref[:2])
            hv_xz = hypervolume_2d(points[:, [0, 2]], ref[[0, 2]])
            hv_yz = hypervolume_2d(points[:, [1, 2]], ref[[1, 2]])
            # Heuristic combination
            return (hv_xy + hv_xz + hv_yz) / 3.0

        def ehvi_score(x_candidate, X_train, F_train):
            """
            Calculate Expected Hypervolume Improvement.
            Simplified: Use GP to predict mean and std, then estimate HV improvement.
            """
            x_c = np.array([x_candidate])
            try:
                mean, var = gp_predict(x_c, X_train, F_train)
                mu = mean[0]
                std = np.sqrt(var[0])
            except:
                return 0.0
                
            # Current HV of non-dominated front
            # Identify non-dominated points in F_train
            F_nd = []
            for i in range(len(F_train)):
                is_dom = False
                for j in range(len(F_train)):
                    if i == j: continue
                    if np.all(F_train[j] <= F_train[i]) and np.any(F_train[j] < F_train[i]):
                        is_dom = True
                        break
                if not is_dom:
                    F_nd.append(F_train[i])
            F_nd = np.array(F_nd)
            current_hv = hypervolume_3d_approx(F_nd, ref_point)
            
            # Expected HV with new point
            # Sample from GP posterior
            n_samples = 50
            total_improvement = 0.0
            for _ in range(n_samples):
                # Sample objective values
                f_sample = mu + std * np.random.randn(self.n_obj)
                # Check if strictly better than ref
                if np.any(f_sample >= ref_point):
                    continue
                # Add to front
                new_front = np.vstack([F_nd, f_sample.reshape(1, -1)])
                # Remove dominated
                nd_front = []
                for i in range(len(new_front)):
                    is_dom = False
                    for j in range(len(new_front)):
                        if i == j: continue
                        if np.all(new_front[j] <= new_front[i]) and np.any(new_front[j] < new_front[i]):
                            is_dom = True
                            break
                    if not is_dom:
                        nd_front.append(new_front[i])
                nd_front = np.array(nd_front)
                new_hv = hypervolume_3d_approx(nd_front, ref_point)
                total_improvement += max(0, new_hv - current_hv)
                
            return total_improvement / n_samples
            
        # 2. Iterative Search
        while used < self.budget:
            if time.time() > deadline - 10:
                break
                
            # Select next point using EHVI
            # We need to optimize the acquisition function.
            # Use a multi-start local optimization on the acquisition function.
            
            best_x = None
            best_score = -np.inf
            
            # Generate random starting points for local optimization
            n_starts = 10
            starts = np.random.uniform(lo, hi, (n_starts, self.dim))
            
            for start in starts:
                def neg_ehvi(x):
                    return -ehvi_score(x, np.array(X_data), np.array(F_data))
                    
                res = minimize(neg_ehvi, start, bounds=list(zip(lo, hi)), method='L-BFGS-B', options={'maxiter': 20})
                if res.success:
                    score = -res.fun
                    if score > best_score:
                        best_score = score
                        best_x = res.x
                        
            if best_x is None:
                # Fallback to random if optimization fails
                best_x = np.random.uniform(lo, hi)
                
            f_new = evaluate(best_x)
            used += 1
            
        return X_data[-1] if X_data else None
