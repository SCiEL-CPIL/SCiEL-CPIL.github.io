"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        
        eval_count = 0
        X_data = []
        F_data = []
        
        def eval_and_record(x):
            nonlocal eval_count
            if time.time() > deadline:
                return None
            fx = problem(x)
            eval_count += 1
            X_data.append(x.copy())
            F_data.append(fx)
            return fx

        # 1. Initial Sampling
        n_init = 20
        rng = np.random.default_rng(42)
        X_init = lo + (hi - lo) * rng.random((n_init, self.dim))
        for i in range(n_init):
            fx = eval_and_record(X_init[i])
            if fx is None:
                break
        
        X_data = np.array(X_data)
        F_data = np.array(F_data)
        
        # 2. GP Surrogate Setup
        def fit_gp(X, y):
            X_mean = X.mean(axis=0)
            X_std = X.std(axis=0) + 1e-10
            X_norm = (X - X_mean) / X_std
            
            y_mean = y.mean()
            y_std = y.std() + 1e-10
            y_norm = (y - y_mean) / y_std
            
            def rbf(X1, X2, length_scale=1.0):
                sq_dists = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T
                return np.exp(-0.5 * sq_dists / (length_scale**2))
                
            K = rbf(X_norm, X_norm, length_scale=1.0) + 1e-8 * np.eye(len(X))
            try:
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_norm))
            except np.linalg.LinAlgError:
                K += 1e-6 * np.eye(len(X))
                alpha = np.linalg.solve(K, y_norm)
                
            return X_mean, X_std, y_mean, y_std, X_norm, alpha, L, rbf

        def predict_gp(params, X_new):
            X_mean, X_std, y_mean, y_std, X_train, alpha, L, rbf = params
            X_new_norm = (X_new - X_mean) / X_std
            K_s = rbf(X_new_norm, X_train, length_scale=1.0)
            K_ss = rbf(X_new_norm, X_new_norm, length_scale=1.0)
            
            mu = K_s @ alpha
            v = np.linalg.solve(L, K_s.T)
            var = np.diag(K_ss) - np.sum(v**2, axis=0)
            var = np.maximum(var, 1e-8)
            
            mu_orig = mu * y_std + y_mean
            std_orig = np.sqrt(var) * y_std
            return mu_orig, std_orig

        # Fit initial GPs
        gps = []
        for j in range(self.n_obj):
            gps.append(fit_gp(X_data, F_data[:, j]))
            
        # 3. Hypervolume Calculation
        ref_point = np.array([3000.0, 0.06])
        
        def compute_hypervolume(F, ref):
            mask = np.all(F < ref, axis=1)
            F = F[mask]
            if len(F) == 0:
                return 0.0
            
            F = F[np.argsort(F[:, 0])]
            
            pf = []
            min_f2 = ref[1]
            for f in F:
                if f[1] < min_f2:
                    pf.append(f)
                    min_f2 = f[1]
            
            if not pf:
                return 0.0
                
            total_hv = 0.0
            for i in range(len(pf)):
                f1_curr, f2_curr = pf[i]
                if i == len(pf) - 1:
                    f1_next = ref[0]
                else:
                    f1_next = pf[i+1][0]
                    
                width = f1_next - f1_curr
                height = ref[1] - f2_curr
                total_hv += width * height
                
            return total_hv

        current_hv = compute_hypervolume(F_data, ref_point)
        
        # 4. Iterative Optimization
        while eval_count < self.budget and time.time() < deadline - 10:
            n_candidates = 100
            if len(X_data) > 5:
                indices = np.random.choice(len(X_data), size=n_candidates//2, replace=True)
                X_local = X_data[indices] + rng.normal(0, 0.1 * (hi - lo), size=(n_candidates//2, self.dim))
                X_local = np.clip(X_local, lo, hi)
                X_global = lo + (hi - lo) * rng.random((n_candidates//2, self.dim))
                X_cands = np.vstack([X_local, X_global])
            else:
                X_cands = lo + (hi - lo) * rng.random((n_candidates, self.dim))
                
            # Vectorized prediction
            mus = []
            stds = []
            for j in range(self.n_obj):
                mu, std = predict_gp(gps[j], X_cands)
                mus.append(mu)
                stds.append(std)
                
            n_samples = 10
            ehvis = np.zeros(len(X_cands))
            
            # Generate samples in bulk
            samples = np.zeros((len(X_cands), self.n_obj, n_samples))
            for j in range(self.n_obj):
                samples[:, j, :] = rng.normal(mus[j][:, None], stds[j][:, None])
                
            for k in range(n_samples):
                for i in range(len(X_cands)):
                    new_F = np.vstack([F_data, samples[i, :, k].reshape(1, -1)])
                    new_hv = compute_hypervolume(new_F, ref_point)
                    ehvis[i] += (new_hv - current_hv)
                    
            ehvis /= n_samples
            
            best_cand_idx = np.argmax(ehvis)
            if ehvis[best_cand_idx] <= 0:
                best_cand_idx = rng.integers(len(X_cands))
                
            x_next = X_cands[best_cand_idx]
            fx = eval_and_record(x_next)
            if fx is None:
                break
                
            X_data = np.vstack([X_data, x_next])
            F_data = np.vstack([F_data, fx])
            
            # Update GPs
            gps = []
            for j in range(self.n_obj):
                gps.append(fit_gp(X_data, F_data[:, j]))
                
            current_hv = compute_hypervolume(F_data, ref_point)

        return X_data[-1] if len(X_data) > 0 else lo

# EVOLVE-BLOCK-END
