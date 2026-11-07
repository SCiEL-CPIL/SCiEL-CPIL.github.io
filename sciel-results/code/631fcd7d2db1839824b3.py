"""Rewritten optimizer: Gaussian Process Surrogate + Bayesian Optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import qmc
from scipy.linalg import cho_factor, cho_solve


class Optimizer:
    """GP Surrogate with LCB Acquisition."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        n = dim

        # Hyperparameters for GP
        gamma = 0.5  # RBF length scale parameter
        noise = 1e-6
        beta = 2.0  # Exploration weight for LCB

        # --- Phase 1: Initial Sobol Sample ---
        n_init = 64
        sobol = qmc.Sobol(d=n, scramble=True, seed=42)
        points_unit = sobol.random(n_init)
        points = -5.0 + 10.0 * points_unit
        X = np.array([points[i] for i in range(n_init)])
        Y = np.array([func(x) for x in points])
        
        calls_made = n_init
        best_idx = np.argmin(Y)
        best_x = X[best_idx].copy()
        best_y = Y[best_idx]

        # Pre-generate a large pool of global candidates from Sobol
        sobol_global = qmc.Sobol(d=n, scramble=True, seed=123)
        global_pool_unit = sobol_global.random(1000)
        global_pool = -5.0 + 10.0 * global_pool_unit
        global_idx = 0

        # Main Loop
        while calls_made < budget:
            # 1. Update GP Model
            X2 = np.sum(X**2, axis=1)
            D2 = X2[:, None] + X2[None, :] - 2 * X @ X.T
            D2 = np.maximum(D2, 0)
            K = np.exp(-gamma * D2)
            K[np.diag_indices_from(K)] += noise
            
            try:
                L = np.linalg.cholesky(K)
                alpha = cho_solve((L, True), Y)
            except np.linalg.LinAlgError:
                alpha = np.linalg.lstsq(K, Y, rcond=None)[0]

            # 2. Generate Candidates
            n_global_cands = 50
            n_local_cands = 50
            
            if global_idx + n_global_cands > len(global_pool):
                global_pool_unit = sobol_global.random(1000)
                global_pool = -5.0 + 10.0 * global_pool_unit
                global_idx = 0
            
            g_cands = global_pool[global_idx : global_idx + n_global_cands]
            global_idx += n_global_cands
            
            progress = calls_made / budget
            step_size = 1.0 + 2.0 * (1.0 - progress)
            l_cands = best_x + step_size * np.random.randn(n_local_cands, n)
            l_cands = np.clip(l_cands, -5, 5)
            
            candidates = np.vstack([g_cands, l_cands])
            
            # 3. Predict Mean and Variance for Candidates
            C2 = np.sum(candidates**2, axis=1)
            DC2 = C2[:, None] + X2[None, :] - 2 * candidates @ X.T
            DC2 = np.maximum(DC2, 0)
            KC = np.exp(-gamma * DC2)
            
            mu = KC @ alpha
            var = 1.0 - KC @ alpha
            var = np.maximum(var, 0)
            std = np.sqrt(var)
            
            # 4. Acquisition Function: LCB
            acq = mu - beta * std
            best_cand_idx = np.argmin(acq)
            x_new = candidates[best_cand_idx]
            
            # 5. Evaluate
            y_new = func(x_new)
            calls_made += 1
            
            # 6. Update Data
            X = np.vstack([X, x_new])
            Y = np.append(Y, y_new)
            
            if y_new < best_y:
                best_y = y_new
                best_x = x_new.copy()

        return best_x, best_y
# EVOLVE-BLOCK-END
