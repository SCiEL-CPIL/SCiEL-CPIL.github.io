"""Symbolic regression via multi-model oracle selection with LOO-CV."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import cdist

def fit(X_train, y_train):
    start_time = time.time()
    DEADLINE = 325.0  # Safety margin for 330s limit
    
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    
    n, d = X.shape
    
    # Normalize inputs to [0, 1] for better kernel conditioning
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_range = X_max - X_min
    X_range[X_range == 0] = 1.0
    X_norm = (X - X_min) / X_range
    
    # Normalize target for numerical stability
    y_mean = np.mean(y)
    y_std = np.std(y)
    if y_std < 1e-12:
        y_std = 1.0
    y_norm = (y - y_mean) / y_std
    
    candidates = []
    
    # --- Model 1: Polynomial Kernel Ridge Regression ---
    def fit_krr():
        if time.time() > DEADLINE - 10:
            return None
        
        # Degree 4 Polynomial Kernel
        degree = 4
        gamma = 1.0 / d
        K = (gamma * (X_norm @ X_norm.T) + 1.0) ** degree
        
        # Regularization
        alpha = 1e-5
        K_reg = K + alpha * np.eye(n)
        
        try:
            w = np.linalg.solve(K_reg, y_norm)
            
            # LOO-CV Score (using the fast formula for kernel ridge)
            # h_ii = K_ii / (1 + alpha * (K_inv @ K_ii)) -- wait, simpler formula exists
            # LOO residual = (y_i - f(-i)_i) / (1 - h_ii)
            # h_ii = (K @ inv(K+alpha I) @ e_i)_i = (inv(K+alpha I) @ K @ e_i)_i
            # Actually, for KRR, h_ii = K_ii / (1 + alpha * (K_inv @ K_ii)) is not quite right.
            # Correct: h_ii = (K @ (K + alpha I)^-1)_{ii}
            # Let's just compute H = K @ inv(K + alpha I)
            # This is O(n^3) which is fine for n=200.
            inv_K_reg = np.linalg.inv(K_reg)
            H = K @ inv_K_reg
            diag_H = np.diag(H)
            
            # LOO predictions
            loo_preds = np.zeros(n)
            for i in range(n):
                if abs(1 - diag_H[i]) < 1e-10:
                    loo_preds[i] = 0
                else:
                    loo_preds[i] = (y_norm[i] - (K @ w)[i] / (1 - diag_H[i])) / (1 - diag_H[i])
                    # Wait, the formula is: f(-i)_i = (f(x_i) - h_ii * (y_i - f(x_i))) / (1 - h_ii)
                    # And LOO error is y_i - f(-i)_i.
                    # Let's stick to the standard formula:
                    # f(-i)_i = (f(x_i) - h_ii * y_i) / (1 - h_ii) is incorrect.
                    # Correct LOO prediction for KRR:
                    # f(-i)_i = (f(x_i) - h_ii * (y_i - f(x_i))) / (1 - h_ii) ? No.
                    # Let's use the residual form.
                    # y_i - f(-i)_i = (y_i - f(x_i)) / (1 - h_ii)
                    # So f(-i)_i = y_i - (y_i - f(x_i)) / (1 - h_ii)
                    loo_preds[i] = y_norm[i] - (y_norm[i] - (K @ w)[i]) / (1 - diag_H[i])
            
            mse = np.mean((y_norm - loo_preds) ** 2)
            
            def predict(X_test):
                X_test = np.asarray(X_test, dtype=np.float64)
                X_test_norm = (X_test - X_min) / X_range
                K_test = (gamma * (X_test_norm @ X_norm.T) + 1.0) ** degree
                preds_norm = K_test @ w
                return preds_norm * y_std + y_mean
            
            return mse, predict
        except Exception:
            return None

    res_krr = fit_krr()
    if res_krr:
        candidates.append(res_krr)

    # --- Model 2: Gaussian Process (Matern-5/2) ---
    def fit_gp():
        if time.time() > DEADLINE - 10:
            return None
        
        def matern52_kernel(X1, X2, length_scale, variance):
            # X1: (n1, d), X2: (n2, d)
            # Compute pairwise Euclidean distances
            # For efficiency with n=200, we can do this directly
            # dist^2 = sum(x1^2) + sum(x2^2) - 2 x1.x2
            x1_sq = np.sum(X1 ** 2, axis=1).reshape(-1, 1)
            x2_sq = np.sum(X2 ** 2, axis=1).reshape(1, -1)
            dist_sq = x1_sq + x2_sq - 2 * (X1 @ X2.T)
            dist_sq = np.maximum(dist_sq, 0)
            dist = np.sqrt(dist_sq)
            r = dist / length_scale
            return variance * (1 + r + r**2 / 3) * np.exp(-r)

        def gp_predict(X_train, y_train, length_scale, variance, noise_var):
            K = matern52_kernel(X_train, X_train, length_scale, variance) + noise_var * np.eye(len(X_train))
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
            
            def predict(X_test):
                K_s = matern52_kernel(X_test, X_train, length_scale, variance)
                preds = K_s @ alpha
                return preds
            
            # Compute LOO MSE
            # For GP, LOO prediction is:
            # mu_-i = mu_i - (K_inv @ e_i) * (y_i - mu_i) / (K_inv @ e_i + 1/noise) ... complicated.
            # Easier: Use the formula:
            # mu_-i = mu_i - (K_inv @ e_i) * (y_i - mu_i) / (K_inv @ e_i) ... 
            # Actually, let's just do it via the Cholesky factor.
            # The diagonal of the posterior variance is related to the inverse of K.
            # Let's use the standard LOO formula for GPs:
            # y_i - mu_-i = (y_i - mu_i) / (1 + v_i) where v_i is the diagonal of the posterior covariance?
            # No, v_i is the predictive variance.
            # The formula is: mu_-i = mu_i - (K_inv @ e_i) * (y_i - mu_i) / (K_inv @ e_i)
            # Let's compute K_inv = inv(K).
            try:
                K_inv = np.linalg.inv(K)
                diag_K_inv = np.diag(K_inv)
                # Predictive mean at training points
                mu = K @ alpha
                # LOO prediction
                # mu_-i = mu_i - (K_inv @ e_i) * (y_i - mu_i) / (K_inv @ e_i)
                # This simplifies to: mu_-i = mu_i - (y_i - mu_i) / (1 + 1/(noise_var * K_inv_ii)) ?
                # Let's use the robust formula:
                # mu_-i = mu_i - (K_inv @ e_i) * (y_i - mu_i) / (K_inv @ e_i)
                # Wait, K_inv @ e_i is the i-th column of K_inv.
                # Let's just compute the full LOO matrix if n is small enough?
                # n=200, so O(n^3) is fine.
                # H = K @ inv(K + noise I)
                # mu_-i = mu_i - H_ii * (y_i - mu_i) / (1 - H_ii) ?
                # Let's stick to the simpler KRR-like formula if we treat GP as KRR with noise.
                # H = K @ inv(K + noise I)
                H = K @ K_inv
                diag_H = np.diag(H)
                loo_preds = np.zeros(n)
                for i in range(n):
                    denom = 1 - diag_H[i]
                    if abs(denom) < 1e-10:
                        loo_preds[i] = 0
                    else:
                        loo_preds[i] = y_norm[i] - (y_norm[i] - mu[i]) / denom
                mse = np.mean((y_norm - loo_preds) ** 2)
                return mse, predict
            except Exception:
                return None

        # Optimize hyperparameters
        # Parameters: log(length_scale), log(variance), log(noise_var)
        def objective(params):
            ls = np.exp(params[0])
            var = np.exp(params[1])
            noise = np.exp(params[2])
            res = gp_predict(X_norm, y_norm, ls, var, noise)
            if res is None:
                return 1e10
            return res[0]

        # Initial guess
        x0 = [0.0, 0.0, -3.0] # ls=1, var=1, noise=0.05
        bounds = [(-3, 3), (-3, 3), (-6, 0)]
        
        try:
            res_opt = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 100, 'xatol': 1e-3, 'fatol': 1e-4})
            best_ls = np.exp(res_opt.x[0])
            best_var = np.exp(res_opt.x[1])
            best_noise = np.exp(res_opt.x[2])
            
            res_gp = gp_predict(X_norm, y_norm, best_ls, best_var, best_noise)
            if res_gp:
                return res_gp
        except Exception:
            pass
        
        # Fallback to default hyperparams
        res_gp = gp_predict(X_norm, y_norm, 1.0, 1.0, 0.01)
        if res_gp:
            return res_gp
        return None

    res_gp = fit_gp()
    if res_gp:
        candidates.append(res_gp)

    # --- Model 3: Rational Function Approximator (Ratio of Linear Basis) ---
    # y = (a^T x + b) / (c^T x + d)
    # This is equivalent to fitting y * (c^T x + d) = a^T x + b
    # y * c^T x + y * d = a^T x + b
    # This is linear in parameters if we rearrange, but y is on both sides.
    # Let's use a different approach: Fit a linear model on an augmented basis that includes terms like x_i / (1 + x_j).
    # Or simply fit a polynomial of degree 2 or 3, which often captures rational-like behavior.
    # Given the "Rational Function Approximator" prompt, let's try a specific structure:
    # y = sum(a_i x_i + b) / (sum(c_i x_i + d))
    # We can use non-linear least squares (Levenberg-Marquardt) to fit this.
    
    def fit_rational():
        if time.time() > DEADLINE - 10:
            return None
            
        # Parameters: a (d), b, c (d), d_const
        # Total params: 2*d + 2
        n_params = 2 * d + 2
        
        def predict_params(X_in, params):
            a = params[:d]
            b = params[d]
            c = params[d+1:d+1+d]
            d_const = params[d+1+d]
            
            num = X_in @ a + b
            den = X_in @ c + d_const
            # Avoid division by zero
            den_safe = np.where(np.abs(den) < 1e-6, 1e-6 * np.sign(den + 1e-10), den)
            return num / den_safe

        def objective(params):
            try:
                preds = predict_params(X_norm, params)
                if not np.all(np.isfinite(preds)):
                    return 1e10
                return np.sum((y_norm - preds) ** 2)
            except Exception:
                return 1e10

        # Initial guess: Linear model
        # y = w^T x + w0
        # a = w, b = w0, c = 0, d = 1
        try:
            A_lin = np.column_stack([X_norm, np.ones(n)])
            w_lin, *_ = np.linalg.lstsq(A_lin, y_norm, rcond=None)
            w = w_lin[:d]
            w0 = w_lin[d]
            x0 = np.concatenate([w, [w0], np.zeros(d), [1.0]])
        except Exception:
            x0 = np.zeros(n_params)
            x0[d] = 0
            x0[-1] = 1

        # Use differential_evolution for robustness
        bounds = [(-10, 10)] * n_params
        
        try:
            res_de = differential_evolution(
                objective, bounds, maxiter=20, popsize=10, tol=0.01, seed=42, polish=False
            )
            best_params = res_de.x
            mse = objective(best_params)
            
            if mse < 1e8: # Check if it's a reasonable fit
                def predict(X_test):
                    X_test = np.asarray(X_test, dtype=np.float64)
                    X_test_norm = (X_test - X_min) / X_range
                    preds_norm = predict_params(X_test_norm, best_params)
                    if not np.all(np.isfinite(preds_norm)):
                        preds_norm = np.where(np.isfinite(preds_norm), preds_norm, 0)
                    return preds_norm * y_std + y_mean
                return mse, predict
        except Exception:
            pass
        return None

    res_rat = fit_rational()
    if res_rat:
        candidates.append(res_rat)

    # Select best model based on LOO MSE
    if not candidates:
        # Fallback to simple linear
        A_lin = np.column_stack([X_norm, np.ones(n)])
        try:
            w, *_ = np.linalg.lstsq(A_lin, y_norm, rcond=None)
            def predict(X_test):
                X_test = np.asarray(X_test, dtype=np.float64)
                X_test_norm = (X_test - X_min) / X_range
                A_test = np.column_stack([X_test_norm, np.ones(len(X_test))])
                preds_norm = A_test @ w
                return preds_norm * y_std + y_mean
            return predict
        except Exception:
            def predict(X_test):
                return np.full(len(X_test), y_mean)
            return predict

    best_mse, best_predict = min(candidates, key=lambda x: x[0])
    return best_predict

# EVOLVE-BLOCK-END
