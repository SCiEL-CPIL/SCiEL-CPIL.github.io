"""Symbolic regression via multi-model oracle selection with LOO-CV."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

def fit(X_train, y_train):
    start_time = time.time()
    DEADLINE = 325.0  # Safety margin for 330s limit
    
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    
    n, d = X.shape
    
    # Normalize inputs to [0, 1]
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_range = X_max - X_min
    X_range[X_range == 0] = 1.0
    X_norm = (X - X_min) / X_range
    
    # Normalize target for numerical stability
    y_mean = np.mean(y)
    y_std = np.std(y)
    if y_std == 0:
        y_std = 1.0
    y_norm = (y - y_mean) / y_std

    # Precompute pairwise distances for kernels
    # Using cdist is often faster and more stable for large N
    D = cdist(X_norm, X_norm, metric='euclidean')
    D2 = D**2

    models = []

    # --- Model 1: Polynomial Kernel Ridge Regression ---
    # K(x, x') = (x^T x' + c)^p
    # We try a few degrees and coefficients
    poly_configs = [
        (3, 1.0), (5, 1.0), (3, 2.0), (5, 2.0), (7, 1.0)
    ]
    
    def fit_poly_kernel(deg, coef):
        K = (X_norm @ X_norm.T + coef) ** deg
        # Regularization
        alpha_vals = [1e-4, 1e-3, 1e-2, 1e-1]
        best_w = None
        best_score = np.inf
        
        for alpha in alpha_vals:
            try:
                w = np.linalg.solve(K + alpha * np.eye(n), y_norm)
                # LOO-CV Score: MSE = sum((y_i - y_hat_i)^2) / n
                # For KRR, y_hat_i (leave-one-out) = (y_i - (K + alpha*I)^{-1}_{ii} * y_i) / (1 - (K + alpha*I)^{-1}_{ii})? 
                # Actually, simpler: Compute full prediction, then adjust? 
                # Standard LOO for KRR: y_loo_i = (y_i - K_ii * w_i) / (1 - K_ii * w_i) is wrong.
                # Correct formula: y_loo_i = (y_i - (K + alpha I)^{-1}_{ii}^{-1} * y_i) ... no.
                # Let's use the standard identity: 
                # y_loo = (y - diag(K) * w) / (1 - diag(K) * w) is for unregularized? 
                # No, for regularized K: K_reg = K + alpha I.
                # w = K_reg^{-1} y.
                # y_loo_i = (y_i - (K_reg^{-1})_{ii}^{-1} * y_i) / (1 - (K_reg^{-1})_{ii}^{-1} * (K_reg^{-1})_{ii}^{-1} ...)?
                # Easier: Just compute predictions on training set and use a proxy? 
                # Or compute LOO properly.
                # y_loo_i = (y_i - (K_reg^{-1})_{ii} * y_i) / (1 - (K_reg^{-1})_{ii} * (K_reg^{-1})_{ii})? 
                # Actually: y_loo = (y - K_ii * w) / (1 - K_ii * w) is only valid if K is the full kernel and we are doing standard KRR?
                # Let's stick to a robust proxy if LOO is too complex to derive quickly, or use a simple holdout if n is large? 
                # n=200 is small. LOO is cheap if we have the inverse.
                # Let's compute the inverse once.
                K_inv = np.linalg.inv(K + alpha * np.eye(n))
                w = K_inv @ y_norm
                # LOO prediction formula:
                # y_loo_i = (y_i - (K_inv)_{ii} * y_i) / (1 - (K_inv)_{ii} * (K_inv)_{ii})? 
                # No. 
                # y_loo = (y - diag(K) * w) / (1 - diag(K) * w) is for KRR with K + lambda I? 
                # Let's use the explicit formula:
                # y_loo_i = (y_i - (K_reg^{-1})_{ii}^{-1} * y_i) / (1 - (K_reg^{-1})_{ii}^{-1} * (K_reg^{-1})_{ii}) 
                # This is getting messy. Let's use a simpler proxy: 
                # Since n is small, we can just do a 5-fold CV to be safe and simple.
                pass
            except np.linalg.LinAlgError:
                continue
        
        # Actually, let's just use a simple 5-fold CV for selection to avoid LOO derivation errors.
        # But wait, 5-fold CV on 200 points is 40 points per fold. That's fine.
        # However, for the final model, we retrain on all data.
        
        # Let's implement a generic CV scorer.
        def cv_score(fit_func, X, y, k=5):
            indices = np.arange(n)
            np.random.shuffle(indices)
            folds = np.array_split(indices, k)
            scores = []
            for i in range(k):
                val_idx = folds[i]
                trn_idx = np.concatenate([folds[j] for j in range(k) if j != i])
                X_tr, y_tr = X[trn_idx], y[trn_idx]
                X_val, y_val = X[val_idx], y[val_idx]
                
                # Fit on training fold
                try:
                    pred = fit_func(X_tr, y_tr, X_val)
                    mse = np.mean((pred - y_val)**2)
                    scores.append(mse)
                except Exception:
                    return np.inf
            return np.mean(scores)

        # Define the fitter for Poly Kernel
        def poly_fitter(X_tr, y_tr, X_val, deg, coef, alpha):
            K_tr = (X_tr @ X_tr.T + coef) ** deg
            w = np.linalg.solve(K_tr + alpha * np.eye(len(y_tr)), y_tr)
            K_val = (X_val @ X_tr.T + coef) ** deg
            return K_val @ w

        # We need to wrap this to pass params.
        # To keep it fast, we'll just pick the best alpha based on a quick heuristic or single CV run.
        # Given the time limit, we'll do a coarse search.
        
        best_score = np.inf
        best_params = (deg, coef, 1e-3)
        
        for alpha in alpha_vals:
            # Simple proxy: Check condition number and residual
            K_reg = (X_norm @ X_norm.T + coef) ** deg + alpha * np.eye(n)
            try:
                w = np.linalg.solve(K_reg, y_norm)
                pred = ((X_norm @ X_norm.T + coef) ** deg) @ w
                # Use training MSE as a rough guide, penalized by alpha
                train_mse = np.mean((pred - y_norm)**2)
                # If training MSE is very low, it might be overfitting, but for symbolic regression, 
                # low training error is a good sign. 
                # Let's use a simple LOO approximation if possible.
                # For KRR, LOO error can be computed from the inverse.
                K_inv = np.linalg.inv(K_reg)
                diag_inv = np.diag(K_inv)
                # y_loo_i = (y_i - diag_inv_i * y_i) / (1 - diag_inv_i * diag_inv_i)? 
                # Correct formula for KRR LOO:
                # y_loo = (y - diag(K) * w) / (1 - diag(K) * w) is WRONG for regularized.
                # y_loo_i = (y_i - (K_reg^{-1})_{ii} * y_i) / (1 - (K_reg^{-1})_{ii} * (K_reg^{-1})_{ii})?
                # Let's just use 5-fold CV for the final selection among alphas.
                pass
            except:
                continue
        
        # To save code space and time, I will implement the CV loop directly for the candidates.
        # This is getting long. I will simplify the candidate list.
        
        # Candidate: Poly Kernel (deg=3, coef=1, alpha=1e-3)
        def fit_pred_poly_3(X_tr, y_tr, X_val):
            K_tr = (X_tr @ X_tr.T + 1.0) ** 3
            w = np.linalg.solve(K_tr + 1e-3 * np.eye(len(y_tr)), y_tr)
            K_val = (X_val @ X_tr.T + 1.0) ** 3
            return K_val @ w
        models.append(('poly3', fit_pred_poly_3))
        
        # Candidate: Poly Kernel (deg=5, coef=1, alpha=1e-3)
        def fit_pred_poly_5(X_tr, y_tr, X_val):
            K_tr = (X_tr @ X_tr.T + 1.0) ** 5
            w = np.linalg.solve(K_tr + 1e-3 * np.eye(len(y_tr)), y_tr)
            K_val = (X_val @ X_tr.T + 1.0) ** 5
            return K_val @ w
        models.append(('poly5', fit_pred_poly_5))
        
        # Candidate: Poly Kernel (deg=3, coef=2, alpha=1e-2)
        def fit_pred_poly_3c2(X_tr, y_tr, X_val):
            K_tr = (X_tr @ X_tr.T + 2.0) ** 3
            w = np.linalg.solve(K_tr + 1e-2 * np.eye(len(y_tr)), y_tr)
            K_val = (X_val @ X_tr.T + 2.0) ** 3
            return K_val @ w
        models.append(('poly3c2', fit_pred_poly_3c2))

    # --- Model 2: RBF Kernel Ridge Regression ---
    rbf_ls_vals = [0.1, 0.3, 0.5, 1.0]
    rbf_alpha_vals = [1e-4, 1e-3, 1e-2]
    
    for ls in rbf_ls_vals:
        for alpha in rbf_alpha_vals:
            def make_rbf_fitter(ls_val, alpha_val):
                def fitter(X_tr, y_tr, X_val):
                    D_tr = cdist(X_tr, X_tr, metric='euclidean')
                    K_tr = np.exp(-0.5 * (D_tr**2) / (ls_val**2))
                    w = np.linalg.solve(K_tr + alpha_val * np.eye(len(y_tr)), y_tr)
                    D_val = cdist(X_val, X_tr, metric='euclidean')
                    K_val = np.exp(-0.5 * (D_val**2) / (ls_val**2))
                    return K_val @ w
                return fitter
            
            name = f'rbf_ls{ls}_a{alpha}'
            models.append((name, make_rbf_fitter(ls, alpha)))

    # --- Model 3: Gaussian Process (Matern 5/2) ---
    # We'll optimize hyperparameters using L-BFGS on NLL
    def matern52_kernel(D, ls, var):
        r = D
        return var * (1.0 + np.sqrt(3.0) * r / ls) * np.exp(-np.sqrt(3.0) * r / ls)

    def fit_gp(X_tr, y_tr, X_val):
        # Optimize hyperparameters on training data
        n_tr = len(y_tr)
        D_tr = cdist(X_tr, X_tr, metric='euclidean')
        
        def nll(params):
            ls = np.exp(params[0])
            var = np.exp(params[1])
            noise = np.exp(params[2])
            K = matern52_kernel(D_tr, ls, var) + noise * np.eye(n_tr)
            try:
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_tr))
                log_det = 2.0 * np.sum(np.log(np.diag(L)))
                ll = -0.5 * (y_tr @ alpha + log_det + n_tr * np.log(2 * np.pi))
                return -ll
            except:
                return 1e10

        x0 = np.array([0.0, 0.0, -10.0])
        try:
            res = minimize(nll, x0, method='L-BFGS-B', options={'maxiter': 20, 'maxfun': 20})
            ls = np.exp(res.x[0])
            var = np.exp(res.x[1])
            noise = np.exp(res.x[2])
            
            K = matern52_kernel(D_tr, ls, var) + noise * np.eye(n_tr)
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_tr))
            
            D_val = cdist(X_val, X_tr, metric='euclidean')
            K_star = matern52_kernel(D_val, ls, var)
            mean = K_star @ alpha
            return mean
        except:
            return np.full(len(y_tr), np.mean(y_tr))

    models.append(('gp', fit_gp))

    # --- Selection via 5-Fold CV ---
    # We have a list of (name, fitter_func). 
    # fitter_func(X_tr, y_tr, X_val) -> predictions for X_val
    
    indices = np.arange(n)
    np.random.shuffle(indices)
    folds = np.array_split(indices, 5)
    
    cv_scores = {}
    
    for name, fitter in models:
        scores = []
        valid = True
        for i in range(5):
            val_idx = folds[i]
            trn_idx = np.concatenate([folds[j] for j in range(5) if j != i])
            
            X_tr = X_norm[trn_idx]
            y_tr = y_norm[trn_idx]
            X_val = X_norm[val_idx]
            y_val = y_norm[val_idx]
            
            try:
                pred = fitter(X_tr, y_tr, X_val)
                # Ensure finite
                if not np.all(np.isfinite(pred)):
                    valid = False
                    break
                mse = np.mean((pred - y_val)**2)
                scores.append(mse)
            except Exception:
                valid = False
                break
            
            if time.time() > DEADLINE - 10:
                valid = False
                break
        
        if valid and scores:
            cv_scores[name] = np.mean(scores)
        else:
            cv_scores[name] = np.inf

    # Find best model
    best_name = min(cv_scores, key=cv_scores.get)
    best_fitter = [f for n, f in models if n == best_name][0]
    
    # Retrain best model on all data
    # We need a version of the fitter that takes X_train and y_train and returns a predict function
    # The current fitters take X_tr, y_tr, X_val and return predictions.
    # We need to wrap the best fitter to create a closure that holds the trained weights/params.
    
    # Since the fitters above are stateless (they retrain every time), 
    # for the final prediction, we will call the fitter with X_train, y_train, and the test X.
    # This is inefficient (retraining on every predict call) but correct for the interface.
    # However, the interface requires `fit` to return a `predict` callable.
    # If we retrain on every predict call, it might be too slow if predict is called many times?
    # The task says "200 separate test points". It's likely called once or a few times.
    # But to be safe, we should train once in `fit` and return a closure.
    
    # Let's refactor: Train the best model once on all data.
    
    if best_name.startswith('poly'):
        # Parse params from name? No, let's just re-implement the specific best fitter logic here.
        # This is fragile. Better approach: 
        # The fitters in `models` are generic. 
        # We can just use the fitter function directly in the predict closure, passing X_train, y_train.
        
        def predict(X_test):
            X_test = np.asarray(X_test, dtype=np.float64)
            X_test_norm = (X_test - X_min) / X_range
            # Call the fitter with training data and test data
            return best_fitter(X_norm, y_norm, X_test_norm) * y_std + y_mean

    elif best_name.startswith('rbf'):
        def predict(X_test):
            X_test = np.asarray(X_test, dtype=np.float64)
            X_test_norm = (X_test - X_min) / X_range
            return best_fitter(X_norm, y_norm, X_test_norm) * y_std + y_mean

    elif best_name == 'gp':
        def predict(X_test):
            X_test = np.asarray(X_test, dtype=np.float64)
            X_test_norm = (X_test - X_min) / X_range
            return best_fitter(X_norm, y_norm, X_test_norm) * y_std + y_mean

    else:
        # Fallback to mean
        def predict(X_test):
            return np.full(len(X_test), y_mean)

    # Optimization: The fitters above retrain on every call. 
    # For GP, this is expensive (minimization). 
    # We should train the GP once here.
    
    if best_name == 'gp':
        # Train GP once
        n_tr = n
        D_tr = cdist(X_norm, X_norm, metric='euclidean')
        
        def nll(params):
            ls = np.exp(params[0])
            var = np.exp(params[1])
            noise = np.exp(params[2])
            K = matern52_kernel(D_tr, ls, var) + noise * np.eye(n_tr)
            try:
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_norm))
                log_det = 2.0 * np.sum(np.log(np.diag(L)))
                ll = -0.5 * (y_norm @ alpha + log_det + n_tr * np.log(2 * np.pi))
                return -ll
            except:
                return 1e10

        x0 = np.array([0.0, 0.0, -10.0])
        try:
            res = minimize(nll, x0, method='L-BFGS-B', options={'maxiter': 50, 'maxfun': 50})
            ls = np.exp(res.x[0])
            var = np.exp(res.x[1])
            noise = np.exp(res.x[2])
            
            K = matern52_kernel(D_tr, ls, var) + noise * np.eye(n_tr)
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_norm))
            
            def predict_gp(X_test):
                X_test = np.asarray(X_test, dtype=np.float64)
                X_test_norm = (X_test - X_min) / X_range
                D_val = cdist(X_test_norm, X_norm, metric='euclidean')
                K_star = matern52_kernel(D_val, ls, var)
                mean = K_star @ alpha
                return mean * y_std + y_mean
            return predict_gp
        except:
            pass

    # For Poly and RBF, we can also train once to avoid re-computation in predict.
    # But the fitters in `models` are stateless. 
    # Let's just use the stateless fitter in the closure for Poly/RBF. 
    # It's O(N^2) per predict call. N=200. 200^2 = 40k ops. Very fast.
    
    def predict_generic(X_test):
        X_test = np.asarray(X_test, dtype=np.float64)
        X_test_norm = (X_test - X_min) / X_range
        return best_fitter(X_norm, y_norm, X_test_norm) * y_std + y_mean

    return predict_generic

# EVOLVE-BLOCK-END
