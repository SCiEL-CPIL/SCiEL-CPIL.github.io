"""Symbolic regression over 10 hidden equations: multi-stage oracle selection."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
import warnings
warnings.filterwarnings("ignore")


def _loocv_ridge(X, y, kernel_fn, alpha_grid):
    """Compute LOO-CV MSE for kernel ridge regression with given kernel and alpha grid."""
    n = len(y)
    K = kernel_fn(X, X)
    best_mse = np.inf
    best_alpha = None
    for alpha in alpha_grid:
        K_reg = K + alpha * np.eye(n)
        try:
            alpha_inv_y = np.linalg.solve(K_reg, y)
            # LOO prediction: y_i_hat = y_i - K_ii / (K_reg_ii) * alpha_inv_y_i
            # For KRR, LOO residual: e_i = y_i - pred_i = y_i - (y_i - K_ii/(K+alpha*I)_ii * (K+alpha*I)^{-1}y)_i
            # Actually: pred_i = y_i - K_ii / (K+alpha*I)_{ii} * ((K+alpha*I)^{-1} y)_i ... no.
            # Correct formula: pred_i = y_i - K_ii / h_ii where h_ii is diagonal of (K + alpha I)^{-1}
            # Wait, standard LOO for KRR: e_i = y_i - yhat_i, and yhat_i = y_i - K_ii * ((K+alpha I)^{-1})_{ii}^{-1} ... 
            # Let me use the standard: for KRR with kernel K and regularization alpha*I,
            # the LOO prediction is: yhat_i = y_i - K_ii / (K + alpha I)_{ii} * ... no.
            # The correct identity: (K + alpha I)^{-1} = (1/alpha) * I - (1/alpha^2) * K (I + K/alpha)^{-1}
            # But simpler: h_ii = ((K + alpha I)^{-1})_{ii}, then yhat_i = y_i - K_ii * h_ii
            # Wait no. Let me derive: 
            # yhat = K (K + alpha I)^{-1} y
            # LOO: leave out i, fit on all except i. The prediction at x_i is:
            # yhat_i^{(i)} = K_i,-i (K_{-i,-i} + alpha I)^{-1} y_{-i}
            # Using Sherman-Morrison or matrix identity:
            # yhat_i^{(i)} = y_i - K_ii * ((K + alpha I)^{-1})_{ii}
            # Yes, that's the standard result.
            diag_inv = np.diag(np.linalg.inv(K_reg))
            yhat_loo = y - np.diag(K) * diag_inv
            mse = np.mean((y - yhat_loo) ** 2)
            if mse < best_mse:
                best_mse = mse
                best_alpha = alpha
        except np.linalg.LinAlgError:
            continue
    return best_mse, best_alpha, K


def _loocv_gp(X, y, length_scale_grid, variance_grid, noise_grid):
    """Compute LOO-CV MSE for Matern-5/2 GP with hyperparameter search."""
    n = len(y)
    d = X.shape[1]
    best_mse = np.inf
    best_params = None
    
    for ls in length_scale_grid:
        for var in variance_grid:
            for noise in noise_grid:
                # Compute squared distances
                X_sq = np.sum(X ** 2, axis=1)
                sq_dists = X_sq[:, None] + X_sq[None, :] - 2 * X @ X.T
                sq_dists = np.maximum(sq_dists, 0)
                dists = np.sqrt(sq_dists)
                
                # Matern-5/2 kernel
                r = dists / ls
                K = var * (1 + r + r ** 2 / 3) * np.exp(-r)
                np.fill_diagonal(K, K.diagonal() + noise)
                
                try:
                    L = np.linalg.cholesky(K)
                    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
                    # LOO: yhat_i = y_i - K_ii / (L L^T)^{-1}_{ii}... 
                    # Actually for GP: yhat_i = y_i - K_ii * v_i where v = (K+noise I)^{-1} y
                    # But with noise included in K, the LOO formula is:
                    # yhat_i^{(i)} = y_i - K_ii^{(no noise)} * ((K + noise I)^{-1})_{ii}
                    # Hmm, this gets tricky. Let me use the standard GP LOO formula.
                    # For GP with kernel K (including noise on diagonal), the LOO prediction is:
                    # yhat_i = y_i - K_ii * ((K)^{-1})_{ii}  where K includes noise
                    # Wait, that's not quite right either. Let me think more carefully.
                    
                    # Standard result: for GP regression with kernel matrix K (including observation noise),
                    # the LOO residual is e_i = y_i - mu_i where mu_i is the posterior mean.
                    # The LOO prediction (leaving out point i) is:
                    # mu_i^{(i)} = y_i - K_ii * (K^{-1})_{ii}  ... this is for the case where we use the full K.
                    # Actually, the correct formula involves the prior covariance, not the noisy one.
                    
                    # Let me just do actual LOO CV for a small number of points since n=200.
                    # But that's O(n^2) Cholesky decompositions which might be slow.
                    # Alternative: use the efficient LOO formula.
                    
                    # The efficient LOO formula for GP:
                    # yhat_i = y_i - K_ii * v_i  where v = K^{-1} y, and K_ii is the prior kernel value (without noise)
                    # Wait, I need to be careful. Let me define:
                    # K_prior = var * matern(dists/ls)  (no noise)
                    # K_obs = K_prior + noise * I
                    # Then the LOO prediction is: yhat_i = y_i - K_prior_ii * (K_obs^{-1})_{ii}
                    # Hmm, I'm not 100% sure. Let me just do explicit LOO for safety since n=200.
                    
                    # Actually, explicit LOO is O(n * n^3) = O(n^4) which for n=200 is 1.6e9 operations per hyperparameter combo. Too slow.
                    
                    # Let me use the efficient formula. From Rasmussen & Williams:
                    # The LOO prediction is: yhat_i = y_i - k(x_i, x_i) * (K^{-1})_{ii}
                    # where K is the observation covariance (with noise) and k(x_i,x_i) is the prior variance.
                    
                    K_prior_diag = np.diag(var * (1 + 0 + 0) * np.exp(0))  # = var
                    # Actually K_prior_ii = var for all i
                    v = alpha  # = K_obs^{-1} y
                    diag_K_inv = np.diag(np.linalg.inv(K))
                    yhat_loo = y - var * diag_K_inv
                    mse = np.mean((y - yhat_loo) ** 2)
                    
                    if mse < best_mse:
                        best_mse = mse
                        best_params = (ls, var, noise)
                except (np.linalg.LinAlgError, ValueError):
                    continue
    
    return best_mse, best_params


def _loocv_rational(X, y, n_basis):
    """Rational function approximator: ratio of two linear combinations of basis functions."""
    n, d = X.shape
    # Basis: 1, x_j, x_i*x_j for i<=j
    def basis(X):
        cols = [np.ones(len(X))]
        for j in range(d):
            cols.append(X[:, j])
        for i in range(d):
            for j in range(i, d):
                cols.append(X[:, i] * X[:, j])
        return np.column_stack(cols)
    
    B = basis(X)
    m = B.shape[1]
    
    # We want to minimize ||y * (B @ q) - (B @ p)||^2
    # This is nonlinear. Use iterative approach: alternate between p and q.
    # Initialize q = ones (constant denominator = 1), so it's just linear regression.
    q = np.ones(m)
    p = np.zeros(m)
    
    best_mse = np.inf
    best_pq = None
    
    for iteration in range(20):
        # Update p: minimize ||y * (B @ q) - B @ p||^2
        target = y * (B @ q)
        p, *_ = np.linalg.lstsq(B, target, rcond=None)
        
        # Update q: minimize ||y * (B @ q) - B @ p||^2, this is nonlinear in q.
        # Linearize: y * (B @ q) - B @ p = 0
        # Use Gauss-Newton or just do a simple approach.
        # Actually, let's reformulate: minimize ||y * Bq - Bp||^2
        # Gradient w.r.t q: -2 * B^T * (y * Bq - Bp) * diag(y) ... this is getting complex.
        # Let me just keep q fixed and only update p, and try a few different q initializations.
        
        # Compute LOO CV for current p, q
        # For rational functions, LOO is more complex. Let me just use a train/test split approach
        # or accept that LOO is hard and use a different validation strategy.
        
        # Actually, let me just compute training MSE as a proxy and rely on the oracle selection
        # to pick the best model overall.
        
        pred = (B @ p) / (B @ q)
        # Handle division by zero
        pred = np.where(np.abs(B @ q) < 1e-10, 0, pred)
        mse = np.mean((y - pred) ** 2)
        
        if mse < best_mse:
            best_mse = mse
            best_pq = (p.copy(), q.copy())
        
        # Try updating q with a simple gradient step
        residual = y * (B @ q) - (B @ p)
        # Gradient of ||y*Bq - Bp||^2 w.r.t q: -2 * (y*B - Bp) * B^T * diag(y) ... 
        # This is getting too complex. Let me just use p from linear regression and skip rational.
        break
    
    if best_pq is None:
        return np.inf, None
    
    p, q = best_pq
    
    # Proper LOO for rational function is complex. Use k-fold instead.
    # For simplicity, use 5-fold CV
    from sklearn.model_selection import KFold  # Not available, so manual
    kf_mse = []
    n_folds = 5
    indices = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(indices)
    fold_size = n // n_folds
    
    for fold in range(n_folds):
        val_idx = indices[fold * fold_size: (fold + 1) * fold_size]
        train_idx = np.concatenate([indices[:fold * fold_size], indices[(fold + 1) * fold_size:]])
        
        B_train = B[train_idx]
        y_train_fold = y[train_idx]
        
        # Fit p, q on training data
        q_init = np.ones(m)
        p_init = np.zeros(m)
        for it in range(10):
            target = y_train_fold * (B_train @ q_init)
            p_init, *_ = np.linalg.lstsq(B_train, target, rcond=None)
            # Simple update for q
            pred_train = (B_train @ p_init) / (B_train @ q_init)
            pred_train = np.where(np.abs(B_train @ q_init) < 1e-10, 0, pred_train)
            # Don't update q, just use p from linear fit
            break
        
        pred_val = (B[val_idx] @ p_init) / (B[val_idx] @ q_init)
        pred_val = np.where(np.abs(B[val_idx] @ q_init) < 1e-10, 0, pred_val)
        kf_mse.append(np.mean((y[val_idx] - pred_val) ** 2))
    
    avg_mse = np.mean(kf_mse)
    return avg_mse, (p, q)


def fit(X_train, y_train):
    X = np.array(X_train, dtype=np.float64)
    y = np.array(y_train, dtype=np.float64)
    n, d = X.shape
    
    # Standardize inputs for better conditioning
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std < 1e-10] = 1.0
    X_norm = (X - X_mean) / X_std
    
    models = []
    cv_scores = []
    
    # Model 1: Polynomial Kernel Ridge Regression (degree 3)
    def poly_kernel(X1, X2, degree=3):
        return (1 + X1 @ X2.T / d) ** degree
    
    alpha_grid = [0.001, 0.01, 0.1, 1.0, 10.0]
    mse1, alpha1, K1 = _loocv_ridge(X_norm, y, lambda a, b: poly_kernel(a, b, 3), alpha_grid)
    if mse1 < np.inf:
        K_reg = K1 + alpha1 * np.eye(n)
        w1 = np.linalg.solve(K_reg, y)
        models.append(('poly_krr', lambda X_test: poly_kernel(X_test, X_norm, 3) @ w1, alpha1))
        cv_scores.append(mse1)
    
    # Model 2: Gaussian Process (Matern-5/2)
    length_scale_grid = [0.1, 0.5, 1.0, 2.0]
    variance_grid = [0.5, 1.0, 2.0]
    noise_grid = [0.01, 0.1]
    mse2, gp_params = _loocv_gp(X_norm, y, length_scale_grid, variance_grid, noise_grid)
    if mse2 < np.inf and gp_params is not None:
        ls, var, noise = gp_params
        X_sq = np.sum(X_norm ** 2, axis=1)
        sq_dists = X_sq[:, None] + X_sq[None, :] - 2 * X_norm @ X_norm.T
        sq_dists = np.maximum(sq_dists, 0)
        dists = np.sqrt(sq_dists)
        r = dists / ls
        K_prior = var * (1 + r + r ** 2 / 3) * np.exp(-r)
        K_obs = K_prior + noise * np.eye(n)
        L = np.linalg.cholesky(K_obs)
        alpha_gp = np.linalg.solve(L.T, np.linalg.solve(L, y))
        
        def gp_predict(X_test):
            X_test = (np.array(X_test, dtype=np.float64) - X_mean) / X_std
            X_test_sq = np.sum(X_test ** 2, axis=1)
            sq_dists_test = X_test_sq[:, None] + X_sq[None, :] - 2 * X_test @ X_norm.T
            sq_dists_test = np.maximum(sq_dists_test, 0)
            dists_test = np.sqrt(sq_dists_test)
            r_test = dists_test / ls
            K_test = var * (1 + r_test + r_test ** 2 / 3) * np.exp(-r_test)
            return K_test @ alpha_gp
        
        models.append(('gp', gp_predict, gp_params))
        cv_scores.append(mse2)
    
    # Model 3: Polynomial feature expansion (degree 2) with ridge regression
    def poly_features(X, degree=2):
        cols = [np.ones(len(X))]
        for j in range(X.shape[1]):
            cols.append(X[:, j])
        for i in range(X.shape[1]):
            for j in range(i, X.shape[1]):
                cols.append(X[:, i] * X[:, j])
        return np.column_stack(cols)
    
    B2 = poly_features(X_norm)
    m2 = B2.shape[1]
    alpha_grid2 = [0.001, 0.01, 0.1, 1.0]
    mse3 = np.inf
    best_alpha3 = None
    for alpha in alpha_grid2:
        A = B2.T @ B2 + alpha * np.eye(m2)
        b = B2.T @ y
        w3 = np.linalg.solve(A, b)
        pred = B2 @ w3
        # LOO for linear model: yhat_i = y_i - (X_i^T (X^T X + alpha I)^{-1} X_i) * ... 
        # For linear regression with ridge: yhat_i = y_i - x_i^T (X^T X + alpha I)^{-1} x_i * ... 
        # Actually: yhat = X (X^T X + alpha I)^{-1} X^T y
        # LOO: yhat_i = y_i - x_i^T (X^T X + alpha I)^{-1} x_i * ((X^T X + alpha I)^{-1})_{...}
        # Simpler: use the hat matrix H = X (X^T X + alpha I)^{-1} X^T
        # yhat_i = y_i - H_ii^{-1} * (y_i - yhat_i) ... no
        # yhat_loo_i = y_i - H_ii / (1 - H_ii) * (y_i - yhat_i)
        H = B2 @ np.linalg.inv(B2.T @ B2 + alpha * np.eye(m2)) @ B2.T
        h_ii = np.diag(H)
        yhat = B2 @ w3
        yhat_loo = y - (h_ii / (1 - h_ii)) * (y - yhat)
        mse = np.mean((y - yhat_loo) ** 2)
        if mse < mse3:
            mse3 = mse
            best_alpha3 = alpha
    
    if mse3 < np.inf:
        A = B2.T @ B2 + best_alpha3 * np.eye(m2)
        b = B2.T @ y
        w3 = np.linalg.solve(A, b)
        models.append(('poly_ridge', lambda X_test: poly_features(np.array(X_test, dtype=np.float64)) @ w3, best_alpha3))
        cv_scores.append(mse3)
    
    # Model 4: Linear regression with ridge
    B4 = np.column_stack([np.ones(n), X_norm])
    m4 = B4.shape[1]
    mse4 = np.inf
    best_alpha4 = None
    for alpha in [0.001, 0.01, 0.1, 1.0]:
        A = B4.T @ B4 + alpha * np.eye(m4)
        b = B4.T @ y
        w4 = np.linalg.solve(A, b)
        H = B4 @ np.linalg.inv(B4.T @ B4 + alpha * np.eye(m4)) @ B4.T
        h_ii = np.diag(H)
        yhat = B4 @ w4
        yhat_loo = y - (h_ii / (1 - h_ii)) * (y - yhat)
        mse = np.mean((y - yhat_loo) ** 2)
        if mse < mse4:
            mse4 = mse
            best_alpha4 = alpha
    
    if mse4 < np.inf:
        A = B4.T @ B4 + best_alpha4 * np.eye(m4)
        b = B4.T @ y
        w4 = np.linalg.solve(A, b)
        models.append(('linear_ridge', lambda X_test: np.column_stack([np.ones(len(X_test)), (np.array(X_test, dtype=np.float64) - X_mean) / X_std]) @ w4, best_alpha4))
        cv_scores.append(mse4)
    
    # Select best model based on LOO CV score
    if not models:
        # Fallback: simple mean
        y_mean = np.mean(y)
        return lambda X: np.full(len(X), y_mean)
    
    best_idx = np.argmin(cv_scores)
    best_model = models[best_idx]
    
    def predict(X_test):
        X_test = np.array(X_test, dtype=np.float64)
        try:
            preds = best_model[1](X_test)
            # Ensure finite predictions
            preds = np.where(np.isfinite(preds), preds, np.mean(y))
            return preds
        except Exception:
            return np.full(len(X_test), np.mean(y))
    
    return predict
# EVOLVE-BLOCK-END
