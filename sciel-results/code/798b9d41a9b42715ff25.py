import time
import numpy as np
import scipy.linalg

def fit(X_train, y_train):
    """
    Fit a regression model and return a predict callable.
    Uses Kernel Ridge Regression (RBF) with automated hyperparameter selection
    via Leave-One-Out Cross-Validation, combined with robust parametric baselines.
    """
    start_time = time.time()
    deadline = start_time + 280  # 20s safety margin

    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n, d = X.shape
    x = X[:, 0]

    def elapsed():
        return time.time() - start_time

    def remaining():
        return deadline - time.time()

    best_score = np.inf
    best_predictor = None

    # Helper to check time
    def time_check():
        return time.time() < deadline - 5

    # 1. Kernel Ridge Regression (RBF)
    # K(x, x') = exp(-||x - x'||^2 / (2 * sigma^2))
    # Prediction: y = K(X_new, X_train) @ (K(X_train, X_train) + alpha*I)^-1 @ y_train
    # LOO CV Score for KRR:
    # e_i = (y_i - y_hat_i) / (1 - (K_inv @ y)_i / y_i) ... actually simpler:
    # Let A = K + alpha*I. Let B = A^-1.
    # y_hat_i (LOO) = y_i - (y_i - y_hat_i_full) / (1 - B_ii)
    # Actually, the standard formula for LOO error in KRR is:
    # e_i = (y_i - y_hat_i) / (1 - h_ii) where h_ii is the i-th diagonal element of H = K @ (K + alpha*I)^-1
    # Or more directly:
    # y_hat_loo = y - (K + alpha*I)^-1 (y - y_hat_full) ... no.
    
    # Efficient LOO for KRR:
    # Let H = K @ inv(K + alpha * I).
    # y_hat_loo_i = y_i - (y_i - y_hat_full_i) / (1 - H_ii)
    # MSE_loo = mean( (y - y_hat_loo)^2 )
    
    def build_krr_rbf(sigma, alpha):
        """
        Returns a tuple (lo_score, predict_func) or (inf, None) if failed.
        """
        try:
            # Precompute Kernel Matrix for training
            # x is (n, 1) or (n,). Here x is 1D array of length n.
            # K_ij = exp(-|x_i - x_j|^2 / (2*sigma^2))
            x_col = x.reshape(-1, 1)
            # Use broadcasting for distance
            diff = x_col - x_col.T
            sq_dist = np.sum(diff ** 2, axis=1)
            
            if sigma <= 0:
                return np.inf, None
                
            K = np.exp(-sq_dist / (2 * sigma ** 2))
            
            # Regularization
            A = K + alpha * np.eye(n)
            
            # Solve for weights w = A^-1 y
            # We need H = K A^-1 for LOO, or just compute y_hat and use the formula.
            # Let's compute w = A^-1 y
            try:
                w = scipy.linalg.solve(A, y, assume_a='pos')
            except:
                # Fallback to lstsq if singular
                w, *_ = np.linalg.lstsq(A, y, rcond=None)
                
            if not np.all(np.isfinite(w)):
                return np.inf, None

            # Full predictions
            y_hat = K @ w
            
            # LOO CV calculation
            # H = K @ A^-1
            # We can compute H_ii = sum_j K_ij * (A^-1)_ji
            # Or use the identity: y_hat_loo = y - (y - y_hat) / (1 - H_ii)
            # Let's compute H_diag.
            # H = K @ A^-1.
            # Instead of full H, note that (K + alpha I) w = y => K w + alpha w = y.
            # This doesn't directly give H_diag easily without inverse.
            # Let's just compute A_inv = inv(A) since n=200 is small.
            try:
                A_inv = scipy.linalg.inv(A)
            except:
                A_inv = np.linalg.pinv(A)
                
            H_diag = np.sum(K * A_inv, axis=1) # Element-wise product then sum? No.
            # H = K @ A_inv. H_ii = sum_j K_ij A_inv_ji.
            # So H_diag = np.sum(K * A_inv, axis=1) is correct if A_inv is symmetric (it is).
            
            # Check for numerical stability in denominator
            denom = 1 - H_diag
            # If denom is close to 0, LOO is unstable.
            if np.any(np.abs(denom) < 1e-10):
                # Fall back to simple MSE or k-fold?
                # For now, return inf to be safe, or use a small epsilon.
                denom = np.where(np.abs(denom) < 1e-10, 1e-10, denom)
            
            y_hat_loo = y - (y - y_hat) / denom
            loo_mse = np.mean((y - y_hat_loo) ** 2)
            
            if not np.isfinite(loo_mse):
                return np.inf, None

            # Define predictor
            def predict(X_in):
                X_in = np.asarray(X_in, dtype=np.float64)
                xi = X_in[:, 0].reshape(-1, 1)
                # Compute kernel between test and train
                diff_test = xi - x_col.T
                sq_dist_test = np.sum(diff_test ** 2, axis=1)
                K_test = np.exp(-sq_dist_test / (2 * sigma ** 2))
                pred = K_test @ w
                return pred

            return loo_mse, predict

        except Exception:
            return np.inf, None

    # 2. Polynomial Basis
    def build_poly(deg):
        try:
            # Check condition number to avoid instability
            x_fit = x
            A = np.vander(x_fit, deg + 1, increasing=True)
            # Simple check
            if deg > 10:
                 # High degree polys are unstable, use regularization or skip
                 pass
            
            # Use LOO for polynomial too?
            # For linear models, LOO is cheap if we have A_inv.
            # Let's just use 5-fold CV for polys to be safe and distinct from KRR logic
            # Or just use full train MSE as a proxy? No, CV is better.
            
            # Let's stick to a quick 5-fold CV
            indices = np.arange(n)
            np.random.seed(42)
            np.random.shuffle(indices)
            folds = np.array_split(indices, 5)
            errors = []
            for i in range(5):
                val_idx = folds[i]
                tr_idx = np.concatenate([folds[j] for j in range(5) if j != i])
                
                x_tr = x[tr_idx]
                y_tr = y[tr_idx]
                x_val = x[val_idx]
                y_val = y[val_idx]
                
                A_tr = np.vander(x_tr, deg + 1, increasing=True)
                w, *_ = np.linalg.lstsq(A_tr, y_tr, rcond=None)
                if not np.all(np.isfinite(w)):
                    return np.inf, None
                A_val = np.vander(x_val, deg + 1, increasing=True)
                pred = A_val @ w
                if not np.all(np.isfinite(pred)):
                    return np.inf, None
                mse = np.mean((pred - y_val) ** 2)
                errors.append(mse)
            
            cv_mse = np.mean(errors)
            
            # Final fit
            A_full = np.vander(x, deg + 1, increasing=True)
            w_full, *_ = np.linalg.lstsq(A_full, y, rcond=None)
            
            def predict(X_in):
                xi = X_in[:, 0]
                A_in = np.vander(xi, deg + 1, increasing=True)
                return A_in @ w_full

            return cv_mse, predict
        except Exception:
            return np.inf, None

    # 3. Fourier Basis
    def build_fourier(max_k):
        try:
            indices = np.arange(n)
            np.random.seed(42)
            np.random.shuffle(indices)
            folds = np.array_split(indices, 5)
            errors = []
            
            for i in range(5):
                val_idx = folds[i]
                tr_idx = np.concatenate([folds[j] for j in range(5) if j != i])
                
                x_tr = x[tr_idx]
                y_tr = y[tr_idx]
                x_val = x[val_idx]
                y_val = y[val_idx]
                
                cols_tr = [np.ones(len(x_tr))]
                cols_val = [np.ones(len(x_val))]
                for k in range(1, max_k + 1):
                    cols_tr.append(np.sin(k * x_tr))
                    cols_tr.append(np.cos(k * x_tr))
                    cols_val.append(np.sin(k * x_val))
                    cols_val.append(np.cos(k * x_val))
                
                A_tr = np.column_stack(cols_tr)
                A_val = np.column_stack(cols_val)
                
                w, *_ = np.linalg.lstsq(A_tr, y_tr, rcond=None)
                if not np.all(np.isfinite(w)):
                    return np.inf, None
                pred = A_val @ w
                if not np.all(np.isfinite(pred)):
                    return np.inf, None
                mse = np.mean((pred - y_val) ** 2)
                errors.append(mse)
            
            cv_mse = np.mean(errors)
            
            # Final fit
            cols_full = [np.ones(n)]
            for k in range(1, max_k + 1):
                cols_full.append(np.sin(k * x))
                cols_full.append(np.cos(k * x))
            A_full = np.column_stack(cols_full)
            w_full, *_ = np.linalg.lstsq(A_full, y, rcond=None)
            
            def predict(X_in):
                xi = X_in[:, 0]
                cols_in = [np.ones(len(xi))]
                for k in range(1, max_k + 1):
                    cols_in.append(np.sin(k * xi))
                    cols_in.append(np.cos(k * xi))
                A_in = np.column_stack(cols_in)
                return A_in @ w_full

            return cv_mse, predict
        except Exception:
            return np.inf, None

    # Grid Search for KRR
    sigmas = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
    alphas = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
    
    # Prioritize KRR as it's the most powerful generic model
    for sigma in sigmas:
        if not time_check():
            break
        for alpha in alphas:
            if not time_check():
                break
            score, pred_func = build_krr_rbf(sigma, alpha)
            if score < best_score:
                best_score = score
                best_predictor = pred_func

    # Grid Search for Polynomials
    for deg in range(1, 11):
        if not time_check():
            break
        score, pred_func = build_poly(deg)
        if score < best_score:
            best_score = score
            best_predictor = pred_func

    # Grid Search for Fourier
    for max_k in range(1, 8):
        if not time_check():
            break
        score, pred_func = build_fourier(max_k)
        if score < best_score:
            best_score = score
            best_predictor = pred_func

    # Fallback: Mean
    if best_predictor is None:
        mean_y = np.mean(y)
        best_predictor = lambda X_in: np.full(len(X_in), mean_y)
        best_score = np.mean((y - mean_y) ** 2)

    # Final Safety Wrapper
    def final_predict(X_in):
        X_in = np.asarray(X_in, dtype=np.float64)
        try:
            res = best_predictor(X_in)
            res = np.asarray(res, dtype=np.float64)
            if res.shape != (len(X_in),):
                res = np.atleast_1d(res).flatten()
                if len(res) != len(X_in):
                    res = np.resize(res, len(X_in))
            if not np.all(np.isfinite(res)):
                # Replace non-finite values with mean
                res = np.where(np.isfinite(res), res, np.mean(y))
            return res
        except Exception:
            return np.full(len(X_in), np.mean(y))

    return final_predict
