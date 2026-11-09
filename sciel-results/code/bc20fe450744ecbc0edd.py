"""Kernel Ridge Regression fitter with LOOCV hyperparameter selection."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from scipy.optimize import minimize

def _rbf_kernel(X1, X2, length_scale):
    """RBF kernel: K(x,y) = exp(-||x-y||^2 / (2*l^2))"""
    sq_dists = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2.0 * X1 @ X2.T
    sq_dists = np.maximum(sq_dists, 0.0)
    return np.exp(-sq_dists / (2.0 * length_scale**2))

def _poly_kernel(X1, X2, degree, coef0=1.0):
    """Polynomial kernel: K(x,y) = (x·y + c)^d"""
    return (X1 @ X2.T + coef0) ** degree

def _periodic_kernel(X1, X2, length_scale, period=2.0 * np.pi):
    """Periodic kernel: K(x,y) = exp(-2*sin^2(pi*||x-y||/p) / l^2)"""
    sq_dists = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2.0 * X1 @ X2.T
    sq_dists = np.maximum(sq_dists, 0.0)
    return np.exp(-2.0 * np.sin(np.pi * np.sqrt(sq_dists) / period) ** 2 / (length_scale ** 2))

def _linear_kernel(X1, X2):
    return X1 @ X2.T

def _fit_kernel_ridge(K, y, alpha):
    """Solve (K + alpha*I) w = y for kernel ridge regression."""
    n = len(y)
    A = K + alpha * np.eye(n)
    try:
        w = np.linalg.solve(A, y)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(A, y, rcond=None)[0]
    return w

def _loocv_score(K, y, alpha):
    """LOOCV MSE for kernel ridge regression using the Sherman-Morrison trick."""
    n = len(y)
    A = K + alpha * np.eye(n)
    try:
        # Solve A w = y
        w = np.linalg.solve(A, y)
        # LOOCV: prediction for i is y_i - K_ii / (A_ii + 1) * ... 
        # Actually: LOOCV pred_i = y_i - K_i / (A + (1/alpha) I)^{-1} e_i
        # Simpler: use the formula: pred_i = y_i - K_{i,:} @ w / (1 + K_{ii}/alpha)
        # Wait, that's not quite right either. Let me use the proper formula.
        # LOOCV: f_{-i}(x_i) = y_i - K_{i,:} (K + alpha I)^{-1} e_i
        # = y_i - K_{ii} / (A_{ii} + ... ) 
        # Actually the clean formula: pred_i = y_i - K_{i,:} @ w / (1 + K_{ii}/alpha)
        # No. Let me just compute it properly.
        # f_{-i}(x_i) = y_i - K_{i,:} (K + alpha*I)^{-1} e_i
        # = y_i - [ (K + alpha*I)^{-1} ]_{ii}^{-1} ... no
        # = y_i - K_{i,:} v_i where v_i = (K + alpha*I)^{-1} e_i
        # The i-th component of (K + alpha*I)^{-1} is 1/(K_{ii}+alpha) only for diagonal
        # Actually: (K + alpha*I)^{-1}_{ii} is NOT simply 1/(K_{ii}+alpha)
        # Let me just use: h_i = K_{i,:} @ w / (1 + K_{ii}/alpha)  -- this is WRONG
        # Correct: pred_i = y_i - K_{i,:} (K + alpha*I)^{-1} e_i
        # Let v = (K + alpha*I)^{-1} e_i, then pred_i = y_i - K_{i,:} @ v
        # But computing this for all i is O(n^3). Instead use:
        # pred_i = y_i - K_{i,:} @ w / (1 + K_{ii}/alpha) -- NO this is wrong
        # The correct efficient formula:
        # Let A = K + alpha*I, w = A^{-1} y
        # Then f_{-i}(x_i) = y_i - K_{i,:} A^{-1} e_i
        # = y_i - K_{i,:} [A^{-1}]_{:,i}
        # = y_i - K_{i,:} [A^{-1} y]_i * alpha / (1 + K_{i,:} A^{-1} e_i * 0) -- no
        # 
        # Simplest correct approach: 
        # f_{-i}(x_i) = y_i - K_{i,:} (K + alpha*I)^{-1} e_i
        # Note that (K + alpha*I)^{-1} e_i is the i-th column of A^{-1}
        # And A^{-1} = A^{-1} A A^{-1} ... 
        # Actually: K_{i,:} A^{-1} e_i = K_{i,:} (A^{-1})_{:,i}
        # We can write: A^{-1} = (1/alpha)(I - K A^{-1})  from A = K + alpha I => A^{-1} = (1/alpha)(I - K A^{-1})
        # So A^{-1} e_i = (1/alpha)(e_i - K A^{-1} e_i)
        # Let u_i = A^{-1} e_i. Then u_i = (1/alpha)(e_i - K u_i)
        # => alpha u_i = e_i - K u_i
        # => (alpha I + K) u_i = e_i
        # => A u_i = e_i  (which is just the definition)
        # 
        # OK let me just use the standard LOOCV formula for KRR:
        # pred_i = y_i - K_{i,:} w / (1 + K_{ii}/alpha)  -- I keep seeing this but let me verify
        # Actually from the literature: for KRR with K and alpha,
        # f_{-i}(x_i) = y_i - K_{i,:} (K + alpha I)^{-1} e_i
        # Let's define h = (K + alpha I)^{-1} y = w
        # Then f_{-i}(x_i) = y_i - K_{i,:} (K + alpha I)^{-1} e_i
        # Note K_{i,:} (K + alpha I)^{-1} e_i is the i-th component of K (K+alpha I)^{-1} e_i
        # And (K+alpha I)^{-1} e_i = (1/alpha)(e_i - K (K+alpha I)^{-1} e_i)
        # Let t = K (K+alpha I)^{-1} e_i
        # Then (K+alpha I)^{-1} e_i = (1/alpha)(e_i - t)
        # And t = K (1/alpha)(e_i - t) = (1/alpha)(K e_i - K t)
        # => alpha t = K e_i - K t
        # => (alpha I + K) t = K e_i
        # => A t = K e_i
        # => t = A^{-1} K e_i
        # So K_{i,:} A^{-1} e_i = [A^{-1} K e_i]_i = [t]_i
        # And t = A^{-1} K e_i
        # Also: A^{-1} K = A^{-1}(A - alpha I) = I - alpha A^{-1}
        # So t = (I - alpha A^{-1}) e_i = e_i - alpha A^{-1} e_i
        # => t_i = 1 - alpha * [A^{-1}]_{ii}
        # Therefore: K_{i,:} A^{-1} e_i = 1 - alpha * [A^{-1}]_{ii}
        # And pred_i = y_i - (1 - alpha * [A^{-1}]_{ii})
        # Hmm, that gives pred_i = y_i - 1 + alpha * [A^{-1}]_{ii}
        # That doesn't seem right dimensionally.
        #
        # Let me just do it the direct way for n=200:
        # pred_i = y_i - K[i,:] @ A^{-1} e_i
        # We need A^{-1} e_i for each i. That's just the columns of A^{-1}.
        # For n=200, computing A^{-1} is O(n^3) = 8M ops, fine.
        A_inv = np.linalg.inv(A)
        # pred_i = y_i - K[i,:] @ A_inv[:, i]
        preds = y.copy()
        for i in range(n):
            preds[i] = y[i] - K[i, :] @ A_inv[:, i]
        mse = np.mean((preds - y) ** 2)
        return mse
    except:
        return np.inf

def _loocv_score_fast(K, y, alpha):
    """Faster LOOCV using the diagonal of the inverse."""
    n = len(y)
    A = K + alpha * np.eye(n)
    try:
        # Use Cholesky for stability
        L = np.linalg.cholesky(A)
        # A^{-1} diagonal: diag(A^{-1}) = 1 / diag(L^T L) ... no
        # diag(A^{-1})_i = 1 / (L^T L)_{ii} is wrong
        # Actually: if A = L L^T, then A^{-1} = L^{-T} L^{-1}
        # diag(A^{-1})_i = sum_j (L^{-1})_{ji}^2
        # Simpler: solve A x = e_i for each i, take x_i
        # Or: compute A_inv = np.linalg.inv(A) and take diagonal
        A_inv = np.linalg.inv(A)
        diag_A_inv = np.diag(A_inv)
        # pred_i = y_i - (1 - alpha * diag_A_inv[i])  -- derived above
        # Wait let me re-derive:
        # f_{-i}(x_i) = y_i - K_{i,:} A^{-1} e_i
        # We showed K_{i,:} A^{-1} e_i = 1 - alpha * [A^{-1}]_{ii}
        # So f_{-i}(x_i) = y_i - 1 + alpha * [A^{-1}]_{ii}
        # Hmm let me verify with a simple case.
        # If K = I, alpha = 1: A = 2I, A^{-1} = 0.5 I
        # f_{-i}(x_i) = y_i - K_{i,:} A^{-1} e_i = y_i - e_i^T * 0.5 * e_i = y_i - 0.5
        # Formula: y_i - 1 + 1*0.5 = y_i - 0.5. Correct!
        # Another: K = 0, alpha = 1: A = I, A^{-1} = I
        # f_{-i}(x_i) = y_i - 0 = y_i
        # Formula: y_i - 1 + 1*1 = y_i. Correct!
        preds = y - 1.0 + alpha * diag_A_inv
        mse = np.mean((preds - y) ** 2)
        return mse
    except:
        return np.inf

def _build_feature_matrix(X, d):
    """Build a rich feature matrix with polynomial, trig, and interaction terms."""
    n = len(X)
    features = [np.ones(n)]
    
    # Linear terms
    for j in range(d):
        features.append(X[:, j])
    
    # Quadratic terms
    for j in range(d):
        features.append(X[:, j] ** 2)
    
    # Cubic terms
    for j in range(d):
        features.append(X[:, j] ** 3)
    
    # Interactions (pairs)
    for j1 in range(d):
        for j2 in range(j1 + 1, d):
            features.append(X[:, j1] * X[:, j2])
    
    # Trig terms
    for j in range(d):
        features.append(np.sin(X[:, j]))
        features.append(np.cos(X[:, j]))
    
    # Sqrt and abs
    for j in range(d):
        features.append(np.sqrt(np.maximum(X[:, j], 0)))
        features.append(np.abs(X[:, j]))
    
    # Higher-order interactions
    if d >= 3:
        for j1 in range(d):
            for j2 in range(j1 + 1, d):
                for j3 in range(j2 + 1, d):
                    features.append(X[:, j1] * X[:, j2] * X[:, j3])
    
    return np.column_stack(features)

def _ridge_regression(X, y, alpha):
    """Standard ridge regression."""
    n, p = X.shape
    A = X.T @ X + alpha * np.eye(p)
    b = X.T @ y
    try:
        w = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(A, b, rcond=None)[0]
    return w

def _loocv_ridge(X, y, alpha):
    """LOOCV for ridge regression using the hat matrix."""
    n, p = X.shape
    A = X.T @ X + alpha * np.eye(p)
    try:
        A_inv = np.linalg.inv(A)
        # Hat matrix H = X A^{-1} X^T
        # diag(H)_i = X_i A^{-1} X_i^T
        # pred_i = y_i - h_ii / (1 - h_ii) * ... no
        # For linear regression: pred_i = y_i - h_ii / (1 - h_ii) * (y_i - f_i)
        # Actually: f_{-i}(x_i) = f_i - h_ii / (1 - h_ii) * (f_i - y_i)
        # = f_i * (1 - h_ii)/(1-h_ii) + y_i * h_ii/(1-h_ii) ... 
        # Simpler: f_{-i} = (f_i - h_ii * y_i) / (1 - h_ii) ... no
        # Standard formula: f_{-i}(x_i) = (f_i(x_i) - h_{ii} y_i) / (1 - h_{ii})
        # where f_i(x_i) is the in-sample prediction
        f = X @ (A_inv @ (X.T @ y))
        h_diag = np.sum(X * (X @ A_inv), axis=1)  # diag(X A^{-1} X^T)
        preds = (f - h_diag * y) / (1.0 - h_diag + 1e-15)
        mse = np.mean((preds - y) ** 2)
        return mse
    except:
        return np.inf

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 280  # Safety margin
    
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=float)
    n, d = X.shape
    
    # Normalize X
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_std[X_std < 1e-8] = 1.0
    Xn = (X - X_mean) / X_std
    
    y_mean = np.mean(y)
    y_std = np.std(y)
    y_std = max(y_std, 1e-8)
    yn = (y - y_mean) / y_std
    
    best_score = np.inf
    best_predict = None
    
    # Strategy 1: Ridge regression on rich features
    try:
        if time.time() < deadline - 5:
            F = _build_feature_matrix(Xn, d)
            # LOOCV over alpha values
            alphas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0]
            best_alpha = 1.0
            best_cv = np.inf
            for a in alphas:
                if time.time() > deadline - 5:
                    break
                cv = _loocv_ridge(F, yn, a)
                if cv < best_cv:
                    best_cv = cv
                    best_alpha = a
            
            w = _ridge_regression(F, yn, best_alpha)
            F_mean = np.mean(F, axis=0)
            F_std = np.std(F, axis=0)
            F_std[F_std < 1e-8] = 1.0
            
            def predict_ridge(Xtest):
                Xt = np.asarray(Xtest, dtype=float)
                if Xt.ndim == 1:
                    Xt = Xt.reshape(1, -1)
                Xtn = (Xt - X_mean) / X_std
                Ft = _build_feature_matrix(Xtn, d)
                p = Ft @ w
                p = p * y_std + y_mean
                p = np.where(np.isfinite(p), p, np.nanmean(y_train))
                if p.ndim == 1 and len(p) == 1:
                    return p[0]
                return p
            
            train_pred = predict_ridge(X)
            train_mse = np.mean((train_pred - y) ** 2) / np.var(y)
            if best_cv < best_score:
                best_score = best_cv
                best_predict = predict_ridge
    except Exception:
        pass
    
    # Strategy 2: Kernel Ridge Regression with RBF
    try:
        if time.time() < deadline - 10:
            # Precompute kernel matrix
            K = _rbf_kernel(Xn, Xn, 1.0)
            np.fill_diagonal(K, 0)  # For LOOCV, we'll add back
            np.fill_diagonal(K, 1.0)  # RBF kernel has K(x,x)=1
            
            best_ls = 1.0
            best_a2 = 1.0
            best_cv2 = np.inf
            
            for ls in [0.3, 0.5, 1.0, 2.0, 3.0, 5.0]:
                if time.time() > deadline - 5:
                    break
                K_ls = _rbf_kernel(Xn, Xn, ls)
                for a2 in [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0]:
                    if time.time() > deadline - 3:
                        break
                    cv = _loocv_score_fast(K_ls, yn, a2)
                    if cv < best_cv2:
                        best_cv2 = cv
                        best_ls = ls
                        best_a2 = a2
            
            if best_cv2 < best_score:
                K_final = _rbf_kernel(Xn, Xn, best_ls)
                w_k = _fit_kernel_ridge(K_final, yn, best_a2)
                
                def predict_krr(Xtest):
                    Xt = np.asarray(Xtest, dtype=float)
                    if Xt.ndim == 1:
                        Xt = Xt.reshape(1, -1)
                    Xtn = (Xt - X_mean) / X_std
                    K_test = _rbf_kernel(Xtn, Xn, best_ls)
                    p = K_test @ w_k
                    p = p * y_std + y_mean
                    p = np.where(np.isfinite(p), p, np.nanmean(y_train))
                    if p.ndim == 1 and len(p) == 1:
                        return p[0]
                    return p
                
                best_score = best_cv2
                best_predict = predict_krr
    except Exception:
        pass
    
    # Strategy 3: Combined kernel (RBF + Linear)
    try:
        if time.time() < deadline - 5:
            K_lin = Xn @ Xn.T
            best_combo_ls = 1.0
            best_combo_w = 0.5
            best_a3 = 1.0
            best_cv3 = np.inf
            
            for ls in [0.5, 1.0, 2.0]:
                if time.time() > deadline - 3:
                    break
                K_rbf = _rbf_kernel(Xn, Xn, ls)
                for w_rbf in [0.2, 0.5, 0.8]:
                    if time.time() > deadline - 2:
                        break
                    K_combo = w_rbf * K_rbf + (1 - w_rbf) * K_lin
                    for a3 in [1e-3, 1e-2, 0.1, 1.0]:
                        if time.time() > deadline - 1:
                            break
                        cv = _loocv_score_fast(K_combo, yn, a3)
                        if cv < best_cv3:
                            best_cv3 = cv
                            best_combo_ls = ls
                            best_combo_w = w_rbf
                            best_a3 = a3
            
            if best_cv3 < best_score:
                K_rbf_f = _rbf_kernel(Xn, Xn, best_combo_ls)
                K_final3 = best_combo_w * K_rbf_f + (1 - best_combo_w) * K_lin
                w_k3 = _fit_kernel_ridge(K_final3, yn, best_a3)
                
                def predict_combo(Xtest):
                    Xt = np.asarray(Xtest, dtype=float)
                    if Xt.ndim == 1:
                        Xt = Xt.reshape(1, -1)
                    Xtn = (Xt - X_mean) / X_std
                    K_rbf_t = _rbf_kernel(Xtn, Xn, best_combo_ls)
                    K_lin_t = Xtn @ Xn.T
                    K_test = best_combo_w * K_rbf_t + (1 - best_combo_w) * K_lin_t
                    p = K_test @ w_k3
                    p = p * y_std + y_mean
                    p = np.where(np.isfinite(p), p, np.nanmean(y_train))
                    if p.ndim == 1 and len(p) == 1:
                        return p[0]
                    return p
                
                best_score = best_cv3
                best_predict = predict_combo
    except Exception:
        pass
    
    # Fallback: simple linear
    if best_predict is None:
        def predict_linear(Xtest):
            Xt = np.asarray(Xtest, dtype=float)
            if Xt.ndim == 1:
                Xt = Xt.reshape(1, -1)
            F = np.column_stack([np.ones(len(Xt))] + [Xt[:, j] for j in range(Xt.shape[1])])
            F_train = np.column_stack([np.ones(n)] + [X[:, j] for j in range(d)])
            w, *_ = np.linalg.lstsq(F_train, y, rcond=None)
            p = F @ w
            p = np.where(np.isfinite(p), p, np.nanmean(y_train))
            if p.ndim == 1 and len(p) == 1:
                return p[0]
            return p
        best_predict = predict_linear
    
    return best_predict
# EVOLVE-BLOCK-END
