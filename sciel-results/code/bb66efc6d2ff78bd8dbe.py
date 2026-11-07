# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve
import warnings

warnings.filterwarnings('ignore')

def fit(X_train, y_train):
    t_start = time.time()
    deadline = t_start + 280  # safety margin
    
    n, d = X_train.shape
    y = np.asarray(y_train, dtype=np.float64).ravel()
    
    # Normalize inputs to roughly [0, 1] based on data range
    X_min = X_train.min(axis=0)
    X_max = X_train.max(axis=0)
    X_range = X_max - X_min
    X_range[X_range < 1e-12] = 1.0
    Xn = (X_train - X_min) / X_range
    
    def build_features(X_raw, Xn_arr):
        """Build a rich feature dictionary from raw and normalized inputs."""
        Xr = np.asarray(X_raw, dtype=np.float64)
        Xn_arr = np.asarray(Xn_arr, dtype=np.float64)
        nr = Xr.shape[0]
        dr = Xr.shape[1]
        
        feats = [np.ones(nr)]
        names = ['const']
        
        # Linear terms (raw)
        for j in range(dr):
            feats.append(Xr[:, j])
            names.append(f'x{j}')
        
        # Quadratic and cubic polynomial terms (raw)
        for i in range(dr):
            for j in range(i, dr):
                feats.append(Xr[:, i] * Xr[:, j])
                names.append(f'x{i}*x{j}')
        
        # Cubic terms
        for i in range(dr):
            for j in range(i, dr):
                for k in range(j, dr):
                    feats.append(Xr[:, i] * Xr[:, j] * Xr[:, k])
                    names.append(f'x{i}*x{j}*x{k}')
        
        # Fourth degree (only for small d)
        if dr <= 2:
            for i in range(dr):
                for j in range(i, dr):
                    for k in range(j, dr):
                        for m in range(k, dr):
                            feats.append(Xr[:, i] * Xr[:, j] * Xr[:, k] * Xr[:, m])
                            names.append(f'x{i}*x{j}*x{k}*x{m}')
        
        # Sin and cos of raw inputs
        for j in range(dr):
            feats.append(np.sin(Xr[:, j]))
            names.append(f'sin(x{j})')
            feats.append(np.cos(Xr[:, j]))
            names.append(f'cos(x{j})')
        
        # Sin and cos of 2x, 3x
        for j in range(dr):
            feats.append(np.sin(2 * Xr[:, j]))
            names.append(f'sin(2x{j})')
            feats.append(np.cos(2 * Xr[:, j]))
            names.append(f'cos(2x{j})')
        
        # Exp and log of raw inputs (with safety)
        for j in range(dr):
            # exp only if values are in reasonable range
            if np.max(np.abs(Xr[:, j])) < 20:
                feats.append(np.exp(Xr[:, j]))
                names.append(f'exp(x{j})')
            # log only if all positive
            if np.min(Xr[:, j]) > 0:
                feats.append(np.log(Xr[:, j]))
                names.append(f'log(x{j})')
                feats.append(np.log(Xr[:, j]) ** 2)
                names.append(f'log(x{j})^2')
        
        # Interactions: x_i * sin(x_j), x_i * cos(x_j)
        for i in range(dr):
            for j in range(dr):
                if i != j:
                    feats.append(Xr[:, i] * np.sin(Xr[:, j]))
                    names.append(f'x{i}*sin(x{j})')
                    feats.append(Xr[:, i] * np.cos(Xr[:, j]))
                    names.append(f'x{i}*cos(x{j})')
        
        # sin(x_i)*sin(x_j), cos(x_i)*cos(x_j), sin(x_i)*cos(x_j)
        for i in range(dr):
            for j in range(i+1, dr):
                feats.append(np.sin(Xr[:, i]) * np.sin(Xr[:, j]))
                names.append(f'sin(x{i})*sin(x{j})')
                feats.append(np.cos(Xr[:, i]) * np.cos(Xr[:, j]))
                names.append(f'cos(x{i})*cos(x{j})')
                feats.append(np.sin(Xr[:, i]) * np.cos(Xr[:, j]))
                names.append(f'sin(x{i})*cos(x{j})')
        
        # x_i^2 * sin(x_j), etc.
        for i in range(dr):
            for j in range(dr):
                if i != j:
                    feats.append(Xr[:, i]**2 * np.sin(Xr[:, j]))
                    names.append(f'x{i}^2*sin(x{j})')
        
        # sin(x_i + x_j), cos(x_i + x_j)
        for i in range(dr):
            for j in range(i+1, dr):
                feats.append(np.sin(Xr[:, i] + Xr[:, j]))
                names.append(f'sin(x{i}+x{j})')
                feats.append(np.cos(Xr[:, i] + Xr[:, j]))
                names.append(f'cos(x{i}+x{j})')
                feats.append(np.sin(Xr[:, i] - Xr[:, j]))
                names.append(f'sin(x{i}-x{j})')
                feats.append(np.cos(Xr[:, i] - Xr[:, j]))
                names.append(f'cos(x{i}-x{j})')
        
        # Product of all inputs
        if dr > 0:
            prod = np.ones(nr)
            for j in range(dr):
                prod = prod * Xr[:, j]
            feats.append(prod)
            names.append('prod_all')
        
        # sqrt of inputs (if positive)
        for j in range(dr):
            if np.min(Xr[:, j]) > 0:
                feats.append(np.sqrt(Xr[:, j]))
                names.append(f'sqrt(x{j})')
        
        # x^3, x^4 for single input
        if dr == 1:
            feats.append(Xr[:, 0]**3)
            names.append('x0^3')
            feats.append(Xr[:, 0]**4)
            names.append('x0^4')
            feats.append(np.sin(Xr[:, 0]) * Xr[:, 0])
            names.append('x0*sin(x0)')
            feats.append(np.cos(Xr[:, 0]) * Xr[:, 0])
            names.append('x0*cos(x0)')
        
        return np.column_stack(feats), names
    
    def make_predictor(F, w, X_min, X_range, feature_names):
        """Create a predict function that uses the learned weights."""
        def predict(X):
            Xr = np.asarray(X, dtype=np.float64)
            Xn_arr = (Xr - X_min) / X_range
            F_pred, _ = build_features(Xr, Xn_arr)
            return F_pred @ w
        return predict
    
    # Try LASSO-like approach: use ridge regression with very small lambda,
    # then zero out small coefficients, then refit
    def solve_with_regularization(F, y, alpha):
        """Ridge regression with given alpha."""
        n, p = F.shape
        # Add regularization
        A = F.T @ F + alpha * np.eye(p)
        b = F.T @ y
        try:
            w = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(F, y, rcond=None)[0]
        return w
    
    def solve_lstsq(F, y):
        """Plain least squares."""
        w, residuals, rank, sv = np.linalg.lstsq(F, y, rcond=None)
        return w
    
    # Build features
    F, feat_names = build_features(X_train, Xn)
    p = F.shape[1]
    
    # Remove constant/linearly dependent features
    # Center features for numerical stability
    F_mean = F.mean(axis=0)
    F_std = F.std(axis=0)
    F_std[F_std < 1e-12] = 1.0
    F_centered = (F - F_mean) / F_std
    
    y_mean = y.mean()
    y_std = y.std()
    if y_std < 1e-12:
        y_std = 1.0
    y_centered = (y - y_mean) / y_std
    
    # Strategy: Try multiple alphas for ridge, pick best training fit
    # Then try sparse selection
    
    best_w = None
    best_train_mse = np.inf
    best_alpha = None
    
    # Try a range of alphas
    alphas = [0.0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 0.1, 1.0]
    
    for alpha in alphas:
        if time.time() > deadline:
            break
        if alpha == 0.0:
            w_c = solve_lstsq(F_centered, y_centered)
        else:
            w_c = solve_with_regularization(F_centered, y_centered, alpha)
        
        # Compute training MSE
        pred_c = F_centered @ w_c
        mse = np.mean((pred_c - y_centered)**2)
        
        if mse < best_train_mse:
            best_train_mse = mse
            best_w = w_c.copy()
            best_alpha = alpha
    
    # Now try sparse selection: zero out small coefficients and refit
    if best_w is not None and time.time() < deadline:
        # Sort coefficients by magnitude
        abs_w = np.abs(best_w)
        # Try different sparsity levels
        for frac in [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]:
            if time.time() > deadline:
                break
            if frac == 0.0:
                mask = np.ones(p, dtype=bool)
            else:
                threshold = np.percentile(abs_w, frac * 100)
                mask = abs_w > threshold
            
            if mask.sum() < 1:
                mask = abs_w == np.max(abs_w)
            
            F_sub = F_centered[:, mask]
            w_sub, *_ = np.linalg.lstsq(F_sub, y_centered, rcond=None)
            
            pred_c = F_sub @ w_sub
            mse = np.mean((pred_c - y_centered)**2)
            
            if mse < best_train_mse:
                best_train_mse = mse
                # Reconstruct full w
                best_w = np.zeros(p)
                best_w[mask] = w_sub
                best_alpha = 'sparse'
    
    # Also try: select top-k features by correlation with y
    if time.time() < deadline:
        # Compute correlations
        F_std2 = F_centered.std(axis=0)
        F_std2[F_std2 < 1e-12] = 1.0
        corr = np.abs(F_centered.T @ y_centered) / (n * F_std2)
        
        for k in [10, 20, 30, 50, 80, 100]:
            if time.time() > deadline:
                break
            if k >= p:
                k = p
            top_k = np.argsort(corr)[-k:]
            F_sub = F_centered[:, top_k]
            w_sub, *_ = np.linalg.lstsq(F_sub, y_centered, rcond=None)
            
            pred_c = F_sub @ w_sub
            mse = np.mean((pred_c - y_centered)**2)
            
            if mse < best_train_mse:
                best_train_mse = mse
                best_w = np.zeros(p)
                best_w[top_k] = w_sub
                best_alpha = f'top_k_{k}'
    
    # Final: also try plain LSTSQ on original (non-centered) features
    if time.time() < deadline:
        w_plain, *_ = np.linalg.lstsq(F, y, rcond=None)
        pred = F @ w_plain
        mse = np.mean((pred - y)**2)
        if mse < best_train_mse * y_std**2:  # compare in original scale
            best_w = w_plain
            best_alpha = 'plain_lstsq'
            # Need to handle differently - store flag
            best_w = None  # signal to use plain
    
    # Build final predictor
    # Determine which approach won
    # Re-evaluate in original scale
    def compute_original_mse(w_centered, w_plain):
        """Compute MSE in original scale."""
        if w_plain is not None:
            pred = F @ w_plain
            return np.mean((pred - y)**2)
        else:
            pred_c = F_centered @ w_centered
            pred = pred_c * y_std + y_mean
            return np.mean((pred - y)**2)
    
    # Let me redo this more carefully
    # Keep track of best in original scale
    best_mse_orig = np.inf
    best_approach = None
    best_params = None
    
    # 1. Plain LSTSQ on F
    if time.time() < deadline:
        try:
            w_plain, *_ = np.linalg.lstsq(F, y, rcond=None)
            pred = F @ w_plain
            mse = np.mean((pred - y)**2)
            if mse < best_mse_orig:
                best_mse_orig = mse
                best_approach = 'plain'
                best_params = w_plain
        except:
            pass
    
    # 2. Ridge on centered features, various alphas
    for alpha in [0.0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 0.1]:
        if time.time() > deadline:
            break
        try:
            if alpha == 0.0:
                w_c, *_ = np.linalg.lstsq(F_centered, y_centered, rcond=None)
            else:
                A = F_centered.T @ F_centered + alpha * np.eye(p)
                b = F_centered.T @ y_centered
                w_c = np.linalg.solve(A, b)
            
            pred_c = F_centered @ w_c
            pred = pred_c * y_std + y_mean
            mse = np.mean((pred - y)**2)
            if mse < best_mse_orig:
                best_mse_orig = mse
                best_approach = 'ridge_centered'
                best_params = (w_c, alpha)
        except:
            pass
    
    # 3. Sparse selection from best ridge
    if best_approach == 'ridge_centered' and time.time() < deadline:
        w_c_best = best_params[0]
        abs_w = np.abs(w_c_best)
        for frac in [0.1, 0.2, 0.3, 0.5]:
            if time.time() > deadline:
                break
            threshold = np.percentile(abs_w, frac * 100)
            mask = abs_w > threshold
            if mask.sum() < 2:
                mask = abs_w >= np.max(abs_w) * 0.1
            F_sub = F_centered[:, mask]
            w_sub, *_ = np.linalg.lstsq(F_sub, y_centered, rcond=None)
            pred_c = F_sub @ w_sub
            pred = pred_c * y_std + y_mean
            mse = np.mean((pred - y)**2)
            if mse < best_mse_orig:
                best_mse_orig = mse
                best_approach = 'sparse'
                best_params = (w_sub, mask)
    
    # 4. Top-k correlation features
    if time.time() < deadline:
        F_std2 = F_centered.std(axis=0)
        F_std2[F_std2 < 1e-12] = 1.0
        corr = np.abs(F_centered.T @ y_centered) / (n * F_std2)
        for k in [5, 10, 20, 30, 50, 80, min(100, p)]:
            if time.time() > deadline:
                break
            if k >= p:
                k = p - 1
            top_k = np.argsort(corr)[-k:]
            F_sub = F_centered[:, top_k]
            w_sub, *_ = np.linalg.lstsq(F_sub, y_centered, rcond=None)
            pred_c = F_sub @ w_sub
            pred = pred_c * y_std + y_mean
            mse = np.mean((pred - y)**2)
            if mse < best_mse_orig:
                best_mse_orig = mse
                best_approach = 'topk'
                best_params = (w_sub, top_k)
    
    # 5. Try even more features if time permits: higher degree polys, more trig
    if time.time() < deadline - 5:
        # Build extended features
        def build_extended_features(X_raw):
            Xr = np.asarray(X_raw, dtype=np.float64)
            nr = Xr.shape[0]
            dr = Xr.shape[1]
            
            feats = [np.ones(nr)]
            for j in range(dr):
                feats.append(Xr[:, j])
                feats.append(Xr[:, j]**2)
                feats.append(Xr[:, j]**3)
                if dr == 1:
                    feats.append(Xr[:, j]**4)
                    feats.append(Xr[:, j]**5)
                feats.append(np.sin(Xr[:, j]))
                feats.append(np.cos(Xr[:, j]))
                feats.append(np.sin(2*Xr[:, j]))
                feats.append(np.cos(2*Xr[:, j]))
                if np.max(np.abs(Xr[:, j])) < 20:
                    feats.append(np.exp(Xr[:, j]))
                if np.min(Xr[:, j]) > 0:
                    feats.append(np.log(Xr[:, j]))
                    feats.append(np.sqrt(Xr[:, j]))
            
            for i in range(dr):
                for j in range(i, dr):
                    feats.append(Xr[:, i] * Xr[:, j])
                    feats.append(Xr[:, i]**2 * Xr[:, j])
                    feats.append(Xr[:, i] * Xr[:, j]**2)
                    feats.append(np.sin(Xr[:, i]) * np.cos(Xr[:, j]))
                    feats.append(np.sin(Xr[:, i] + Xr[:, j]))
                    feats.append(np.cos(Xr[:, i] + Xr[:, j]))
                    feats.append(np.sin(Xr[:, i] - Xr[:, j]))
                    feats.append(np.cos(Xr[:, i] - Xr[:, j]))
                    feats.append(Xr[:, i] * np.sin(Xr[:, j]))
                    feats.append(Xr[:, i] * np.cos(Xr[:, j]))
            
            for i in range(dr):
                for j in range(i, dr):
                    for k in range(j, dr):
                        feats.append(Xr[:, i] * Xr[:, j] * Xr[:, k])
            
            return np.column_stack(feats)
        
        F_ext, _ = build_extended_features(X_train), None
        F_ext = build_extended_features(X_train)
        p_ext = F_ext.shape[1]
        
        # Center
        F_ext_mean = F_ext.mean(axis=0)
        F_ext_std = F_ext.std(axis=0)
        F_ext_std[F_ext_std < 1e-12] = 1.0
        F_ext_c = (F_ext - F_ext_mean) / F_ext_std
        
        # Plain LSTSQ
        try:
            w_ext, *_ = np.linalg.lstsq(F_ext, y, rcond=None)
            pred = F_ext @ w_ext
            mse = np.mean((pred - y)**2)
            if mse < best_mse_orig:
                best_mse_orig = mse
                best_approach = 'ext_plain'
                best_params = w_ext
        except:
            pass
        
        # Ridge
        for alpha in [1e-8, 1e-6, 1e-4, 1e-3]:
            if time.time() > deadline:
                break
            try:
                A = F_ext_c.T @ F_ext_c + alpha * np.eye(p_ext)
                b = F_ext_c.T @ y_centered
                w_c = np.linalg.solve(A, b)
                pred_c = F_ext_c @ w_c
                pred = pred_c * y_std + y_mean
                mse = np.mean((pred - y)**2)
                if mse < best_mse_orig:
                    best_mse_orig = mse
                    best_approach = 'ext_ridge'
                    best_params = (w_c, alpha)
            except:
                pass
        
        # Top-k from extended
        if time.time() < deadline:
            F_ext_std2 = F_ext_c.std(axis=0)
            F_ext_std2[F_ext_std2 < 1e-12] = 1.0
            corr_ext = np.abs(F_ext_c.T @ y_centered) / (n * F_ext_std2)
            for k in [20, 40, 60, 80, 100, min(150, p_ext)]:
                if time.time() > deadline:
                    break
                if k >= p_ext:
                    k = p_ext - 1
                top_k = np.argsort(corr_ext)[-k:]
                F_sub = F_ext_c[:, top_k]
                w_sub, *_ = np.linalg.lstsq(F_sub, y_centered, rcond=None)
                pred_c = F_sub @ w_sub
                pred = pred_c * y_std + y_mean
                mse = np.mean((pred - y)**2)
                if mse < best_mse_orig:
                    best_mse_orig = mse
                    best_approach = 'ext_topk'
                    best_params = (w_sub, top_k)
    
    # Build final predict function
    if best_approach is None:
        # Fallback: simple linear
        w_fallback, *_ = np.linalg.lstsq(np.column_stack([np.ones(n), X_train]), y, rcond=None)
        def predict(X):
            Xr = np.asarray(X, dtype=np.float64)
            return np.column_stack([np.ones(Xr.shape[0]), Xr]) @ w_fallback
        return predict
    
    if best_approach == 'plain':
        w_final = best_params
        def predict(X):
            Xr = np.asarray(X, dtype=np.float64)
            F_pred, _ = build_features(Xr, (Xr - X_min) / X_range)
            return F_pred @ w_final
        return predict
    
    elif best_approach == 'ridge_centered':
        w_c, alpha = best_params
        def predict(X):
            Xr = np.asarray(X, dtype=np.float64)
            F_pred, _ = build_features(Xr, (Xr - X_min) / X_range)
            F_pred_mean = F_pred.mean(axis=0)
            F_pred_std = F_pred.std(axis=0)
            F_pred_std[F_pred_std < 1e-12] = 1.0
            F_pred_c = (F_pred - F_mean) / F_std  # Use training stats
            pred_c = F_pred_c @ w_c
            return pred_c * y_std + y_mean
        return predict
    
    elif best_approach == 'sparse':
        w_sub, mask = best_params
        def predict(X):
            Xr = np.asarray(X, dtype=np.float64)
            F_pred, _ = build_features(Xr, (Xr - X_min) / X_range)
            F_pred_c = (F_pred - F_mean) / F_std
            pred_c = F_pred_c[:, mask] @ w_sub
            return pred_c * y_std + y_mean
        return predict
    
    elif best_approach == 'topk':
        w_sub, top_k = best_params
        def predict(X):
            Xr = np.asarray(X, dtype=np.float64)
            F_pred, _ = build_features(Xr, (Xr - X_min) / X_range)
            F_pred_c = (F_pred - F_mean) / F_std
            pred_c = F_pred_c[:, top_k] @ w_sub
            return pred_c * y_std + y_mean
        return predict
    
    elif best_approach == 'ext_plain':
        w_final = best_params
        def predict(X):
            Xr = np.asarray(X, dtype=np.float64)
            F_pred = build_extended_features(Xr)
            return F_pred @ w_final
        return predict
    
    elif best_approach == 'ext_ridge':
        w_c, alpha = best_params
        def predict(X):
            Xr = np.asarray(X, dtype=np.float64)
            F_pred = build_extended_features(Xr)
            F_pred_c = (F_pred - F_ext_mean) / F_ext_std
            pred_c = F_pred_c @ w_c
            return pred_c * y_std + y_mean
        return predict
    
    elif best_approach == 'ext_topk':
        w_sub, top_k = best_params
        def predict(X):
            Xr = np.asarray(X, dtype=np.float64)
            F_pred = build_extended_features(Xr)
            F_pred_c = (F_pred - F_ext_mean) / F_ext_std
            pred_c = F_pred_c[:, top_k] @ w_sub
            return pred_c * y_std + y_mean
        return predict
    
    # Fallback
    w_fallback, *_ = np.linalg.lstsq(np.column_stack([np.ones(n), X_train]), y, rcond=None)
    def predict(X):
        Xr = np.asarray(X, dtype=np.float64)
        return np.column_stack([np.ones(Xr.shape[0]), Xr]) @ w_fallback
    return predict
# EVOLVE-BLOCK-END
