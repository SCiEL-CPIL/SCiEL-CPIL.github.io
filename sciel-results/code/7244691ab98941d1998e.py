import numpy as np
import time
from scipy.interpolate import CubicSpline, PchipInterpolator

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 280  # Safety margin
    
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n, d = X.shape
    
    # Since all inputs are 1D, we can exploit this.
    # If d > 1, we fall back to a general method.
    
    def get_rich_features(X_1d):
        x = X_1d.flatten()
        n = len(x)
        cols = [np.ones(n)]
        # Polynomials
        xk = np.ones(n)
        for k in range(1, 20):
            xk = xk * x
            cols.append(xk.copy())
        # Trig
        for freq in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]:
            cols.append(np.sin(freq * x))
            cols.append(np.cos(freq * x))
        # Exp
        for c in [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]:
            cols.append(np.exp(c * x))
        # Other
        cols.append(np.abs(x))
        cols.append(np.sqrt(np.abs(x) + 1e-9))
        cols.append(np.log(np.abs(x) + 1e-9))
        cols.append(1.0 / (x + 1e-9))
        return np.column_stack(cols)

    # Model 1: Cubic Spline
    def create_spline_model(X_1d, y_1d):
        try:
            idx = np.argsort(X_1d.flatten())
            X_sorted = X_1d[idx]
            y_sorted = y_1d[idx]
            cs = CubicSpline(X_sorted.flatten(), y_sorted)
            def predict(X_new):
                return cs(X_new.flatten()).reshape(X_new.shape[0])
            return predict
        except Exception:
            return None

    # Model 2: PCHIP (Piecewise Cubic Hermite Interpolating Polynomial)
    def create_pchip_model(X_1d, y_1d):
        try:
            idx = np.argsort(X_1d.flatten())
            X_sorted = X_1d[idx]
            y_sorted = y_1d[idx]
            pchip = PchipInterpolator(X_sorted.flatten(), y_sorted)
            def predict(X_new):
                return pchip(X_new.flatten()).reshape(X_new.shape[0])
            return predict
        except Exception:
            return None

    # Model 3: Regularized Least Squares (Ridge-like) with Rich Features
    def create_ridge_model(X_1d, y_1d, alpha=1.0):
        try:
            Xf = get_rich_features(X_1d)
            n_features = Xf.shape[1]
            I = np.eye(n_features)
            I[0, 0] = 0.0  # Do not regularize intercept
            
            A = Xf.T @ Xf + alpha * I
            b = Xf.T @ y_1d
            
            try:
                w = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                w, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            
            def predict(X_new):
                Xf_new = get_rich_features(X_new)
                return Xf_new @ w
            return predict
        except Exception:
            return None

    # Model 4: RBF Interpolation (Gaussian)
    def create_rbf_model(X_1d, y_1d, epsilon=1.0):
        try:
            x = X_1d.flatten()
            n = len(x)
            
            diff = x[:, None] - x[None, :]
            K = np.exp(-(epsilon ** 2) * (diff ** 2))
            
            reg = 1e-8 * np.eye(n)
            K_reg = K + reg
            
            try:
                w = np.linalg.solve(K_reg, y_1d)
            except np.linalg.LinAlgError:
                w, _, _, _ = np.linalg.lstsq(K_reg, y_1d, rcond=None)
            
            def predict(X_new):
                x_new = np.asarray(X_new, dtype=np.float64).flatten()
                diff_new = x_new[:, None] - x[None, :]
                K_new = np.exp(-(epsilon ** 2) * (diff_new ** 2))
                return K_new @ w
            return predict
        except Exception:
            return None

    # Model 5: Polynomial Kernel Regression (Linear in Kernel Space)
    # This is effectively polynomial regression, but framed as kernel.
    # We use explicit polynomial features up to degree 15 to avoid kernel matrix issues
    # and leverage the fact that polynomial functions are common.
    def create_poly_kernel_model(X_1d, y_1d, degree=10, alpha=1.0):
        try:
            x = X_1d.flatten()
            n = len(x)
            # Build polynomial feature matrix [1, x, x^2, ..., x^degree]
            V = np.vander(x, N=degree+1, increasing=True)
            
            # Ridge regression
            A = V.T @ V + alpha * np.eye(degree+1)
            A[0, 0] += 0 # Don't regularize intercept
            b = V.T @ y_1d
            
            try:
                w = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                w, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            
            def predict(X_new):
                x_new = np.asarray(X_new, dtype=np.float64).flatten()
                V_new = np.vander(x_new, N=degree+1, increasing=True)
                return V_new @ w
            return predict
        except Exception:
            return None

    # Model 6: LOESS (Local Polynomial Regression)
    def create_loess_model(X_1d, y_1d, frac=0.5, degree=2):
        try:
            x = X_1d.flatten()
            n = len(x)
            idx = np.argsort(x)
            x_sorted = x[idx]
            y_sorted = y[idx]
            
            k = max(3, int(n * frac))
            
            def predict(X_new):
                x_new = np.asarray(X_new, dtype=np.float64).flatten()
                n_new = len(x_new)
                y_pred = np.zeros(n_new)
                
                for i in range(n_new):
                    xi = x_new[i]
                    dists = np.abs(x_sorted - xi)
                    k_idx = np.argpartition(dists, k)[:k]
                    k_idx = k_idx[np.argsort(dists[k_idx])]
                    
                    x_loc = x_sorted[k_idx]
                    y_loc = y_sorted[k_idx]
                    d_loc = dists[k_idx]
                    
                    max_d = d_loc[-1]
                    if max_d < 1e-10:
                        w_loc = np.ones(k)
                    else:
                        w_loc = 1.0 - d_loc / max_d
                        w_loc = np.maximum(w_loc, 1e-10)
                    
                    dx = x_loc - xi
                    V = np.vander(dx, N=degree+1, increasing=True)
                    W = np.diag(w_loc)
                    
                    try:
                        VtWV = V.T @ W @ V
                        VtWy = V.T @ W @ y_loc
                        VtWV += 1e-8 * np.eye(degree+1)
                        theta = np.linalg.solve(VtWV, VtWy)
                        y_pred[i] = theta[0]
                    except np.linalg.LinAlgError:
                        y_pred[i] = np.sum(w_loc * y_loc) / np.sum(w_loc)
                
                return y_pred
            return predict
        except Exception:
            return None

    # If d == 1, use the 1D specific models with validation
    if d == 1:
        # Split for validation
        idx = np.arange(n)
        np.random.seed(42)
        np.random.shuffle(idx)
        val_size = 20
        val_idx = idx[:val_size]
        train_idx = idx[val_size:]
        
        X_val = X[val_idx]
        y_val = y[val_idx]
        X_tr = X[train_idx]
        y_tr = y[train_idx]
        
        best_model = None
        best_val_mse = np.inf
        best_params = None
        
        # Helper to evaluate a model
        def evaluate_model(model, X_val, y_val):
            if model is None:
                return np.inf
            try:
                pred_val = model(X_val)
                if not np.all(np.isfinite(pred_val)):
                    return np.inf
                mse = np.mean((pred_val - y_val)**2)
                return mse
            except Exception:
                return np.inf

        # Candidate models to try
        candidates = []
        
        # Spline
        m = create_spline_model(X_tr, y_tr)
        if m: candidates.append(('spline', m, {}))
        
        # PCHIP
        m = create_pchip_model(X_tr, y_tr)
        if m: candidates.append(('pchip', m, {}))
        
        # Ridge (multiple alphas)
        for alpha in [0.1, 1.0, 10.0]:
            m = create_ridge_model(X_tr, y_tr, alpha=alpha)
            if m: candidates.append(('ridge', m, {'alpha': alpha}))
            
        # RBF (multiple epsilons)
        for eps in [0.5, 1.0, 2.0, 5.0, 10.0]:
            m = create_rbf_model(X_tr, y_tr, epsilon=eps)
            if m: candidates.append(('rbf', m, {'epsilon': eps}))
            
        # Polynomial Kernel (multiple degrees)
        for deg in [5, 10, 15]:
            m = create_poly_kernel_model(X_tr, y_tr, degree=deg, alpha=1.0)
            if m: candidates.append(('poly', m, {'degree': deg, 'alpha': 1.0}))
            
        # LOESS (multiple fracs)
        for frac in [0.3, 0.5, 0.8]:
            m = create_loess_model(X_tr, y_tr, frac=frac, degree=2)
            if m: candidates.append(('loess', m, {'frac': frac, 'degree': 2}))
            
        # Evaluate all candidates
        for name, model, params in candidates:
            mse = evaluate_model(model, X_val, y_val)
            if np.isfinite(mse) and mse < best_val_mse:
                best_val_mse = mse
                best_model = name
                best_params = params
        
        # Refit best model on all data
        if best_model == 'spline':
            final_model = create_spline_model(X, y)
        elif best_model == 'pchip':
            final_model = create_pchip_model(X, y)
        elif best_model == 'ridge':
            alpha = best_params.get('alpha', 1.0)
            final_model = create_ridge_model(X, y, alpha=alpha)
        elif best_model == 'rbf':
            eps = best_params.get('epsilon', 1.0)
            final_model = create_rbf_model(X, y, epsilon=eps)
        elif best_model == 'poly':
            deg = best_params.get('degree', 10)
            alpha = best_params.get('alpha', 1.0)
            final_model = create_poly_kernel_model(X, y, degree=deg, alpha=alpha)
        elif best_model == 'loess':
            frac = best_params.get('frac', 0.5)
            deg = best_params.get('degree', 2)
            final_model = create_loess_model(X, y, frac=frac, degree=deg)
        else:
            # Fallback: Linear regression
            try:
                X_aug = np.column_stack([np.ones(n), X])
                w, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
                final_model = lambda X_new: np.column_stack([np.ones(len(X_new)), X_new]) @ w
            except Exception:
                y_mean = np.mean(y)
                final_model = lambda X_new: np.full(len(X_new), y_mean)
        
        def predict(X_new):
            return final_model(np.asarray(X_new, dtype=np.float64))
            
        return predict
    
    else:
        # Fallback for d > 1
        try:
            X_aug = np.column_stack([np.ones(n), X])
            w, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
            def predict(X_new):
                X_new = np.asarray(X_new, dtype=np.float64)
                X_new_aug = np.column_stack([np.ones(X_new.shape[0]), X_new])
                return X_new_aug @ w
            return predict
        except Exception:
            y_mean = np.mean(y)
            def predict(X_new):
                return np.full(len(X_new), y_mean)
            return predict
