"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize

def fit(X_train, y_train):
    start_time = time.time()
    DEADLINE_MARGIN = 10.0  # seconds before the 330s wall-clock limit to stop
    
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n, d = X.shape
    
    def time_left():
        return (start_time + 330.0 - DEADLINE_MARGIN) - time.time()

    # Handle constant y
    y_mean = np.mean(y)
    y_var = np.var(y)
    if y_var < 1e-12:
        c = y_mean
        return lambda X: np.full(len(X), c)

    # Strategy 1: Non-parametric Interpolation (CubicSpline)
    # This is highly effective for smooth, noiseless functions where we have enough points (200).
    # It captures local behavior better than global polynomials or fixed basis sets.
    if d == 1:
        x = X[:, 0]
        
        # Sort by x for interpolation
        sort_idx = np.argsort(x)
        x_sorted = x[sort_idx]
        y_sorted = y[sort_idx]
        
        # Handle duplicates by averaging y
        unique_x, inverse, counts = np.unique(x_sorted, return_inverse=True, return_counts=True)
        if len(unique_x) < n:
            y_sum = np.zeros(len(unique_x))
            np.add.at(y_sum, inverse, y_sorted)
            y_unique = y_sum / counts
            x_unique = unique_x
        else:
            x_unique = x_sorted
            y_unique = y_sorted
            
        # Need at least 4 points for cubic spline, otherwise fall back to linear or poly
        if len(x_unique) >= 4:
            try:
                # Use 'not-a-knot' boundary condition which is standard for smooth functions
                cs = CubicSpline(x_unique, y_unique, bc_type='not-a-knot')
                
                # Verify fit quality on training data
                y_pred_train = cs(x)
                sse = np.sum((y - y_pred_train)**2)
                sst = np.sum((y - y_mean)**2)
                r2 = 1 - sse / sst if sst > 0 else 1.0
                
                # If the spline fits well (which it should for smooth noiseless data), return it
                # Threshold 0.99 is safe; if it's not smooth, R2 will be lower and we fallback
                if r2 > 0.99:
                    def predict_spline(X_pred):
                        X_pred = np.asarray(X_pred, dtype=np.float64)
                        if X_pred.ndim == 1:
                            X_pred = X_pred.reshape(-1, 1)
                        x_pred = X_pred[:, 0]
                        # Clip to training range to avoid extrapolation issues if needed, 
                        # though CubicSpline extrapolates by default. 
                        # For symbolic regression, extrapolation is often dangerous.
                        # However, test points are likely within or near the training range.
                        # Let's allow extrapolation but check for non-finite values.
                        y_pred = cs(x_pred)
                        # Replace any non-finite values with the mean or clamped value
                        y_pred = np.nan_to_num(y_pred, nan=y_mean, posinf=np.max(y_unique), neginf=np.min(y_unique))
                        return y_pred
                    
                    return predict_spline
            except Exception:
                pass

    # Strategy 2: Parametric Basis Search (from Inspiration 1)
    # If interpolation fails (e.g., non-smooth, or multi-dimensional), try a rich basis.
    
    def get_basis_functions(X):
        n = len(X)
        cols = [np.ones(n)]
        
        # Polynomial up to degree 12
        for p in range(1, 13):
            cols.append(X ** p)
        
        # Trigonometric with more harmonics
        for k in [1, 2, 3, 4, 5]:
            cols.append(np.sin(k * X))
            cols.append(np.cos(k * X))
        
        # Exponential with various rates
        for rate in [0.5, 1.0, 1.5, 2.0]:
            cols.append(np.exp(rate * X))
            cols.append(np.exp(-rate * X))
        
        # Logarithmic
        cols.append(np.log1p(np.abs(X) + 1e-12))
        cols.append(np.log(np.abs(X) + 1e-12))
        
        # Inverse
        cols.append(1.0 / (1.0 + X**2))
        cols.append(1.0 / (1.0 + np.abs(X)))
        cols.append(1.0 / (1.0 + X**4))
        
        # Smooth step / Sigmoid
        cols.append(1.0 / (1.0 + np.exp(-X)))
        cols.append(np.tanh(X))
        cols.append(np.tanh(2 * X))
        cols.append(np.tanh(0.5 * X))
        
        # Sqrt
        cols.append(np.sqrt(np.abs(X) + 1e-12))
        cols.append(np.cbrt(X))
        
        # x * trig
        cols.append(X * np.sin(X))
        cols.append(X * np.cos(X))
        cols.append(X * np.sin(2*X))
        cols.append(X * np.cos(2*X))
        
        # Absolute value combinations
        cols.append(np.abs(X))
        cols.append(np.abs(X) * np.sign(X))
        
        # Rational functions
        cols.append(X / (1.0 + X**2))
        cols.append(X / (1.0 + np.abs(X)))
        
        # Gaussian-like bumps
        for center in [-2, -1, 0, 1, 2]:
            for width in [0.5, 1.0, 2.0]:
                cols.append(np.exp(-((X - center) ** 2) / (2 * width**2)))
        
        return np.column_stack(cols)

    # Only apply basis search if d==1 for now, as multi-dim basis explosion is hard
    if d == 1:
        x = X[:, 0]
        B = get_basis_functions(x)
        B = np.nan_to_num(B, nan=0.0, posinf=0.0, neginf=0.0)
        
        y_centered = y - y_mean
        y_norm = np.linalg.norm(y_centered)
        if y_norm < 1e-12:
            return lambda X: np.full(len(X), y_mean)
            
        # Compute R-squared for each column individually
        best_r2 = -np.inf
        best_idx = -1
        B_centered = B - np.mean(B, axis=0)
        
        for j in range(B.shape[1]):
            b_norm = np.linalg.norm(B_centered[:, j])
            if b_norm < 1e-12:
                continue
            corr = np.dot(B_centered[:, j], y_centered) / (b_norm * y_norm)
            r2 = corr * corr
            if r2 > best_r2:
                best_r2 = r2
                best_idx = j
                
        # If a single feature explains > 99.99% of variance, use it
        if best_r2 > 0.9999:
            X_f = B[:, best_idx]
            X_aug = np.column_stack([np.ones(n), X_f])
            w, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
            
            def predict_single(X_pred):
                X_pred = np.asarray(X_pred, dtype=np.float64)
                if X_pred.ndim == 1:
                    X_pred = X_pred.reshape(-1, 1)
                B_pred = get_basis_functions(X_pred[:, 0])
                B_pred = np.nan_to_num(B_pred, nan=0.0, posinf=0.0, neginf=0.0)
                X_f_pred = B_pred[:, best_idx]
                X_aug_pred = np.column_stack([np.ones(len(X_pred)), X_f_pred])
                return X_aug_pred @ w
                
            return predict_single

        # Try pairs if single wasn't enough
        if best_r2 < 0.999:
            best_pair_r2 = best_r2
            best_pair = (best_idx, -1)
            
            # Limit search to top 15 individual features
            r2_scores = []
            for j in range(B.shape[1]):
                b_norm = np.linalg.norm(B_centered[:, j])
                if b_norm < 1e-12:
                    r2_scores.append((0, j))
                else:
                    corr = np.dot(B_centered[:, j], y_centered) / (b_norm * y_norm)
                    r2_scores.append((corr*corr, j))
            
            r2_scores.sort(reverse=True)
            top_k = 15
            top_indices = [idx for _, idx in r2_scores[:top_k]]
            
            for i in range(len(top_indices)):
                for j in range(i+1, len(top_indices)):
                    idx1, idx2 = top_indices[i], top_indices[j]
                    X_pair = np.column_stack([np.ones(n), B[:, idx1], B[:, idx2]])
                    try:
                        w_pair, *_ = np.linalg.lstsq(X_pair, y, rcond=None)
                        y_hat = X_pair @ w_pair
                        sse = np.sum((y - y_hat)**2)
                        r2_pair = 1 - sse / np.sum(y_centered**2)
                        if r2_pair > best_pair_r2:
                            best_pair_r2 = r2_pair
                            best_pair = (idx1, idx2)
                    except:
                        pass
                    if time_left() < 5.0:
                        break
                if time_left() < 5.0:
                    break
            
            if best_pair_r2 > 0.9999:
                idx1, idx2 = best_pair
                X_f = np.column_stack([np.ones(n), B[:, idx1], B[:, idx2]])
                w, *_ = np.linalg.lstsq(X_f, y, rcond=None)
                
                def predict_pair(X_pred):
                    X_pred = np.asarray(X_pred, dtype=np.float64)
                    if X_pred.ndim == 1:
                        X_pred = X_pred.reshape(-1, 1)
                    B_pred = get_basis_functions(X_pred[:, 0])
                    B_pred = np.nan_to_num(B_pred, nan=0.0, posinf=0.0, neginf=0.0)
                    X_f_pred = np.column_stack([np.ones(len(X_pred)), B_pred[:, idx1], B_pred[:, idx2]])
                    return X_f_pred @ w
                    
                return predict_pair
            
            # Try triples from top 8 features
            if best_pair_r2 < 0.9999 and time_left() > 10.0:
                best_triple_r2 = best_pair_r2
                best_triple = (best_idx, -1, -1)
                top_k3 = 8
                top_indices3 = [idx for _, idx in r2_scores[:top_k3]]
                
                for i in range(len(top_indices3)):
                    for j in range(i+1, len(top_indices3)):
                        for k in range(j+1, len(top_indices3)):
                            idx1, idx2, idx3 = top_indices3[i], top_indices3[j], top_indices3[k]
                            X_triple = np.column_stack([np.ones(n), B[:, idx1], B[:, idx2], B[:, idx3]])
                            try:
                                w_triple, *_ = np.linalg.lstsq(X_triple, y, rcond=None)
                                y_hat = X_triple @ w_triple
                                sse = np.sum((y - y_hat)**2)
                                r2_triple = 1 - sse / np.sum(y_centered**2)
                                if r2_triple > best_triple_r2:
                                    best_triple_r2 = r2_triple
                                    best_triple = (idx1, idx2, idx3)
                            except:
                                pass
                            if time_left() < 5.0:
                                break
                        if time_left() < 5.0:
                            break
                    if time_left() < 5.0:
                        break
                
                if best_triple_r2 > 0.9999:
                    idx1, idx2, idx3 = best_triple
                    X_f = np.column_stack([np.ones(n), B[:, idx1], B[:, idx2], B[:, idx3]])
                    w, *_ = np.linalg.lstsq(X_f, y, rcond=None)
                    
                    def predict_triple(X_pred):
                        X_pred = np.asarray(X_pred, dtype=np.float64)
                        if X_pred.ndim == 1:
                            X_pred = X_pred.reshape(-1, 1)
                        B_pred = get_basis_functions(X_pred[:, 0])
                        B_pred = np.nan_to_num(B_pred, nan=0.0, posinf=0.0, neginf=0.0)
                        X_f_pred = np.column_stack([np.ones(len(X_pred)), B_pred[:, idx1], B_pred[:, idx2], B_pred[:, idx3]])
                        return X_f_pred @ w
                        
                    return predict_triple

    # Fallback: Simple Polynomial Regression
    # This handles cases where interpolation failed (e.g., multi-dim) or basis search didn't find a perfect fit
    y_centered = y - y_mean
    y_norm = np.linalg.norm(y_centered)
    if y_norm < 1e-12:
        return lambda X: np.full(len(X), y_mean)
    
    best_w = None
    best_r2 = -np.inf
    best_degree = 1
    
    # Try polynomial degrees from 1 to 12 (keep it low to avoid overfitting/instability if spline failed)
    max_degree = 12
    for degree in range(1, max_degree + 1):
        if d == 1:
            x = X[:, 0]
            X_poly = np.column_stack([x**p for p in range(degree + 1)])
        else:
            # For multi-dimensional, use simple polynomial features
            X_poly = np.ones((n, 1))
            for p in range(1, degree + 1):
                for i in range(d):
                    X_poly = np.column_stack([X_poly, X[:, i]**p])
        
        try:
            w, *_ = np.linalg.lstsq(X_poly, y, rcond=None)
            y_hat = X_poly @ w
            sse = np.sum((y - y_hat)**2)
            r2 = 1 - sse / np.sum(y_centered**2)
            
            if r2 > best_r2:
                best_r2 = r2
                best_w = w
                best_degree = degree
            
            # If we have a very good fit, stop early
            if r2 > 0.9999:
                break
                
        except:
            continue
        
        if time_left() < 5.0:
            break
    
    # Create prediction function
    def predict_poly(X_pred):
        X_pred = np.asarray(X_pred, dtype=np.float64)
        if X_pred.ndim == 1:
            X_pred = X_pred.reshape(-1, 1)
        n_pred = len(X_pred)
        d_pred = X_pred.shape[1]
        
        if d_pred == 1:
            x_pred = X_pred[:, 0]
            X_poly_pred = np.column_stack([x_pred**p for p in range(best_degree + 1)])
        else:
            X_poly_pred = np.ones((n_pred, 1))
            for p in range(1, best_degree + 1):
                for i in range(d_pred):
                    X_poly_pred = np.column_stack([X_poly_pred, X_pred[:, i]**p])
        
        return X_poly_pred @ best_w
    
    # If polynomial fit is still not great, try Gaussian Process regression for 1D case
    if d == 1 and best_r2 < 0.999 and time_left() > 20.0:
        try:
            x = X[:, 0]
            # Use GP with RBF kernel
            # Standardize x
            x_mean = np.mean(x)
            x_std = np.std(x)
            if x_std < 1e-10:
                x_std = 1.0
            x_norm = (x - x_mean) / x_std
            
            # Compute kernel matrix
            X_grid = x_norm[:, None]
            X_grid_T = x_norm[None, :]
            sq_dist = np.sum(X_grid**2, axis=1)[:, None] + np.sum(X_grid_T**2, axis=1)[None, :] - 2 * X_grid @ X_grid_T.T
            sq_dist = np.maximum(sq_dist, 0)
            
            # Try different length scales
            best_lscale = 1.0
            best_gp_r2 = best_r2
            for ls in [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]:
                K = np.exp(-sq_dist / (2 * ls**2)) + 1e-8 * np.eye(n)
                try:
                    L = np.linalg.cholesky(K)
                    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
                    # Predict at training points to check R2
                    y_pred_train = K @ alpha
                    sse_gp = np.sum((y - y_pred_train)**2)
                    r2_gp = 1 - sse_gp / np.sum(y_centered**2)
                    if r2_gp > best_gp_r2:
                        best_gp_r2 = r2_gp
                        best_lscale = ls
                except:
                    continue
            
            if best_gp_r2 > 0.9999:
                # Use the best length scale for prediction
                def predict_gp(X_pred):
                    X_pred = np.asarray(X_pred, dtype=np.float64)
                    if X_pred.ndim == 1:
                        X_pred = X_pred.reshape(-1, 1)
                    x_pred = X_pred[:, 0]
                    x_pred_norm = (x_pred - x_mean) / x_std
                    
                    # Compute kernel between test and train
                    X_test = x_pred_norm[:, None]
                    X_train = x_norm[None, :]
                    sq_dist_test = np.sum(X_test**2, axis=1)[:, None] + np.sum(X_train**2, axis=1)[None, :] - 2 * X_test @ X_train.T
                    sq_dist_test = np.maximum(sq_dist_test, 0)
                    K_test = np.exp(-sq_dist_test / (2 * best_lscale**2))
                    
                    y_pred = K_test @ alpha
                    return y_pred
                
                return predict_gp
        except:
            pass
    
    return predict_poly

# EVOLVE-BLOCK-END
