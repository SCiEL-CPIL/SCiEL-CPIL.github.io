# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.linalg import solve
from scipy.optimize import differential_evolution
import itertools

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 290.0

    n, d = X_train.shape
    y_mean = np.mean(y_train)
    y_var = np.var(y_train)
    if y_var < 1e-12:
        return lambda X: np.full(len(X), y_mean)
    
    y_std = np.std(y_train)
    if y_std < 1e-10:
        y_std = 1.0
    y_norm = (y_train - y_mean) / y_std

    X_min = X_train.min(axis=0)
    X_max = X_train.max(axis=0)
    X_range = X_max - X_min
    X_range[X_range < 1e-10] = 1.0

    def check_time():
        return time.time() < deadline

    best_predict = None
    best_mse = np.inf

    def update_best(model):
        nonlocal best_predict, best_mse
        if model is None:
            return
        try:
            preds = model(X_train)
            preds = np.asarray(preds, dtype=float)
            if preds.shape != y_train.shape:
                return
            if np.any(np.isnan(preds)) or np.any(np.isinf(preds)):
                return
            mse = np.mean((preds - y_train) ** 2)
            if np.isnan(mse) or np.isinf(mse):
                return
            if mse < best_mse:
                best_mse = mse
                best_predict = model
        except Exception:
            pass

    # Strategy 1: Fourier basis with cross terms
    # This captures smooth nonlinear and interactive patterns well for bounded inputs
    def make_fourier_features(X, freqs_per_dim):
        # Normalize to [0, 1]
        Xn = (X - X_min) / X_range
        
        cols = [np.ones(len(Xn))]
        
        # Generate frequency sets
        freqs = list(range(1, freqs_per_dim + 1))
        
        # For each variable, add sin and cos terms
        var_terms = []
        for j in range(d):
            terms_j = []
            for f in freqs:
                terms_j.append((j, f, 'sin'))
                terms_j.append((j, f, 'cos'))
            var_terms.append(terms_j)
        
        # Add all single-variable terms
        for j in range(d):
            for term in var_terms[j]:
                j_idx, f, typ = term
                val = np.sin(2 * np.pi * f * Xn[:, j_idx]) if typ == 'sin' else np.cos(2 * np.pi * f * Xn[:, j_idx])
                cols.append(val)
        
        # Add cross terms (products of single-variable terms)
        if d >= 2:
            # Limit cross terms to keep dimension manageable
            max_cross = 50
            cross_count = 0
            for j1 in range(d):
                for j2 in range(j1 + 1, d):
                    for t1 in var_terms[j1]:
                        if cross_count >= max_cross:
                            break
                        for t2 in var_terms[j2]:
                            if cross_count >= max_cross:
                                break
                            j1_idx, f1, typ1 = t1
                            j2_idx, f2, typ2 = t2
                            val1 = np.sin(2 * np.pi * f1 * Xn[:, j1_idx]) if typ1 == 'sin' else np.cos(2 * np.pi * f1 * Xn[:, j1_idx])
                            val2 = np.sin(2 * np.pi * f2 * Xn[:, j2_idx]) if typ2 == 'sin' else np.cos(2 * np.pi * f2 * Xn[:, j2_idx])
                            cols.append(val1 * val2)
                            cross_count += 1
        
        return np.column_stack(cols)

    for freqs in [2, 3, 4, 5]:
        if not check_time():
            break
        try:
            F = make_fourier_features(X_train, freqs)
            if F.shape[1] > 150:
                continue
            lam = 1e-4
            A = F.T @ F + lam * np.eye(F.shape[1])
            A[0, 0] -= lam
            w = solve(A, F.T @ y_norm)
            
            def predict_fourier(X, w, y_mean, y_std, freqs):
                F_test = make_fourier_features(X, freqs)
                return y_mean + y_std * (F_test @ w)
            
            update_best(lambda X: predict_fourier(X, w, y_mean, y_std, freqs))
        except Exception:
            pass

    # Strategy 2: Polynomial regression with Chebyshev basis
    def make_cheb_features(X, degree):
        Xn = (X - X_min) / X_range
        Xn = 2 * Xn - 1  # Map to [-1, 1]
        cols = [np.ones(len(Xn))]
        
        def cheb_vals(x, deg):
            T0 = np.ones_like(x)
            if deg == 0:
                return [T0]
            T1 = x.copy()
            result = [T0, T1]
            for k in range(2, deg + 1):
                Tk = 2 * x * T1 - T0
                result.append(Tk)
                T0, T1 = T1, Tk
            return result
        
        cheb_sets = []
        for j in range(d):
            cheb_sets.append(cheb_vals(Xn[:, j], degree))
        
        # Add all combinations up to total degree
        for total_deg in range(1, degree + 1):
            for combo in _cheb_combinations(d, total_deg):
                term = np.ones(len(Xn))
                for j, k in combo:
                    term *= cheb_sets[j][k]
                cols.append(term)
        
        return np.column_stack(cols)
    
    def _cheb_combinations(d, total_deg):
        if d == 1:
            return [(0, total_deg)]
        results = []
        for k in range(total_deg + 1):
            rest = _cheb_combinations(d - 1, total_deg - k)
            for r in rest:
                results.append((0, k) + r)
        return results
    
    for deg in [2, 3, 4]:
        if not check_time():
            break
        try:
            F = make_cheb_features(X_train, deg)
            if F.shape[1] > 100:
                continue
            lam = 1e-4
            A = F.T @ F + lam * np.eye(F.shape[1])
            A[0, 0] -= lam
            w = solve(A, F.T @ y_norm)
            
            def predict_cheb(X, w, y_mean, y_std, deg):
                F_test = make_cheb_features(X, deg)
                return y_mean + y_std * (F_test @ w)
            
            update_best(lambda X: predict_cheb(X, w, y_mean, y_std, deg))
        except Exception:
            pass

    # Strategy 3: RBF with grid of centers
    def make_rbf_model(gamma, n_centers_per_dim):
        if d == 1:
            centers = np.linspace(X_min[0], X_max[0], n_centers_per_dim).reshape(-1, 1)
        elif d == 2:
            x1_grid = np.linspace(X_min[0], X_max[0], n_centers_per_dim)
            x2_grid = np.linspace(X_min[1], X_max[1], n_centers_per_dim)
            X1, X2 = np.meshgrid(x1_grid, x2_grid)
            centers = np.column_stack([X1.ravel(), X2.ravel()])
        else:
            idx = np.random.choice(n, min(n_centers_per_dim ** 2, n), replace=False)
            centers = X_train[idx]
        
        n_c = len(centers)
        diffs = X_train[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sum(diffs ** 2, axis=2)
        F = np.exp(-gamma * dists)
        
        lam = 1e-4
        A = F.T @ F + lam * np.eye(n_c + 1)
        A[0, 0] -= lam
        b = np.concatenate([np.ones(n_c), F.T @ y_norm])
        w = solve(A, b)
        
        def predict_rbf(X, centers, gamma, w, y_mean, y_std):
            diffs = X[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dists = np.sum(diffs ** 2, axis=2)
            F_test = np.exp(-gamma * dists)
            F_test = np.column_stack([np.ones(len(F_test)), F_test])
            return y_mean + y_std * (F_test @ w)
        
        return lambda X: predict_rbf(X, centers, gamma, w, y_mean, y_std)
    
    X_range_mean = np.mean(X_range)
    base_gamma = 1.0 / (d * X_range_mean ** 2 + 1e-10)
    
    for g_mult in [0.1, 0.5, 1.0, 2.0, 5.0]:
        if not check_time():
            break
        for nc in [5, 8, 12]:
            if not check_time():
                break
            try:
                model = make_rbf_model(base_gamma * g_mult, nc)
                update_best(model)
            except Exception:
                pass

    # Strategy 4: Simple functional form search with linear fit
    simple_forms_2d = [
        lambda x1, x2: x1 + x2,
        lambda x1, x2: x1 - x2,
        lambda x1, x2: x2 - x1,
        lambda x1, x2: x1 * x2,
        lambda x1, x2: x1 / np.maximum(x2, 1e-10),
        lambda x1, x2: x2 / np.maximum(x1, 1e-10),
        lambda x1, x2: x1 ** 2 + x2 ** 2,
        lambda x1, x2: x1 ** 2 - x2 ** 2,
        lambda x1, x2: x2 ** 2 - x1 ** 2,
        lambda x1, x2: x1 ** 2 * x2,
        lambda x1, x2: x1 * x2 ** 2,
        lambda x1, x2: np.sin(x1) * np.cos(x2),
        lambda x1, x2: np.cos(x1) * np.sin(x2),
        lambda x1, x2: np.sin(x1 + x2),
        lambda x1, x2: np.cos(x1 + x2),
        lambda x1, x2: np.exp(-x1) * x2,
        lambda x1, x2: np.exp(-x2) * x1,
        lambda x1, x2: np.sqrt(np.maximum(x1, 0)) * x2,
        lambda x1, x2: x1 * np.sqrt(np.maximum(x2, 0)),
        lambda x1, x2: np.abs(x1 - x2),
        lambda x1, x2: np.maximum(x1, x2),
        lambda x1, x2: np.minimum(x1, x2),
        lambda x1, x2: (x1 + x2) ** 2,
        lambda x1, x2: (x1 - x2) ** 2,
        lambda x1, x2: x1 ** 3 + x2,
        lambda x1, x2: x1 + x2 ** 3,
        lambda x1, x2: x1 ** 2 * x2 ** 2,
        lambda x1, x2: np.sin(x1) + np.cos(x2),
        lambda x1, x2: x1 * np.sin(x2),
        lambda x1, x2: x2 * np.sin(x1),
    ]
    
    for i in range(d):
        for j in range(i + 1, d):
            if not check_time():
                break
            for form_fn in simple_forms_2d:
                if not check_time():
                    break
                try:
                    x1 = X_train[:, i]
                    x2 = X_train[:, j]
                    y_pred = form_fn(x1, x2)
                    A_mat = np.column_stack([y_pred, np.ones(n)])
                    lam = 1e-6
                    A = A_mat.T @ A_mat + lam * np.eye(2)
                    A[1, 1] -= lam
                    w = solve(A, A_mat.T @ y_norm)
                    a, b = w[0], w[1]
                    mse = np.mean((a * y_pred + b - y_norm) ** 2)
                    if mse < best_mse:
                        best_mse = mse
                        def predict_form(X, i, j, form_fn, a, b, y_mean, y_std):
                            x1 = X[:, i]
                            x2 = X[:, j]
                            return y_mean + y_std * (a * form_fn(x1, x2) + b)
                        best_predict = lambda X, i=i, j=j, f=form_fn, a=a, b=b: predict_form(X, i, j, f, a, b, y_mean, y_std)
                except Exception:
                    pass
    
    # 1D simple forms
    simple_forms_1d = [
        lambda x: x,
        lambda x: x ** 2,
        lambda x: x ** 3,
        lambda x: np.sin(x),
        lambda x: np.cos(x),
        lambda x: np.exp(x),
        lambda x: np.exp(-x),
        lambda x: np.sqrt(np.maximum(x, 0)),
        lambda x: np.abs(x),
        lambda x: np.log(np.maximum(x, 1e-10)),
        lambda x: 1.0 / np.maximum(x, 1e-10),
        lambda x: x * np.sin(x),
        lambda x: x * np.cos(x),
        lambda x: np.sin(x) / np.maximum(x, 1e-10),
        lambda x: x ** 2 * np.sin(x),
        lambda x: np.exp(-x ** 2 / 2),
    ]
    
    for i in range(d):
        if not check_time():
            break
        for form_fn in simple_forms_1d:
            if not check_time():
                break
            try:
                x1 = X_train[:, i]
                y_pred = form_fn(x1)
                A_mat = np.column_stack([y_pred, np.ones(n)])
                lam = 1e-6
                A = A_mat.T @ A_mat + lam * np.eye(2)
                A[1, 1] -= lam
                w = solve(A, A_mat.T @ y_norm)
                a, b = w[0], w[1]
                mse = np.mean((a * y_pred + b - y_norm) ** 2)
                if mse < best_mse:
                    best_mse = mse
                    def predict_1d_form(X, i, form_fn, a, b, y_mean, y_std):
                        x1 = X[:, i]
                        return y_mean + y_std * (a * form_fn(x1) + b)
                    best_predict = lambda X, i=i, f=form_fn, a=a, b=b: predict_1d_form(X, i, f, a, b, y_mean, y_std)
            except Exception:
                pass

    # Fallback
    if best_predict is None:
        return lambda X: np.full(len(X), y_mean)
    
    return best_predict

# EVOLVE-BLOCK-END
