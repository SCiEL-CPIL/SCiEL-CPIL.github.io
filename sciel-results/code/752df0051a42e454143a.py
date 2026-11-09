# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 285  # Safety margin for 330s limit
    
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=float).flatten()
    n = len(y)
    
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    x = X[:, 0]
    x_min = x.min()
    x_max = x.max()
    x_range = x_max - x_min
    if x_range < 1e-12:
        x_range = 1.0
        x_min = x[0] - 0.5 if n > 0 else 0.0
        x_max = x[0] + 0.5 if n > 0 else 1.0
    
    y_mean = float(np.mean(y))
    y_std = float(np.std(y))
    if y_std < 1e-12:
        y_std = 1.0
    
    candidates = []
    
    def elapsed():
        return time.time() - start_time
    
    def add_candidate(name, pred_func):
        if elapsed() > deadline:
            return
        try:
            yp = np.asarray(pred_func(X), dtype=float).flatten()
            if yp.shape[0] != n or not np.all(np.isfinite(yp)):
                return
            # Use Cross-Validation MSE for selection to avoid overfitting
            # Simple 5-fold CV
            indices = np.arange(n)
            np.random.shuffle(indices)
            folds = np.array_split(indices, 5)
            cv_mses = []
            for i in range(5):
                val_idx = folds[i]
                tr_idx = np.concatenate([folds[j] for j in range(5) if j != i])
                # Refit is expensive, so for simple models we use train MSE as proxy if CV is too slow?
                # Actually, for linear models, we can compute CV efficiently or just use a hold-out set.
                # Let's use a fixed 80/20 split for speed.
                pass
            
            # To keep it fast, let's just use a hold-out set for model selection
            # Shuffle and split
            idx = np.arange(n)
            np.random.shuffle(idx)
            split = int(0.8 * n)
            tr_idx = idx[:split]
            val_idx = idx[split:]
            
            # We need to refit the model on tr_idx to get val_mse?
            # That's expensive if we do it for every candidate.
            # Alternative: Use training MSE but penalize complexity?
            # Or, just pick the best training MSE but cap the complexity?
            # Given the time limit, let's stick to Training MSE but be smarter about which models we generate.
            # The main issue with the previous code was likely the NN and Local Poly.
            
            mse = float(np.mean((yp - y) ** 2))
            if np.isfinite(mse):
                candidates.append((name, pred_func, mse))
        except Exception:
            pass
    
    # --- Strategy 1: Polynomial regression ---
    x_scaled = (x - x_min) / x_range
    for deg in range(1, 15): # Reduced max degree to avoid overfitting/instability
        if elapsed() > deadline:
            break
        V = np.vander(x_scaled, N=deg + 1, increasing=True)
        try:
            w, *_ = np.linalg.lstsq(V, y, rcond=None)
            def pred_poly(Xp, deg=deg, w=w, x_min=x_min, x_range=x_range):
                Xp = np.asarray(Xp, dtype=float).reshape(-1)
                Xs = (Xp - x_min) / x_range
                Vp = np.vander(Xs, N=deg + 1, increasing=True)
                return Vp @ w
            add_candidate(f"poly_{deg}", pred_poly)
        except Exception:
            pass
    
    # --- Strategy 2: Fourier basis ---
    omegas = [np.pi / x_range, 2 * np.pi / x_range, np.pi, 2 * np.pi, 1.0, 0.5, 0.25, 4.0, 0.1, 5.0]
    for nf in range(1, 8): # Reduced max harmonics
        if elapsed() > deadline:
            break
        for omega in omegas[:5]: # Reduced omegas
            if elapsed() > deadline:
                break
            cols = [np.ones(n)]
            for k in range(1, nf + 1):
                cols.append(np.sin(k * omega * x))
                cols.append(np.cos(k * omega * x))
            V = np.column_stack(cols)
            try:
                w, *_ = np.linalg.lstsq(V, y, rcond=None)
                def pred_fourier(Xp, nf=nf, omega=omega, w=w):
                    Xp = np.asarray(Xp, dtype=float).reshape(-1)
                    cols = [np.ones(len(Xp))]
                    for k in range(1, nf + 1):
                        cols.append(np.sin(k * omega * Xp))
                        cols.append(np.cos(k * omega * Xp))
                    Vp = np.column_stack(cols)
                    return Vp @ w
                add_candidate(f"four_nf{nf}_om{omega:.4f}", pred_fourier)
            except Exception:
                pass

    # --- Strategy 3: Rich basis with greedy selection ---
    basis_funcs = {}
    basis_funcs['1'] = lambda x: np.ones_like(x)
    basis_funcs['x'] = lambda x: x
    basis_funcs['x2'] = lambda x: x ** 2
    basis_funcs['x3'] = lambda x: x ** 3
    basis_funcs['x4'] = lambda x: x ** 4
    basis_funcs['abs_x'] = lambda x: np.abs(x)
    basis_funcs['sqrt_abs'] = lambda x: np.sqrt(np.abs(x))
    basis_funcs['log1p_abs'] = lambda x: np.log1p(np.abs(x))
    basis_funcs['tanh'] = lambda x: np.tanh(x)
    basis_funcs['tanh_2x'] = lambda x: np.tanh(2 * x)
    basis_funcs['exp_05x'] = lambda x: np.exp(0.5 * x)
    basis_funcs['exp_neg05x'] = lambda x: np.exp(-0.5 * x)
    basis_funcs['exp_x'] = lambda x: np.exp(x)
    basis_funcs['exp_negx'] = lambda x: np.exp(-x)
    basis_funcs['x_over_1x2'] = lambda x: x / (1 + x ** 2)
    basis_funcs['sinc'] = lambda x: np.sinc(x)
    basis_funcs['sin_x'] = lambda x: np.sin(x)
    basis_funcs['cos_x'] = lambda x: np.cos(x)
    basis_funcs['sin_2x'] = lambda x: np.sin(2 * x)
    basis_funcs['cos_2x'] = lambda x: np.cos(2 * x)
    basis_funcs['exp_negx2'] = lambda x: np.exp(-x ** 2)
    basis_funcs['exp_neg05x2'] = lambda x: np.exp(-0.5 * x ** 2)
    
    basis_names = list(basis_funcs.keys())
    B = np.zeros((n, len(basis_names)))
    for j, name in enumerate(basis_names):
        try:
            B[:, j] = basis_funcs[name](x)
            B[:, j] = np.where(np.isfinite(B[:, j]), B[:, j], 0.0)
        except Exception:
            pass
    
    if elapsed() < deadline:
        # Greedy forward selection
        selected = []
        remaining = list(range(len(basis_names)))
        for step in range(min(8, len(basis_names))): # Reduced max steps
            if elapsed() > deadline:
                break
            best_j = None
            best_rss = np.inf
            for j in remaining:
                try:
                    if selected:
                        V = np.column_stack([B[:, i] for i in selected] + [B[:, j]])
                    else:
                        V = B[:, j:j+1]
                    w, *_ = np.linalg.lstsq(V, y, rcond=None)
                    rss = float(np.sum((V @ w - y) ** 2))
                    if np.isfinite(rss) and rss < best_rss:
                        best_rss = rss
                        best_j = j
                except Exception:
                    pass
            if best_j is None:
                break
            selected.append(best_j)
            remaining.remove(best_j)
            
            sel_names = [basis_names[i] for i in selected]
            try:
                V = np.column_stack([B[:, i] for i in selected])
                w, *_ = np.linalg.lstsq(V, y, rcond=None)
                def pred_sel(Xp, sel_names=sel_names, w=w):
                    Xp = np.asarray(Xp, dtype=float).reshape(-1)
                    cols = []
                    for nm in sel_names:
                        c = basis_funcs[nm](Xp)
                        c = np.where(np.isfinite(c), c, 0.0)
                        cols.append(c)
                    Vp = np.column_stack(cols)
                    return Vp @ w
                add_candidate(f"sel_{'_'.join(sel_names[:3])}", pred_sel)
            except Exception:
                pass
    
    # --- Strategy 4: Gaussian Process with Matern 5/2 ---
    if elapsed() < deadline:
        def matern52_kernel(X1, X2, ls):
            X1 = np.asarray(X1, dtype=float).reshape(-1, 1)
            X2 = np.asarray(X2, dtype=float).reshape(-1, 1)
            D = np.sqrt(np.sum((X1[:, None, :] - X2[None, :, :]) ** 2, axis=2))
            sqrt3d = np.sqrt(3.0) * D / ls
            return (1 + sqrt3d + sqrt3d ** 2 / 3.0) * np.exp(-sqrt3d)
        
        y_var = float(np.var(y))
        if y_var < 1e-12:
            y_var = 1.0
        
        # Grid search for length scale
        ls_candidates = [x_range / 3.0, x_range / 5.0, x_range / 10.0, 0.1, 0.2, 0.5, 1.0, 2.0]
        ls_candidates = [ls for ls in ls_candidates if ls > 0]
        
        for ls in ls_candidates:
            if elapsed() > deadline:
                break
            try:
                K_kernel = matern52_kernel(x, x, ls)
                K = y_var * K_kernel + 1e-8 * np.eye(n)
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
                
                def pred_gp(Xp, x=x, L=L, alpha=alpha, ls=ls, y_var=y_var, matern52_kernel=matern52_kernel):
                    Xp = np.asarray(Xp, dtype=float).reshape(-1)
                    Ks = matern52_kernel(Xp, x, ls)
                    return y_var * (Ks @ alpha)
                
                add_candidate(f"gp_ls{ls:.4f}", pred_gp)
            except Exception:
                pass
        
    # --- Strategy 5: RBF kernel regression ---
    if elapsed() < deadline:
        def rbf_kernel(X1, X2, width):
            X1 = np.asarray(X1, dtype=float).reshape(-1, 1)
            X2 = np.asarray(X2, dtype=float).reshape(-1, 1)
            D2 = np.sum((X1[:, None, :] - X2[None, :, :]) ** 2, axis=2)
            return np.exp(-D2 / (2 * width ** 2))
        
        width_candidates = [x_range / 4.0, x_range / 8.0, x_range / 16.0, 0.1, 0.2, 0.5, 1.0]
        width_candidates = [w for w in width_candidates if w > 0]
        
        for width in width_candidates:
            if elapsed() > deadline:
                break
            try:
                K = rbf_kernel(x, x, width)
                K += 1e-8 * np.eye(n)
                w, *_ = np.linalg.lstsq(K, y, rcond=None)
                
                def pred_rbf(Xp, x=x, w=w, width=width, rbf_kernel=rbf_kernel):
                    Xp = np.asarray(Xp, dtype=float).reshape(-1)
                    Ks = rbf_kernel(Xp, x, width)
                    return Ks @ w
                
                add_candidate(f"rbf_w{width:.4f}", pred_rbf)
            except Exception:
                pass

    # --- Strategy 6: Kernel Ridge Regression (New) ---
    if elapsed() < deadline:
        # Define kernels
        def kernel_rbf(X1, X2, gamma):
            sq_dists = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T
            sq_dists = np.maximum(sq_dists, 0)
            return np.exp(-gamma * sq_dists)
        
        def kernel_poly(X1, X2, degree, gamma, coef0):
            return (gamma * X1 @ X2.T + coef0) ** degree
        
        def kernel_sigmoid(X1, X2, gamma, coef0):
            return np.tanh(gamma * X1 @ X2.T + coef0)
        
        X_norm = (X - x_min) / x_range
        y_norm = (y - y_mean) / y_std
        
        # Hyperparameters
        gammas = [0.1, 0.5, 1.0, 5.0, 10.0]
        alphas = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
        
        # RBF KRR
        for gamma in gammas:
            if elapsed() > deadline: break
            for alpha in alphas:
                if elapsed() > deadline: break
                try:
                    K = kernel_rbf(X_norm, X_norm, gamma)
                    w = np.linalg.solve(K + alpha * np.eye(n), y_norm)
                    
                    def pred_krr_rbf(Xp, X_norm=X_norm, w=w, gamma=gamma, y_mean=y_mean, y_std=y_std, x_min=x_min, x_range=x_range):
                        Xp = np.asarray(Xp, dtype=float).reshape(-1, 1)
                        Xp_norm = (Xp - x_min) / x_range
                        Ks = kernel_rbf(Xp_norm, X_norm, gamma)
                        return y_mean + y_std * (Ks @ w)
                    
                    add_candidate(f"krr_rbf_g{gamma}_a{alpha}", pred_krr_rbf)
                except Exception:
                    pass
        
        # Poly KRR
        for degree in [2, 3]:
            if elapsed() > deadline: break
            for gamma in [0.5, 1.0]:
                if elapsed() > deadline: break
                for alpha in [1e-2, 1e-1, 1.0]:
                    if elapsed() > deadline: break
                    try:
                        K = kernel_poly(X_norm, X_norm, degree, gamma, 1.0)
                        w = np.linalg.solve(K + alpha * np.eye(n), y_norm)
                        
                        def pred_krr_poly(Xp, X_norm=X_norm, w=w, degree=degree, gamma=gamma, y_mean=y_mean, y_std=y_std, x_min=x_min, x_range=x_range):
                            Xp = np.asarray(Xp, dtype=float).reshape(-1, 1)
                            Xp_norm = (Xp - x_min) / x_range
                            Ks = kernel_poly(Xp_norm, X_norm, degree, gamma, 1.0)
                            return y_mean + y_std * (Ks @ w)
                        
                        add_candidate(f"krr_poly_d{degree}_g{gamma}_a{alpha}", pred_krr_poly)
                    except Exception:
                        pass

    # --- Strategy 7: Spline interpolation via piecewise polynomial ---
    if elapsed() < deadline:
        try:
            from scipy.interpolate import UnivariateSpline
            for s in [0.0, 0.01, 0.1, 1.0, 10.0]:
                if elapsed() > deadline:
                    break
                try:
                    spl = UnivariateSpline(x, y, s=s)
                    def pred_spline(Xp, spl=spl):
                        Xp = np.asarray(Xp, dtype=float).reshape(-1)
                        return spl(Xp)
                    add_candidate(f"spline_s{s}", pred_spline)
                except Exception:
                    pass
        except Exception:
            pass
    
    # Select best candidate
    if not candidates:
        # Fallback: linear regression
        A = np.column_stack([np.ones(n), x])
        w, *_ = np.linalg.lstsq(A, y, rcond=None)
        def pred_lin(Xp, w=w):
            Xp = np.asarray(Xp, dtype=float).reshape(-1)
            return np.column_stack([np.ones(len(Xp)), Xp]) @ w
        return pred_lin
    
    candidates.sort(key=lambda t: t[2])
    best_name, best_pred, best_mse = candidates[0]
    
    def final_predict(Xp):
        out = best_pred(Xp)
        out = np.asarray(out, dtype=float).flatten()
        out = np.where(np.isfinite(out), out, 0.0)
        return out
    
    return final_predict
# EVOLVE-BLOCK-END
