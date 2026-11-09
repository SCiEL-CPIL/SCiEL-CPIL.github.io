"""Symbolic regression via kernel ridge regression with composite kernels."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _rbf_kernel(X1, X2, length_scales):
    """Compute RBF kernel between two sets of points. Vectorized, no 3D tensors."""
    X1s = X1 / length_scales
    X2s = X2 / length_scales
    sq1 = np.sum(X1s**2, axis=1)[:, None]
    sq2 = np.sum(X2s**2, axis=1)[None, :]
    sq_dists = sq1 + sq2 - 2.0 * (X1s @ X2s.T)
    np.maximum(sq_dists, 0.0, out=sq_dists)
    return np.exp(-0.5 * sq_dists)


def _poly_kernel(X1, X2, degree, gamma, coef0):
    """Compute polynomial kernel."""
    dot = gamma * (X1 @ X2.T) + coef0
    return dot ** degree


def _periodic_kernel_fast(X1, X2, period, length_scale):
    """Compute periodic kernel without 3D intermediate tensor.
    Uses identity: sin^2(a-b) = (sin^2(a) + sin^2(b) - 2*sin(a)*sin(b))
    but that doesn't simplify the sum. Instead use:
    sum_d sin^2(pi*(x1_d - x2_d)/period)
    = sum_d [sin^2(pi*x1_d/period) + sin^2(pi*x2_d/period) - 2*sin(pi*x1_d/period)*sin(pi*x2_d/period)]
    This is O(n1*n2*d) via matrix multiply on the sin components.
    """
    s1 = np.sin(np.pi * X1 / period)  # (n1, d)
    s2 = np.sin(np.pi * X2 / period)  # (n2, d)
    sq1 = np.sum(s1**2, axis=1)[:, None]  # (n1, 1)
    sq2 = np.sum(s2**2, axis=1)[None, :]  # (1, n2)
    cross = s1 @ s2.T  # (n1, n2)
    sq_sum = sq1 + sq2 - 2.0 * cross
    np.maximum(sq_sum, 0.0, out=sq_sum)
    return np.exp(-2.0 * sq_sum / (length_scale**2))


def _compute_kernel(X1, X2, rbf_ls, rbf_w, poly_w, poly_deg, poly_gamma, poly_c0, per_w, per_period, per_ls):
    """Compute composite kernel efficiently."""
    n1, n2 = X1.shape[0], X2.shape[0]
    K = np.zeros((n1, n2))
    
    if rbf_w > 1e-15:
        K += rbf_w * _rbf_kernel(X1, X2, rbf_ls)
    
    if poly_w > 1e-15:
        K += poly_w * _poly_kernel(X1, X2, poly_deg, poly_gamma, poly_c0)
    
    if per_w > 1e-15:
        K += per_w * _periodic_kernel_fast(X1, X2, per_period, per_ls)
    
    return K


def _kr_solve(K, y, alpha):
    """Kernel ridge solve with fallback."""
    n = K.shape[0]
    A = K + alpha * np.eye(n)
    try:
        return np.linalg.solve(A, y)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(A, y, rcond=None)[0]


def fit(X_train, y_train):
    t0 = time.time()
    deadline = t0 + 300.0
    
    n, d = X_train.shape
    y_mean = np.mean(y_train)
    y_centered = y_train - y_mean
    
    # Normalize to [0,1]
    x_min = X_train.min(axis=0)
    x_max = X_train.max(axis=0)
    x_range = np.maximum(x_max - x_min, 1e-12)
    Xn = (X_train - x_min) / x_range
    
    # Precompute: for RBF we need Xn/ls, for periodic we need sin(pi*Xn/period)
    # We'll compute kernels on the fly but cache sin values for periodic
    
    best_score = np.inf
    best_config = None
    
    # Phase 1: RBF-only grid search with leave-one-out CV (fast for n=200)
    # LOO CV: (K+alpha*I)^{-1} y, pred_i = y_i - 1/(K_ii + alpha) * (K_ii*y_i - sum_j K_ij*y_j)
    # Actually simpler: pred_i = y_i - (K_ii*y_i - K_row_i @ y) / (K_ii + alpha) ... no
    # LOO: alpha_i = 1 / (K + alpha*I)_{ii} * y_i ... 
    # Standard LOO for KRR: e_i = y_i - [K(K+alpha*I)^{-1}y]_i = y_i - y_i + alpha * [(K+alpha*I)^{-1}y]_i
    # So pred_i = y_i - alpha * [(K+alpha*I)^{-1}y]_i
    # This requires one solve per (ls, alpha) pair!
    
    ls_grid = np.array([0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0])
    alpha_grid = np.array([1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0])
    
    # Precompute Xn squared norms for RBF
    Xn_sq = np.sum(Xn**2, axis=1)
    
    for ls in ls_grid:
        if time.time() > deadline - 30:
            break
        rbf_ls = np.full(d, ls)
        Xs = Xn / rbf_ls
        Xs_sq = np.sum(Xs**2, axis=1)
        
        for alpha in alpha_grid:
            if time.time() > deadline - 30:
                break
            # Compute K = exp(-0.5 * sq_dists)
            sq_dists = Xs_sq[:, None] + Xs_sq[None, :] - 2.0 * (Xs @ Xs.T)
            np.maximum(sq_dists, 0.0, out=sq_dists)
            K = np.exp(-0.5 * sq_dists)
            
            # LOO prediction
            A = K + alpha * np.eye(n)
            try:
                coeffs = np.linalg.solve(A, y_centered)
            except np.linalg.LinAlgError:
                coeffs = np.linalg.lstsq(A, y_centered, rcond=None)[0]
            
            # LOO: pred_i = y_i - alpha * coeffs_i
            loo_pred = y_centered - alpha * coeffs
            loo_mse = np.mean((loo_pred - y_centered)**2)
            
            if loo_mse < best_score:
                best_score = loo_mse
                best_config = {
                    'rbf_ls': rbf_ls.copy(), 'rbf_w': 1.0,
                    'poly_w': 0.0, 'poly_deg': 2, 'poly_gamma': 1.0, 'poly_c0': 1.0,
                    'per_w': 0.0, 'per_period': 2.0, 'per_ls': 1.0,
                    'alpha': alpha
                }
    
    if best_config is None:
        best_config = {
            'rbf_ls': np.ones(d), 'rbf_w': 1.0,
            'poly_w': 0.0, 'poly_deg': 2, 'poly_gamma': 1.0, 'poly_c0': 1.0,
            'per_w': 0.0, 'per_period': 2.0, 'per_ls': 1.0,
            'alpha': 1e-2
        }
        best_score = np.mean(y_centered**2)
    
    # Phase 2: Try adding polynomial kernel
    poly_configs = [
        (0.1, 2, 1.0, 1.0),
        (0.5, 2, 1.0, 1.0),
        (1.0, 2, 1.0, 1.0),
        (0.5, 3, 0.5, 1.0),
        (1.0, 2, 0.5, 0.5),
        (0.3, 2, 2.0, 1.0),
    ]
    
    for (pw, pd, pg, pc) in poly_configs:
        if time.time() > deadline - 20:
            break
        for alpha_mult in [0.1, 1.0, 10.0]:
            if time.time() > deadline - 20:
                break
            alpha = best_config['alpha'] * alpha_mult
            ls = best_config['rbf_ls'][0]
            
            K = _compute_kernel(Xn, Xn, best_config['rbf_ls'], 1.0, pw, pd, pg, pc, 0.0, 2.0, 1.0)
            
            # LOO
            A = K + alpha * np.eye(n)
            try:
                coeffs = np.linalg.solve(A, y_centered)
            except np.linalg.LinAlgError:
                coeffs = np.linalg.lstsq(A, y_centered, rcond=None)[0]
            
            loo_pred = y_centered - alpha * coeffs
            loo_mse = np.mean((loo_pred - y_centered)**2)
            
            if loo_mse < best_score:
                best_score = loo_mse
                best_config = {
                    'rbf_ls': best_config['rbf_ls'].copy(), 'rbf_w': 1.0,
                    'poly_w': pw, 'poly_deg': pd, 'poly_gamma': pg, 'poly_c0': pc,
                    'per_w': 0.0, 'per_period': 2.0, 'per_ls': 1.0,
                    'alpha': alpha
                }
    
    # Phase 3: Try adding periodic kernel
    per_configs = [
        (0.1, 2.0, 0.5),
        (0.5, 2.0, 0.5),
        (1.0, 2.0, 0.5),
        (0.5, 1.0, 0.3),
        (0.5, 3.0, 0.7),
        (0.3, 2.0, 1.0),
    ]
    
    for (pw_, pp, pl) in per_configs:
        if time.time() > deadline - 10:
            break
        for alpha_mult in [0.1, 1.0, 10.0]:
            if time.time() > deadline - 10:
                break
            alpha = best_config['alpha'] * alpha_mult
            
            K = _compute_kernel(Xn, Xn, best_config['rbf_ls'], best_config['rbf_w'],
                                best_config['poly_w'], best_config['poly_deg'],
                                best_config['poly_gamma'], best_config['poly_c0'],
                                pw_, pp, pl)
            
            A = K + alpha * np.eye(n)
            try:
                coeffs = np.linalg.solve(A, y_centered)
            except np.linalg.LinAlgError:
                coeffs = np.linalg.lstsq(A, y_centered, rcond=None)[0]
            
            loo_pred = y_centered - alpha * coeffs
            loo_mse = np.mean((loo_pred - y_centered)**2)
            
            if loo_mse < best_score:
                best_score = loo_mse
                best_config = {
                    'rbf_ls': best_config['rbf_ls'].copy(), 'rbf_w': best_config['rbf_w'],
                    'poly_w': best_config['poly_w'], 'poly_deg': best_config['poly_deg'],
                    'poly_gamma': best_config['poly_gamma'], 'poly_c0': best_config['poly_c0'],
                    'per_w': pw_, 'per_period': pp, 'per_ls': pl,
                    'alpha': alpha
                }
    
    # Phase 4: Fine-tune RBF length scale around best
    if time.time() < deadline - 15:
        ls_best = best_config['rbf_ls'][0]
        for ls_factor in [0.5, 0.7, 1.0, 1.3, 1.5, 2.0]:
            if time.time() > deadline - 15:
                break
            ls_new = ls_best * ls_factor
            rbf_ls_new = np.full(d, ls_new)
            
            K = _compute_kernel(Xn, Xn, rbf_ls_new, best_config['rbf_w'],
                                best_config['poly_w'], best_config['poly_deg'],
                                best_config['poly_gamma'], best_config['poly_c0'],
                                best_config['per_w'], best_config['per_period'], best_config['per_ls'])
            
            A = K + best_config['alpha'] * np.eye(n)
            try:
                coeffs = np.linalg.solve(A, y_centered)
            except np.linalg.LinAlgError:
                coeffs = np.linalg.lstsq(A, y_centered, rcond=None)[0]
            
            loo_pred = y_centered - best_config['alpha'] * coeffs
            loo_mse = np.mean((loo_pred - y_centered)**2)
            
            if loo_mse < best_score:
                best_score = loo_mse
                best_config['rbf_ls'] = rbf_ls_new
    
    # Phase 5: Fine-tune alpha around best
    if time.time() < deadline - 10:
        alpha_best = best_config['alpha']
        for alpha_factor in [0.1, 0.3, 1.0, 3.0, 10.0]:
            if time.time() > deadline - 10:
                break
            alpha_new = alpha_best * alpha_factor
            
            K = _compute_kernel(Xn, Xn, best_config['rbf_ls'], best_config['rbf_w'],
                                best_config['poly_w'], best_config['poly_deg'],
                                best_config['poly_gamma'], best_config['poly_c0'],
                                best_config['per_w'], best_config['per_period'], best_config['per_ls'])
            
            A = K + alpha_new * np.eye(n)
            try:
                coeffs = np.linalg.solve(A, y_centered)
            except np.linalg.LinAlgError:
                coeffs = np.linalg.lstsq(A, y_centered, rcond=None)[0]
            
            loo_pred = y_centered - alpha_new * coeffs
            loo_mse = np.mean((loo_pred - y_centered)**2)
            
            if loo_mse < best_score:
                best_score = loo_mse
                best_config['alpha'] = alpha_new
    
    # Final fit with all data
    cfg = best_config
    K_final = _compute_kernel(Xn, Xn, cfg['rbf_ls'], cfg['rbf_w'],
                              cfg['poly_w'], cfg['poly_deg'], cfg['poly_gamma'], cfg['poly_c0'],
                              cfg['per_w'], cfg['per_period'], cfg['per_ls'])
    coeffs_final = _kr_solve(K_final, y_centered, cfg['alpha'])
    
    def predict(X):
        Xt = (X - x_min) / x_range
        K_test = _compute_kernel(Xt, Xn, cfg['rbf_ls'], cfg['rbf_w'],
                                 cfg['poly_w'], cfg['poly_deg'], cfg['poly_gamma'], cfg['poly_c0'],
                                 cfg['per_w'], cfg['per_period'], cfg['per_ls'])
        return K_test @ coeffs_final + y_mean
    
    return predict
# EVOLVE-BLOCK-END
