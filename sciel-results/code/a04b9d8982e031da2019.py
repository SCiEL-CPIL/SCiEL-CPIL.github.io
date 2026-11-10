import numpy as np
import scipy.special as sp
import time
import sys

def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill
    
    x = np.asarray(training_x, dtype=np.float64).flatten()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).flatten()
    
    # Filter non-finite
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    
    n = len(x)
    if n == 0:
        return np.zeros(6)
    if n < 10:
        return np.array([np.mean(y), 0, 0, 0, 0, 0])
    
    # Check time
    if time.time() > deadline - 10.0:
        # Fallback: OLS linear on original x
        coeff = np.polynomial.polynomial.polyfit(x, y, 1)
        return coeff

    # Scaling to [-1, 1] for Legendre basis
    x_min = np.min(x)
    x_max = np.max(x)
    if x_max - x_min < 1e-10:
        x_max = x_min + 1.0
    
    t = 2.0 * (x - x_min) / (x_max - x_min) - 1.0
    
    deg = 5
    
    # Subsample for CV if large to ensure speed
    max_cv_samples = 50000
    if n > max_cv_samples:
        np.random.seed(42)
        idx = np.random.choice(n, max_cv_samples, replace=False)
        t_cv = t[idx]
        y_cv = y[idx]
    else:
        t_cv = t
        y_cv = y
        
    n_cv = len(t_cv)
    
    # Precompute Legendre basis for CV data
    # L[i, j] = P_j(t_i)
    L_cv = np.zeros((n_cv, deg + 1))
    for j in range(deg + 1):
        L_cv[:, j] = sp.eval_legendre(j, t_cv)
        
    lambdas = np.logspace(-8, 4, 60)
    k = 5
    indices = np.arange(n_cv)
    np.random.seed(42)
    np.random.shuffle(indices)
    folds = np.array_split(indices, k)
    
    best_lambda = 1e-8
    min_cv_mse = np.inf
    
    # Precompute fold splits
    fold_data = []
    for i in range(k):
        val_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        fold_data.append((train_idx, val_idx))
        
    # We can precompute L_train^T L_train and L_train^T y_train for each fold
    # to speed up the lambda loop?
    # Actually, L_train changes per fold, but we can compute the Gram matrix once per fold.
    
    gram_matrices = []
    rhs_vectors = []
    L_vals = []
    y_vals = []
    
    for train_idx, val_idx in fold_data:
        L_train = L_cv[train_idx]
        y_train = y_cv[train_idx]
        L_val = L_cv[val_idx]
        y_val = y_cv[val_idx]
        
        G = L_train.T @ L_train
        g = L_train.T @ y_train
        
        gram_matrices.append(G)
        rhs_vectors.append(g)
        L_vals.append(L_val)
        y_vals.append(y_val)
        
    # Loop over lambdas
    for lam in lambdas:
        if time.time() > deadline - 5.0:
            break
            
        mse_sum = 0.0
        for i in range(k):
            G = gram_matrices[i]
            g = rhs_vectors[i]
            L_val = L_vals[i]
            y_val = y_vals[i]
            
            A = G + lam * np.eye(deg + 1)
            
            try:
                beta = np.linalg.solve(A, g)
            except np.linalg.LinAlgError:
                beta = np.linalg.lstsq(A, g, rcond=None)[0]
                
            y_pred = L_val @ beta
            mse = np.mean((y_val - y_pred)**2)
            mse_sum += mse
            
        mean_mse = mse_sum / k
        if mean_mse < min_cv_mse:
            min_cv_mse = mean_mse
            best_lambda = lam
            
    # Final Fit on all data
    L_full = np.zeros((n, deg + 1))
    for j in range(deg + 1):
        L_full[:, j] = sp.eval_legendre(j, t)
        
    G_full = L_full.T @ L_full
    g_full = L_full.T @ y
    
    A_full = G_full + best_lambda * np.eye(deg + 1)
    
    try:
        beta_legendre = np.linalg.solve(A_full, g_full)
    except np.linalg.LinAlgError:
        beta_legendre = np.linalg.lstsq(A_full, g_full, rcond=None)[0]
        
    # Convert Legendre coefficients to Monomial coefficients in t
    # beta_legendre = [b0, b1, ..., b5]
    # We want a_t such that sum(b_j * P_j(t)) = sum(a_k * t^k)
    
    a_t = np.zeros(deg + 1)
    for j in range(deg + 1):
        # Get monomial coefficients of P_j(t)
        # P_j(t) = sum_{k=0}^j d_{jk} t^k
        # Use leg2poly
        poly_coeffs = sp.leg2poly(beta_legendre[j])
        # poly_coeffs is [c_j, ..., c_0] (high to low)
        # leg2poly returns coefficients in descending order?
        # Documentation: "Returns the coefficients of the polynomial in descending order."
        # So poly_coeffs[0] is coeff of t^j, poly_coeffs[j] is coeff of t^0.
        
        for k in range(j + 1):
            # index in poly_coeffs: j - k
            # index in a_t: k
            a_t[k] += beta_legendre[j] * poly_coeffs[j - k]
            
    # Now a_t are coefficients in t: sum_{k=0}^5 a_t[k] t^k
    # We need coefficients c in x: sum_{m=0}^5 c[m] x^m
    # t = (x - x_min) * 2 / (x_max - x_min) - 1
    # Let A = 2 / (x_max - x_min) and B = -x_min * 2 / (x_max - x_min) - 1
    # t = A * x + B
    
    A_coeff = 2.0 / (x_max - x_min)
    B_coeff = -x_min * A_coeff - 1.0
    
    c = np.zeros(deg + 1)
    
    # Expand (A*x + B)^k
    # (A*x + B)^k = sum_{m=0}^k binom(k, m) (A*x)^m B^{k-m}
    #             = sum_{m=0}^k binom(k, m) A^m B^{k-m} x^m
    
    for k in range(deg + 1):
        coeff_t_k = a_t[k]
        if np.abs(coeff_t_k) < 1e-15:
            continue
        for m in range(k + 1):
            binom_val = sp.comb(k, m, exact=True)
            term = coeff_t_k * binom_val * (A_coeff ** m) * (B_coeff ** (k - m))
            c[m] += term
            
    return c
