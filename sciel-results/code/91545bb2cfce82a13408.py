import numpy as np
import time
import math
from numpy.polynomial.chebyshev import cheb2poly

def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin
    
    x = np.array(training_x, dtype=np.float64)
    y = np.array(training_subgrid_tendency, dtype=np.float64)
    
    # Flatten just in case
    x = x.flatten()
    y = y.flatten()
    
    n_samples = len(x)
    if n_samples == 0:
        return np.zeros(6)
        
    # If data is extremely small, just do a simple polyfit
    if n_samples < 10:
        try:
            coeff = np.polynomial.polynomial.polyfit(x, y, 5)
            return coeff
        except:
            return np.zeros(6)

    # Standardization
    x_mean = np.mean(x)
    x_std = np.std(x)
    if x_std < 1e-10:
        x_std = 1.0
    x_scaled = (x - x_mean) / x_std
    
    # Build Chebyshev basis matrix for degree 5
    # T_0(z) = 1
    # T_1(z) = z
    # T_k(z) = 2z T_{k-1}(z) - T_{k-2}(z)
    T = np.zeros((n_samples, 6))
    T[:, 0] = 1.0
    if n_samples > 0:
        T[:, 1] = x_scaled
    for k in range(2, 6):
        T[:, k] = 2.0 * x_scaled * T[:, k-1] - T[:, k-2]
        
    # Cross-validation to select lambda
    # Split data into 10 folds for better selection
    indices = np.arange(n_samples)
    np.random.seed(42)
    np.random.shuffle(indices)
    n_folds = 10
    folds = np.array_split(indices, n_folds)
    
    # Wider lambda grid, including 0
    lambdas = np.logspace(-6, 4, 20)
    lambdas = np.concatenate(([0.0], lambdas))
    
    best_lambda = 0.0
    best_cv_mse = np.inf
    
    for lam in lambdas:
        if time.time() > deadline - 20.0:
            break
            
        mses = []
        for i in range(n_folds):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
            
            T_train = T[train_idx]
            y_train = y[train_idx]
            T_val = T[val_idx]
            y_val = y[val_idx]
            
            # Solve Ridge: (T^T T + lambda * D) c = T^T y
            # D is diag(0, 1, 1, 1, 1, 1)
            TtT = T_train.T @ T_train
            Tty = T_train.T @ y_train
            
            # Add penalty
            # Do not penalize the intercept (index 0)
            TtT_p = TtT.copy()
            if lam > 0:
                TtT_p[1:, 1:] += lam * np.eye(5)
            
            try:
                c_cheb = np.linalg.solve(TtT_p, Tty)
            except np.linalg.LinAlgError:
                c_cheb = np.linalg.lstsq(TtT_p, Tty, rcond=None)[0]
                
            # Predict on val
            y_pred = T_val @ c_cheb
            mse = np.mean((y_val - y_pred)**2)
            mses.append(mse)
            
        mean_mse = np.mean(mses)
        if mean_mse < best_cv_mse:
            best_cv_mse = mean_mse
            best_lambda = lam
            
    # Refit with best lambda on all data
    TtT = T.T @ T
    Tty = T.T @ y
    TtT_p = TtT.copy()
    if best_lambda > 0:
        TtT_p[1:, 1:] += best_lambda * np.eye(5)
    
    try:
        c_cheb = np.linalg.solve(TtT_p, Tty)
    except np.linalg.LinAlgError:
        c_cheb = np.linalg.lstsq(TtT_p, Tty, rcond=None)[0]
        
    # Convert Chebyshev coefficients to Monomial coefficients
    # Use numpy's built-in function for stability
    c_mon_z = cheb2poly(c_cheb)
    
    # Transform coefficients from scaled domain (z) to original domain (x)
    # z = (x - x_mean) / x_std
    # P(z) = sum c_mon_z_i z^i
    # We want Q(x) = P((x - x_mean) / x_std)
    # Let a = 1/x_std, b = -x_mean/x_std
    # Q(x) = P(a*x + b)
    # c_mon_z are coefficients for z.
    # Q(x) = sum_{k=0}^5 c_mon_z_k (a*x + b)^k
    #      = sum_{k=0}^5 c_mon_z_k sum_{j=0}^k binom(k, j) (a*x)^j b^{k-j}
    #      = sum_{j=0}^5 ( sum_{k=j}^5 c_mon_z_k binom(k, j) a^j b^{k-j} ) x^j
    
    a = 1.0 / x_std
    b = -x_mean / x_std
    
    c_orig = np.zeros(6)
    for j in range(6):
        s = 0.0
        for k in range(j, 6):
            binom_k_j = math.comb(k, j)
            term = c_mon_z[k] * binom_k_j * (a**j) * (b**(k-j))
            s += term
        c_orig[j] = s
        
    return c_orig
