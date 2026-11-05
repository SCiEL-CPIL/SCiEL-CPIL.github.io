import numpy as np
import time
from scipy.optimize import nnls # Not needed, least squares is linear
# from sklearn.linear_model import Ridge # Can't assume sklearn is available? "Python standard library, NumPy and SciPy are available". So no sklearn.

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
    
    # Build Vandermonde matrix for degree 5
    # V[i, j] = x_scaled[i]^j
    V = np.vander(x_scaled, N=6, increasing=True)
    
    # Cross-validation to select lambda
    # Split data into 5 folds
    indices = np.arange(n_samples)
    np.random.shuffle(indices) # Use a fixed seed for reproducibility? 
    # The evaluator might run multiple times or with different data. 
    # Using a fixed seed for the split ensures consistent behavior for the same input.
    np.random.seed(42)
    np.random.shuffle(indices)
    folds = np.array_split(indices, 5)
    
    lambdas = np.logspace(-4, 2, 10)
    best_lambda = 0.0
    best_cv_mse = np.inf
    
    # Precompute X^T X and X^T y parts? 
    # X^T X is 6x6. X^T y is 6x1.
    # For each fold, we recompute on the subset.
    
    for lam in lambdas:
        if time.time() > deadline - 20.0:
            break
            
        mses = []
        for i in range(5):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(5) if j != i])
            
            V_train = V[train_idx]
            y_train = y[train_idx]
            V_val = V[val_idx]
            y_val = y[val_idx]
            
            # Solve Ridge: (V^T V + lambda * D) c = V^T y
            # D is diag(0, 1, 1, 1, 1, 1)
            VtV = V_train.T @ V_train
            Vty = V_train.T @ y_train
            
            # Add penalty
            # Do not penalize the intercept (index 0)
            VtV_p = VtV.copy()
            VtV_p[1:, 1:] += lam * np.eye(5)
            
            try:
                c = np.linalg.solve(VtV_p, Vty)
            except np.linalg.LinAlgError:
                c = np.linalg.lstsq(VtV_p, Vty, rcond=None)[0]
                
            # Predict on val
            y_pred = V_val @ c
            mse = np.mean((y_val - y_pred)**2)
            mses.append(mse)
            
        mean_mse = np.mean(mses)
        if mean_mse < best_cv_mse:
            best_cv_mse = mean_mse
            best_lambda = lam
            
    # Refit with best lambda on all data
    VtV = V.T @ V
    Vty = V.T @ y
    VtV_p = VtV.copy()
    VtV_p[1:, 1:] += best_lambda * np.eye(5)
    
    try:
        c_scaled = np.linalg.solve(VtV_p, Vty)
    except np.linalg.LinAlgError:
        c_scaled = np.linalg.lstsq(VtV_p, Vty, rcond=None)[0]
        
    # Transform coefficients from scaled domain (z) to original domain (x)
    # z = (x - x_mean) / x_std
    # P(z) = sum c_i z^i
    # We want Q(x) = P((x - x_mean) / x_std)
    # Let a = 1/x_std, b = -x_mean/x_std
    # Q(x) = P(a*x + b)
    # c_scaled are coefficients for z.
    # Q(x) = sum_{k=0}^5 c_k (a*x + b)^k
    #      = sum_{k=0}^5 c_k sum_{j=0}^k binom(k, j) (a*x)^j b^{k-j}
    #      = sum_{j=0}^5 ( sum_{k=j}^5 c_k binom(k, j) a^j b^{k-j} ) x^j
    
    a = 1.0 / x_std
    b = -x_mean / x_std
    
    c_orig = np.zeros(6)
    for j in range(6):
        s = 0.0
        for k in range(j, 6):
            # binom(k, j)
            binom_k_j = 1.0
            for m in range(j):
                binom_k_j *= (k - m)
            for m in range(1, j+1):
                binom_k_j /= m
            # Wait, simpler: math.comb
            import math
            binom_k_j = math.comb(k, j)
            
            term = c_scaled[k] * binom_k_j * (a**j) * (b**(k-j))
            s += term
        c_orig[j] = s
        
    return c_orig
