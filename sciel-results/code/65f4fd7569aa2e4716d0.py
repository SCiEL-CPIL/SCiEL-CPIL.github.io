"""Symbolic regression via Sparse GP (LASSO-GP) with adaptive hyperparameter search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import operator
from scipy.optimize import minimize

def fit(X_train, y_train):
    start_time = time.time()
    # 300s limit. We reserve 30s for safety/overhead, leaving 270s for computation.
    deadline = start_time + 270.0
    
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64).flatten()
    n_samples, n_features = X.shape
    
    # Normalize inputs to [0, 1] for numerical stability in distance calculations
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_range = X_max - X_min
    X_range[X_range == 0] = 1.0
    X_norm = (X - X_min) / X_range
    
    # Standardize y
    y_mean = y.mean()
    y_std = y.std()
    if y_std == 0:
        y_std = 1.0
    y_norm = (y - y_mean) / y_std

    # Strategy: Sparse GP (LASSO-GP)
    # This approach fits a GP with a linear combination of basis functions.
    # It is robust to non-smooth functions if the basis set is rich enough,
    # and the sparsity constraint (L1) helps select the most relevant features.
    
    # Define a set of candidate basis functions.
    # We include polynomials, sines, cosines, and RBF-like bumps.
    # Since dimensions vary (1 to 3), we generate bases per dimension.
    
    # For each dimension d_j, we create a set of functions.
    # To keep the basis size manageable for LASSO (which is O(N * M)), 
    # we limit the number of basis functions per dimension.
    
    basis_funcs = []
    
    # 1. Constant
    basis_funcs.append(lambda X: np.ones((X.shape[0], 1)))
    
    # 2. Linear terms (already in X_norm, but let's be explicit)
    for j in range(n_features):
        basis_funcs.append(lambda X, j=j: X[:, j:j+1])
        
    # 3. Polynomial terms (degree 2)
    for j in range(n_features):
        basis_funcs.append(lambda X, j=j: X[:, j:j+1] ** 2)
        
    # 4. Interaction terms (if d > 1)
    if n_features > 1:
        for i in range(n_features):
            for j in range(i+1, n_features):
                basis_funcs.append(lambda X, i=i, j=j: X[:, i:i+1] * X[:, j:j+1])
                
    # 5. Trigonometric terms (to capture periodic or oscillatory behavior)
    # Frequencies: 1, 2, 3 (scaled to [0, 1] range, so 2*pi * freq * x)
    freqs = [1.0, 2.0, 3.0]
    for j in range(n_features):
        for f in freqs:
            basis_funcs.append(lambda X, j=j, f=f: np.sin(2 * np.pi * f * X[:, j:j+1]))
            basis_funcs.append(lambda X, j=j, f=f: np.cos(2 * np.pi * f * X[:, j:j+1]))
            
    # 6. RBF-like bumps (localized features)
    # Centers: 0.25, 0.5, 0.75 for each dimension
    centers = [0.25, 0.5, 0.75]
    widths = [0.1, 0.2, 0.4]
    for j in range(n_features):
        for c in centers:
            for w in widths:
                # RBF: exp(-0.5 * (x-c)^2 / w^2)
                basis_funcs.append(lambda X, j=j, c=c, w=w: np.exp(-0.5 * ((X[:, j:j+1] - c) / w) ** 2))

    # Build the design matrix Phi
    # Phi shape: (n_samples, n_basis)
    Phi = np.column_stack([f(X_norm) for f in basis_funcs])
    
    # Handle potential NaNs or infs in Phi (shouldn't happen with bounded inputs, but safe)
    Phi = np.nan_to_num(Phi, nan=0.0, posinf=0.0, neginf=0.0)
    
    # LASSO-GP Objective: Minimize ||y - Phi @ w||^2 + lambda * ||w||_1
    # We can solve this using scipy.optimize or a simple iterative LASSO solver.
    # For speed and robustness, we use a coordinate descent or simply use 
    # scipy.optimize.minimize with L1 penalty if available, or a simple Lasso via 
    # iterative reweighted least squares (IRLS) or just use a library if allowed.
    # Since we only have standard library, numpy, scipy:
    # We can use scipy.optimize.minimize with method='L-BFGS-B' but L1 is non-smooth.
    # Alternative: Use a simple L1-regularized least squares via coordinate descent.
    
    def solve_lasso(Phi, y, lam):
        n, m = Phi.shape
        w = np.zeros(m)
        # Precompute column norms for coordinate descent
        col_norms = np.sum(Phi**2, axis=0)
        col_norms[col_norms == 0] = 1e-10
        
        # Simple coordinate descent for LASSO
        # This might be slow if m is large. 
        # m is roughly 1 + d + d + d(d-1)/2 + 6d + 9d = ~20d + interactions.
        # For d=3, m ~ 1+3+3+3+18+27 = 55. Very small.
        # For d=1, m ~ 1+1+1+0+6+9 = 18.
        
        max_iter = 100
        for _ in range(max_iter):
            w_old = w.copy()
            for j in range(m):
                # Residual without current feature
                r = y - Phi @ w + Phi[:, j] * w[j]
                # Update w[j]
                # Minimize ||r - Phi_j * w_j||^2 + lam * |w_j|
                # Solution: soft_threshold(<Phi_j, r> / ||Phi_j||^2, lam / ||Phi_j||^2)
                inner = np.dot(Phi[:, j], r)
                denom = col_norms[j]
                z = inner / denom
                threshold = lam / denom
                if z > threshold:
                    w[j] = z - threshold
                elif z < -threshold:
                    w[j] = z + threshold
                else:
                    w[j] = 0.0
            
            if np.max(np.abs(w - w_old)) < 1e-6:
                break
                
        return w

    # We want to find a good lambda. 
    # Instead of a full cross-validation which is expensive, 
    # we can try a few candidate lambdas and pick the one with lowest training error 
    # (since noise is 0, training error is a good proxy, but LASSO might underfit if lambda is too high).
    # Actually, with noiseless data, we want the simplest model that fits perfectly.
    # So we should try small lambdas.
    
    candidate_lambdas = [1e-4, 1e-5, 1e-6, 1e-7]
    
    best_w = None
    best_score = np.inf
    
    for lam in candidate_lambdas:
        if time.time() > deadline - 10:
            break
            
        w = solve_lasso(Phi, y_norm, lam)
        pred_norm = Phi @ w
        mse = np.mean((pred_norm - y_norm)**2)
        
        # We prefer low MSE. If multiple have similar MSE, prefer sparser (larger lambda)
        # But for now, just pick lowest MSE.
        if mse < best_score:
            best_score = mse
            best_w = w
            
    if best_w is None:
        # Fallback to OLS if LASSO fails or times out
        try:
            best_w, *_ = np.linalg.lstsq(Phi, y_norm, rcond=None)
        except:
            best_w = np.zeros(Phi.shape[1])

    def predict(X_test):
        X_test = np.asarray(X_test, dtype=np.float64)
        # Normalize test data using train stats
        X_test_norm = (X_test - X_min) / X_range
        
        # Build test design matrix
        Phi_test = np.column_stack([f(X_test_norm) for f in basis_funcs])
        Phi_test = np.nan_to_num(Phi_test, nan=0.0, posinf=0.0, neginf=0.0)
        
        pred_norm = Phi_test @ best_w
        pred = pred_norm * y_std + y_mean
        
        return pred

    return predict
# EVOLVE-BLOCK-END
