"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 280.0  # Leave 20s safety margin for 300s total limit (assuming ~20s per eq overhead or just general safety)
    # Actually, the limit is 300s for ALL 10 equations. 
    # So per equation we have roughly 30s. 
    # Let's be conservative: 25s per equation.
    deadline = start_time + 25.0
    
    d = X_train.shape[1]
    n = X_train.shape[0]
    y_var = np.var(y_train)
    if y_var < 1e-12:
        mean_y = np.mean(y_train)
        return lambda X: np.full(X.shape[0], mean_y)

    # Standardize X for kernel stability
    X_mean = np.mean(X_train, axis=0)
    X_std = np.std(X_train, axis=0)
    X_std[X_std < 1e-8] = 1.0
    X_norm = (X_train - X_mean) / X_std

    best_predict = None
    best_score = np.inf

    # Grid search for hyperparameters
    # RBF: length_scale (sigma)
    # Poly: degree
    # Periodic: period
    
    # We will try a combination of kernels.
    # K = a * K_rbf + b * K_poly + c * K_periodic
    # For simplicity and robustness, we'll just sum them with fixed weights or tune one main one.
    # Given the time constraint, let's tune the RBF length scale and Poly degree primarily,
    # as these cover most smooth and polynomial-like functions.
    # Periodic is useful if there are sines.
    
    # Hyperparameters to search:
    # 1. RBF length scale (sigma)
    # 2. Polynomial degree (2, 3, 4)
    # 3. RBF weight (alpha_rbf)
    # 4. Poly weight (alpha_poly)
    # 5. Periodic weight (alpha_per) and period (T)
    
    # To keep it fast, we'll fix some and search others.
    # Let's search:
    # - RBF sigma: [0.1, 0.5, 1.0, 2.0, 5.0]
    # - Poly degree: [2, 3, 4]
    # - Weights: Fixed sum to 1? Or just tune alpha_rbf and assume poly=1, per=0?
    # Let's try a composite kernel: K = w_rbf * K_rbf + w_poly * K_poly
    # And maybe add periodic if it helps.
    
    # Simplified Grid:
    sigmas = [0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
    degrees = [2, 3, 4]
    # Weights: Let's try (1, 0), (0.5, 0.5), (0, 1) effectively by scaling.
    # Actually, KRR solves (K + lambda I) w = y.
    # We can just try different kernels and pick the best.
    
    kernels_to_try = []
    
    # Generate RBF kernels
    for sigma in sigmas:
        kernels_to_try.append(('rbf', sigma))
        
    # Generate Poly kernels
    for deg in degrees:
        kernels_to_try.append(('poly', deg))
        
    # Generate Periodic kernels (useful for sine/cosine hidden equations)
    # Periods: 2*pi is common, maybe pi, 4*pi
    for period in [np.pi, 2*np.pi, 4*np.pi]:
        kernels_to_try.append(('periodic', period))
        
    # Composite: RBF + Poly
    # We can't easily search weights in a small grid without exploding combinations.
    # Instead, we'll rely on the fact that if a function is polynomial, Poly kernel wins.
    # If it's smooth but not polynomial, RBF wins.
    # If it's periodic, Periodic wins.
    # A mix might be best, but let's first try individual strong kernels.
    # If time permits, we can try a fixed mix like 0.5*RBF(1) + 0.5*Poly(3).
    
    # Add some fixed composites to the list
    kernels_to_try.append(('composite1', 1.0)) # 0.5 RBF(1) + 0.5 Poly(3)
    kernels_to_try.append(('composite2', 0.5)) # 0.5 RBF(0.5) + 0.5 Poly(2)

    def compute_kernel(X1, X2, kernel_type, param):
        """Compute kernel matrix between X1 and X2."""
        if kernel_type == 'rbf':
            # Squared Euclidean distance
            sq_dists = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1.dot(X2.T)
            sq_dists = np.maximum(sq_dists, 0)
            return np.exp(-sq_dists / (2 * param**2))
        elif kernel_type == 'poly':
            # Polynomial kernel: (X1 dot X2 + c)^degree
            c = 1.0
            return (X1.dot(X2.T) + c) ** param
        elif kernel_type == 'periodic':
            # Periodic kernel: cos(pi * ||x1 - x2||^2 / T)
            # ||x1 - x2||^2 = ||x1||^2 + ||x2||^2 - 2 x1.x2
            sq_dists = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1.dot(X2.T)
            sq_dists = np.maximum(sq_dists, 0)
            return np.cos(np.pi * sq_dists / param)
        elif kernel_type == 'composite1':
            # 0.5 * RBF(1.0) + 0.5 * Poly(3)
            k_rbf = np.exp(-np.maximum(np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1.dot(X2.T), 0) / 2.0)
            k_poly = (X1.dot(X2.T) + 1.0) ** 3
            return 0.5 * k_rbf + 0.5 * k_poly
        elif kernel_type == 'composite2':
            # 0.5 * RBF(0.5) + 0.5 * Poly(2)
            sq_dists = np.maximum(np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1.dot(X2.T), 0)
            k_rbf = np.exp(-sq_dists / (2 * 0.5**2))
            k_poly = (X1.dot(X2.T) + 1.0) ** 2
            return 0.5 * k_rbf + 0.5 * k_poly
        else:
            return np.ones((X1.shape[0], X2.shape[0]))

    # Regularization lambda
    # We need to pick a good lambda. 
    # Common practice: small value like 1e-2 to 1e-1.
    # Let's try a few lambdas as well, but to save time, we'll fix it or search a small set.
    lambdas = [1e-2, 1e-1, 1.0]
    
    # To keep the search space manageable:
    # 6 sigmas * 1 + 3 degrees * 1 + 3 periods * 1 + 2 composites = 14 kernel types
    # 3 lambdas
    # Total 42 configurations.
    # For n=200, 42 * (200x200 matrix ops) is very fast.
    
    for k_type, k_param in kernels_to_try:
        if time.time() > deadline:
            break
            
        for lam in lambdas:
            if time.time() > deadline:
                break
                
            try:
                K = compute_kernel(X_norm, X_norm, k_type, k_param)
                # Add regularization
                K_reg = K + lam * np.eye(n)
                
                # Solve for weights
                # (K + lambda I) w = y
                # Using np.linalg.solve or lstsq
                w = np.linalg.solve(K_reg, y_train)
                
                # Calculate training error
                pred_train = K.dot(w)
                mse_train = np.mean((pred_train - y_train)**2)
                norm_mse_train = mse_train / y_var
                
                # We don't have validation set, so we use training error + complexity penalty?
                # Or just pick the one with lowest training error?
                # With regularization, lowest training error might overfit if lambda is too small.
                # However, KRR with fixed lambda is stable.
                # Let's use a heuristic: prefer lower training error, but penalize high complexity (small lambda).
                # Actually, let's just pick the best training error among reasonable lambdas.
                # If lambda is too large, error is high. If too small, it fits noise (but noise is 0).
                # Since noise is 0, the best fit is the one that interpolates best without numerical issues.
                # So lowest training MSE is a good proxy.
                
                if np.isfinite(norm_mse_train) and norm_mse_train < best_score:
                    best_score = norm_mse_train
                    # Store parameters for prediction
                    best_k_type = k_type
                    best_k_param = k_param
                    best_lam = lam
                    best_w = w
                    best_predict = None # Reset to force creation below if this is best
                    
                    # We need to capture the kernel type and params for the predict function
                    # We'll create the predict function after the loop using the best found params
                    # But we need to store the best params.
                    # Let's store a tuple of (k_type, k_param, lam, w)
                    # Actually, w depends on the kernel. So we store w.
                    # But we need to recompute K for new X in predict.
                    
            except np.linalg.LinAlgError:
                continue
            except Exception:
                continue

    # If no kernel worked (should not happen), fallback to mean
    if best_score == np.inf:
        mean_y = np.mean(y_train)
        return lambda X: np.full(X.shape[0], mean_y)

    # Define the predict function using the best found kernel and weights
    def predict(X):
        X_test = (X - X_mean) / X_std
        K_test = compute_kernel(X_test, X_norm, best_k_type, best_k_param)
        return K_test.dot(best_w)
        
    return predict

# EVOLVE-BLOCK-END
