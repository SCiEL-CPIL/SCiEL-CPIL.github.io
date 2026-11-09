# EVOLVE-BLOCK-START
import numpy as np


def fit(X_train, y_train):
    n, d = X_train.shape
    
    # Define frequency grid
    # Ranges are up to 10, so frequencies up to ~5-10 rad/unit are reasonable.
    # Using a log-spaced grid might be better to cover low and high freqs.
    # Let's try linear for now, similar to seed but optimized.
    freqs = np.linspace(0.1, 5.0, 20)
    n_freqs = len(freqs)
    
    # Precompute sin and cos for training data
    # Shape: (n, d, n_freqs)
    # X_train[:, j] is (n,)
    # freqs is (n_freqs,)
    # We need cos(freqs[k] * X_train[i, j])
    
    # Efficient way:
    # X_train_expanded: (n, d, 1)
    # freqs_expanded: (1, 1, n_freqs)
    # product: (n, d, n_freqs)
    
    X_exp = X_train[:, :, np.newaxis]
    F_exp = freqs[np.newaxis, np.newaxis, :]
    
    angles = X_exp * F_exp
    cos_train = np.cos(angles)
    sin_train = np.sin(angles)
    
    # Compute dimensional kernels K_dim[j] of shape (n, n)
    # K_dim[j][i, l] = sum_k (cos_train[i, j, k] * cos_train[l, j, k] + sin_train[i, j, k] * sin_train[l, j, k])
    
    K_dims = []
    for j in range(d):
        C = cos_train[:, j, :]  # (n, n_freqs)
        S = sin_train[:, j, :]  # (n, n_freqs)
        K_dim = C @ C.T + S @ S.T
        K_dims.append(K_dim)
    
    # Additive kernel
    K_add = np.zeros((n, n))
    for K_dim in K_dims:
        K_add += K_dim
        
    # Multiplicative kernel
    K_mul = np.ones((n, n))
    for K_dim in K_dims:
        K_mul *= K_dim
        
    # Combined kernel
    # Add a bias term (constant kernel)
    K = K_add + K_mul + np.ones((n, n))
    
    # Regularization
    # Choose alpha based on trace or fixed
    # Trace of K is sum of K[i, i]
    # K[i, i] = sum_j sum_f 1 + prod_j sum_f 1 + 1 = d*n_freqs + n_freqs^d + 1
    # For d=3, n_freqs=20: 60 + 8000 + 1 = 8061
    # alpha should be comparable to the scale of K
    # Let's use a small fraction of the mean diagonal
    diag_mean = np.mean(np.diag(K))
    alpha = 1e-3 * diag_mean
    
    K_reg = K + alpha * np.eye(n)
    
    # Solve
    try:
        w = np.linalg.solve(K_reg, y_train)
    except np.linalg.LinAlgError:
        # Fallback to least squares if singular
        w, _, _, _ = np.linalg.lstsq(K_reg, y_train, rcond=None)
        
    def predict(X):
        n_test = X.shape[0]
        
        # Precompute sin and cos for test data
        X_test_exp = X[:, :, np.newaxis]
        angles_test = X_test_exp * F_exp
        
        cos_test = np.cos(angles_test)
        sin_test = np.sin(angles_test)
        
        # Compute dimensional kernels for test vs train
        K_dims_test = []
        for j in range(d):
            C_test = cos_test[:, j, :]  # (n_test, n_freqs)
            S_test = sin_test[:, j, :]  # (n_test, n_freqs)
            
            C_train = cos_train[:, j, :]  # (n, n_freqs)
            S_train = sin_train[:, j, :]  # (n, n_freqs)
            
            K_dim_test = C_test @ C_train.T + S_test @ S_train.T
            K_dims_test.append(K_dim_test)
            
        K_add_test = np.zeros((n_test, n))
        for K_dim_test in K_dims_test:
            K_add_test += K_dim_test
            
        K_mul_test = np.ones((n_test, n))
        for K_dim_test in K_dims_test:
            K_mul_test *= K_dim_test
            
        K_test = K_add_test + K_mul_test + np.ones((n_test, n))
        
        return K_test @ w

    return predict
# EVOLVE-BLOCK-END
