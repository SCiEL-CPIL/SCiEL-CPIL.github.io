# EVOLVE-BLOCK-START
import numpy as np
from scipy.linalg import solve

def fit(X_train, y_train):
    n, d = X_train.shape
    
    # Number of frequency components
    M = 30
    # Frequencies: geometric spacing from low to high
    # Max frequency ~ 20 rad/unit covers variations on scale 0.3 units
    freqs = np.geomspace(0.5, 20.0, M)
    
    # Compute the full kernel matrix K of shape (n, n)
    # K(x, x') = prod_j sum_f cos(f * (x_j - x'_j))
    # We normalize by M^d so that K(x, x) = 1
    
    K = np.ones((n, n))
    
    for dim in range(d):
        x_dim = X_train[:, dim]
        # Compute pairwise differences
        diff = x_dim[:, None] - x_dim[None, :]
        
        # Compute sum of cosines
        # angles shape (M, n, n)
        angles = diff[None, :, :] * freqs[:, None, None]
        k_dim = np.sum(np.cos(angles), axis=0)
        
        # Multiply into total kernel
        K *= k_dim
        
    # Normalize kernel so diagonal is 1
    # Diagonal is sum_f cos(0) = M for each dim, so product is M^d
    K /= (M ** d)
    
    # Regularization parameter
    # With normalized kernel, alpha controls the trade-off.
    # For noiseless data, small alpha is good, but we need stability.
    alpha = 1e-3
    
    I = np.eye(n)
    A = K + alpha * I
    
    # Solve for weights
    try:
        weights = solve(A, y_train, assume_a='pos')
    except np.linalg.LinAlgError:
        # Fallback
        weights = np.linalg.lstsq(A, y_train, rcond=None)[0]
        
    def predict(X):
        n_test = X.shape[0]
        
        # Compute test kernel matrix K_test of shape (n_test, n_train)
        K_test = np.ones((n_test, n))
        
        for dim in range(d):
            x_train_dim = X_train[:, dim]
            x_test_dim = X[:, dim]
            
            # diff shape (n_test, n_train)
            diff = x_test_dim[:, None] - x_train_dim[None, :]
            
            # angles shape (M, n_test, n_train)
            angles = diff[None, :, :] * freqs[:, None, None]
            
            # Sum of cosines
            k_dim = np.sum(np.cos(angles), axis=0)
            
            # Multiply into total kernel
            K_test *= k_dim
            
        # Normalize
        K_test /= (M ** d)
        
        return K_test @ weights
        
    return predict
# EVOLVE-BLOCK-END
