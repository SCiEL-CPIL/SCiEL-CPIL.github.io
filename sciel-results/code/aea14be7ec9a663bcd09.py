import numpy as np
import time
from scipy.optimize import minimize_scalar
import itertools

def _train_nn(X_train, y_train, deadline):
    """Train a small neural network with Adam optimizer."""
    n, d = X_train.shape
    
    # Normalize inputs
    x_mean = X_train.mean(axis=0)
    x_std = X_train.std(axis=0)
    x_std[x_std < 1e-8] = 1.0
    X_norm = (X_train - x_mean) / x_std
    
    # Normalize outputs
    y_mean = y_train.mean()
    y_std = y_train.std()
    if y_std < 1e-8:
        y_std = 1.0
    y_norm = (y_train - y_mean) / y_std
    
    # Architecture: d -> 32 -> 32 -> 1
    hidden1 = 32
    hidden2 = 32
    
    # Initialize weights with He initialization
    scale1 = np.sqrt(2.0 / d)
    scale2 = np.sqrt(2.0 / hidden1)
    scale3 = np.sqrt(2.0 / hidden2)
    
    W1 = np.random.randn(d, hidden1) * scale1
    b1 = np.zeros(hidden1)
    W2 = np.random.randn(hidden1, hidden2) * scale2
    b2 = np.zeros(hidden2)
    W3 = np.random.randn(hidden2, 1) * scale3
    b3 = np.zeros(1)
    
    # Adam optimizer parameters
    lr = 0.01
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    
    # Adam state
    mW1 = np.zeros_like(W1); vW1 = np.zeros_like(W1)
    mb1 = np.zeros_like(b1); vb1 = np.zeros_like(b1)
    mW2 = np.zeros_like(W2); vW2 = np.zeros_like(W2)
    mb2 = np.zeros_like(b2); vb2 = np.zeros_like(b2)
    mW3 = np.zeros_like(W3); vW3 = np.zeros_like(W3)
    mb3 = np.zeros_like(b3); vb3 = np.zeros_like(b3)
    
    t = 0
    best_loss = np.inf
    best_params = (W1.copy(), b1.copy(), W2.copy(), b2.copy(), W3.copy(), b3.copy())
    
    # Training loop
    while time.time() < deadline - 10:
        t += 1
        # Forward pass
        z1 = X_norm @ W1 + b1
        a1 = np.tanh(z1)
        z2 = a1 @ W2 + b2
        a2 = np.tanh(z2)
        z3 = a2 @ W3 + b3
        pred = z3.flatten()
        
        # Loss (MSE)
        loss = np.mean((pred - y_norm) ** 2)
        
        # Backward pass
        dz3 = 2 * (pred - y_norm) / n
        dW3 = a2.T @ dz3
        db3 = np.sum(dz3, axis=0)
        
        da2 = dz3 @ W3.T
        dz2 = da2 * (1 - a2**2)
        dW2 = a1.T @ dz2
        db2 = np.sum(dz2, axis=0)
        
        da1 = dz2 @ W2.T
        dz1 = da1 * (1 - a1**2)
        dW1 = X_norm.T @ dz1
        db1 = np.sum(dz1, axis=0)
        
        # Adam update
        t += 1
        alpha_t = lr * np.sqrt(1 - beta2**t) / (1 - beta1**t)
        
        mW1 = beta1 * mW1 + (1 - beta1) * dW1
        vW1 = beta2 * vW1 + (1 - beta2) * dW1**2
        mW1_hat = mW1 / (1 - beta1**t)
        vW1_hat = vW1 / (1 - beta2**t)
        W1 -= alpha_t * mW1_hat / (np.sqrt(vW1_hat) + eps)
        
        mb1 = beta1 * mb1 + (1 - beta1) * db1
        vb1 = beta2 * vb1 + (1 - beta2) * db1**2
        mb1_hat = mb1 / (1 - beta1**t)
        vb1_hat = vb1 / (1 - beta2**t)
        b1 -= alpha_t * mb1_hat / (np.sqrt(vb1_hat) + eps)
        
        mW2 = beta1 * mW2 + (1 - beta1) * dW2
        vW2 = beta2 * vW2 + (1 - beta2) * dW2**2
        mW2_hat = mW2 / (1 - beta1**t)
        vW2_hat = vW2 / (1 - beta2**t)
        W2 -= alpha_t * mW2_hat / (np.sqrt(vW2_hat) + eps)
        
        mb2 = beta1 * mb2 + (1 - beta1) * db2
        vb2 = beta2 * vb2 + (1 - beta2) * db2**2
        mb2_hat = mb2 / (1 - beta1**t)
        vb2_hat = vb2 / (1 - beta2**t)
        b2 -= alpha_t * mb2_hat / (np.sqrt(vb2_hat) + eps)
        
        mW3 = beta1 * mW3 + (1 - beta1) * dW3
        vW3 = beta2 * vW3 + (1 - beta2) * dW3**2
        mW3_hat = mW3 / (1 - beta1**t)
        vW3_hat = vW3 / (1 - beta2**t)
        W3 -= alpha_t * mW3_hat / (np.sqrt(vW3_hat) + eps)
        
        mb3 = beta1 * mb3 + (1 - beta1) * db3
        vb3 = beta2 * vb3 + (1 - beta2) * db3**2
        mb3_hat = mb3 / (1 - beta1**t)
        vb3_hat = vb3 / (1 - beta2**t)
        b3 -= alpha_t * mb3_hat / (np.sqrt(vb3_hat) + eps)
        
        # Track best
        if loss < best_loss:
            best_loss = loss
            best_params = (W1.copy(), b1.copy(), W2.copy(), b2.copy(), W3.copy(), b3.copy())
    
    return best_params, x_mean, x_std, y_mean, y_std

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 300.0  # Leave 30s buffer

    X = np.array(X_train, dtype=float)
    y = np.array(y_train, dtype=float)
    n, d = X.shape

    # Normalize X to unit variance to make sigma selection more robust
    x_mean = X.mean(axis=0)
    x_std = X.std(axis=0)
    x_std[x_std < 1e-8] = 1.0
    X_norm = (X - x_mean) / x_std
    
    y_mean = y.mean()
    y_std = y.std()
    if y_std < 1e-8:
        y_std = 1.0
    y_norm = (y - y_mean) / y_std

    def _time_left():
        return deadline - time.time()

    # Strategy: Try neural network first if time allows, otherwise fall back to kernel ridge
    # The NN approach may handle complex functions better than kernel ridge for some equations
    
    # First, let's try a neural network approach
    # We'll use a small fraction of the time budget for NN training
    nn_deadline = start_time + 120.0  # Give NN 120s
    
    best_score = np.inf
    best_predict_fn = None
    
    # Try neural network
    try:
        nn_params, nn_x_mean, nn_x_std, nn_y_mean, nn_y_std = _train_nn(X, y, nn_deadline)
        
        W1, b1, W2, b2, W3, b3 = nn_params
        
        def nn_predict(X_test):
            X_test = np.array(X_test, dtype=float)
            if X_test.ndim == 1:
                X_test = X_test.reshape(1, -1)
            
            # Normalize test data
            X_test_norm = (X_test - nn_x_mean) / nn_x_std
            
            # Forward pass
            z1 = X_test_norm @ W1 + b1
            a1 = np.tanh(z1)
            z2 = a1 @ W2 + b2
            a2 = np.tanh(z2)
            z3 = a2 @ W3 + b3
            pred = z3.flatten()
            
            # Denormalize
            y_pred = pred * nn_y_std + nn_y_mean
            return y_pred
        
        # Evaluate NN on training data to get a score
        y_pred_train = nn_predict(X)
        nn_mse = np.mean((y_pred_train - y)**2) / (np.var(y) + 1e-8)
        
        if nn_mse < best_score:
            best_score = nn_mse
            best_predict_fn = nn_predict
            
    except Exception:
        pass
    
    # If we have time, also try kernel ridge regression
    if _time_left() > 30:
        # Generate a grid of sigma values
        sigmas = np.logspace(-1, 1, 15)
        lambdas = np.logspace(-4, 4, 15)
        
        def compute_loocv(sigma, lam):
            X_sq = np.sum(X_norm**2, axis=1)
            sq_dists = X_sq[:, None] + X_sq[None, :] - 2 * (X_norm @ X_norm.T)
            sq_dists = np.maximum(sq_dists, 0)
            
            K = np.exp(-sq_dists / (2 * sigma**2))
            K_reg = K + lam * np.eye(n)
            
            try:
                w = np.linalg.solve(K_reg, y_norm)
                y_pred = K @ w
                
                inv_K_reg = np.linalg.inv(K_reg)
                H_diag = np.einsum('ij,ji->i', K, inv_K_reg)
                
                denom = 1 - H_diag
                denom[denom < 1e-8] = 1e-8
                
                loocv_errors = (y_norm - y_pred) / denom
                mse = np.mean(loocv_errors**2)
                return mse
            except:
                return np.inf
        
        best_kernel_score = np.inf
        best_sigma = 1.0
        best_lambda = 1.0
        
        for sigma in sigmas:
            if _time_left() < 10:
                break
            for lam in lambdas:
                if _time_left() < 5:
                    break
                mse = compute_loocv(sigma, lam)
                if mse < best_kernel_score:
                    best_kernel_score = mse
                    best_sigma = sigma
                    best_lambda = lam
        
        # Compare with NN score and use the better one
        if best_kernel_score < best_score:
            # Use kernel ridge
            X_sq = np.sum(X_norm**2, axis=1)
            sq_dists = X_sq[:, None] + X_sq[None, :] - 2 * (X_norm @ X_norm.T)
            sq_dists = np.maximum(sq_dists, 0)
            K = np.exp(-sq_dists / (2 * best_sigma**2))
            K_reg = K + best_lambda * np.eye(n)
            
            try:
                weights = np.linalg.solve(K_reg, y_norm)
            except:
                weights = np.zeros(n)
            
            def kernel_predict(X_test):
                X_test = np.array(X_test, dtype=float)
                if X_test.ndim == 1:
                    X_test = X_test.reshape(1, -1)
                
                X_test_norm = (X_test - x_mean) / x_std
                X_test_sq = np.sum(X_test_norm**2, axis=1)
                X_train_sq = np.sum(X_norm**2, axis=1)
                sq_dists_test = X_test_sq[:, None] + X_train_sq[None, :] - 2 * (X_test_norm @ X_norm.T)
                sq_dists_test = np.maximum(sq_dists_test, 0)
                
                K_test = np.exp(-sq_dists_test / (2 * best_sigma**2))
                
                y_pred_norm = K_test @ weights
                y_pred = y_pred_norm * y_std + y_mean
                return y_pred
            
            best_predict_fn = kernel_predict
    
    # If no valid predict function was created, fall back to mean prediction
    if best_predict_fn is None:
        def fallback_predict(X_test):
            return np.full(len(X_test), y_mean)
        best_predict_fn = fallback_predict
    
    return best_predict_fn
