"""
Strategy: Deep Feedforward Neural Network (MLP) with Adam Optimization.
Rationale: 
1. The previous strategies (splines, polynomial/Fourier bases) struggle with 
   arbitrary functional forms that might be highly non-linear, oscillatory, 
   or have complex interactions (though here d=1, so interactions are trivial).
2. A small MLP (e.g., 2-3 hidden layers, 64-128 units) is a universal approximator.
3. With 200 noiseless points, an MLP can interpolate the training data almost perfectly.
4. Since the test points are separate but drawn from the same distribution, 
   and the functions are likely smooth, an MLP that fits the training data 
   well will also generalize well to the test points, provided it doesn't 
   overfit to noise (there is none) or oscillate wildly between points.
5. We use Early Stopping based on a small holdout set from the training data 
   to ensure we don't train too long, which could lead to numerical instability 
   or overfitting if the network capacity is too high.
6. This is structurally different from basis expansion (which is linear in parameters) 
   and splines (which are local). It is a global, non-linear, iterative optimization.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 290  # Leave 10s buffer
    
    X = X_train[:, 0].reshape(-1, 1)
    y = y_train
    
    n = len(X)
    
    # Normalize input to [0, 1] to help with gradient flow
    x_min, x_max = X.min(), X.max()
    if x_max == x_min:
        x_max = x_min + 1e-6
    X_norm = (X - x_min) / (x_max - x_min)
    
    # Normalize output for better training stability
    y_mean, y_std = np.mean(y), np.std(y)
    if y_std < 1e-8:
        y_std = 1.0
    y_norm = (y - y_mean) / y_std
    
    # Define a simple MLP architecture
    # Input: 1
    # Hidden 1: 32 units, tanh
    # Hidden 2: 32 units, tanh
    # Output: 1, linear
    # This is small enough to train fast, large enough to fit smooth 1D functions.
    
    def init_weights(fan_in, fan_out):
        # He initialization for tanh is okay, or Xavier
        scale = np.sqrt(2.0 / fan_in)
        return np.random.randn(fan_in, fan_out) * scale
    
    # Parameters: W1, b1, W2, b2, W3, b3
    W1 = init_weights(1, 32)
    b1 = np.zeros((1, 32))
    W2 = init_weights(32, 32)
    b2 = np.zeros((1, 32))
    W3 = init_weights(32, 1)
    b3 = np.zeros((1, 1))
    
    params = [W1, b1, W2, b2, W3, b3]
    
    # Adam Optimizer state
    m_params = [np.zeros_like(p) for p in params]
    v_params = [np.zeros_like(p) for p in params]
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0
    lr = 1e-2
    
    def forward(X_in):
        h1 = np.tanh(X_in @ W1 + b1)
        h2 = np.tanh(h1 @ W2 + b2)
        out = h2 @ W3 + b3
        return out, h1, h2
    
    def train_epoch():
        nonlocal t, lr
        t += 1
        lr_t = lr * (0.99 ** t) # Simple decay
        
        # Forward pass
        pred, h1, h2 = forward(X_norm)
        
        # Loss: MSE
        loss = np.mean((pred - y_norm.reshape(-1, 1)) ** 2)
        
        # Backward pass
        # dL/dout
        d_out = 2 * (pred - y_norm.reshape(-1, 1)) / n
        
        # dL/dW3, db3
        d_W3 = h2.T @ d_out
        d_b3 = np.sum(d_out, axis=0, keepdims=True)
        
        # dL/dh2
        d_h2 = d_out @ W3.T
        # tanh derivative
        d_h2 = d_h2 * (1 - h2 ** 2)
        
        # dL/dW2, db2
        d_W2 = h1.T @ d_h2
        d_b2 = np.sum(d_h2, axis=0, keepdims=True)
        
        # dL/dh1
        d_h1 = d_h2 @ W2.T
        # tanh derivative
        d_h1 = d_h1 * (1 - h1 ** 2)
        
        # dL/dW1, db1
        d_W1 = X_norm.T @ d_h1
        d_b1 = np.sum(d_h1, axis=0, keepdims=True)
        
        grads = [d_W1, d_b1, d_W2, d_b2, d_W3, d_b3]
        
        # Adam update
        for i, p in enumerate(params):
            m_params[i] = beta1 * m_params[i] + (1 - beta1) * grads[i]
            v_params[i] = beta2 * v_params[i] + (1 - beta2) * (grads[i] ** 2)
            m_hat = m_params[i] / (1 - beta1 ** t)
            v_hat = v_params[i] / (1 - beta2 ** t)
            params[i] -= lr_t * m_hat / (np.sqrt(v_hat) + eps)
            
        return loss
    
    # Early stopping setup
    # Use last 10% of data as validation
    val_size = int(0.1 * n)
    val_idx = np.random.choice(n, val_size, replace=False)
    train_idx = np.setdiff1d(np.arange(n), val_idx)
    
    X_tr = X_norm[train_idx]
    y_tr = y_norm[train_idx]
    X_val = X_norm[val_idx]
    y_val = y_norm[val_idx]
    
    best_val_loss = np.inf
    best_params = [p.copy() for p in params]
    patience = 50
    patience_counter = 0
    max_epochs = 2000
    
    for epoch in range(max_epochs):
        if time.time() > deadline:
            break
            
        # Train on training split
        # Note: We are using the full batch here. With 180 points, this is fine.
        # To speed up, we could use mini-batches, but full batch is stable.
        
        # Temporarily swap data for training
        # Actually, let's just train on ALL data for simplicity and speed, 
        # but monitor val loss. 
        # Wait, if we train on all data, val loss will drop along with train loss.
        # Better to train on train_idx and evaluate on val_idx.
        
        # Forward pass on train data
        pred_tr, h1_tr, h2_tr = forward(X_tr)
        loss_tr = np.mean((pred_tr - y_tr.reshape(-1, 1)) ** 2)
        
        # Backward pass on train data
        d_out = 2 * (pred_tr - y_tr.reshape(-1, 1)) / len(X_tr)
        d_W3 = h2_tr.T @ d_out
        d_b3 = np.sum(d_out, axis=0, keepdims=True)
        d_h2 = d_out @ W3.T
        d_h2 = d_h2 * (1 - h2_tr ** 2)
        d_W2 = h1_tr.T @ d_h2
        d_b2 = np.sum(d_h2, axis=0, keepdims=True)
        d_h1 = d_h2 @ W2.T
        d_h1 = d_h1 * (1 - h1_tr ** 2)
        d_W1 = X_tr.T @ d_h1
        d_b1 = np.sum(d_h1, axis=0, keepdims=True)
        
        grads = [d_W1, d_b1, d_W2, d_b2, d_W3, d_b3]
        
        t += 1
        lr_t = lr * (0.995 ** t)
        
        for i, p in enumerate(params):
            m_params[i] = beta1 * m_params[i] + (1 - beta1) * grads[i]
            v_params[i] = beta2 * v_params[i] + (1 - beta2) * (grads[i] ** 2)
            m_hat = m_params[i] / (1 - beta1 ** t)
            v_hat = v_params[i] / (1 - beta2 ** t)
            params[i] -= lr_t * m_hat / (np.sqrt(v_hat) + eps)
            
        # Evaluate on validation data
        if epoch % 10 == 0:
            pred_val, _, _ = forward(X_val)
            val_loss = np.mean((pred_val - y_val.reshape(-1, 1)) ** 2)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_params = [p.copy() for p in params]
                patience_counter = 0
            else:
                patience_counter += 10
                if patience_counter > patience:
                    break
    
    # Restore best parameters
    W1, b1, W2, b2, W3, b3 = best_params
    
    def predict(X):
        X_flat = X.flatten().reshape(-1, 1)
        X_norm_pred = (X_flat - x_min) / (x_max - x_min)
        
        h1 = np.tanh(X_norm_pred @ W1 + b1)
        h2 = np.tanh(h1 @ W2 + b2)
        out = h2 @ W3 + b3
        
        # Denormalize
        pred = out.flatten() * y_std + y_mean
        
        return pred.reshape(X.shape)
    
    return predict
# EVOLVE-BLOCK-END
