import numpy as np

def fit(X_train, y_train):
    X = X_train
    y = y_train
    
    # Normalize X and y
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_std[X_std == 0] = 1.0
    X_norm = (X - X_mean) / X_std
    
    y_mean = np.mean(y)
    y_std = np.std(y)
    y_std[y_std == 0] = 1.0
    y_norm = (y - y_mean) / y_std

    n_samples, n_features = X_norm.shape
    
    # Neural Network Parameters
    n_hidden = 20
    lr = 0.01
    epochs = 1500
    batch_size = 32
    l2_reg = 1e-4

    # Use a local RNG for reproducibility
    rng = np.random.default_rng(42)
    
    # Initialize weights
    # He initialization for ReLU
    W1 = rng.normal(0, np.sqrt(2.0 / n_features), (n_features, n_hidden))
    b1 = rng.normal(0, 0.1, (1, n_hidden))
    W2 = rng.normal(0, np.sqrt(2.0 / n_hidden), (n_hidden, 1))
    b2 = rng.normal(0, 0.1, (1, 1))
    
    # Adam optimizer states
    m_W1 = np.zeros_like(W1); v_W1 = np.zeros_like(W1)
    m_b1 = np.zeros_like(b1); v_b1 = np.zeros_like(b1)
    m_W2 = np.zeros_like(W2); v_W2 = np.zeros_like(W2)
    m_b2 = np.zeros_like(b2); v_b2 = np.zeros_like(b2)
    
    t = 0
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    
    # Training loop
    for epoch in range(epochs):
        # Shuffle
        indices = np.arange(n_samples)
        rng.shuffle(indices)
        
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_idx = indices[start:end]
            X_batch = X_norm[batch_idx]
            y_batch = y_norm[batch_idx]
            
            # Forward pass
            z1 = X_batch @ W1 + b1
            a1 = np.maximum(0, z1) # ReLU
            z2 = a1 @ W2 + b2
            a2 = z2 # Linear output
            
            # Loss and Gradients
            error = a2 - y_batch
            dL_da2 = error / len(X_batch)
            
            dz2 = dL_da2
            dW2 = a1.T @ dz2 + l2_reg * W2
            db2 = np.sum(dz2, axis=0, keepdims=True)
            
            da1 = dz2 @ W2.T
            dz1 = da1 * (z1 > 0) # ReLU derivative
            
            dW1 = X_batch.T @ dz1 + l2_reg * W1
            db1 = np.sum(dz1, axis=0, keepdims=True)
            
            # Adam update
            t += 1
            m_W1 = beta1 * m_W1 + (1 - beta1) * dW1
            v_W1 = beta2 * v_W1 + (1 - beta2) * (dW1 ** 2)
            m_W1_hat = m_W1 / (1 - beta1 ** t)
            v_W1_hat = v_W1 / (1 - beta2 ** t)
            W1 -= lr * m_W1_hat / (np.sqrt(v_W1_hat) + epsilon)
            
            m_b1 = beta1 * m_b1 + (1 - beta1) * db1
            v_b1 = beta2 * v_b1 + (1 - beta2) * (db1 ** 2)
            m_b1_hat = m_b1 / (1 - beta1 ** t)
            v_b1_hat = v_b1 / (1 - beta2 ** t)
            b1 -= lr * m_b1_hat / (np.sqrt(v_b1_hat) + epsilon)
            
            m_W2 = beta1 * m_W2 + (1 - beta1) * dW2
            v_W2 = beta2 * v_W2 + (1 - beta2) * (dW2 ** 2)
            m_W2_hat = m_W2 / (1 - beta1 ** t)
            v_W2_hat = v_W2 / (1 - beta2 ** t)
            W2 -= lr * m_W2_hat / (np.sqrt(v_W2_hat) + epsilon)
            
            m_b2 = beta1 * m_b2 + (1 - beta1) * db2
            v_b2 = beta2 * v_b2 + (1 - beta2) * (db2 ** 2)
            m_b2_hat = m_b2 / (1 - beta1 ** t)
            v_b2_hat = v_b2 / (1 - beta2 ** t)
            b2 -= lr * m_b2_hat / (np.sqrt(v_b2_hat) + epsilon)

    def predict(X):
        X_test = (X - X_mean) / X_std
        z1 = X_test @ W1 + b1
        a1 = np.maximum(0, z1)
        z2 = a1 @ W2 + b2
        return (z2.flatten() * y_std) + y_mean

    return predict
