"""Rewritten program: ensemble of models for symbolic regression over 10 hidden equations."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _time_remaining(deadline, margin=5.0):
    return deadline - time.time() - margin


def _poly_features(X, degree):
    """Generate polynomial features up to given degree."""
    d = X.shape[1]
    n = X.shape[0]
    cols = [np.ones(n)]
    for deg in range(1, degree + 1):
        for i in range(d):
            for j in range(i, d):
                # Only include pairs where i <= j for degree 2, but for higher degrees
                # we need all combinations with replacement
                pass
    # Simpler: generate all monomials up to degree
    from itertools import combinations_with_replacement
    cols = [np.ones(n)]
    for deg in range(1, degree + 1):
        for combo in combinations_with_replacement(range(d), deg):
            feat = np.ones(n)
            for idx in combo:
                feat *= X[:, idx]
            cols.append(feat)
    return np.column_stack(cols)


def _fit_poly(X, y, degree):
    """Fit polynomial regression."""
    try:
        F = _poly_features(X, degree)
        w, *_ = np.linalg.lstsq(F, y, rcond=None)
        return lambda X: _poly_features(X, degree) @ w
    except Exception:
        return None


def _fit_rbf_kernel(X_train, y_train, gamma=None, epsilon=0.1):
    """Fit using RBF kernel ridge regression."""
    try:
        if gamma is None:
            # Estimate gamma from data scale
            d = X_train.shape[1]
            gamma = 1.0 / (d * np.var(X_train, axis=0).mean() + 1e-10)
        
        # RBF kernel matrix
        def rbf_kernel(A, B, gamma):
            A2 = np.sum(A ** 2, axis=1)
            B2 = np.sum(B ** 2, axis=1)
            dists = A2[:, None] + B2[None, :] - 2 * A @ B.T
            dists = np.maximum(dists, 0)
            return np.exp(-gamma * dists)
        
        K = rbf_kernel(X_train, X_train, gamma)
        n = len(X_train)
        alpha = np.linalg.solve(K + epsilon * np.eye(n), y_train)
        
        def predict(X):
            K_test = rbf_kernel(X, X_train, gamma)
            return K_test @ alpha
        return predict
    except Exception:
        return None


def _fit_neural_network(X_train, y_train, hidden_sizes=(64, 64), epochs=500, lr=0.01, seed=42):
    """Fit a simple neural network with manual implementation."""
    try:
        np.random.seed(seed)
        n, d = X_train.shape
        
        # Normalize inputs
        X_mean = X_train.mean(axis=0)
        X_std = X_train.std(axis=0) + 1e-8
        X_norm = (X_train - X_mean) / X_std
        
        y_mean = y_train.mean()
        y_std = y_train.std() + 1e-8
        y_norm = (y_train - y_mean) / y_std
        
        # Initialize weights
        layers = [d] + list(hidden_sizes) + [1]
        W = []
        b = []
        for i in range(len(layers) - 1):
            fan_in = layers[i]
            fan_out = layers[i + 1]
            W.append(np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in))
            b.append(np.zeros(fan_out))
        
        def forward(X):
            a = X
            activations = [a]
            for i in range(len(W)):
                z = a @ W[i] + b[i]
                if i < len(W) - 1:
                    a = np.tanh(z)
                else:
                    a = z  # linear output
                activations.append(a)
            return activations
        
        def compute_loss_and_grads(X, y):
            activations = forward(X)
            y_pred = activations[-1].flatten()
            loss = np.mean((y_pred - y) ** 2)
            
            # Backprop
            m = X.shape[0]
            grads_W = [np.zeros_like(W[i]) for i in range(len(W))]
            grads_b = [np.zeros_like(b[i]) for i in range(len(b))]
            
            delta = 2.0 * (y_pred - y) / m
            for i in range(len(W) - 1, -1, -1):
                grads_W[i] = activations[i].T @ delta
                grads_b[i] = np.sum(delta, axis=0)
                if i > 0:
                    delta = (delta @ W[i].T) * (1 - activations[i + 1] ** 2)
            
            return loss, grads_W, grads_b
        
        # Adam optimizer
        t = 0
        beta1, beta2 = 0.9, 0.999
        eps = 1e-8
        m_W = [np.zeros_like(W[i]) for i in range(len(W))]
        v_W = [np.zeros_like(W[i]) for i in range(len(W))]
        m_b = [np.zeros_like(b[i]) for i in range(len(b))]
        v_b = [np.zeros_like(b[i]) for i in range(len(b))]
        
        for epoch in range(epochs):
            loss, grads_W, grads_b = compute_loss_and_grads(X_norm, y_norm)
            t += 1
            for i in range(len(W)):
                m_W[i] = beta1 * m_W[i] + (1 - beta1) * grads_W[i]
                v_W[i] = beta2 * v_W[i] + (1 - beta2) * grads_W[i] ** 2
                m_hat = m_W[i] / (1 - beta1 ** t)
                v_hat = v_W[i] / (1 - beta2 ** t)
                W[i] -= lr * m_hat / (np.sqrt(v_hat) + eps)
                
                m_b[i] = beta1 * m_b[i] + (1 - beta1) * grads_b[i]
                v_b[i] = beta2 * v_b[i] + (1 - beta2) * grads_b[i] ** 2
                m_hat_b = m_b[i] / (1 - beta1 ** t)
                v_hat_b = v_b[i] / (1 - beta2 ** t)
                b[i] -= lr * m_hat_b / (np.sqrt(v_hat_b) + eps)
        
        def predict(X):
            X_norm_pred = (X - X_mean) / X_std
            activations = forward(X_norm_pred)
            return (activations[-1].flatten() * y_std) + y_mean
        
        return predict
    except Exception:
        return None


def _fit_gaussian_process(X_train, y_train, length_scale=1.0, noise=1e-6):
    """Fit a Gaussian Process regression."""
    try:
        d = X_train.shape[1]
        
        def rbf_kernel(A, B, ls):
            A2 = np.sum(A ** 2, axis=1)
            B2 = np.sum(B ** 2, axis=1)
            dists = A2[:, None] + B2[None, :] - 2 * A @ B.T
            dists = np.maximum(dists, 0)
            return np.exp(-0.5 * dists / (ls ** 2))
        
        K = rbf_kernel(X_train, X_train, length_scale) + noise * np.eye(len(X_train))
        L = np.linalg.cholesky(K)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
        
        def predict(X):
            K_s = rbf_kernel(X, X_train, length_scale)
            return K_s @ alpha
        return predict
    except Exception:
        return None


def _ensemble_predict(models, X):
    """Average predictions from valid models."""
    preds = []
    for model in models:
        if model is not None:
            try:
                p = model(X)
                if np.all(np.isfinite(p)):
                    preds.append(p)
            except Exception:
                pass
    if not preds:
        # Fallback: constant predictor
        return np.zeros(X.shape[0])
    return np.mean(preds, axis=0)


def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 320.0  # 320s limit with 10s margin
    
    # Try multiple models and ensemble them
    models = []
    
    # 1. Polynomial regression (degree 2)
    if _time_remaining(deadline) > 0:
        model = _fit_poly(X_train, y_train, degree=2)
        if model is not None:
            models.append(model)
    
    # 2. Polynomial regression (degree 3)
    if _time_remaining(deadline) > 0:
        model = _fit_poly(X_train, y_train, degree=3)
        if model is not None:
            models.append(model)
    
    # 3. RBF kernel ridge regression
    if _time_remaining(deadline) > 0:
        model = _fit_rbf_kernel(X_train, y_train, epsilon=0.1)
        if model is not None:
            models.append(model)
    
    # 4. Neural network
    if _time_remaining(deadline) > 30:
        model = _fit_neural_network(X_train, y_train, hidden_sizes=(64, 64), epochs=300, lr=0.01, seed=42)
        if model is not None:
            models.append(model)
    
    # 5. Gaussian Process
    if _time_remaining(deadline) > 0:
        model = _fit_gaussian_process(X_train, y_train, length_scale=1.0, noise=1e-6)
        if model is not None:
            models.append(model)
    
    # If no models worked, use a simple constant
    if not models:
        mean_y = np.mean(y_train)
        return lambda X: np.full(X.shape[0], mean_y)
    
    # Return ensemble predictor
    def predict(X):
        return _ensemble_predict(models, X)
    
    return predict
# EVOLVE-BLOCK-END
