# EVOLVE-BLOCK-START
import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = _START_TIME + 310  # safety margin before 330s kill


def _time_left():
    return _DEADLINE - time.time()


class _SmallNet:
    """Small fully-connected neural network trained with manual Adam."""
    
    def __init__(self, d_in, d_hidden=32, d_out=1):
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.d_out = d_out
        
        # Xavier initialization
        scale1 = np.sqrt(2.0 / (d_in + d_hidden))
        scale2 = np.sqrt(2.0 / (d_hidden + d_out))
        
        self.W1 = np.random.randn(d_in, d_hidden) * scale1
        self.b1 = np.zeros(d_hidden)
        self.W2 = np.random.randn(d_hidden, d_out) * scale2
        self.b2 = np.zeros(d_out)
        
        # Adam state
        self.mW1 = np.zeros_like(self.W1)
        self.vW1 = np.zeros_like(self.W1)
        self.mb1 = np.zeros_like(self.b1)
        self.vb1 = np.zeros_like(self.b1)
        self.mW2 = np.zeros_like(self.W2)
        self.vW2 = np.zeros_like(self.W2)
        self.mb2 = np.zeros_like(self.b2)
        self.vb2 = np.zeros_like(self.b2)
        self.t = 0
        
    def _tanh(self, x):
        return np.tanh(x)
    
    def _tanh_deriv(self, x):
        return 1.0 - x ** 2
    
    def forward(self, X):
        """Forward pass, returns (output, cache)."""
        h1 = self._tanh(X @ self.W1 + self.b1)
        out = h1 @ self.W2 + self.b2
        cache = (X, h1)
        return out, cache
    
    def backward(self, out, y, cache):
        """Compute gradients for MSE loss."""
        X, h1 = cache
        N = X.shape[0]
        
        # dL/dout = 2 * (out - y) / N
        dout = 2.0 * (out - y) / N
        
        # Backprop to layer 2
        dW2 = h1.T @ dout
        db2 = np.sum(dout, axis=0)
        
        # Backprop to layer 1
        dh1 = dout @ self.W2.T
        dz1 = dh1 * self._tanh_deriv(h1)
        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0)
        
        return dW1, db1, dW2, db2
    
    def adam_step(self, dW1, db1, dW2, db2, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        self.t += 1
        b1c = 1 - beta1 ** self.t
        b2c = 1 - beta2 ** self.t
        
        # W1
        self.mW1 = beta1 * self.mW1 + (1 - beta1) * dW1
        self.vW1 = beta2 * self.vW1 + (1 - beta2) * dW1 ** 2
        self.W1 -= lr * (self.mW1 / b1c) / (np.sqrt(self.vW1 / b2c) + eps)
        
        # b1
        self.mb1 = beta1 * self.mb1 + (1 - beta1) * db1
        self.vb1 = beta2 * self.vb1 + (1 - beta2) * db1 ** 2
        self.b1 -= lr * (self.mb1 / b1c) / (np.sqrt(self.vb1 / b2c) + eps)
        
        # W2
        self.mW2 = beta1 * self.mW2 + (1 - beta1) * dW2
        self.vW2 = beta2 * self.vW2 + (1 - beta2) * dW2 ** 2
        self.W2 -= lr * (self.mW2 / b1c) / (np.sqrt(self.vW2 / b2c) + eps)
        
        # b2
        self.mb2 = beta1 * self.mb2 + (1 - beta1) * db2
        self.vb2 = beta2 * self.vb2 + (1 - beta2) * db2 ** 2
        self.b2 -= lr * (self.mb2 / b1c) / (np.sqrt(self.vb2 / b2c) + eps)
    
    def predict(self, X):
        out, _ = self.forward(X)
        return out.flatten()


def _normalize(X, y):
    """Normalize inputs and output for better training."""
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_std[X_std < 1e-10] = 1.0
    X_norm = (X - X_mean) / X_std
    
    y_mean = np.mean(y)
    y_std = np.std(y)
    if y_std < 1e-10:
        y_std = 1.0
    y_norm = (y - y_mean) / y_std
    
    return X_norm, y_norm, X_mean, X_std, y_mean, y_std


def _denormalize(y_norm, y_mean, y_std):
    """Denormalize predictions."""
    return y_norm * y_std + y_mean


def fit(X_train, y_train):
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    N, d = X_train.shape
    
    # Handle edge cases
    if N == 0:
        return lambda X: np.zeros(len(X))
    
    # Normalize
    X_norm, y_norm, X_mean, X_std, y_mean, y_std = _normalize(X_train, y_train)
    
    # Train multiple networks with different random seeds and keep the best
    best_net = None
    best_train_loss = np.inf
    best_final_pred = None
    
    # Time budget per network
    time_per_net = min(25.0, max(5.0, _time_left() / 4.0))
    max_nets = min(4, int(_time_left() / 10.0))
    max_nets = max(1, max_nets)
    
    for seed in range(max_nets):
        if _time_left() < 5.0:
            break
        
        np.random.seed(42 + seed)
        net = _SmallNet(d, d_hidden=32, d_out=1)
        
        # Training loop
        lr = 0.01
        batch_size = min(64, N)
        n_epochs = 500
        start_time = time.time()
        
        for epoch in range(n_epochs):
            if _time_left() < 10.0:
                break
            if (time.time() - start_time) > time_per_net:
                break
            
            # Mini-batch
            if batch_size < N:
                idx = np.random.permutation(N)[:batch_size]
                Xb = X_norm[idx]
                yb = y_norm[idx]
            else:
                Xb = X_norm
                yb = y_norm
            
            out, cache = net.forward(Xb)
            dW1, db1, dW2, db2 = net.backward(out, yb.reshape(-1, 1), cache)
            net.adam_step(dW1, db1, dW2, db2, lr=lr)
            
            # Adaptive learning rate
            if epoch > 100 and epoch % 100 == 0:
                lr *= 0.5
        
        # Evaluate on full training set
        train_pred_norm, _ = net.forward(X_norm)
        train_pred = _denormalize(train_pred_norm.flatten(), y_mean, y_std)
        train_mse = np.mean((train_pred - y_train) ** 2)
        y_var = np.var(y_train)
        if y_var > 0:
            norm_mse = train_mse / y_var
        else:
            norm_mse = train_mse
        
        if train_mse < best_train_loss:
            best_train_loss = train_mse
            best_net = net
            best_final_pred = train_pred
    
    # If no network was trained (shouldn't happen), use mean prediction
    if best_net is None:
        return lambda X: np.full(len(X), np.mean(y_train))
    
    # Final refinement: a few more epochs with lower learning rate
    if _time_left() > 15.0:
        lr = 0.001
        for _ in range(200):
            if _time_left() < 10.0:
                break
            idx = np.random.permutation(N)
            Xb = X_norm[idx]
            yb = y_norm[idx]
            out, cache = best_net.forward(Xb)
            dW1, db1, dW2, db2 = best_net.backward(out, yb.reshape(-1, 1), cache)
            best_net.adam_step(dW1, db1, dW2, db2, lr=lr)
    
    # Capture the network and normalization params
    net = best_net
    x_mean = X_mean
    x_std = X_std
    y_mean_val = y_mean
    y_std_val = y_std
    
    def predict(X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X_norm_pred = (X - x_mean) / x_std
        out, _ = net.forward(X_norm_pred)
        return _denormalize(out.flatten(), y_mean_val, y_std_val)
    
    return predict
# EVOLVE-BLOCK-END
