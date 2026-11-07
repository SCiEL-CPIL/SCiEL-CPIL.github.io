"""Symbolic regression over 10 hidden equations: Neural net + comprehensive features + kernel ensemble."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import scipy.linalg

_START = time.time()
_DEADLINE = 300.0

def _remaining():
    return _DEADLINE - (time.time() - _START)

# --- Neural Network (manual implementation) ---

def _relu(x):
    return np.maximum(0, x)

def _relu_grad(x):
    return (x > 0).astype(float)

def _sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))

def _sigmoid_grad(x):
    s = _sigmoid(x)
    return s * (1 - s)

class _SmallNN:
    """Small fully-connected neural network with manual backprop + Adam."""
    
    def __init__(self, input_dim, hidden_dims, output_dim=1, seed=42):
        np.random.seed(seed)
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            W = np.random.randn(prev_dim, h) * np.sqrt(2.0 / prev_dim)
            b = np.zeros(h)
            self.layers.append((W, b))
            prev_dim = h
        W_out = np.random.randn(prev_dim, output_dim) * np.sqrt(2.0 / prev_dim)
        b_out = np.zeros(output_dim)
        self.layers.append((W_out, b_out))
        
        # Adam state
        self.m_W = [np.zeros_like(W) for W, b in self.layers]
        self.v_W = [np.zeros_like(W) for W, b in self.layers]
        self.m_b = [np.zeros_like(b) for W, b in self.layers]
        self.v_b = [np.zeros_like(b) for W, b in self.layers]
        self.t = 0
        
    def forward(self, X):
        """Forward pass, cache activations for backprop."""
        self.cached = []
        a = X
        for i, (W, b) in enumerate(self.layers):
            z = a @ W + b
            if i < len(self.layers) - 1:
                a_out = _relu(z)
            else:
                a_out = z  # linear output
            self.cached.append((a, z, a_out))
            a = a_out
        return a
    
    def predict(self, X):
        """Predict without caching (for inference)."""
        a = X
        for i, (W, b) in enumerate(self.layers):
            z = a @ W + b
            if i < len(self.layers) - 1:
                a = _relu(z)
            else:
                a = z
        return a.flatten()
    
    def train_step(self, X, y, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        """One full-batch gradient descent step with Adam."""
        self.t += 1
        n = X.shape[0]
        
        # Forward
        self.forward(X)
        pred = self.cached[-1][2]
        
        # Backward
        d_out = 2.0 * (pred - y) / n  # dL/doutput
        
        grads_W = [None] * len(self.layers)
        grads_b = [None] * len(self.layers)
        
        d_a = d_out
        for i in range(len(self.layers) - 1, -1, -1):
            a_prev, z, a_out = self.cached[i]
            W, b = self.layers[i]
            
            grads_W[i] = a_prev.T @ d_a
            grads_b[i] = np.sum(d_a, axis=0)
            
            if i > 0:
                if i - 1 < len(self.layers) - 1:
                    d_z = d_a * _relu_grad(z)
                else:
                    d_z = d_a
                d_a = d_z @ W.T
        
        # Adam update
        for i in range(len(self.layers)):
            W, b = self.layers[i]
            gW = grads_W[i]
            gb = grads_b[i]
            
            self.m_W[i] = beta1 * self.m_W[i] + (1 - beta1) * gW
            self.v_W[i] = beta2 * self.v_W[i] + (1 - beta2) * gW**2
            m_hat = self.m_W[i] / (1 - beta1**self.t)
            v_hat = self.v_W[i] / (1 - beta2**self.t)
            self.layers[i] = (W - lr * m_hat / (np.sqrt(v_hat) + eps), b)
            
            self.m_b[i] = beta1 * self.m_b[i] + (1 - beta1) * gb
            self.v_b[i] = beta2 * self.v_b[i] + (1 - beta2) * gb**2
            m_hat_b = self.m_b[i] / (1 - beta1**self.t)
            v_hat_b = self.v_b[i] / (1 - beta2**self.t)
            self.layers[i] = (self.layers[i][0], b - lr * m_hat_b / (np.sqrt(v_hat_b) + eps))
    
    def _make_predict(self):
        def predict(X):
            X = np.asarray(X, dtype=float)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            return self.predict(X)
        return predict

def _train_nn(X, y, hidden_dims=(32, 32), n_epochs=500, lr=0.01, seed=42, time_budget=20.0):
    """Train a small NN and return predict function."""
    d = X.shape[1]
    nn = _SmallNN(d, hidden_dims, output_dim=1, seed=seed)
    
    start = time.time()
    best_loss = float('inf')
    best_layers = [(W.copy(), b.copy()) for W, b in nn.layers]
    
    for epoch in range(n_epochs):
        if time.time() - start > time_budget:
            break
        nn.train_step(X, y, lr=lr)
        pred = nn.predict(X)
        loss = np.mean((pred - y)**2)
        if loss < best_loss:
            best_loss = loss
            best_layers = [(W.copy(), b.copy()) for W, b in nn.layers]
        
        # Learning rate decay
        if epoch % 100 == 0 and epoch > 0:
            lr *= 0.5
    
    # Restore best layers
    nn.layers = best_layers
    return nn._make_predict(), best_loss

# --- Comprehensive Feature Builder ---

def _safe_div(a, b, eps=1e-8):
    return a / np.where(np.abs(b) < eps, np.sign(b) * eps + (b == 0) * eps, b)

def _build_features(X):
    """Build a comprehensive feature matrix."""
    n, d = X.shape
    cols = [np.ones(n)]
    
    # Linear
    for j in range(d):
        cols.append(X[:, j])
    
    # Squares, cubes
    for j in range(d):
        cols.append(X[:, j] ** 2)
        cols.append(X[:, j] ** 3)
    
    # Pairwise products
    for i in range(d):
        for j in range(i + 1, d):
            cols.append(X[:, i] * X[:, j])
    
    # Pairwise sums and differences
    for i in range(d):
        for j in range(i + 1, d):
            cols.append(X[:, i] + X[:, j])
            cols.append(X[:, i] - X[:, j])
    
    # Trig on individual inputs
    for j in range(d):
        cols.append(np.sin(X[:, j]))
        cols.append(np.cos(X[:, j]))
        cols.append(np.sin(2 * X[:, j]))
        cols.append(np.cos(2 * X[:, j]))
    
    # Trig on pairwise sums/diffs
    for i in range(d):
        for j in range(i + 1, d):
            s = X[:, i] + X[:, j]
            diff = X[:, i] - X[:, j]
            cols.append(np.sin(s))
            cols.append(np.cos(s))
            cols.append(np.sin(diff))
            cols.append(np.cos(diff))
    
    # Exponentials (clipped)
    for j in range(d):
        x_clip = np.clip(X[:, j], -50, 50)
        cols.append(np.exp(x_clip))
        cols.append(np.exp(-x_clip))
    
    # Tanh, sigmoid
    for j in range(d):
        cols.append(np.tanh(X[:, j]))
        x_clip = np.clip(X[:, j], -500, 500)
        cols.append(1.0 / (1.0 + np.exp(-x_clip)))
    
    # Log, sqrt
    for j in range(d):
        x_pos = np.maximum(X[:, j], 1e-8)
        cols.append(np.log(x_pos))
        cols.append(np.sqrt(x_pos))
    
    # Rational functions
    for i in range(d):
        for j in range(i + 1, d):
            cols.append(_safe_div(X[:, i], X[:, j]))
            cols.append(_safe_div(X[:, j], X[:, i]))
    for j in range(d):
        cols.append(_safe_div(1.0, X[:, j]))
    
    # Product of all inputs
    if d >= 1:
        p = np.ones(n)
        for j in range(d):
            p = p * X[:, j]
        cols.append(p)
    
    # Sum, norm
    s = np.sum(X, axis=1)
    cols.append(s)
    cols.append(s ** 2)
    cols.append(np.sin(s))
    cols.append(np.cos(s))
    cols.append(np.tanh(s))
    norm = np.sqrt(np.sum(X ** 2, axis=1))
    cols.append(norm)
    cols.append(np.sin(norm))
    cols.append(np.cos(norm))
    
    A = np.column_stack(cols)
    mask = np.isfinite(A)
    A_clean = np.where(mask, A, 0.0)
    col_valid = mask.all(axis=0)
    A_clean = A_clean[:, col_valid]
    return A_clean

# --- Ridge Regression ---

def _ridge_fit(A, y, alpha):
    n, p = A.shape
    try:
        AtA = A.T @ A
        Aty = A.T @ y
        reg = (alpha + 1e-10) * np.eye(p)
        w = np.linalg.solve(AtA + reg, Aty)
        pred = A @ w
        if not np.all(np.isfinite(pred)):
            return None
        err = np.mean((pred - y) ** 2)
        return w, err
    except Exception:
        return None

def _select_alpha(A, y, alphas=None):
    if alphas is None:
        alphas = [1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0]
    best = None
    for alpha in alphas:
        res = _ridge_fit(A, y, alpha)
        if res is not None:
            w, err = res
            if best is None or err < best[1]:
                best = (w, err, alpha)
    return best

# --- RBF Kernel ---

def _make_predict_rbf(X_train, y_train, gamma, alpha):
    diff = X_train[:, np.newaxis, :] - X_train[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=-1)
    K = np.exp(-gamma * sq_dists)
    K_reg = K + alpha * np.eye(len(X_train))
    try:
        w = scipy.linalg.solve(K_reg, y_train)
    except Exception:
        w = np.linalg.pinv(K_reg) @ y_train
    
    def predict(X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        diff_test = X[:, np.newaxis, :] - X_train[np.newaxis, :, :]
        sq_dists_test = np.sum(diff_test**2, axis=-1)
        K_test = np.exp(-gamma * sq_dists_test)
        return K_test @ w
    return predict

def _cv_rbf_score(X, y, gamma, alpha, n_folds=5):
    n = len(y)
    indices = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(indices)
    fold_size = n // n_folds
    cv_errors = []
    for fold in range(n_folds):
        val_start = fold * fold_size
        val_end = val_start + fold_size if fold < n_folds - 1 else n
        val_idx = indices[val_start:val_end]
        train_idx = np.concatenate([indices[:val_start], indices[val_end:]])
        if len(train_idx) < 10:
            continue
        try:
            pred = _make_predict_rbf(X[train_idx], y[train_idx], gamma, alpha)(X[val_idx])
            err = np.mean((pred - y[val_idx])**2)
            if np.isfinite(err):
                cv_errors.append(err)
        except Exception:
            continue
    if cv_errors:
        return np.mean(cv_errors)
    return float('inf')

# --- Exact Fit with Simple Bases ---

def _basis_simple(X):
    d = X.shape[1]
    cols = [np.ones(len(X))]
    for j in range(d):
        cols.append(X[:, j])
        cols.append(X[:, j] ** 2)
    for i in range(d):
        for j in range(i, d):
            cols.append(X[:, i] * X[:, j])
    return np.column_stack(cols)

def _try_exact_fit(basis_func, X, y, tol=1e-6):
    try:
        A = basis_func(X)
        if not np.all(np.isfinite(A)):
            return None
        w, residuals, rank, s = np.linalg.lstsq(A, y, rcond=None)
        pred = A @ w
        if not np.all(np.isfinite(pred)):
            return None
        mse = np.mean((pred - y)**2)
        scale = max(1.0, np.max(np.abs(y)))
        if mse <= (tol * scale)**2:
            return w, mse
    except Exception:
        pass
    return None

def _make_predict_linear(basis_func, w):
    def predict(X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        A = basis_func(X)
        return A @ w
    return predict

def fit(X_train, y_train):
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    y = y.reshape(-1)
    
    n_samples = len(X)
    d = X.shape[1]
    
    candidates = []  # list of (predict_fn, train_mse)
    
    # 1. Try exact fit with simple basis
    if _remaining() > 5.0:
        result = _try_exact_fit(_basis_simple, X, y)
        if result is not None:
            w, err = result
            candidates.append((_make_predict_linear(_basis_simple, w), err))
    
    # 2. Neural Network
    if _remaining() > 30.0:
        try:
            nn_predict, nn_loss = _train_nn(X, y, hidden_dims=(32, 32), n_epochs=500, lr=0.01, time_budget=25.0)
            if np.isfinite(nn_loss):
                candidates.append((nn_predict, nn_loss))
        except Exception:
            pass
        
        # Try a second NN with different architecture
        if _remaining() > 20.0:
            try:
                nn_predict2, nn_loss2 = _train_nn(X, y, hidden_dims=(64, 32), n_epochs=400, lr=0.005, seed=123, time_budget=15.0)
                if np.isfinite(nn_loss2):
                    candidates.append((nn_predict2, nn_loss2))
            except Exception:
                pass
    
    # 3. Ridge on comprehensive features
    if _remaining() > 10.0:
        try:
            A = _build_features(X)
            if A.shape[1] > 0:
                best = _select_alpha(A, y)
                if best is not None:
                    w, err, alpha = best
                    def predict_ridge(X_test):
                        X_test = np.asarray(X_test, dtype=float)
                        if X_test.ndim == 1:
                            X_test = X_test.reshape(-1, 1)
                        A_test = _build_features(X_test)
                        if A_test.shape[1] != len(w):
                            # Fallback
                            return np.ones(len(X_test)) * np.mean(y)
                        return A_test @ w
                    candidates.append((predict_ridge, err))
        except Exception:
            pass
    
    # 4. RBF Kernel with CV
    if _remaining() > 15.0:
        best_rbf_err = float('inf')
        best_rbf_predict = None
        gammas_rbf = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
        alphas = [1e-4, 1e-3, 1e-2, 0.1, 1.0]
        
        for gamma in gammas_rbf:
            if _remaining() < 5.0:
                break
            for alpha in alphas:
                if _remaining() < 3.0:
                    break
                try:
                    cv_err = _cv_rbf_score(X, y, gamma, alpha)
                    if cv_err < best_rbf_err:
                        best_rbf_err = cv_err
                        best_rbf_predict = _make_predict_rbf(X, y, gamma, alpha)
                except Exception:
                    continue
        
        if best_rbf_predict is not None:
            candidates.append((best_rbf_predict, best_rbf_err))
    
    # Select best candidate
    if candidates:
        candidates.sort(key=lambda x: x[1])
        best_predict, best_err = candidates[0]
        
        # If best is very good, return it
        y_var = np.var(y)
        if y_var > 0 and best_err / y_var < 0.01:
            return best_predict
        
        # Ensemble: average top candidates if they're all reasonable
        if len(candidates) >= 2 and _remaining() > 5.0:
            top_n = min(3, len(candidates))
            top_errs = [c[1] for c in candidates[:top_n]]
            # Only ensemble if errors are within 5x of each other
            if max(top_errs) < 5 * min(top_errs) + 1e-10:
                top_predicts = [c[0] for c in candidates[:top_n]]
                def ensemble_predict(X_test):
                    X_test = np.asarray(X_test, dtype=float)
                    if X_test.ndim == 1:
                        X_test = X_test.reshape(-1, 1)
                    preds = np.array([p(X_test) for p in top_predicts])
                    return np.mean(preds, axis=0)
                return ensemble_predict
        
        return best_predict
    
    # Ultimate fallback
    try:
        A = _basis_simple(X)
        w, *_ = np.linalg.lstsq(A, y, rcond=None)
        return _make_predict_linear(_basis_simple, w)
    except Exception:
        mean_y = np.mean(y)
        def predict_mean(X):
            X = np.asarray(X, dtype=float)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            return np.full(len(X), mean_y)
        return predict_mean

# EVOLVE-BLOCK-END
