"""Neural ODE-based regression with fallback and time management."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

# Global time tracking
_START_TIME = None
_TIME_LIMIT = 320.0  # Leave 10s safety margin for 330s limit


def _get_elapsed():
    if _START_TIME is None:
        return 0.0
    return time.time() - _START_TIME


def _remaining_time():
    return _TIME_LIMIT - _get_elapsed()


def _init_weights(n_in, n_out, seed=0):
    rng = np.random.default_rng(seed)
    W1 = rng.normal(0, 1.0 / np.sqrt(n_in), size=(n_in, 64))
    b1 = np.zeros(64)
    W2 = rng.normal(0, 1.0 / np.sqrt(64), size=(64, n_out))
    b2 = np.zeros(n_out)
    return W1, b1, W2, b2


def _neural_ode_predict(x, W1, b1, W2, b2, T=1.0):
    """
    Given input x (scalar or array), compute prediction via neural ODE.
    State is [x_t, y_t] where y starts at 0 and evolves to f(x).
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 0:
        x = x.reshape(1)
    
    n = len(x)
    # State: [x_1, ..., x_n, y_1, ..., y_n]
    state0 = np.zeros(2 * n)
    state0[:n] = x
    state0[n:] = 0.0  # y starts at 0
    
    def ode(t, state):
        x_state = state[:n]
        y_state = state[n:]
        # Compute f(x_state) using MLP
        h = np.tanh(x_state @ W1 + b1)
        f_vals = h @ W2 + b2
        # dx/dt = 0 (x is constant), dy/dt = f(x)
        return np.concatenate([np.zeros(n), f_vals])
    
    sol = solve_ivp(ode, [0, T], state0, method='RK45', rtol=1e-8, atol=1e-10)
    if not sol.success:
        # Fallback: just use final state if available
        if len(sol.y) > 0:
            return sol.y[:, n:].flatten()
        return np.zeros(n)
    
    return sol.y[:, n:, -1].flatten()


def _neural_ode_predict_batch(X, W1, b1, W2, b2, T=1.0, batch_size=50):
    """Batch prediction for neural ODE to improve efficiency."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    n_samples = X.shape[0]
    predictions = np.zeros(n_samples)
    
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        X_batch = X[start:end]
        n = X_batch.shape[0]
        
        state0 = np.zeros(2 * n)
        state0[:n] = X_batch.flatten()
        state0[n:] = 0.0
        
        def ode(t, state):
            x_state = state[:n].reshape(X_batch.shape)
            h = np.tanh(x_state @ W1 + b1)
            f_vals = h @ W2 + b2
            return np.concatenate([np.zeros(n), f_vals.flatten()])
        
        sol = solve_ivp(ode, [0, T], state0, method='RK45', rtol=1e-8, atol=1e-10)
        if sol.success and len(sol.y) > 0:
            predictions[start:end] = sol.y[:, n:, -1].flatten()
        elif len(sol.y) > 0:
            predictions[start:end] = sol.y[:, n:].flatten()
        else:
            predictions[start:end] = 0.0
    
    return predictions


def _fit_neural_ode(X_train, y_train, max_time=60.0):
    """Fit a neural ODE model."""
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    
    n_in = X_train.shape[1]
    n_out = 1
    
    # Normalize inputs
    x_mean = X_train.mean(axis=0)
    x_std = X_train.std(axis=0) + 1e-8
    y_mean = y_train.mean()
    y_std = y_train.std() + 1e-8
    
    X_norm = (X_train - x_mean) / x_std
    y_norm = (y_train - y_mean) / y_std
    
    W1, b1, W2, b2 = _init_weights(n_in, n_out, seed=42)
    params = np.concatenate([W1.flatten(), b1, W2.flatten(), b2])
    
    def loss_func(params):
        W1_r = params[:n_in*64].reshape(n_in, 64)
        b1_r = params[n_in*64:n_in*64+64]
        W2_r = params[n_in*64+64:n_in*64+64+64].reshape(64, 1)
        b2_r = params[n_in*64+64+64:]
        
        pred = _neural_ode_predict_batch(X_norm, W1_r, b1_r, W2_r, b2_r, T=1.0)
        return np.mean((pred - y_norm) ** 2)
    
    # Try a few optimization steps
    try:
        result = minimize(loss_func, params, method='L-BFGS-B', 
                         options={'maxiter': 50, 'maxfun': 100})
        best_params = result.x
    except Exception:
        best_params = params
    
    W1_best = best_params[:n_in*64].reshape(n_in, 64)
    b1_best = best_params[n_in*64:n_in*64+64]
    W2_best = best_params[n_in*64+64:n_in*64+64+64].reshape(64, 1)
    b2_best = best_params[n_in*64+64+64:]
    
    def predict(X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        X_norm = (X - x_mean) / x_std
        pred_norm = _neural_ode_predict_batch(X_norm, W1_best, b1_best, W2_best, b2_best, T=1.0)
        return pred_norm * y_std + y_mean
    
    return predict


def _fit_gaussian_process(X_train, y_train, max_time=30.0):
    """Gaussian Process regression with RBF kernel."""
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    
    n = len(X_train)
    
    # RBF kernel
    def rbf_kernel(X1, X2, length_scale=1.0, variance=1.0):
        X1 = X1[:, :, None]
        X2 = X2[:, None, :]
        sq_dist = np.sum((X1 - X2) ** 2, axis=-1)
        return variance * np.exp(-0.5 * sq_dist / length_scale**2)
    
    # Simple GP with fixed hyperparameters
    length_scale = 0.5
    variance = np.var(y_train) + 1e-8
    noise = 1e-6
    
    K = rbf_kernel(X_train, X_train, length_scale, variance) + noise * np.eye(n)
    
    try:
        L = np.linalg.cholesky(K)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
    except np.linalg.LinAlgError:
        # Fallback to linear
        w, *_ = np.linalg.lstsq(X_train, y_train, rcond=None)
        return lambda X: np.asarray(X) @ w
    
    def predict(X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        K_s = rbf_kernel(X, X_train, length_scale, variance)
        return K_s @ alpha
    
    return predict


def _fit_high_degree_poly(X_train, y_train, max_degree=10):
    """Higher degree polynomial fit with regularization."""
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    
    d = X_train.shape[1]
    
    # Build feature matrix with higher degree terms
    cols = [np.ones(len(X_train))]
    for deg in range(1, max_degree + 1):
        for i in range(d):
            cols.append(X_train[:, i] ** deg)
    
    F = np.column_stack(cols)
    
    # Ridge regression for stability
    lambda_val = 1e-4
    w = np.linalg.solve(F.T @ F + lambda_val * np.eye(F.shape[1]), F.T @ y_train)
    
    def features(X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        cols = [np.ones(len(X))]
        for deg in range(1, max_degree + 1):
            for i in range(d):
                cols.append(X[:, i] ** deg)
        return np.column_stack(cols)
    
    return lambda X: features(X) @ w


def _fit_ensemble(X_train, y_train):
    """Combine multiple models with validation."""
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    
    n = len(X_train)
    
    # Split for validation
    rng = np.random.default_rng(123)
    indices = rng.permutation(n)
    val_size = max(20, n // 5)
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]
    
    X_tr, y_tr = X_train[train_idx], y_train[train_idx]
    X_val, y_val = X_train[val_idx], y_train[val_idx]
    
    models = []
    scores = []
    
    # Model 1: High degree poly
    try:
        if _remaining_time() > 5:
            pred_func = _fit_high_degree_poly(X_tr, y_tr, max_degree=8)
            val_pred = pred_func(X_val)
            score = np.mean((val_pred - y_val) ** 2)
            models.append((score, pred_func, 'poly8'))
            scores.append(score)
    except Exception:
        pass
    
    # Model 2: Gaussian Process
    try:
        if _remaining_time() > 5:
            pred_func = _fit_gaussian_process(X_tr, y_tr)
            val_pred = pred_func(X_val)
            score = np.mean((val_pred - y_val) ** 2)
            models.append((score, pred_func, 'gp'))
            scores.append(score)
    except Exception:
        pass
    
    # Model 3: Neural ODE (if time permits)
    try:
        if _remaining_time() > 30:
            pred_func = _fit_neural_ode(X_tr, y_tr, max_time=25)
            val_pred = pred_func(X_val)
            score = np.mean((val_pred - y_val) ** 2)
            models.append((score, pred_func, 'neural_ode'))
            scores.append(score)
    except Exception:
        pass
    
    # Select best model or ensemble
    if not models:
        # Fallback to simple linear
        w, *_ = np.linalg.lstsq(X_train, y_train, rcond=None)
        return lambda X: np.asarray(X) @ w
    
    # Weighted ensemble based on validation scores
    if len(models) > 1:
        inv_scores = [1.0 / (s + 1e-10) for s, _, _ in models]
        total = sum(inv_scores)
        weights = [s / total for s in inv_scores]
        
        def predict(X):
            X = np.asarray(X, dtype=float)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            pred = np.zeros(len(X))
            for w, (_, func, _), _ in zip(weights, models, range(len(models))):
                pred += w * func(X)
            return pred
        
        return predict
    else:
        return models[0][1]


def fit(X_train, y_train):
    """Main fit function with time management."""
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    
    # Try ensemble approach
    try:
        predict_func = _fit_ensemble(X_train, y_train)
    except Exception:
        # Fallback to simple polynomial
        def features(X):
            X = np.asarray(X, dtype=float)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            d = X.shape[1]
            cols = [np.ones(len(X))] + [X[:, j] for j in range(d)] + \
                   [X[:, i] * X[:, j] for i in range(d) for j in range(i, d)]
            return np.column_stack(cols)
        
        w, *_ = np.linalg.lstsq(features(X_train), y_train, rcond=None)
        predict_func = lambda X: features(X) @ w
    
    # Ensure predictions are finite
    def safe_predict(X):
        try:
            pred = predict_func(X)
            pred = np.asarray(pred, dtype=float)
            # Replace any non-finite values
            pred = np.where(np.isfinite(pred), pred, 0.0)
            return pred
        except Exception:
            X = np.asarray(X, dtype=float)
            return np.zeros(len(X))
    
    return safe_predict
# EVOLVE-BLOCK-END
