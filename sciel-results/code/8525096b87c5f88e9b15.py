"""Symbolic regression over hidden equations using ensemble of diverse regressors."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings
from scipy.optimize import minimize
from scipy.spatial import distance
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern, WhiteKernel
from sklearn.model_selection import cross_val_predict
import warnings
warnings.filterwarnings('ignore')

_START_TIME = time.time()
_TIME_LIMIT = 320.0  # seconds, safety margin from 330s


def _remaining_time():
    return max(0.0, _TIME_LIMIT - (time.time() - _START_TIME))


def _fit_ridge_poly(X, y, degree=3):
    """Fit polynomial regression with Ridge regularization."""
    pf = PolynomialFeatures(degree=degree, include_bias=True)
    Xp = pf.fit_transform(X)
    # Add regularization to avoid overfitting
    alpha = 1.0
    n, p = Xp.shape
    I = np.eye(p)
    I[0, 0] = 0  # Don't regularize bias
    w = np.linalg.solve(Xp.T @ Xp + alpha * I, Xp.T @ y)
    return lambda X_new: pf.transform(X_new) @ w


def _fit_kernel_ridge(X, y):
    """Fit Kernel Ridge Regression with RBF kernel."""
    # Try multiple length scales and pick best via cross-validation
    best_predictor = None
    best_score = np.inf
    for ls in [0.1, 0.3, 0.5, 1.0, 2.0, 5.0]:
        try:
            kr = KernelRidge(alpha=1e-5, kernel='rbf', gamma=1.0 / (2 * ls * ls))
            # Use cross-validation to estimate performance
            from sklearn.model_selection import KFold
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            cv_preds = cross_val_predict(kr, X, y, cv=kf)
            mse = np.mean((cv_preds - y) ** 2)
            if mse < best_score:
                best_score = mse
                best_predictor = kr
        except Exception:
            continue
    if best_predictor is None:
        best_predictor = KernelRidge(alpha=1e-5, kernel='rbf', gamma=0.5)
    best_predictor.fit(X, y)
    return lambda X_new: best_predictor.predict(X_new)


def _fit_svr(X, y):
    """Fit Support Vector Regression."""
    # Scale features to [0, 1] for better SVR performance
    X_min = X.min(axis=0, keepdims=True)
    X_max = X.max(axis=0, keepdims=True)
    X_range = X_max - X_min
    X_range[X_range < 1e-10] = 1.0
    X_scaled = (X - X_min) / X_range
    
    svr = SVR(kernel='rbf', C=100.0, epsilon=0.01, gamma='scale')
    svr.fit(X_scaled, y)
    
    def predict(X_new):
        X_new_scaled = (X_new - X_min) / X_range
        return svr.predict(X_new_scaled)
    return predict


def _fit_gp(X, y):
    """Fit Gaussian Process Regression."""
    # Normalize X for better GP performance
    X_mean = X.mean(axis=0, keepdims=True)
    X_std = X.std(axis=0, keepdims=True)
    X_std[X_std < 1e-10] = 1.0
    X_norm = (X - X_mean) / X_std
    
    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-6)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-10, n_restarts_optimizer=5,
                                  normalize_y=True, random_state=42)
    gp.fit(X_norm, y)
    
    def predict(X_new):
        X_new_norm = (X_new - X_mean) / X_std
        return gp.predict(X_new_norm)
    return predict


def _fit_rbf_network(X, y):
    """Fit a simple RBF network."""
    n, d = X.shape
    # Choose centers as a subset of training points
    n_centers = min(50, n)
    indices = np.random.choice(n, n_centers, replace=False)
    centers = X[indices]
    
    # Compute distances from all training points to centers
    D = distance.cdist(X, centers)
    # Choose gamma based on median distance
    gamma = 1.0 / (np.median(D) + 1e-10)
    
    # Solve for weights
    K = np.exp(-gamma * D ** 2)
    w = np.linalg.lstsq(K, y, rcond=None)[0]
    
    def predict(X_new):
        D_new = distance.cdist(X_new, centers)
        K_new = np.exp(-gamma * D_new ** 2)
        return K_new @ w
    return predict


def _fit_fourier(X, y, max_freq=5):
    """Fit using Fourier basis functions."""
    n, d = X.shape
    # Normalize X to [0, 1]
    X_min = X.min(axis=0, keepdims=True)
    X_max = X.max(axis=0, keepdims=True)
    X_range = X_max - X_min
    X_range[X_range < 1e-10] = 1.0
    X_norm = (X - X_min) / X_range
    
    # Build Fourier basis
    cols = [np.ones(n)]
    for j in range(d):
        for k in range(1, max_freq + 1):
            cols.append(np.sin(k * np.pi * X_norm[:, j]))
            cols.append(np.cos(k * np.pi * X_norm[:, j]))
    # Cross terms
    for i in range(d):
        for j in range(i, d):
            for k in range(1, 3):
                cols.append(np.sin(k * np.pi * X_norm[:, i]) * np.sin(k * np.pi * X_norm[:, j]))
                cols.append(np.cos(k * np.pi * X_norm[:, i]) * np.cos(k * np.pi * X_norm[:, j]))
    
    Xf = np.column_stack(cols)
    # Ridge regression
    alpha = 0.1
    p = Xf.shape[1]
    I = np.eye(p)
    I[0, 0] = 0
    w = np.linalg.solve(Xf.T @ Xf + alpha * I, Xf.T @ y)
    
    def predict(X_new):
        X_new_norm = (X_new - X_min) / X_range
        cols_new = [np.ones(len(X_new))]
        for j in range(d):
            for k in range(1, max_freq + 1):
                cols_new.append(np.sin(k * np.pi * X_new_norm[:, j]))
                cols_new.append(np.cos(k * np.pi * X_new_norm[:, j]))
        for i in range(d):
            for j in range(i, d):
                for k in range(1, 3):
                    cols_new.append(np.sin(k * np.pi * X_new_norm[:, i]) * np.sin(k * np.pi * X_new_norm[:, j]))
                    cols_new.append(np.cos(k * np.pi * X_new_norm[:, i]) * np.cos(k * np.pi * X_new_norm[:, j]))
        Xf_new = np.column_stack(cols_new)
        return Xf_new @ w
    return predict


def _fit_nn(X, y):
    """Fit a simple neural network using scipy optimization."""
    n, d = X.shape
    # Normalize
    X_mean = X.mean(axis=0, keepdims=True)
    X_std = X.std(axis=0, keepdims=True)
    X_std[X_std < 1e-10] = 1.0
    X_norm = (X - X_mean) / X_std
    y_mean = y.mean()
    y_std = y.std()
    if y_std < 1e-10:
        y_std = 1.0
    y_norm = (y - y_mean) / y_std
    
    # Simple 1 hidden layer network
    n_hidden = 32
    
    def _net_forward(params, X):
        W1 = params[:d * n_hidden].reshape(d, n_hidden)
        b1 = params[d * n_hidden:d * n_hidden + n_hidden]
        W2 = params[d * n_hidden + n_hidden:d * n_hidden + n_hidden + n_hidden].reshape(n_hidden, 1)
        b2 = params[d * n_hidden + n_hidden + n_hidden:]
        h = np.tanh(X @ W1 + b1)
        return h @ W2 + b2
    
    def _loss(params):
        pred = _net_forward(params, X_norm)
        return np.mean((pred - y_norm) ** 2)
    
    # Initialize
    np.random.seed(42)
    params0 = np.random.randn(d * n_hidden + n_hidden + n_hidden + 1) * 0.1
    
    result = minimize(_loss, params0, method='L-BFGS-B', options={'maxiter': 500})
    params = result.x
    
    def predict(X_new):
        X_new_norm = (X_new - X_mean) / X_std
        pred = _net_forward(params, X_new_norm)
        return pred * y_std + y_mean
    return predict


def fit(X_train, y_train):
    """Main fit function: try multiple models and return the best ensemble."""
    np.random.seed(42)
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    
    # Ensure 2D
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    n, d = X.shape
    
    # Try multiple models and collect predictions
    models = []
    model_names = []
    
    # 1. Polynomial Ridge
    try:
        if _remaining_time() > 10:
            pred_func = _fit_ridge_poly(X, y, degree=3)
            models.append(pred_func)
            model_names.append('poly3')
    except Exception:
        pass
    
    # 2. Kernel Ridge
    try:
        if _remaining_time() > 20:
            pred_func = _fit_kernel_ridge(X, y)
            models.append(pred_func)
            model_names.append('kr')
    except Exception:
        pass
    
    # 3. SVR
    try:
        if _remaining_time() > 15:
            pred_func = _fit_svr(X, y)
            models.append(pred_func)
            model_names.append('svr')
    except Exception:
        pass
    
    # 4. GP
    try:
        if _remaining_time() > 30:
            pred_func = _fit_gp(X, y)
            models.append(pred_func)
            model_names.append('gp')
    except Exception:
        pass
    
    # 5. RBF Network
    try:
        if _remaining_time() > 10:
            pred_func = _fit_rbf_network(X, y)
            models.append(pred_func)
            model_names.append('rbf')
    except Exception:
        pass
    
    # 6. Fourier
    try:
        if _remaining_time() > 15:
            pred_func = _fit_fourier(X, y, max_freq=4)
            models.append(pred_func)
            model_names.append('fourier')
    except Exception:
        pass
    
    # 7. Neural Network
    try:
        if _remaining_time() > 60:
            pred_func = _fit_nn(X, y)
            models.append(pred_func)
            model_names.append('nn')
    except Exception:
        pass
    
    # If no models worked, fall back to simple linear
    if not models:
        def fallback_predict(X_new):
            X_new = np.asarray(X_new, dtype=np.float64)
            if X_new.ndim == 1:
                X_new = X_new.reshape(-1, 1)
            # Simple mean prediction
            return np.full(len(X_new), np.mean(y))
        return fallback_predict
    
    # Evaluate each model on training data to get weights
    train_preds = []
    for model in models:
        try:
            pred = model(X)
            train_preds.append(pred)
        except Exception:
            train_preds.append(None)
    
    # Compute MSE for each model
    mses = []
    for pred in train_preds:
        if pred is not None:
            mse = np.mean((pred - y) ** 2)
            mses.append(mse)
        else:
            mses.append(np.inf)
    
    # Weight models inversely proportional to their MSE
    mses = np.array(mses)
    valid = mses < np.inf
    if np.any(valid):
        # Use inverse MSE as weights, with a floor
        weights = 1.0 / (mses[valid] + 1e-10)
        weights = weights / weights.sum()
        # Only keep valid models
        valid_models = [m for m, v in zip(models, valid) if v]
        valid_weights = weights
    else:
        # Equal weights
        valid_models = models
        valid_weights = np.ones(len(models)) / len(models)
    
    # Create ensemble predictor
    def predict(X_new):
        X_new = np.asarray(X_new, dtype=np.float64)
        if X_new.ndim == 1:
            X_new = X_new.reshape(-1, 1)
        
        all_preds = []
        for model in valid_models:
            try:
                p = model(X_new)
                if p.ndim == 0:
                    p = np.array([p])
                all_preds.append(p)
            except Exception:
                all_preds.append(None)
        
        # Combine predictions
        result = np.zeros(len(X_new))
        total_weight = 0.0
        for i, pred in enumerate(all_preds):
            if pred is not None:
                result += valid_weights[i] * pred
                total_weight += valid_weights[i]
        
        if total_weight > 0:
            result /= total_weight
        else:
            result = np.full(len(X_new), np.mean(y))
        
        return result
    
    return predict
# EVOLVE-BLOCK-END
