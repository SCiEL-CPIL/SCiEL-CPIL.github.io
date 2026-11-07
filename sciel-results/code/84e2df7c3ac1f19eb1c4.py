import numpy as np
import time
import warnings
from sklearn.model_selection import KFold
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler
import sklearn.metrics

warnings.filterwarnings("ignore")

_START_TIME = None
TIME_LIMIT = 300.0
SAFETY_MARGIN = 20.0

def _time_left():
    if _START_TIME is None:
        return TIME_LIMIT - SAFETY_MARGIN
    return TIME_LIMIT - SAFETY_MARGIN - (time.time() - _START_TIME)

def _fit_kernel_ridge(X, y, kernel_type, gamma, alpha):
    """
    Fit Kernel Ridge Regression.
    Returns a predict function.
    """
    # Standardize X for better kernel performance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create KernelRidge model
    # Note: KernelRidge in sklearn uses 'alpha' as regularization parameter (inverse of C in SVM terminology)
    # Lower alpha means less regularization (closer to interpolation)
    try:
        model = KernelRidge(alpha=alpha, kernel=kernel_type, gamma=gamma)
        model.fit(X_scaled, y)
    except Exception:
        # Fallback to linear if kernel fails
        model = KernelRidge(alpha=alpha, kernel='linear', gamma=gamma)
        model.fit(X_scaled, y)
        
    def predict(X_in):
        X_in_scaled = scaler.transform(X_in)
        return model.predict(X_in_scaled)
    
    return predict

def _cv_score(X, y, kernel_type, gamma, alpha, n_splits=5):
    """
    Compute Mean Cross-Validation MSE.
    Returns inf if it fails.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    mse_scores = []
    
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        if _time_left() < 10:
            return np.inf
            
        try:
            pred = _fit_kernel_ridge(X_train, y_train, kernel_type, gamma, alpha)
            y_pred = pred(X_val)
            mse = np.mean((y_val - y_pred) ** 2)
            mse_scores.append(mse)
        except Exception:
            return np.inf
            
    if len(mse_scores) == n_splits:
        return np.mean(mse_scores)
    return np.inf

def fit(X_train, y_train):
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
        
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=float)
    
    # Handle edge cases
    if len(X) == 0:
        return lambda X_in: np.zeros(len(X_in))
    
    # Ensure 2D
    if X.ndim == 1:
        X = X.reshape(-1, 1)
        
    # Remove non-finite
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[mask], y[mask]
    
    if len(X) == 0:
        return lambda X_in: np.zeros(len(X_in))
        
    # Normalize Y for scoring consistency in selection (though KRR handles raw y fine)
    # We want to minimize MSE on raw y, so CV on raw y is correct.
    
    # Define candidate kernels and hyperparameters
    # Since data is noiseless, we want low alpha (high capacity) but CV will penalize overfitting.
    # RBF is universal approximator.
    kernels = [
        ('rbf', 0.1, 1e-4),
        ('rbf', 0.5, 1e-4),
        ('rbf', 1.0, 1e-4),
        ('rbf', 2.0, 1e-4),
        ('rbf', 5.0, 1e-4),
        ('rbf', 10.0, 1e-4),
        ('rbf', 0.1, 1e-6),
        ('rbf', 1.0, 1e-6),
        ('poly', 1.0, 1e-4),  # degree is fixed in sklearn KernelRidge 'poly' kernel? No, 'poly' kernel uses gamma, coef0, degree.
        # Actually, sklearn KernelRidge 'poly' kernel uses gamma, coef0, degree. 
        # The constructor accepts 'gamma', 'coef0', 'degree'. 
        # My _fit_kernel_ridge doesn't pass degree/coef0. Let's stick to RBF and Linear/Sine which are simpler.
        # Let's add Sine kernel: sin(gamma * (x+y)) ? No, standard sine kernel is sin(gamma * ||x-y||^2) or similar.
        # sklearn doesn't have 'sine' built-in easily without custom. 
        # Let's stick to RBF with varied gammas and alphas, plus Linear.
        ('linear', 1.0, 1e-4),
        ('linear', 1.0, 1e-6),
        ('poly', 1.0, 1e-4), # This will use default degree=3, coef0=1
    ]
    
    # Add more specific RBFs
    kernels.extend([
        ('rbf', 0.01, 1e-5),
        ('rbf', 0.2, 1e-5),
        ('rbf', 0.8, 1e-5),
        ('rbf', 2.5, 1e-5),
        ('rbf', 0.5, 1e-8),
        ('rbf', 1.5, 1e-8),
    ])
    
    best_score = np.inf
    best_predictor = None
    best_params = None
    
    # Sort candidates? Or just iterate. 
    # We have ~20 candidates. Each CV is 5 folds. 
    # 20 * 5 = 100 fits. 
    # With 200 points, fits are very fast (<0.1s each). 
    # Total time ~10s. Well within limit.
    
    for kernel_type, gamma, alpha in kernels:
        if _time_left() < 5:
            break
            
        score = _cv_score(X, y, kernel_type, gamma, alpha)
        
        if score < best_score:
            best_score = score
            best_params = (kernel_type, gamma, alpha)
            # Don't store predictor yet, re-fit on full data at the end to use all data
            # Or store the model? Re-fitting is cleaner.
            
    if best_params is None:
        # Fallback: Linear regression
        try:
            from sklearn.linear_model import LinearRegression
            lr = LinearRegression()
            lr.fit(X, y)
            def predict(X_in):
                return lr.predict(X_in)
            return predict
        except:
            y_mean = y.mean()
            return lambda X_in: np.full(len(X_in), y_mean)
            
    # Re-fit best model on all training data
    kernel_type, gamma, alpha = best_params
    try:
        final_predictor = _fit_kernel_ridge(X, y, kernel_type, gamma, alpha)
        return final_predictor
    except Exception:
        # Fallback if final fit fails
        y_mean = y.mean()
        return lambda X_in: np.full(len(X_in), y_mean)
