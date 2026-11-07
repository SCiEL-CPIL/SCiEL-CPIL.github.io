"""
Strategy: Shape-Preserving Piecewise Cubic Hermite Interpolation (PCHIP).
Rationale:
1. The data is noiseless (200 points). PCHIP is ideal for noiseless data as it 
   interpolates exactly while preserving local shape (monotonicity) and avoiding 
   the oscillatory artifacts (Gibbs-like phenomena) of standard cubic splines.
2. Unlike GPs, which require kernel selection and Cholesky decompositions that can 
   be numerically unstable or slow for certain point distributions, PCHIP is 
   deterministic, extremely fast (O(n log n) sort + O(n) fit), and robust.
3. By using PCHIP, we effectively treat the 200 training points as a dense, 
   high-fidelity sample of the true underlying function. This approach minimizes 
   bias by not assuming a specific global functional form (like RBF or polynomial) 
   and instead adapts locally to the data's curvature and monotonicity.
4. The predict function uses vectorized binary search to locate intervals and 
   evaluates the cubic Hermite polynomials, ensuring high performance even for 
   large test sets.
"""
import numpy as np
import time
from scipy.interpolate import PchipInterpolator
import warnings
warnings.filterwarnings('ignore')

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 280  # Leave 20s buffer for 300s limit
    
    X = X_train[:, 0]  # 1D input
    y = y_train
    
    # Sort data by X for interpolation
    sort_idx = np.argsort(X)
    x_sorted = X[sort_idx]
    y_sorted = y[sort_idx]
    
    # Remove duplicate X values if any (keep first occurrence)
    # PCHIP requires strictly increasing x values
    unique_mask = np.concatenate([True, x_sorted[1:] != x_sorted[:-1]])
    if not np.all(unique_mask):
        x_sorted = x_sorted[unique_mask]
        y_sorted = y_sorted[unique_mask]
    
    # Fit PCHIP interpolator
    # PchipInterpolator handles the slope calculation to preserve shape
    try:
        pchip = PchipInterpolator(x_sorted, y_sorted)
    except Exception:
        # Fallback: if PCHIP fails (e.g., due to numerical issues), use linear interpolation
        # or a simple polynomial fit if the range is very small
        try:
            # Linear interpolation is safer than high-degree poly for arbitrary data
            def predict_linear(X):
                X_flat = X.flatten()
                return np.interp(X_flat, x_sorted, y_sorted).reshape(X.shape)
            return predict_linear
        except Exception:
            mean_y = np.mean(y)
            return lambda X: np.full(X.shape[0], mean_y)
    
    # Define predict callable
    def predict(X):
        X_flat = X.flatten()
        
        # Check for values outside the training range
        # PCHIP extrapolates linearly by default in scipy (actually, it uses the 
        # boundary slopes, which is linear extrapolation based on the first/last 
        # interval's slope). 
        # np.interp is linear. PchipInterpolator extrapolates using the boundary slopes.
        # This is generally good behavior for smooth functions.
        
        try:
            y_pred = pchip(X_flat)
        except Exception:
            # Fallback to linear interpolation if PCHIP evaluation fails for some reason
            y_pred = np.interp(X_flat, x_sorted, y_sorted)
            
        return y_pred.reshape(X.shape)
    
    return predict
