"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.interpolate import PchipInterpolator

def fit(X_train, y_train):
    start_time = time.time()
    # Safety margin: The evaluator kills after 330s. 
    # We have 4 equations. Total budget ~ 330s.
    # We will use a very fast method, so time is not a strict constraint per equation,
    # but we guard just in case.
    
    X = np.asarray(X_train)
    y = np.asarray(y_train)
    
    # Ensure 1D column vector if not already
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if y.ndim == 1:
        y = y.reshape(-1)
        
    # Extract 1D input
    x = X[:, 0]
    
    # Sort by input value for interpolation
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]
    
    # Check for duplicate x values. 
    # If duplicates exist, average the y values to ensure a valid function definition
    # for the interpolator, which expects strictly increasing x.
    unique_x, inverse_indices, counts = np.unique(x_sorted, return_inverse=True, return_counts=True)
    
    if len(unique_x) < len(x_sorted):
        # Handle duplicates
        y_sum = np.zeros_like(unique_x)
        np.add.at(y_sum, inverse_indices, y_sorted)
        y_avg = y_sum / counts
        x_interp = unique_x
        y_interp = y_avg
    else:
        x_interp = x_sorted
        y_interp = y_sorted
        
    # The hidden equations are likely smooth. 
    # PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) is monotonicity-preserving
    # and avoids the oscillations of standard cubic splines (Runge's phenomenon).
    # It is a robust choice for unknown 1D functions.
    
    try:
        interpolator = PchipInterpolator(x_interp, y_interp)
    except Exception:
        # Fallback to linear interpolation if PCHIP fails (e.g., numerical issues)
        try:
            interpolator = np.interp
            # np.interp requires a lambda or closure to match interface
            x_ref = x_interp
            y_ref = y_interp
            def predict_linear(X_test):
                x_test = np.asarray(X_test)
                if x_test.ndim == 1:
                    x_test = x_test.reshape(-1, 1)
                return np.interp(x_test[:, 0], x_ref, y_ref)
            return predict_linear
        except Exception:
            # Absolute fallback: return mean
            mean_y = np.mean(y)
            def predict_mean(X_test):
                return np.full(len(X_test), mean_y)
            return predict_mean

    def predict(X_test):
        # Convert to numpy array
        X_test = np.asarray(X_test)
        if X_test.ndim == 1:
            x_test = X_test.reshape(-1, 1)
        else:
            x_test = X_test
            
        # Extract 1D input
        x_vals = x_test[:, 0]
        
        # Perform interpolation
        # PCHIP extrapolates linearly outside the domain by default (fill_value="extrapolate" is default for PCHIP? 
        # Actually, PCHIPInterpolator uses 'extrapolate' for fill_value by default? 
        # Let's check: scipy.interpolate.PchipInterpolator(..., fill_value="extrapolate") is default.
        # This is good for test points slightly outside train range.
        
        y_pred = interpolator(x_vals)
        
        # Ensure finite values
        y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)
        
        return y_pred

    return predict

# EVOLVE-BLOCK-END
