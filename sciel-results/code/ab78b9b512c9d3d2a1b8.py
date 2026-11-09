"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.interpolate import make_interp_spline
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')


def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 290.0  # safety margin before 330s kill

    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)

    d = X.shape[1]
    n = X.shape[0]

    # If d > 1, we could try multivariate, but the task says 1 input for all 4 equations
    # So we assume d=1. If d!=1, fallback to poly.
    if d != 1:
        # Fallback for safety, though unlikely
        def features(X):
            cols = [np.ones(len(X))] + [X[:, j] for j in range(X.shape[1])] + [X[:, i] * X[:, j] for i in range(X.shape[1]) for j in range(i, X.shape[1])]
            return np.column_stack(cols)
        w, *_ = np.linalg.lstsq(features(X), y, rcond=None)
        return lambda X: features(X) @ w

    x = X[:, 0]
    y = y.flatten()

    # Sort by x for interpolation and analysis
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]

    # Check for noise: compute local variance or residual from a smooth spline
    # Try a few degrees of polynomial first to get a baseline
    best_model = None
    best_rmse = np.inf

    # 1. Try Polynomial fits of degree 1 to 6
    for deg in range(1, 7):
        try:
            if time.time() > deadline:
                break
            coeffs = np.polyfit(x_sorted, y_sorted, deg)
            y_pred = np.polyval(coeffs, x_sorted)
            rmse = np.sqrt(np.mean((y_sorted - y_pred) ** 2))
            if rmse < best_rmse:
                best_rmse = rmse
                best_model = ('poly', coeffs)
        except Exception:
            pass

    # 2. Try Spline Interpolation (cubic)
    # Since data is noiseless, a cubic spline through the points should be near-perfect
    # unless the function has discontinuities or is not smooth.
    try:
        if time.time() > deadline:
            pass
        else:
            # make_interp_spline requires unique x values.
            # If there are duplicates, we need to handle them.
            unique_x, unique_indices = np.unique(x_sorted, return_index=True)
            if len(unique_x) > 3:
                y_unique = y_sorted[unique_indices]
                spl = make_interp_spline(unique_x, y_unique, k=3)
                y_pred = spl(x_sorted)
                rmse = np.sqrt(np.mean((y_sorted - y_pred) ** 2))
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_model = ('spline', spl, unique_x, y_unique)
    except Exception:
        pass

    # 3. Try specific functional forms if polynomial/spline RMSE is not tiny
    # The input ranges suggest:
    # Eq 1: [-3, 3]
    # Eq 2: [-2, 2]
    # Eq 3: [-2, 2]
    # Eq 4: [0.1, 6]
    
    # If best_rmse is still significant, try curve_fit with common functions
    # However, spline interpolation is extremely strong for noiseless 1D data.
    # Let's check if the spline error is very small.
    
    # If spline failed or is not great, try a high-degree polynomial (up to 10)
    if best_rmse > 1e-6:
        for deg in range(7, 11):
            try:
                if time.time() > deadline:
                    break
                coeffs = np.polyfit(x_sorted, y_sorted, deg)
                y_pred = np.polyval(coeffs, x_sorted)
                rmse = np.sqrt(np.mean((y_sorted - y_pred) ** 2))
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_model = ('poly', coeffs)
            except Exception:
                pass

    # If we still have a high error, it might be a non-polynomial function like sin, exp, log
    # But spline should handle smooth functions well.
    # Let's verify the spline prediction on the training set is good.
    
    # Define predict function
    def predict(X_test):
        X_test = np.asarray(X_test, dtype=np.float64)
        if X_test.ndim == 1:
            X_test = X_test.reshape(-1, 1)
        x_test = X_test[:, 0]
        
        if best_model is None:
            # Fallback to mean
            return np.full(len(x_test), np.mean(y))
            
        model_type = best_model[0]
        
        if model_type == 'poly':
            coeffs = best_model[1]
            return np.polyval(coeffs, x_test)
        elif model_type == 'spline':
            spl = best_model[1]
            # Spline can extrapolate, but it might go wild.
            # For safety, we can clip x_test to the range of training data if extrapolation is risky,
            # but usually splines are okay for small extrapolations.
            # However, if the test points are within the range, it's perfect.
            try:
                return spl(x_test)
            except Exception:
                # Fallback to linear interpolation if spline fails
                return np.interp(x_test, x_sorted, y_sorted)
        else:
            return np.interp(x_test, x_sorted, y_sorted)

    return predict

# EVOLVE-BLOCK-END
