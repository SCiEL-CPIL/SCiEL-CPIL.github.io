"""Exact Interpolation with CV Selection for 1-D Noiseless Symbolic Regression."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.interpolate import CubicSpline, PchipInterpolator, interp1d

# Global deadline guard: 330s wall-clock limit, use 295s for safety
_DEADLINE = time.time() + 295.0

def _build_interpolators(x_train, y_train):
    """
    Builds a list of (name, interpolator_object) tuples.
    Each interpolator is designed to exactly fit the noiseless training data.
    """
    interpolators = []
    
    # Sort by x to ensure monotonicity for PCHIP and avoid issues with some splines
    # Note: The problem doesn't guarantee sorted X_train, so we must sort for fitting
    # but the predict function will handle unsorted X by calling the interpolator directly.
    # Interpolators from scipy generally handle unsorted query points if the model is built correctly.
    
    # 1. Natural Cubic Spline
    try:
        # bc_type='natural' ensures zero second derivative at boundaries
        cs = CubicSpline(x_train, y_train, bc_type='natural')
        interpolators.append(('NaturalCubic', cs))
    except Exception:
        pass

    # 2. PCHIP (Piecewise Cubic Hermite Interpolating Polynomial)
    # Preserves shape, no overshoots. Good for monotonic or locally smooth functions.
    try:
        pchip = PchipInterpolator(x_train, y_train)
        interpolators.append(('PCHIP', pchip))
    except Exception:
        pass

    # 3. Linear Interpolation (Fallback, robust but less accurate for smooth curves)
    try:
        lin = interp1d(x_train, y_train, kind='linear', fill_value='extrapolate')
        interpolators.append(('Linear', lin))
    except Exception:
        pass

    # 4. Multi-Scale RBF (Radial Basis Function) Interpolator
    # We solve for weights w such that sum(w_i * phi(x - x_i)) = y_j
    # phi is Gaussian. We try multiple bandwidths (sigma).
    n = len(x_train)
    if n > 5:
        # Generate candidate sigmas based on the data range
        x_min, x_max = np.min(x_train), np.max(x_train)
        x_range = x_max - x_min
        if x_range > 0:
            # Sigmas relative to range
            sigmas = [0.05 * x_range, 0.1 * x_range, 0.2 * x_range, 0.5 * x_range]
            for sigma in sigmas:
                try:
                    # Build RBF matrix
                    # D[i, j] = exp(-((x_i - x_j)^2) / (2 * sigma^2))
                    # To save memory/time, we might use a subset or smaller n, but n=200 is small.
                    diff = x_train[:, np.newaxis] - x_train[np.newaxis, :]
                    D = np.exp(-(diff ** 2) / (2 * sigma ** 2))
                    
                    # Add a small regularization to avoid singularity if points are too close
                    # Since data is uniform random, points are likely distinct, but just in case.
                    D += 1e-10 * np.eye(n)
                    
                    # Solve for weights
                    # D @ w = y
                    w = np.linalg.solve(D, y_train)
                    
                    # Create a callable for this RBF
                    def rbf_predict(x, w=w, centers=x_train, sigma=sigma):
                        diff = x[:, np.newaxis] - centers[np.newaxis, :]
                        values = np.exp(-(diff ** 2) / (2 * sigma ** 2))
                        return values @ w
                        
                    # Wrap in a class-like object or lambda to match interface?
                    # scipy interp1d and Pchip have __call__. 
                    # We can create a simple object.
                    class RBFModel:
                        def __init__(self, w, centers, sigma):
                            self.w = w
                            self.centers = centers
                            self.sigma = sigma
                        def __call__(self, x):
                            return rbf_predict(x, self.w, self.centers, self.sigma)
                    
                    interpolators.append((f'RBF_{sigma:.4f}', RBFModel(w, x_train, sigma)))
                except np.linalg.LinAlgError:
                    continue
                except Exception:
                    continue

    return interpolators

def fit(X_train, y_train):
    global _DEADLINE
    
    # Ensure 1D input handling
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    
    x_train = X_train[:, 0]
    y_train = y_train
    
    # Sort training data by x for consistent interpolation behavior (especially PCHIP)
    sort_idx = np.argsort(x_train)
    x_sorted = x_train[sort_idx]
    y_sorted = y_train[sort_idx]
    
    # Build candidate interpolators
    candidates = _build_interpolators(x_sorted, y_sorted)
    
    if not candidates:
        # Fallback: constant predictor
        y_mean = np.mean(y_train)
        def predict(X):
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            return np.full(X.shape[0], y_mean)
        return predict

    # If we have time, perform Cross-Validation to select the best interpolator.
    # Since the data is noiseless, the "best" interpolator is the one that generalizes
    # to the underlying smooth function. 
    # Strategy: Hold out a small portion of the training data (e.g., 10%) as a validation set.
    # Fit all candidates on the remaining 90%.
    # Evaluate MSE on the held-out 10%.
    # Select the candidate with the lowest validation MSE.
    # Refit the selected candidate on the FULL training data.
    
    n = len(x_train)
    n_val = max(10, int(n * 0.1))
    n_train_cv = n - n_val
    
    # Random split
    indices = np.arange(n)
    np.random.shuffle(indices)
    val_idx = indices[:n_val]
    train_cv_idx = indices[n_val:]
    
    x_cv_train = x_train[train_cv_idx]
    y_cv_train = y_train[train_cv_idx]
    x_val = x_train[val_idx]
    y_val = y_train[val_idx]
    
    # Sort CV training data
    sort_idx_cv = np.argsort(x_cv_train)
    x_cv_train_sorted = x_cv_train[sort_idx_cv]
    y_cv_train_sorted = y_cv_train[sort_idx_cv]
    
    best_model = None
    best_val_mse = np.inf
    
    # Time budget for CV
    cv_start = time.time()
    cv_time_limit = 10.0 # Spend max 10s on CV
    
    for name, _ in candidates:
        if time.time() > _DEADLINE - 10: # Keep 10s safety margin
            break
        if time.time() - cv_start > cv_time_limit:
            break
            
        # Rebuild the specific candidate on CV train data
        # We need to re-instantiate the specific type
        # This is slightly inefficient but n is small (180)
        
        # Instead of rebuilding from scratch, we can just build a fresh list for the CV set
        # and pick the one with the same name/type logic.
        # For simplicity and robustness, let's just build the set of candidates again for the CV split.
        
        # Optimization: We know the types. Let's just build the specific ones we care about.
        # Actually, rebuilding the full list is fine for n=200.
        
        cv_candidates = _build_interpolators(x_cv_train_sorted, y_cv_train_sorted)
        
        for cv_name, cv_model in cv_candidates:
            if cv_name != name:
                continue
            try:
                pred_val = cv_model(x_val)
                mse = np.mean((pred_val - y_val) ** 2)
                if mse < best_val_mse:
                    best_val_mse = mse
                    best_model = cv_name # Store name, we will refit on full data
            except Exception:
                continue
        break # We only need to evaluate one representative per type? 
        # Actually, for RBFs, the sigma matters. 
        # The loop above breaks after the first candidate type. 
        # This is a flaw. We should evaluate all candidates.
        # Let's restructure: iterate through all candidates built from full data,
        # but we need to refit them on the CV subset to get a fair comparison.
        
    # The previous logic was flawed because it broke early. Let's do a cleaner CV.
    
    # Clean CV Implementation
    best_model_name = None
    best_val_mse = np.inf
    cv_candidates = _build_interpolators(x_cv_train_sorted, y_cv_train_sorted)
    
    for name, model in cv_candidates:
        if time.time() > _DEADLINE - 10:
            break
        try:
            pred_val = model(x_val)
            mse = np.mean((pred_val - y_val) ** 2)
            if mse < best_val_mse:
                best_val_mse = mse
                best_model_name = name
        except Exception:
            continue

    # Refit the best model on the FULL training data
    final_model = None
    if best_model_name:
        # We need to find the model in the original full-data candidates that matches the name
        # OR we can just refit a new one. Refitting is safer to ensure it uses all data.
        # Let's find the type from the name and refit.
        
        # Since names are unique per sigma/type, we can just iterate the original candidates
        # and pick the one with the same name. But the original candidates were built on full data,
        # so they are already the final models!
        # Wait, the CV was done on a subset. The model that won on the subset might not be the best on the full data?
        # No, the "type" (e.g., NaturalCubic) is the same. The parameters (weights) change.
        # So if "NaturalCubic" won the CV, we should use the NaturalCubic built on the FULL data.
        
        for name, model in candidates:
            if name == best_model_name:
                final_model = model
                break
    
    if final_model is None:
        # Fallback to the first candidate if CV failed or no match found
        final_model = candidates[0][1]
        
    def predict(X):
        try:
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            x_test = X[:, 0]
            preds = final_model(x_test)
            # Ensure finite
            preds = np.where(np.isfinite(preds), preds, np.mean(y_train))
            return preds
        except Exception:
            return np.full(X.shape[0], np.mean(y_train))
    
    return predict

# EVOLVE-BLOCK-END
