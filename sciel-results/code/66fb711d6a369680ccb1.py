# EVOLVE-BLOCK-START
import numpy as np
from scipy.interpolate import UnivariateSpline, PchipInterpolator
import warnings

def _fit_piecewise_poly(x, y, max_degree=5, max_depth=10, tol=1e-8):
    """
    Recursively fits piecewise polynomials to (x, y) data.
    Returns a list of (x_start, x_end, coeffs) where coeffs are for a polynomial
    in the form sum(coeffs[i] * (x - x_start)^i).
    """
    x = np.asarray(x)
    y = np.asarray(y)
    
    if len(x) < 2:
        return []
        
    # If very few points, just fit a polynomial of degree len-1
    if len(x) <= max_degree + 1:
        coeffs = np.polyfit(x, y, deg=len(x)-1)
        # Convert to local form around x[0] for stability? 
        # Actually, np.polyfit returns high-to-low power.
        # Let's just store the poly object or coeffs.
        # For prediction, we need to evaluate.
        # Let's store (x_min, x_max, poly_coeffs)
        return [(x.min(), x.max(), coeffs)]

    # Fit a polynomial of degree max_degree
    # Use np.polyfit
    try:
        coeffs = np.polyfit(x, y, deg=max_degree)
        y_pred = np.polyval(coeffs, x)
        residuals = np.abs(y - y_pred)
        max_res = np.max(residuals)
        
        # Scale tolerance by y magnitude
        y_scale = np.max(np.abs(y)) + 1e-10
        if max_res < tol * y_scale:
            return [(x.min(), x.max(), coeffs)]
    except Exception:
        # If polyfit fails (e.g., ill-conditioned), fall back to PCHIP for this segment
        pass

    # Find the point with maximum residual to split
    idx_max_res = np.argmax(residuals)
    x_split = x[idx_max_res]
    
    # Split data
    left_mask = x < x_split
    right_mask = x > x_split
    
    # Ensure we have enough points on both sides
    # If not, shift split point
    if np.sum(left_mask) < max_degree + 1:
        # Try splitting at a different point, e.g., median
        x_split = np.median(x)
        left_mask = x < x_split
        right_mask = x > x_split
        
    if np.sum(left_mask) < 2 or np.sum(right_mask) < 2:
        # Cannot split further effectively, just fit a high-degree poly
        deg = min(max_degree, len(x) - 1)
        coeffs = np.polyfit(x, y, deg=deg)
        return [(x.min(), x.max(), coeffs)]

    left_x, left_y = x[left_mask], y[left_mask]
    right_x, right_y = x[right_mask], y[right_mask]
    
    # Recurse
    left_segs = _fit_piecewise_poly(left_x, left_y, max_degree, max_depth - 1, tol)
    right_segs = _fit_piecewise_poly(right_x, right_y, max_degree, max_depth - 1, tol)
    
    return left_segs + right_segs

def fit(X_train, y_train):
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    
    if X.ndim == 1:
        X = X.reshape(-1, 1)
        
    n_samples, n_features = X.shape
    
    if n_features > 1:
        # Fallback for multi-feature (should not happen per task description)
        # Use simple linear regression
        X_1d = X[:, 0]
        A = np.column_stack([np.ones_like(X_1d), X_1d])
        w, *_ = np.linalg.lstsq(A, y, rcond=None)
        def predict_multi(X_test):
            X_test = np.asarray(X_test, dtype=np.float64)
            if X_test.ndim == 1:
                X_test = X_test.reshape(-1, 1)
            X_1d = X_test[:, 0]
            A_test = np.column_stack([np.ones_like(X_1d), X_1d])
            return A_test @ w
        return predict_multi

    x = X[:, 0]
    
    # Sort
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]
    
    # Handle duplicates
    unique_x, unique_indices = np.unique(x_sorted, return_index=True)
    if len(unique_x) < len(x_sorted):
        y_unique = np.zeros(len(unique_x))
        for i, idx in enumerate(unique_indices):
            if i < len(unique_indices) - 1:
                next_idx = unique_indices[i+1]
                y_unique[i] = y_sorted[idx:next_idx].mean()
            else:
                y_unique[i] = y_sorted[idx:].mean()
        x_sorted = unique_x
        y_sorted = y_unique

    # Try global polynomial first (degrees 1-5)
    for deg in range(1, 6):
        if len(x_sorted) > deg:
            try:
                coeffs = np.polyfit(x_sorted, y_sorted, deg=deg)
                y_pred = np.polyval(coeffs, x_sorted)
                residuals = np.abs(y_sorted - y_pred)
                y_scale = np.max(np.abs(y_sorted)) + 1e-10
                if np.max(residuals) < 1e-8 * y_scale:
                    def predict_poly(X_test):
                        X_test = np.asarray(X_test, dtype=np.float64)
                        if X_test.ndim == 1:
                            X_test = X_test.reshape(-1, 1)
                        x_test = X_test[:, 0]
                        return np.polyval(coeffs, x_test)
                    return predict_poly
            except Exception:
                continue

    # If global poly didn't work, use recursive piecewise poly
    segments = _fit_piecewise_poly(x_sorted, y_sorted, max_degree=4, max_depth=12, tol=1e-9)
    
    # If recursive fitting failed or produced too few segments, fall back to PCHIP
    if not segments:
        try:
            pchip = PchipInterpolator(x_sorted, y_sorted)
            def predict_pchip(X_test):
                X_test = np.asarray(X_test, dtype=np.float64)
                if X_test.ndim == 1:
                    X_test = X_test.reshape(-1, 1)
                x_test = X_test[:, 0]
                return pchip(x_test)
            return predict_pchip
        except Exception:
            # Last resort: linear interp
            from scipy.interpolate import interp1d
            interp_func = interp1d(x_sorted, y_sorted, kind='linear', fill_value='extrapolate')
            def predict_linear(X_test):
                X_test = np.asarray(X_test, dtype=np.float64)
                if X_test.ndim == 1:
                    X_test = X_test.reshape(-1, 1)
                x_test = X_test[:, 0]
                return interp_func(x_test)
            return predict_linear

    # Convert segments to a structure for fast lookup
    # segments is a list of (x_start, x_end, coeffs)
    # We need to handle the boundaries carefully.
    # Sort segments by x_start
    segments.sort(key=lambda s: s[0])
    
    # Pre-compute boundaries for binary search
    bounds = np.array([s[0] for s in segments])
    coeffs_list = [s[2] for s in segments]
    x_starts = np.array([s[0] for s in segments])
    
    def predict(X_test):
        X_test = np.asarray(X_test, dtype=np.float64)
        if X_test.ndim == 1:
            X_test = X_test.reshape(-1, 1)
        x_test = X_test[:, 0]
        
        y_pred = np.zeros_like(x_test)
        
        # For each test point, find the segment
        # Since segments are contiguous and sorted, we can use searchsorted
        # But segments might have gaps or overlaps due to recursion logic.
        # Let's assume they cover the range [min(x), max(x)].
        # For points outside, use the nearest segment's polynomial.
        
        # Find indices
        # We want the segment where x_start <= x < x_end
        # Using searchsorted on x_starts:
        # If x is in [s_i.start, s_i.end), it belongs to i.
        # searchsorted returns the index where x would be inserted to maintain order.
        # So if x is between s_i.start and s_{i+1}.start, it returns i+1?
        # No, searchsorted(bounds, x, side='right') - 1 gives the index of the last bound <= x.
        
        indices = np.searchsorted(bounds, x_test, side='right') - 1
        
        # Clip indices to valid range
        indices = np.clip(indices, 0, len(segments) - 1)
        
        for i in range(len(x_test)):
            seg_idx = indices[i]
            coeffs = coeffs_list[seg_idx]
            x_start = x_starts[seg_idx]
            # Evaluate polynomial: sum(coeffs[j] * (x - x_start)^(deg-j))
            # np.polyval expects coeffs in descending power order.
            # My _fit_piecewise_poly uses np.polyfit which returns descending order.
            # However, I noted in the helper comment "local form" but didn't implement it.
            # np.polyval(coeffs, x_test[i]) works globally.
            # But for stability in piecewise, local is better.
            # Let's just use np.polyval with the global coeffs for simplicity, 
            # as the range is small.
            y_pred[i] = np.polyval(coeffs, x_test[i])
            
        return y_pred

    return predict
# EVOLVE-BLOCK-END
