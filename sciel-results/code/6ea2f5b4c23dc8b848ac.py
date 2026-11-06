import numpy as np
import time

def _nadaraya_watson(x, y, bandwidth=None, n_bins=200):
    """
    Nadaraya-Watson kernel regression for denoising.
    Returns smoothed y values at the original x locations.
    """
    n = len(x)
    if n < 10:
        return y.copy()
    
    if bandwidth is None:
        # Silverman's rule of thumb
        std = np.std(x)
        if std < 1e-12:
            std = 1.0
        bandwidth = 1.06 * std * n ** (-1.0 / 5.0)
    
    if bandwidth < 1e-12:
        bandwidth = 1e-8
    
    # Use binning for efficiency with large datasets
    x_min = np.min(x)
    x_max = np.max(x)
    x_range = x_max - x_min
    
    if x_range < 1e-12:
        return np.full_like(y, np.mean(y))
    
    # Create bin centers
    bin_edges = np.linspace(x_min - bandwidth, x_max + bandwidth, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    # Assign each point to nearest bin
    bin_indices = np.searchsorted(bin_centers, x, side='left')
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    # Compute weighted average per bin using Gaussian kernel
    smoothed = np.zeros(n)
    
    for i in range(n_bins):
        mask = bin_indices == i
        if not np.any(mask):
            continue
        
        x_i = x[mask]
        y_i = y[mask]
        center = bin_centers[i]
        
        # Gaussian kernel weights
        dist = x_i - center
        weights = np.exp(-0.5 * (dist / bandwidth) ** 2)
        total_w = np.sum(weights)
        
        if total_w < 1e-12:
            smoothed[mask] = np.mean(y_i)
        else:
            smoothed[mask] = np.sum(weights * y_i) / total_w
    
    return smoothed

def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = 120.0  # seconds
    
    # Convert to numpy arrays
    x = np.asarray(training_x, dtype=float).flatten()
    y = np.asarray(training_subgrid_tendency, dtype=float).flatten()
    
    # Filter non-finite values
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    
    if len(x) < 10:
        # Fallback to linear if data is too sparse
        coeff = np.zeros(6)
        if len(x) >= 2:
            try:
                coeff[:2] = np.polynomial.polynomial.polyfit(x, y, 1)
            except Exception:
                pass
        return coeff

    # Check time
    if time.time() > deadline - 5:
        # Fallback to linear if running out of time
        coeff = np.zeros(6)
        try:
            coeff[:2] = np.polynomial.polynomial.polyfit(x, y, 1)
        except Exception:
            pass
        return coeff

    # Step 1: Nadaraya-Watson denoising
    y_smooth = _nadaraya_watson(x, y)
    
    # Check time
    if time.time() > deadline - 10:
        # Fallback to simple polynomial fit on smoothed data
        try:
            coeff = np.polynomial.polynomial.polyfit(x, y_smooth, 5)
            if not np.all(np.isfinite(coeff)):
                coeff = np.zeros(6)
                coeff[:2] = np.polynomial.polynomial.polyfit(x, y, 1)
        except Exception:
            coeff = np.zeros(6)
            try:
                coeff[:2] = np.polynomial.polynomial.polyfit(x, y, 1)
            except Exception:
                pass
        return coeff

    # Step 2: Initial degree-5 polynomial fit on smoothed data
    x_min = np.min(x)
    x_max = np.max(x)
    x_range = x_max - x_min
    
    if x_range < 1e-12:
        coeff = np.zeros(6)
        coeff[0] = np.mean(y_smooth)
        return coeff

    # Scale x to [-1, 1] for numerical stability
    s = (2.0 * x - (x_max + x_min)) / x_range
    
    # Fit Legendre polynomial of degree 5 on smoothed data
    try:
        leg_coeffs = np.polynomial.legende.legfit(s, y_smooth, 5)
    except Exception:
        # Fallback to linear if fit fails
        coeff = np.zeros(6)
        try:
            coeff[:2] = np.polynomial.polynomial.polyfit(x, y, 1)
        except Exception:
            pass
        return coeff

    # Check time
    if time.time() > deadline - 5:
        # Convert and return
        leg_poly = np.polynomial.legende.Legendre(leg_coeffs)
        mono_s_poly = leg_poly.convert(kind=np.polynomial.polynomial.Polynomial)
        a = 2.0 / x_range
        b = -(x_max + x_min) / x_range
        transform_poly = np.polynomial.polynomial.Polynomial([b, a])
        final_poly = mono_s_poly.compose(transform_poly)
        coeff = np.zeros(6)
        num_coeffs = min(6, len(final_poly.coef))
        coeff[:num_coeffs] = final_poly.coef[:num_coeffs]
        if not np.all(np.isfinite(coeff)):
            coeff = np.zeros(6)
            try:
                coeff[:2] = np.polynomial.polynomial.polyfit(x, y, 1)
            except Exception:
                pass
        return coeff

    # Step 3: Robust iterative reweighting with Huber loss
    # Compute initial residuals
    leg_poly = np.polynomial.legende.Legendre(leg_coeffs)
    mono_s_poly = leg_poly.convert(kind=np.polynomial.polynomial.Polynomial)
    a = 2.0 / x_range
    b = -(x_max + x_min) / x_range
    transform_poly = np.polynomial.polynomial.Polynomial([b, a])
    final_poly = mono_s_poly.compose(transform_poly)
    
    # Evaluate polynomial at x
    y_pred = final_poly(x)
    residuals = y_smooth - y_pred
    
    # Compute Huber weights
    # Use MAD for scale estimation
    mad = np.median(np.abs(residuals - np.median(residuals)))
    if mad < 1e-12:
        mad = np.std(residuals)
    if mad < 1e-12:
        mad = 1.0
    
    # Huber delta
    delta_huber = 1.5 * 1.4826 * mad  # approximately 1.5 * sigma
    
    # Perform up to 3 iterations of reweighting
    best_coeff = None
    best_mse = np.inf
    
    for iteration in range(3):
        # Check time
        if time.time() > deadline - 3:
            break
        
        # Compute Huber weights
        abs_res = np.abs(residuals)
        weights = np.ones(len(x))
        large = abs_res > delta_huber
        weights[large] = delta_huber / abs_res[large]
        
        # Fit weighted polynomial using Legendre basis
        # Create weighted least squares system in Legendre basis
        # V = [P_0(s), P_1(s), ..., P_5(s)]
        V = np.polynomial.legende.legvander(s, 5)
        
        # Weighted least squares: (V^T W V) c = V^T W y_smooth
        W_sqrt = np.sqrt(weights)
        V_w = V * W_sqrt[:, np.newaxis]
        y_w = y_smooth * W_sqrt
        
        try:
            # Solve the weighted system
            A = V_w.T @ V_w
            B = V_w.T @ y_w
            
            # Add small regularization for stability
            reg = 1e-10 * np.trace(A) / A.shape[0]
            A += reg * np.eye(A.shape[0])
            
            leg_coeffs_new = np.linalg.solve(A, B)
        except Exception:
            # If solve fails, keep previous coefficients
            break
        
        if not np.all(np.isfinite(leg_coeffs_new)):
            break
        
        # Update polynomial
        leg_poly_new = np.polynomial.legende.Legendre(leg_coeffs_new)
        mono_s_poly_new = leg_poly_new.convert(kind=np.polynomial.polynomial.Polynomial)
        final_poly_new = mono_s_poly_new.compose(transform_poly)
        
        # Compute new predictions and residuals
        y_pred_new = final_poly_new(x)
        residuals_new = y_smooth - y_pred_new
        
        # Compute MSE on original (unsmoothed) data for evaluation
        mse_original = np.mean((y - y_pred_new) ** 2)
        
        if mse_original < best_mse:
            best_mse = mse_original
            best_coeff = final_poly_new.coef.copy()
        
        # Update residuals for next iteration
        residuals = residuals_new
        leg_coeffs = leg_coeffs_new
        
        # Check convergence
        if np.max(np.abs(leg_coeffs_new - leg_coeffs)) < 1e-8:
            break
    
    # Use best coefficients found
    if best_coeff is not None and np.all(np.isfinite(best_coeff)):
        coeff = np.zeros(6)
        num_coeffs = min(6, len(best_coeff))
        coeff[:num_coeffs] = best_coeff[:num_coeffs]
    else:
        # Fallback to initial fit
        leg_poly = np.polynomial.legende.Legendre(leg_coeffs)
        mono_s_poly = leg_poly.convert(kind=np.polynomial.polynomial.Polynomial)
        final_poly = mono_s_poly.compose(transform_poly)
        coeff = np.zeros(6)
        num_coeffs = min(6, len(final_poly.coef))
        coeff[:num_coeffs] = final_poly.coef[:num_coeffs]
    
    # Final check for non-finite results
    if not np.all(np.isfinite(coeff)):
        coeff = np.zeros(6)
        try:
            coeff[:2] = np.polynomial.polynomial.polyfit(x, y, 1)
        except Exception:
            pass
    
    return coeff
