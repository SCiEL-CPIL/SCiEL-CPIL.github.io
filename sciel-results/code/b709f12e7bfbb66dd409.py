import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    x = np.asarray(training_x, dtype=float).flatten()
    t = np.asarray(training_subgrid_tendency, dtype=float).flatten()
    
    # Filter non-finite values
    mask = np.isfinite(x) & np.isfinite(t)
    x = x[mask]
    t = t[mask]
    
    n = len(x)
    if n == 0:
        return np.zeros(6)
    
    # If n is very large, subsample for kernel regression to ensure speed
    max_samples_for_smoothing = 5000
    if n > max_samples_for_smoothing:
        idx = np.random.choice(n, max_samples_for_smoothing, replace=False)
        x_smooth = x[idx]
        t_smooth = t[idx]
    else:
        x_smooth = x
        t_smooth = t
    
    # Estimate bandwidth for Nadaraya-Watson
    std_x = np.std(x_smooth)
    if std_x < 1e-10:
        return np.zeros(6)
    n_smooth = len(x_smooth)
    h_base = 1.06 * std_x * (n_smooth ** (-0.2))
    
    # Try a small grid of bandwidths around Silverman's estimate
    # to find the one that gives the best denoised polynomial fit
    bandwidth_factors = [0.5, 0.7, 1.0, 1.3, 1.5]
    
    def _compute_denoised(h_val):
        """Compute Nadaraya-Watson denoised targets for given bandwidth."""
        t_den = np.zeros_like(t_smooth)
        chunk_size = 1000
        for i in range(0, n_smooth, chunk_size):
            end = min(i + chunk_size, n_smooth)
            x_chunk = x_smooth[i:end]
            diff = x_chunk[:, np.newaxis] - x_smooth[np.newaxis, :]
            weights = np.exp(-0.5 * (diff / h_val) ** 2)
            w_sum = np.sum(weights, axis=1)
            wt_sum = np.sum(weights * t_smooth[np.newaxis, :], axis=1)
            t_den[i:end] = wt_sum / np.where(w_sum > 1e-10, w_sum, 1.0)
        return t_den
    
    def _fit_polynomial(x_data, t_data):
        """Fit degree-5 polynomial with Huber-loss reweighting and return coefficients in original x space."""
        mu = np.mean(x_data)
        sigma = np.std(x_data)
        if sigma < 1e-10:
            sigma = 1.0
        x_scaled = (x_data - mu) / sigma
        
        # Build Vandermonde matrix for degree-5 polynomial in scaled space
        V = np.vander(x_scaled, 6, increasing=True)
        
        # Initial least-squares fit
        try:
            coeff_scaled, _, _, _ = np.linalg.lstsq(V, t_data, rcond=None)
        except Exception:
            try:
                coeff_scaled, _, _, _ = np.linalg.lstsq(V[:, :4], t_data, rcond=None)
                temp = np.zeros(6)
                temp[:4] = coeff_scaled
                coeff_scaled = temp
            except:
                return None
        
        # Iterative reweighting with Huber loss
        for _ in range(5):
            if time.time() > deadline - 3.0:
                break
            # Compute residuals
            pred = V @ coeff_scaled
            resid = t_data - pred
            
            # Huber weights: w = 1 if |r| <= delta, else delta/|r|
            # Use MAD-based scale estimate for robustness
            abs_resid = np.abs(resid)
            mad = np.median(abs_resid)
            if mad < 1e-10:
                mad = np.std(resid)
            if mad < 1e-10:
                mad = 1e-10
            # delta = 1.4826 * MAD (consistent estimator of std for Gaussian)
            delta = 1.4826 * mad
            
            weights = np.where(abs_resid <= delta, 1.0, delta / abs_resid)
            # Clip weights to avoid extreme values
            weights = np.clip(weights, 0.01, 1.0)
            
            # Weighted least squares
            W_sqrt = np.sqrt(weights)
            V_w = V * W_sqrt[:, np.newaxis]
            t_w = t_data * W_sqrt
            try:
                coeff_scaled, _, _, _ = np.linalg.lstsq(V_w, t_w, rcond=None)
            except Exception:
                break
        
        # Transform from scaled x to original x
        c_u = coeff_scaled / (sigma ** np.arange(6))
        coeff_original = np.zeros(6)
        for i in range(6):
            for j in range(i+1):
                if j == 0 or j == i:
                    binom_val = 1.0
                else:
                    num = 1
                    for k in range(j):
                        num *= (i - k)
                    den = 1
                    for k in range(j):
                        den *= (k + 1)
                    binom_val = num / den
                coeff_original[j] += c_u[i] * binom_val * ((-mu) ** (i - j))
        return coeff_original
    
    best_coeff = None
    best_score = np.inf
    
    for factor in bandwidth_factors:
        if time.time() > deadline - 5.0:
            break
        
        h_val = h_base * factor
        t_denoised = _compute_denoised(h_val)
        
        # Evaluate the polynomial fit quality on a held-out subset
        # Use the first 80% for fit, last 20% for validation
        split = int(0.8 * n_smooth)
        x_fit = x_smooth[:split]
        t_fit = t_denoised[:split]
        x_val = x_smooth[split:]
        t_val = t_denoised[split:]
        
        # Fit polynomial on fit subset with Huber reweighting
        mu_fit = np.mean(x_fit)
        sigma_fit = np.std(x_fit)
        if sigma_fit < 1e-10:
            sigma_fit = 1.0
        x_scaled_fit = (x_fit - mu_fit) / sigma_fit
        
        V_fit = np.vander(x_scaled_fit, 6, increasing=True)
        try:
            coeff_scaled_fit, _, _, _ = np.linalg.lstsq(V_fit, t_fit, rcond=None)
        except Exception:
            try:
                coeff_scaled_fit, _, _, _ = np.linalg.lstsq(V_fit[:, :4], t_fit, rcond=None)
                temp = np.zeros(6)
                temp[:4] = coeff_scaled_fit
                coeff_scaled_fit = temp
            except:
                continue
        
        # Huber reweighting for fit subset
        for _ in range(3):
            if time.time() > deadline - 3.0:
                break
            pred_fit = V_fit @ coeff_scaled_fit
            resid_fit = t_fit - pred_fit
            abs_resid_fit = np.abs(resid_fit)
            mad_fit = np.median(abs_resid_fit)
            if mad_fit < 1e-10:
                mad_fit = np.std(resid_fit)
            if mad_fit < 1e-10:
                mad_fit = 1e-10
            delta_fit = 1.4826 * mad_fit
            weights_fit = np.where(abs_resid_fit <= delta_fit, 1.0, delta_fit / abs_resid_fit)
            weights_fit = np.clip(weights_fit, 0.01, 1.0)
            W_sqrt_fit = np.sqrt(weights_fit)
            V_fit_w = V_fit * W_sqrt_fit[:, np.newaxis]
            t_fit_w = t_fit * W_sqrt_fit
            try:
                coeff_scaled_fit, _, _, _ = np.linalg.lstsq(V_fit_w, t_fit_w, rcond=None)
            except Exception:
                break
        
        # Transform to original x space
        c_u_fit = coeff_scaled_fit / (sigma_fit ** np.arange(6))
        coeff_orig_fit = np.zeros(6)
        for i in range(6):
            for j in range(i+1):
                if j == 0 or j == i:
                    binom_val = 1.0
                else:
                    num = 1
                    for k in range(j):
                        num *= (i - k)
                    den = 1
                    for k in range(j):
                        den *= (k + 1)
                    binom_val = num / den
                coeff_orig_fit[j] += c_u_fit[i] * binom_val * ((-mu_fit) ** (i - j))
        
        # Evaluate MSE on validation set
        pred_val = np.zeros_like(x_val)
        for i in range(6):
            pred_val += coeff_orig_fit[i] * x_val ** i
        mse_val = np.mean((pred_val - t_val) ** 2)
        
        if mse_val < best_score:
            best_score = mse_val
            # Refit on all data with this bandwidth
            t_denoised_full = _compute_denoised(h_val)
            best_coeff = _fit_polynomial(x_smooth, t_denoised_full)
    
    if best_coeff is None:
        # Fallback: use the original Silverman bandwidth
        t_denoised = _compute_denoised(h_base)
        best_coeff = _fit_polynomial(x_smooth, t_denoised)
    
    if best_coeff is None:
        # Final fallback
        coeff = np.zeros(6)
        try:
            coeff[:2] = np.polynomial.polynomial.polyfit(x, t, 1)
        except:
            pass
        return coeff
    
    return best_coeff
