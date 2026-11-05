"""
Combined strategy:
1. Use Exact Binomial Expansion for coefficient conversion (from Inspiration 1) for numerical stability.
2. Replace K-fold CV with Leave-One-Out Cross-Validation (LOOCV) for Lambda selection.
   LOOCV is computationally cheaper for Ridge regression (using the Hat Matrix) and provides a less biased estimate of generalization error compared to K-fold, which is crucial for the "unseen forcing regime" requirement.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys
from math import comb

def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-five polynomial to the unresolved subgrid tendency.
    
    Args:
        training_x: flattened array of slow-state values
        training_subgrid_tendency: flattened array of corresponding subgrid tendencies
    
    Returns:
        6 finite real coefficients in ascending powers of x
    """
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s limit (and 150s kill)
    
    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()
    
    n = len(x)
    if n == 0:
        return np.zeros(6)
    
    # Remove any NaN/Inf values
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = len(x)
    
    if n == 0:
        return np.zeros(6)
    
    # Normalize x to improve numerical conditioning
    x_mean = np.mean(x)
    x_std = np.std(x)
    if x_std < 1e-12:
        x_std = 1.0
    x_norm = (x - x_mean) / x_std
    
    # Build Vandermonde matrix for normalized x
    def build_vandermonde(xn, degree=5):
        """Build Vandermonde matrix for polynomial fit in normalized coords."""
        V = np.zeros((len(xn), degree + 1))
        for i in range(degree + 1):
            V[:, i] = xn ** i
        return V
    
    V_norm = build_vandermonde(x_norm, 5)
    
    # Regularization: penalize higher-order terms more heavily
    # This helps prevent overfitting to noise in the fast-scale dynamics
    reg_weights = np.array([1.0, 1.0, 10.0, 100.0, 1000.0, 10000.0])
    reg_matrix = np.diag(reg_weights)
    
    # Log-spaced regularization values
    # We include 0.0 for OLS, and a range of Ridge penalties
    lambdas = [0.0] + list(np.logspace(-10, 4, 30))
    
    best_mse = np.inf
    best_coeffs_norm = None
    best_lambda = 0.0
    
    # Precompute VtV for efficiency
    VtV = V_norm.T @ V_norm
    Vty = V_norm.T @ y
    
    for lam in lambdas:
        if time.time() - start_time > deadline - 5:
            break
            
        try:
            if lam == 0.0:
                # OLS
                # Use lstsq for robustness
                coeffs_lam, _, _, _ = np.linalg.lstsq(V_norm, y, rcond=None)
                # For LOOCV, we need the hat matrix diagonal.
                # For OLS, H = V (V'V)^-1 V'. 
                # It's safer to compute predictions and residuals directly if n is small, 
                # but LOOCV formula is preferred.
                # However, for lam=0, the standard LOOCV formula with hat matrix works.
                try:
                    V_inv_VtV = np.linalg.inv(VtV)
                    H = V_norm @ V_inv_VtV @ V_norm.T
                    h_diag = np.diag(H)
                    # Avoid division by zero
                    denom = 1.0 - h_diag
                    denom[denom < 1e-10] = 1e-10
                    y_pred = V_norm @ coeffs_lam
                    residuals = y - y_pred
                    loocv_errors = residuals / denom
                    mse = np.mean(loocv_errors ** 2)
                except np.linalg.LinAlgError:
                    continue
            else:
                # Ridge Regression
                A = VtV + lam * reg_matrix
                try:
                    # Solve for coefficients
                    coeffs_lam = np.linalg.solve(A, Vty)
                    
                    # Calculate Hat Matrix diagonal for LOOCV
                    # H = V (V'V + lam R)^-1 V'
                    # We need diag(H).
                    # H = V A^-1 V'
                    # diag(H) = sum_j V_ij^2 (A^-1)_jj ... this is not quite right.
                    # H_ij = sum_k V_ik (A^-1)_kj V'_jk
                    # H_ii = sum_k,l V_ik (A^-1)_kl V_il
                    # This is computationally expensive to do for all i.
                    # Alternative: Compute full H if n is not too large.
                    # Given n is likely < 10000, computing H (n x n) might be memory heavy but feasible.
                    # However, we can compute H @ 1 or similar? No.
                    # Let's compute A_inv = np.linalg.inv(A)
                    A_inv = np.linalg.inv(A)
                    H = V_norm @ A_inv @ V_norm.T
                    h_diag = np.diag(H)
                    
                    denom = 1.0 - h_diag
                    denom[denom < 1e-10] = 1e-10
                    y_pred = V_norm @ coeffs_lam
                    residuals = y - y_pred
                    loocv_errors = residuals / denom
                    mse = np.mean(loocv_errors ** 2)
                    
                except np.linalg.LinAlgError:
                    continue
            
            if not np.all(np.isfinite(coeffs_lam)) or not np.isfinite(mse):
                continue
                
            if mse < best_mse:
                best_mse = mse
                best_coeffs_norm = coeffs_lam.copy()
                best_lambda = lam
                
        except Exception:
            continue

    # If no valid lambda was found, fallback to OLS or simple Ridge
    if best_coeffs_norm is None:
        try:
            coeffs_norm, _, _, _ = np.linalg.lstsq(V_norm, y, rcond=None)
            if np.all(np.isfinite(coeffs_norm)):
                best_coeffs_norm = coeffs_norm
            else:
                return np.zeros(6)
        except:
            return np.zeros(6)
    
    # Convert normalized coefficients to original x-space using exact binomial expansion
    # P(x) = sum_{k=0}^5 c_k * ((x - x_mean) / x_std)^k
    #      = sum_{k=0}^5 c_k * x_std^{-k} * (x - x_mean)^k
    #      = sum_{k=0}^5 c_k * x_std^{-k} * sum_{j=0}^k binom(k, j) * x^j * (-x_mean)^(k-j)
    #      = sum_{j=0}^5 [ sum_{k=j}^5 c_k * x_std^{-k} * binom(k, j) * (-x_mean)^(k-j) ] * x^j
    
    coeff_orig = np.zeros(6)
    for k in range(6):
        if k >= 6: break
        c_k = best_coeffs_norm[k]
        if not np.isfinite(c_k):
            continue
        inv_std_k = 1.0 / (x_std ** k)
        for j in range(k + 1):
            coeff_orig[j] += c_k * inv_std_k * comb(k, j) * ((-x_mean) ** (k - j))
    
    # Ensure all coefficients are finite
    if not np.all(np.isfinite(coeff_orig)):
        try:
            c_lin = np.polyfit(x, y, 1)
            coeff_orig = np.zeros(6)
            coeff_orig[0] = c_lin[1]
            coeff_orig[1] = c_lin[0]
        except:
            coeff_orig = np.zeros(6)
    
    if not np.all(np.isfinite(coeff_orig)):
        return np.zeros(6)
    
    return coeff_orig
# EVOLVE-BLOCK-END
