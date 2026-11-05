"""
Rewritten program using MODEL-AND-OPTIMIZE with LOOCV for precise lambda selection.
Uses the Hat Matrix formula for efficient Leave-One-Out Cross-Validation of Ridge Regression.
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
    
    # Precompute Gram matrix and V^T y
    # G = V^T V (6x6)
    # vy = V^T y (6x1)
    G = V_norm.T @ V_norm
    vy = V_norm.T @ y
    
    # Regularization: penalize higher-order terms more heavily
    # This helps with stability and generalization, especially for high-degree terms
    reg_weights = np.array([1.0, 1.0, 5.0, 25.0, 125.0, 625.0])
    R = np.diag(reg_weights)
    
    # Log-spaced regularization values
    # We can afford a very fine grid because LOOCV is cheap for p=6
    lambdas = np.logspace(-10, 5, 200)
    
    best_mse = np.inf
    best_lambda = 0.0
    best_coeffs_norm = None
    
    # Precompute V for hat matrix diagonal calculation
    # H_ii = v_i^T (G + lambda*R)^{-1} v_i
    # We can compute M = V @ inv(G + lambda*R) @ V.T
    # Diagonal of M is what we need.
    
    for lam in lambdas:
        if time.time() - start_time > deadline - 10:
            break
        
        A = G + lam * R
        
        try:
            # Factorize A for efficient inversion and solving
            # Since A is 6x6, direct inverse is fine, but LU is safer
            A_inv = np.linalg.inv(A)
            
            # Compute coefficients
            coeffs_lam = A_inv @ vy
            
            if not np.all(np.isfinite(coeffs_lam)):
                continue
            
            # Compute fitted values
            y_pred = V_norm @ coeffs_lam
            
            # Compute diagonal of Hat Matrix H = V (G + lambda R)^{-1} V^T
            # H_ii = row_i(V) @ A_inv @ col_i(V)
            # This is equivalent to np.diag(V @ A_inv @ V.T)
            # But we can compute it more efficiently as:
            # H_ii = sum_k (V_ik * (A_inv @ V.T)_ik)
            # Or simply:
            # Temp = V @ A_inv  (n x 6)
            # H_diag = np.sum(Temp * V, axis=1)
            
            Temp = V_norm @ A_inv
            H_diag = np.sum(Temp * V_norm, axis=1)
            
            # LOOCV Error Formula:
            # e_i^2 / (1 - h_ii)^2
            # Note: If h_ii is close to 1, this can blow up, but for Ridge it's bounded by 1.
            # Actually, for Ridge, h_ii < 1 always.
            
            denom = 1.0 - H_diag
            # Avoid division by zero or near-zero (should not happen for ridge, but safe)
            valid = np.abs(denom) > 1e-12
            if not np.all(valid):
                continue
                
            residuals = y - y_pred
            loocv_errors = (residuals / denom) ** 2
            loocv_mse = np.mean(loocv_errors)
            
            if np.isfinite(loocv_mse) and loocv_mse < best_mse:
                best_mse = loocv_mse
                best_lambda = lam
                best_coeffs_norm = coeffs_lam.copy()
                
        except np.linalg.LinAlgError:
            continue
        except Exception:
            continue

    if best_coeffs_norm is None:
        # Fallback to OLS on all data if LOOCV failed (e.g., singular matrix issues)
        try:
            coeffs_norm, _, _, _ = np.linalg.lstsq(V_norm, y, rcond=None)
            if np.all(np.isfinite(coeffs_norm)):
                best_coeffs_norm = coeffs_norm
            else:
                return np.zeros(6)
        except:
            return np.zeros(6)
    
    # Convert normalized coefficients to original x-space using stable interpolation
    # Evaluate at Chebyshev points and solve for original coefficients
    x_min = np.min(x)
    x_max = np.max(x)
    if x_max - x_min < 1e-12:
        x_min -= 1.0
        x_max += 1.0
    
    # Chebyshev nodes on [-1, 1] mapped to [x_min, x_max]
    cheb_nodes = np.cos(np.pi * (2 * np.arange(6) + 1) / (2 * 6))
    x_points = 0.5 * (x_max - x_min) * cheb_nodes + 0.5 * (x_max + x_min)
    
    # Evaluate normalized polynomial at these points
    x_points_norm = (x_points - x_mean) / x_std
    V_points = build_vandermonde(x_points_norm, 5)
    y_points = V_points @ best_coeffs_norm
    
    # Solve for original coefficients
    V_orig_points = np.zeros((6, 6))
    for k in range(6):
        V_orig_points[:, k] = x_points ** k
    
    try:
        coeff_orig, _, _, _ = np.linalg.lstsq(V_orig_points, y_points, rcond=None)
    except:
        # Fallback: direct binomial expansion
        coeff_orig = np.zeros(6)
        for i in range(6):
            ci_scaled = best_coeffs_norm[i] / (x_std ** i)
            for k in range(i + 1):
                coeff_orig[k] += ci_scaled * comb(i, k) * ((-x_mean) ** (i - k))
    
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
