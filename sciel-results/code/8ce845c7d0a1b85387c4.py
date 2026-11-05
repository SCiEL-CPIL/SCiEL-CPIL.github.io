"""Rewritten program: fit a shared degree-five polynomial to unresolved tendency."""
# EVOLVE-BLOCK-START
import time
import numpy as np

DEADLINE = 145.0  # safety margin before 150s wall-clock kill
START_TIME = time.time()


def _elapsed():
    return time.time() - START_TIME


def _check_deadline():
    return _elapsed() > DEADLINE


def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-five polynomial to the unresolved fast-scale forcing
    of a two-scale Lorenz-96 model using Chebyshev basis for numerical stability.
    
    Returns 6 coefficients in ASCENDING powers of x.
    """
    training_x = np.asarray(training_x, dtype=np.float64).ravel()
    training_subgrid_tendency = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()
    
    n = training_x.size
    if n == 0:
        return np.zeros(6)
    
    # Ensure at least as many points as we need for degree 5
    if n < 6:
        # Fall back to least degree possible
        deg = max(1, n - 1)
        deg = min(deg, 5)
        coeff = np.zeros(6)
        try:
            c = np.polynomial.polynomial.polyfit(training_x, training_subgrid_tendency, deg)
            coeff[:deg+1] = c
        except Exception:
            coeff[0] = np.mean(training_subgrid_tendency)
        return coeff
    
    # Strategy: Fit in Chebyshev basis on normalized domain for superior conditioning
    # 1. Scale x to z in [-1, 1] using min-max normalization
    # 2. Fit Chebyshev series C(z) = sum c_k T_k(z)
    # 3. Convert to monomial basis in z
    # 4. Transform from z-space to x-space
    # Use GCV to select regularization strength and anisotropic regularization
    
    x_min = np.min(training_x)
    x_max = np.max(training_x)
    x_range = x_max - x_min
    
    if x_range < 1e-12:
        # Degenerate case: all x values are the same
        return np.array([np.mean(training_subgrid_tendency)] + [0.0]*5)
    
    # Map x to z in [-1, 1]
    z = 2.0 * (training_x - x_min) / x_range - 1.0
    
    # Precompute Chebyshev Vandermonde
    V_cheb = np.polynomial.chebyshev.chebvander(z, 5)
    n = V_cheb.shape[0]
    
    # Precompute transformation matrix from z-monomial to x-monomial
    a_val = 2.0 / x_range
    b_val = -(2.0 * x_min / x_range) - 1.0
    
    T = np.zeros((6, 6))
    T[0, 0] = 1.0
    for i in range(1, 6):
        for j in range(i):
            T[j, i] = b_val * T[j, i-1]
            if j + 1 <= i:
                T[j+1, i] += a_val * T[j, i-1]
    
    # Anisotropic regularization: penalize higher Chebyshev orders more
    # Weight vector: [1, 1, 1, 2, 4, 8] - exponential growth
    w = np.array([1.0, 1.0, 1.0, 2.0, 4.0, 8.0])
    W = np.diag(w)
    
    # Try multiple regularization strengths with anisotropic Tikhonov
    alphas = [0.0, 1e-10, 1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1.0, 5.0, 10.0]
    
    best_coeff = None
    best_gcv = np.inf
    
    for alpha in alphas:
        if _check_deadline():
            break
        
        try:
            if alpha == 0.0:
                # Pure Chebyshev fit (unregularized)
                c_cheb, residuals, rank, sv = np.linalg.lstsq(V_cheb, training_subgrid_tendency, rcond=None)
                # Hat matrix H = V (V^T V)^-1 V^T
                VtV = V_cheb.T @ V_cheb
                if np.linalg.matrix_rank(VtV) < 6:
                    VtV_inv = np.linalg.pinv(VtV)
                else:
                    VtV_inv = np.linalg.inv(VtV)
                H = V_cheb @ VtV_inv @ V_cheb.T
                trace_H = np.trace(H)
            else:
                # Anisotropic Tikhonov regularization
                # Minimize ||V c - y||^2 + alpha * ||W c||^2
                # (V^T V + alpha W^T W) c = V^T y
                WtW = W.T @ W
                A = V_cheb.T @ V_cheb + alpha * WtW
                b = V_cheb.T @ training_subgrid_tendency
                c_cheb = np.linalg.solve(A, b)
                
                # Hat matrix for regularized least squares: H = V (V^T V + alpha W^T W)^-1 V^T
                A_inv = np.linalg.inv(A)
                H = V_cheb @ A_inv @ V_cheb.T
                trace_H = np.trace(H)
            
            # Compute GCV score
            pred = V_cheb @ c_cheb
            resid = training_subgrid_tendency - pred
            mse = np.mean(resid ** 2)
            
            # GCV = MSE / (1 - trace(H)/n)^2
            denom = (1.0 - trace_H / n)
            if denom < 1e-10:
                gcv = np.inf
            else:
                gcv = mse / (denom ** 2)
            
            if gcv < best_gcv:
                best_gcv = gcv
                # Convert to monomial in z-space
                c_z = np.polynomial.chebyshev.cheb2poly(c_cheb)
                if len(c_z) < 6:
                    c_z_full = np.zeros(6)
                    c_z_full[:len(c_z)] = c_z
                    c_z = c_z_full
                else:
                    c_z = c_z[:6]
                
                # Transform to x-space
                best_coeff = T @ c_z
                
        except Exception:
            continue
    
    if best_coeff is None:
        # Fallback: simple linear fit
        try:
            c = np.polynomial.polynomial.polyfit(training_x, training_subgrid_tendency, 1)
            best_coeff = np.zeros(6)
            best_coeff[:2] = c
        except Exception:
            best_coeff = np.zeros(6)
            best_coeff[0] = np.mean(training_subgrid_tendency)
    
    # Final check: ensure all coefficients are finite
    if not np.all(np.isfinite(best_coeff)):
        best_coeff = np.zeros(6)
        best_coeff[0] = np.mean(training_subgrid_tendency)
    
    return best_coeff
# EVOLVE-BLOCK-END
