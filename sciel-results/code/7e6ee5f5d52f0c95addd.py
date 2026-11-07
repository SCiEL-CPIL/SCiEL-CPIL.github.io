"""
Derivative-Based Symbolic Regression for 1-D Noiseless Equations.

Strategy:
1. Estimate local derivatives (dy/dx, d2y/dx2) from the training data using finite differences.
2. Generate a library of candidate symbolic functions from a set of basis functions.
3. For each candidate function, compute its analytical first and second derivatives.
4. Fit the candidate function to the data using Weighted Least Squares (WLS) that minimizes
   the error in y, y', and y''.
5. Select the candidate with the lowest weighted error.
6. Return the best candidate as the predictor.

This approach leverages the smoothness of the hidden equations to distinguish between
similar-looking curves (e.g., a parabola vs. a high-order polynomial fitting the same points)
by enforcing consistency in the derivatives.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from scipy.optimize import curve_fit

# Global deadline guard: 330s wall-clock limit, use 290s for safety
_DEADLINE = time.time() + 290.0

def _estimate_derivatives(x, y, smooth_window=5):
    """
    Estimate first and second derivatives of y with respect to x.
    Uses central differences with a small smoothing window to reduce noise sensitivity,
    though data is noiseless, numerical stability is key.
    """
    n = len(x)
    if n < 3:
        return np.zeros(n), np.zeros(n)
    
    # Ensure x is sorted for finite difference logic
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]
    
    dx = np.diff(x_sorted)
    # Avoid division by zero if duplicate x exists (shouldn't happen with uniform random, but safe)
    dx_safe = np.where(dx < 1e-9, 1e-9, dx)
    
    # Central differences for interior points
    dy_dx = np.zeros(n)
    d2y_dx2 = np.zeros(n)
    
    # Interior
    if n > 2:
        # Forward/Backward for boundaries, Central for interior
        # For simplicity and speed, we use np.gradient which handles boundaries well
        # np.gradient uses second order accurate central differences for interior points
        dy_dx = np.gradient(y_sorted, x_sorted)
        d2y_dx2 = np.gradient(dy_dx, x_sorted)
        
    # Unsort back to original order
    inv_idx = np.argsort(sort_idx)
    dy_dx_orig = dy_dx[inv_idx]
    d2y_dx2_orig = d2y_dx2[inv_idx]
    
    return dy_dx_orig, d2y_dx2_orig

def _generate_basis_functions(x):
    """
    Generate a set of basis functions and their derivatives.
    Returns a list of tuples: (name, f(x), f'(x), f''(x))
    """
    bases = []
    
    # Constants
    bases.append(('const', np.ones_like(x), np.zeros_like(x), np.zeros_like(x)))
    
    # Linear
    bases.append(('x', x, np.ones_like(x), np.zeros_like(x)))
    
    # Quadratic
    bases.append(('x2', x**2, 2*x, np.ones_like(x)*2))
    
    # Cubic
    bases.append(('x3', x**3, 3*x**2, 6*x))
    
    # Quartic
    bases.append(('x4', x**4, 4*x**3, 12*x**2))
    
    # Trig: sin, cos
    sin_x = np.sin(x)
    cos_x = np.cos(x)
    bases.append(('sin', sin_x, cos_x, -sin_x))
    bases.append(('cos', cos_x, -sin_x, -cos_x))
    
    # Trig: tan (careful with singularities, but WLS will penalize)
    # We'll skip tan in the basis to avoid numerical instability in derivatives near pi/2
    # unless the data specifically demands it, which is rare for smooth 1D tasks.
    
    # Exponential
    exp_x = np.exp(x)
    bases.append(('exp', exp_x, exp_x, exp_x))
    
    # Log
    # Log is undefined for x<=0. We'll use a shifted log or just log(abs(x)) with care?
    # Given ranges [-3,3] and [0.1, 6], log is risky for negative x.
    # Let's include log(x) only if all x > 0? No, the basis must be general.
    # We can use log1p(x) which is safe for x > -1.
    # Or just skip log for now to keep it simple and robust.
    
    # Sqrt
    # Sqrt(x) is undefined for x < 0.
    
    # Let's add some mixed terms if we were doing GP, but for linear combination of bases:
    # The strategy is to find c0*f0 + c1*f1 + ... = y
    # So we just need a rich set of linearly independent functions.
    
    # Add sin(2x), cos(2x) for higher frequency
    bases.append(('sin2x', np.sin(2*x), 2*np.cos(2*x), -4*np.sin(2*x)))
    bases.append(('cos2x', np.cos(2*x), -2*np.sin(2*x), -4*np.cos(2*x)))
    
    # exp(-x)
    exp_neg_x = np.exp(-x)
    bases.append(('exp_neg', exp_neg_x, -exp_neg_x, exp_neg_x))
    
    return bases

def _fit_wls(x, y, dy_dx, d2y_dx2, bases, weights):
    """
    Fit a linear combination of bases to y, dy_dx, d2y_dx2 using Weighted Least Squares.
    Model: y = sum(c_i * f_i(x))
    Residuals:
    R_y = y - sum(c_i * f_i)
    R_dy = dy_dx - sum(c_i * f_i')
    R_d2y = d2y_dx2 - sum(c_i * f_i'')
    
    We minimize: w_y * ||R_y||^2 + w_dy * ||R_dy||^2 + w_d2y * ||R_d2y||^2
    This is a linear least squares problem in c.
    """
    n_points = len(x)
    n_bases = len(bases)
    
    # Build the design matrix for the combined system
    # The system is:
    # [ F_y   0    0   ] [c]   [y]
    # [ 0     F_dy  0  ] [c] = [dy]
    # [ 0     0    F_d2y] [c]   [d2y]
    # Where F_y is n x n_bases, etc.
    
    # To do WLS, we can weight the rows.
    # Simpler approach: Stack the equations with weights incorporated.
    
    # Number of total equations = 3 * n_points
    # Number of unknowns = n_bases
    
    # We can solve this by normal equations or QR decomposition.
    # Let's build the weighted matrix A and vector b.
    
    # A will be (3*n_points, n_bases)
    # b will be (3*n_points,)
    
    A_rows = []
    b_vec = []
    
    # y part
    F_y = np.column_stack([b[1] for b in bases])
    A_rows.append(np.sqrt(weights[0]) * F_y)
    b_vec.append(np.sqrt(weights[0]) * y)
    
    # dy part
    F_dy = np.column_stack([b[2] for b in bases])
    A_rows.append(np.sqrt(weights[1]) * F_dy)
    b_vec.append(np.sqrt(weights[1]) * dy_dx)
    
    # d2y part
    F_d2y = np.column_stack([b[3] for b in bases])
    A_rows.append(np.sqrt(weights[2]) * F_d2y)
    b_vec.append(np.sqrt(weights[2]) * d2y_dx2)
    
    A = np.vstack(A_rows)
    b = np.concatenate(b_vec)
    
    # Solve A c = b
    try:
        c, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
        # Calculate residual error
        pred_y = F_y @ c
        pred_dy = F_dy @ c
        pred_d2y = F_d2y @ c
        
        err_y = np.mean((y - pred_y)**2)
        err_dy = np.mean((dy_dx - pred_dy)**2)
        err_d2y = np.mean((d2y_dx2 - pred_d2y)**2)
        
        total_err = weights[0]*err_y + weights[1]*err_dy + weights[2]*err_d2y
        return c, total_err
    except Exception:
        return None, np.inf

def fit(X_train, y_train):
    global _DEADLINE
    
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    
    x = X_train[:, 0]
    y = y_train
    
    n = len(x)
    
    # 1. Estimate derivatives
    dy_dx, d2y_dx2 = _estimate_derivatives(x, y)
    
    # 2. Generate basis functions
    bases = _generate_basis_functions(x)
    
    # 3. Define weights for WLS
    # We want to balance the magnitude of y, y', y''.
    # Normalize the weights by the inverse variance of the signals to make them comparable.
    var_y = np.var(y) + 1e-9
    var_dy = np.var(dy_dx) + 1e-9
    var_d2y = np.var(d2y_dx2) + 1e-9
    
    # We want the contribution of each term to be roughly equal in the objective function.
    # If y is large, its error will be large. So we scale down its weight.
    # w_i = 1 / var(signal_i)
    w_y = 1.0 / var_y
    w_dy = 1.0 / var_dy
    w_d2y = 1.0 / var_d2y
    
    # We might want to down-weight the second derivative slightly as it's noisier
    # even in noiseless data due to finite difference error.
    w_d2y *= 0.5
    
    weights = np.array([w_y, w_dy, w_d2y])
    
    # 4. Fit the linear combination
    best_c, best_err = _fit_wls(x, y, dy_dx, d2y_dx2, bases, weights)
    
    if best_c is None:
        # Fallback: Mean
        y_mean = np.mean(y)
        def predict(X):
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            return np.full(X.shape[0], y_mean)
        return predict

    # 5. Construct the predictor
    # We have coefficients c for the bases.
    # The prediction is sum(c_i * f_i(x_test))
    
    # Pre-compute the basis matrices for the specific bases used?
    # No, the basis functions are defined by their formula.
    # We need to evaluate them at test points.
    
    # Store the names of the bases and the coefficients
    base_names = [b[0] for b in bases]
    
    def predict(X):
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        x_test = X[:, 0]
        
        # Re-evaluate bases at x_test
        # This is inefficient if called many times, but n is small and test is 200 points.
        # We can cache if needed, but let's keep it simple.
        
        # Generate bases again for x_test
        # Note: _generate_basis_functions returns tuples. We need just the f(x) part.
        # To avoid code duplication, we can just call it and extract.
        
        # Optimization: We can store the type of each base and evaluate directly.
        # But for now, calling the generator is fine for 200 points.
        
        bases_test = _generate_basis_functions(x_test)
        
        # Extract f(x) for each base
        f_vals = np.column_stack([b[1] for b in bases_test])
        
        # Calculate prediction
        preds = f_vals @ best_c
        
        # Ensure finite
        preds = np.where(np.isfinite(preds), preds, np.mean(y))
        
        return preds

    return predict

# EVOLVE-BLOCK-END
