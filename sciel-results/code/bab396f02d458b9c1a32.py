# EVOLVE-BLOCK-START
import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()

    n = len(x)
    if n < 6:
        return np.zeros(6)

    # Filter non-finite values
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = len(x)
    if n < 6:
        return np.zeros(6)

    # --- Step 1: Define Chebyshev Basis on Scaled Domain ---
    # We map x to t in [-1, 1] using the range of x.
    # This allows us to use Chebyshev polynomials T_k(t), which are orthogonal
    # with respect to the weight 1/sqrt(1-t^2) and numerically stable.
    
    x_min = np.min(x)
    x_max = np.max(x)
    x_range = x_max - x_min
    
    if x_range < 1e-10:
        # Degenerate case: all x are the same
        return np.array([np.mean(y), 0.0, 0.0, 0.0, 0.0, 0.0])
        
    x_center = (x_min + x_max) / 2.0
    x_scale = x_range / 2.0
    
    # t = (x - x_center) / x_scale
    t = (x - x_center) / x_scale
    
    # Clip t to [-1, 1] to handle numerical edge cases where x slightly exceeds min/max
    # due to floating point or if we want strict domain adherence
    # However, for polynomial fitting, extrapolation is possible. 
    # But Chebyshev recurrence is stable on [-1,1]. 
    # Since we fit on the data range, t is in [-1, 1] by construction.
    
    # Construct Chebyshev design matrix C (n x 6)
    # C[i, k] = T_k(t_i)
    C = np.zeros((n, 6))
    C[:, 0] = 1.0  # T_0(t) = 1
    C[:, 1] = t    # T_1(t) = t
    
    # Recurrence: T_{k+1}(t) = 2*t*T_k(t) - T_{k-1}(t)
    for k in range(1, 5):
        C[:, k+1] = 2.0 * t * C[:, k] - C[:, k-1]
        
    # --- Step 2: Ridge Regression with Cross-Validation for Lambda ---
    # We minimize ||C @ coeff - y||^2 + lambda * ||coeff||^2
    # Note: In Chebyshev basis, the identity penalty is not isotropic in the monomial basis,
    # but it is a reasonable regularizer for the Chebyshev coefficients themselves,
    # effectively penalizing higher frequency oscillations in the function space.
    
    # Efficient computation of predictions for various lambdas
    # We can use the SVD or eigen-decomposition of C.T @ C for fast ridge solutions.
    # Since n is likely large and p=6 is small, we can just solve the system for each lambda.
    # Or better: Precompute (C.T @ C + lambda * I)^-1 (C.T @ y)
    
    CtC = C.T @ C  # 6x6 matrix
    Cty = C.T @ y  # 6x1 vector
    
    # Eigen-decomposition for fast ridge
    # CtC is symmetric positive semi-definite
    eigvals, eigvecs = np.linalg.eigh(CtC)
    
    # Ensure eigvals are non-negative (numerical noise)
    eigvals = np.maximum(eigvals, 0.0)
    
    # Project y onto eigenvectors
    y_proj = eigvecs.T @ Cty
    
    # Grid of lambdas
    # Scale lambda relative to the variance of y and the scale of CtC
    y_var = np.var(y)
    max_eig = np.max(eigvals) if np.max(eigvals) > 0 else 1.0
    
    # Log-spaced lambdas
    # Start very small (LS) to large (strong regularization)
    log_lambdas = np.linspace(-10, 6, 50)
    lambdas = 10.0 ** log_lambdas * (max_eig / 1e6) # Heuristic scaling
    
    best_lambda = 0.0
    best_mse = np.inf
    best_coeff_cheb = np.zeros(6)
    
    # Use a holdout set for CV to avoid overfitting the polynomial to noise
    # Stratified or random split
    n_holdout = max(10, min(500, n // 10))
    indices = np.random.permutation(n)
    holdout_idx = indices[:n_holdout]
    train_idx = indices[n_holdout:]
    
    C_train = C[train_idx]
    y_train = y[train_idx]
    C_hold = C[holdout_idx]
    y_hold = y[holdout_idx]
    
    # Recalculate CtC and Cty for train set only for CV
    CtC_train = C_train.T @ C_train
    Cty_train = C_train.T @ y_train
    
    # Eigen decomp for train set
    eigvals_tr, eigvecs_tr = np.linalg.eigh(CtC_train)
    eigvals_tr = np.maximum(eigvals_tr, 0.0)
    y_proj_tr = eigvecs_tr.T @ Cty_train
    
    for lam in lambdas:
        # Ridge solution in eigenbasis:
        # coeff = V * ( (V.T y) / (eigvals + lam) )
        denom = eigvals_tr + lam
        # Avoid division by zero
        denom[denom < 1e-15] = 1e-15
        coeff_eig = y_proj_tr / denom
        coeff_cheb_cv = eigvecs_tr @ coeff_eig
        
        # Predict on holdout
        pred_hold = C_hold @ coeff_cheb_cv
        mse = np.mean((pred_hold - y_hold) ** 2)
        
        if mse < best_mse:
            best_mse = mse
            best_lambda = lam
            best_coeff_cheb = coeff_cheb_cv
        
        if time.time() > deadline - 10:
            break

    # --- Step 3: Refit on Full Data with Best Lambda ---
    denom_full = eigvals + best_lambda
    denom_full[denom_full < 1e-15] = 1e-15
    coeff_eig_full = y_proj / denom_full
    coeff_cheb_full = eigvecs @ coeff_eig_full
    
    # Check time
    if time.time() > deadline - 5:
        # If we are running out of time, return a simple LS fit in monomial basis as fallback?
        # No, the Chebyshev conversion is fast. Just return the Chebyshev-based monomial coeffs.
        pass

    # --- Step 4: Convert Chebyshev Coefficients to Monomial Coefficients ---
    # We have f(x) = sum_{k=0}^5 a_k T_k(t(x))
    # where t(x) = a*x + b
    # a = 1/x_scale = 2/x_range
    # b = -x_center/x_scale = -x_center * 2/x_range
    
    a = 1.0 / x_scale
    b = -x_center / x_scale
    
    # We need to expand T_k(a*x + b) into monomial form.
    # T_0(z) = 1
    # T_1(z) = z
    # T_2(z) = 2z^2 - 1
    # T_3(z) = 4z^3 - 3z
    # T_4(z) = 8z^4 - 8z^2 + 1
    # T_5(z) = 16z^5 - 20z^3 + 5z
    
    # Let's compute the expansion programmatically to be safe and general.
    # Represent polynomial in x as array [c0, c1, ..., ck]
    
    def cheb_to_monomial(k, a, b):
        """
        Returns coefficients of T_k(a*x + b) in powers of x.
        T_k(z) is defined by recurrence T_0=1, T_1=z, T_{k+1}=2zT_k - T_{k-1}.
        """
        if k == 0:
            return np.array([1.0])
        if k == 1:
            return np.array([b, a])
        
        # We need polynomials Q_{k-1}(x) and Q_{k-2}(x)
        # Q_k(x) = 2 * (a*x + b) * Q_{k-1}(x) - Q_{k-2}(x)
        
        # To avoid recursion depth issues or repeated work, we can iterate.
        # But for k=5, recursion is fine.
        
        Q_k_minus_1 = cheb_to_monomial(k-1, a, b)
        Q_k_minus_2 = cheb_to_monomial(k-2, a, b)
        
        # Multiply Q_{k-1} by (a*x + b)
        # Let Q_{k-1} = sum c_j x^j
        # (a*x + b) * Q_{k-1} = a * sum c_j x^{j+1} + b * sum c_j x^j
        # = sum (b*c_j) x^j + sum (a*c_j) x^{j+1}
        
        len_q1 = len(Q_k_minus_1)
        prod = np.zeros(len_q1 + 1)
        prod[:len_q1] += b * Q_k_minus_1
        prod[1:] += a * Q_k_minus_1
        
        # Multiply by 2
        prod *= 2.0
        
        # Subtract Q_{k-2}
        # Q_{k-2} has length k-1. prod has length k+1.
        prod[:len(Q_k_minus_2)] -= Q_k_minus_2
        
        return prod

    coeff_monom = np.zeros(6)
    
    for k in range(6):
        c_k = coeff_cheb_full[k]
        if abs(c_k) < 1e-15:
            continue
        expansion = cheb_to_monomial(k, a, b)
        coeff_monom[:len(expansion)] += c_k * expansion
        
    if not np.all(np.isfinite(coeff_monom)):
        # Fallback to simple mean if something went wrong
        return np.array([np.mean(y), 0.0, 0.0, 0.0, 0.0, 0.0])

    return coeff_monom

# EVOLVE-BLOCK-END
