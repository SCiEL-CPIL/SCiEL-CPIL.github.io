"""
Improved program: Centered/Scaled Ridge Regression with K-Fold Cross-Validation.

Strategy:
1. Center and scale the input x to improve numerical stability of the Vandermonde matrix.
2. Perform K-Fold Cross-Validation to select the optimal Ridge regularization parameter (lambda).
3. Fit the final model on all data using the optimal lambda.
4. Transform the coefficients from the scaled/centered basis back to the original x basis.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit (evaluator might have overhead)
    
    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)
    
    n = len(x)
    
    # Sanity checks
    if n < 6:
        return np.zeros(6)
    if np.all(x == x[0]):
        return np.array([np.mean(y)] + [0.0]*5)
    
    # 1. Preprocess: Center and Scale x
    # This helps with the conditioning of the Vandermonde matrix for high-degree polynomials
    mu = np.mean(x)
    sigma = np.std(x)
    if sigma < 1e-12:
        sigma = 1.0
    x_scaled = (x - mu) / sigma
    
    # 2. Build Vandermonde matrix for scaled x
    # V[i, j] = x_scaled[i]^j for j=0..5
    V = np.vander(x_scaled, N=6, increasing=True)
    
    # 3. Select optimal regularization parameter via K-Fold Cross-Validation
    # Ridge Regression: min ||Vc - y||^2 + lambda * ||c||^2
    # Solution: c = (V^T V + lambda I)^-1 V^T y
    
    # Candidate lambda values (log spaced)
    # lambda=0 is least squares.
    # We search a range that covers under-regularized to over-regularized.
    lambdas = np.logspace(-4, 4, 19)  # 1e-4 to 1e4
    
    best_lambda = 0.0
    best_cv_score = np.inf
    
    # K-Fold setup
    k_folds = 5
    indices = np.arange(n)
    # Shuffle indices for better fold distribution? 
    # The data is from trajectories, so it's temporally correlated.
    # Simple random shuffle is standard for CV unless time-series CV is required.
    # Given the "independent validation trajectories" in the task, the training data 
    # might be mixed from two trajectories. Random shuffle is a reasonable assumption 
    # for the internal CV split to estimate generalization error.
    rng = np.random.default_rng(42)
    rng.shuffle(indices)
    folds = np.array_split(indices, k_folds)
    
    # Precompute V^T V for efficiency? 
    # V is n x 6. V^T V is 6x6. This is cheap.
    VtV = V.T @ V
    
    for lam in lambdas:
        if time.time() > deadline - 20:
            break
            
        total_error = 0.0
        valid_folds = 0
        
        for i in range(k_folds):
            val_idx = folds[i]
            trn_idx = np.concatenate([folds[j] for j in range(k_folds) if j != i])
            
            V_trn = V[trn_idx, :]
            y_trn = y[trn_idx]
            y_val = y[val_idx]
            V_val = V[val_idx, :]
            
            # Solve Ridge
            # (V_trn^T V_trn + lam * I) c = V_trn^T y_trn
            # Note: V_trn^T V_trn is not exactly VtV if we don't use all data.
            # For small n, computing V_trn.T @ V_trn is cheap.
            A = V_trn.T @ V_trn + lam * np.eye(6)
            b = V_trn.T @ y_trn
            
            try:
                c = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                continue
            
            if not np.all(np.isfinite(c)):
                continue
                
            pred = V_val @ c
            mse = np.mean((pred - y_val) ** 2)
            total_error += mse
            valid_folds += 1
            
        if valid_folds > 0:
            avg_mse = total_error / valid_folds
            if avg_mse < best_cv_score:
                best_cv_score = avg_mse
                best_lambda = lam
                
    # 4. Final Fit on all data with best_lambda
    A_final = VtV + best_lambda * np.eye(6)
    b_final = V.T @ y
    
    try:
        c_scaled = np.linalg.solve(A_final, b_final)
    except np.linalg.LinAlgError:
        # Fallback to least squares if solve fails (unlikely with lambda > 0)
        try:
            c_scaled = np.linalg.lstsq(V, y, rcond=None)[0]
        except:
            return np.zeros(6)
            
    if not np.all(np.isfinite(c_scaled)):
        return np.zeros(6)
        
    # 5. Transform coefficients from scaled basis to original basis
    # We have P(x) = sum_{i=0}^5 c_scaled[i] * ((x-mu)/sigma)^i
    # We want Q(x) = sum_{j=0}^5 c_orig[j] * x^j
    # Let z = (x-mu)/sigma. Then x = sigma*z + mu.
    # P(x) = sum_{i=0}^5 c_scaled[i] * z^i
    # We need to express z^i in terms of x.
    # z = (1/sigma)*x - (mu/sigma)
    # This is a change of basis.
    # Let T be the transformation matrix such that c_orig = T @ c_scaled
    # Alternatively, we can just evaluate the polynomial at specific points to find c_orig?
    # No, we need the coefficients.
    # Let's use the property of linear combinations.
    # z^0 = 1
    # z^1 = x/sigma - mu/sigma
    # z^2 = (x/sigma - mu/sigma)^2 = x^2/sigma^2 - 2*mu*x/sigma^2 + mu^2/sigma^2
    # ...
    # This is effectively expanding ( (x-mu)/sigma )^i.
    
    # We can construct the transformation matrix M where M[i, j] is the coefficient of x^j in z^i.
    # z^i = ( (1/sigma)*x - (mu/sigma) )^i
    # Using binomial expansion:
    # (A x + B)^i = sum_{j=0}^i binom(i, j) A^j B^{i-j} x^j
    # Here A = 1/sigma, B = -mu/sigma.
    
    M = np.zeros((6, 6))
    for i in range(6):
        for j in range(i + 1):
            # Coefficient of x^j in z^i
            # Term is binom(i, j) * (1/sigma)^j * (-mu/sigma)^(i-j)
            # Wait, (A x + B)^i = sum_{k=0}^i binom(i, k) (A x)^k B^{i-k}
            # So coefficient of x^k is binom(i, k) A^k B^{i-k}
            # Here k corresponds to j in our loop (power of x)
            coeff_val = np.math.comb(i, j) * (1.0/sigma)**j * (-mu/sigma)**(i-j)
            M[i, j] = coeff_val
            
    # c_scaled is a vector of length 6 (coeffs for z^0 ... z^5)
    # c_orig[j] = sum_{i=j}^5 M[i, j] * c_scaled[i]
    # This is equivalent to c_orig = M.T @ c_scaled
    
    c_orig = M.T @ c_scaled
    
    # Final safety check
    if not np.all(np.isfinite(c_orig)):
        # Fallback to unregularized LS in original space if something went wrong
        V_orig = np.vander(x, N=6, increasing=True)
        try:
            c_orig = np.linalg.lstsq(V_orig, y, rcond=None)[0]
        except:
            return np.zeros(6)
            
    # Clip extremely large values
    max_mag = np.max(np.abs(c_orig))
    if max_mag > 1e6:
        c_orig = np.clip(c_orig, -1e6, 1e6)
        
    return c_orig
# EVOLVE-BLOCK-END
