import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = start_time + 145.0 # 5s safety margin

    x = np.asarray(training_x, dtype=np.float64)
    t = np.asarray(training_subgrid_tendency, dtype=np.float64)
    
    # Ensure 1D
    x = x.flatten()
    t = t.flatten()
    
    if len(x) == 0:
        return np.zeros(6)
    
    # Degree 5 polynomial
    degree = 5
    
    # Construct Vandermonde matrix
    # V[:, i] = x**i
    # Using np.vander is convenient but we can also use polynomial.polynomial.polyvander
    V = np.vander(x, N=degree+1, increasing=True)
    
    # We want to find c such that V @ c approx t
    # Regularized least squares (Ridge)
    # Minimize ||V c - t||^2 + lambda ||c||^2
    
    # Try a range of lambda values
    # Log-spaced from 1e-6 to 1e1
    lambdas = np.logspace(-6, 1, 14)
    
    best_coeff = None
    best_mse = np.inf
    
    for lam in lambdas:
        if time.time() > deadline:
            break
            
        # Solve (V^T V + lambda I) c = V^T t
        # Using lstsq on augmented matrix is numerically stable
        # Augmented A = [V; sqrt(lam)*I]
        # Augmented b = [t; 0]
        
        # Efficient way:
        # c = (V^T V + lambda I)^-1 V^T t
        # We can use np.linalg.lstsq on the augmented system for stability
        
        sqrt_lam = np.sqrt(lam)
        # Create identity matrix of size (degree+1)
        I = np.eye(degree + 1)
        
        # Augmented matrix
        # A_aug = vstack([V, sqrt_lam * I])
        # b_aug = vstack([t, zeros(degree+1)])
        
        # To save memory/time for large N, we can compute V^T V and V^T t
        # V^T V is (degree+1) x (degree+1)
        # V^T t is (degree+1) x 1
        
        VtV = V.T @ V
        Vtt = V.T @ t
        
        # Regularized system
        A = VtV + sqrt_lam**2 * I # lambda * I
        b = Vtt
        
        # Solve linear system
        try:
            c = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            continue
            
        # Check for finite values
        if not np.all(np.isfinite(c)):
            continue
            
        # Calculate training MSE to select best lambda
        # pred = V @ c
        # mse = mean((pred - t)**2)
        # For large N, this might be slow if done for every lambda.
        # However, N is likely manageable.
        
        # Optimization: Only compute full MSE if the lambda is promising?
        # Or just compute it.
        
        pred = V @ c
        mse = np.mean((pred - t)**2)
        
        if mse < best_mse:
            best_mse = mse
            best_coeff = c.copy()

    if best_coeff is None:
        # Fallback to linear fit if something went wrong
        try:
            coeff = np.polynomial.polynomial.polyfit(x, t, 1)
            # Pad to degree 5
            full_coeff = np.zeros(6)
            full_coeff[:len(coeff)] = coeff
            return full_coeff
        except Exception:
            return np.zeros(6)

    return best_coeff
