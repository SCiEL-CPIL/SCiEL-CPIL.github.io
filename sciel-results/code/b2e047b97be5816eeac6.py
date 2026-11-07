import numpy as np
import time

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin before 300s limit

    # Extract 1D input
    x_train = X_train[:, 0]
    y_train = y_train.flatten()

    # Normalize x to [-1, 1] for Chebyshev fitting
    # Using training data min/max to define the mapping
    x_min = np.min(x_train)
    x_max = np.max(x_train)
    if x_max == x_min:
        x_max = x_min + 1e-6
    
    def map_to_cheb(x):
        return (2 * (x - x_min) / (x_max - x_min)) - 1

    x_cheb_train = map_to_cheb(x_train)

    # Try increasing degrees of Chebyshev polynomial until convergence or deadline
    # Chebyshev polynomials of the first kind
    best_w = None
    best_mse = np.inf
    best_degree = 0

    # We can compute Chebyshev coefficients using the discrete Chebyshev transform
    # or simply construct the Vandermonde matrix.
    # For N=200 and D=20, constructing the matrix is fast.
    
    max_degree = 30  # Upper bound for search
    
    # Pre-compute cosines for Chebyshev evaluation? 
    # T_n(x) = cos(n * arccos(x))
    # It's more stable to use the recurrence relation or np.polynomial.chebyshev.chebvander
    
    for D in range(1, max_degree + 1):
        if time.time() > deadline:
            break
            
        try:
            # Construct Chebyshev Vandermonde matrix
            # V[:, j] = T_j(x)
            # Using numpy's chebvander for efficiency and stability
            V = np.polynomial.chebyshev.chebvander(x_cheb_train, D)
            
            # Solve least squares
            w, residuals, rank, sv = np.linalg.lstsq(V, y_train, rcond=None)
            
            # Calculate training MSE to check convergence (optional, since noiseless, 
            # but helps detect overfitting if we were to use regularization, 
            # here we just want to see if adding more terms helps or if it's flat)
            # Actually, with noiseless data, we just want the best fit.
            # Since it's symbolic regression, the true function might be low degree.
            # We can stop if the improvement in MSE is negligible.
            
            pred = V @ w
            mse = np.mean((pred - y_train) ** 2)
            
            if mse < best_mse:
                best_mse = mse
                best_w = w
                best_degree = D
            
            # Heuristic: If MSE is very small, we likely found the function
            if mse < 1e-10:
                break
                
        except Exception:
            continue

    # If no degree worked, use degree 1 (linear)
    if best_w is None:
        x_mean = np.mean(x_train)
        x_var = np.var(x_train)
        if x_var > 0:
            slope = np.dot(x_train - x_mean, y_train - np.mean(y_train)) / x_var
            intercept = np.mean(y_train) - slope * x_mean
            best_w = [intercept, slope] # Linear in original space? 
            # No, let's stick to Chebyshev degree 1 in mapped space
            V = np.polynomial.chebyshev.chebvander(x_cheb_train, 1)
            best_w, *_ = np.linalg.lstsq(V, y_train, rcond=None)
            best_degree = 1
        else:
            best_w = [np.mean(y_train)]
            best_degree = 0

    final_degree = best_degree
    final_w = best_w

    def predict(X):
        x = X[:, 0]
        x_cheb = map_to_cheb(x)
        V = np.polynomial.chebyshev.chebvander(x_cheb, final_degree)
        return V @ final_w

    return predict
