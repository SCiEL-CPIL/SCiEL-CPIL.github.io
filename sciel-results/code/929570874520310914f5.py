import numpy as np
import time
from scipy.optimize import minimize

def _compute_entropies(P):
    """
    Computes D = H(x-y) and M = max(H(x), H(x+y), H(x+2y)) for a 10x10 grid P[y, x].
    """
    n = 10
    # Flatten P to get probabilities for each (y, x)
    # P[y, x] corresponds to y in 0..9, x in 0..9
    
    # Projection x - y
    # Values range from -9 to 9
    # We can use a binning approach or just sum over indices
    # For efficiency and precision, we'll iterate or use vectorized sums
    
    # Vectorized approach for projections
    ys = np.arange(n)
    xs = np.arange(n)
    
    # Create meshgrid
    Y, X = np.meshgrid(ys, xs, indexing='ij')
    
    # Flatten
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    P_flat = P.flatten()
    
    # D: distribution of x - y
    val_d = X_flat - Y_flat
    # Sum probabilities for each unique value
    # Values are in [-9, 9]
    min_val = -9
    max_val = 9
    counts_d = np.zeros(max_val - min_val + 1)
    for v in range(min_val, max_val + 1):
        mask = (val_d == v)
        counts_d[v - min_val] = np.sum(P_flat[mask])
    
    # H0: distribution of x
    val_h0 = X_flat
    counts_h0 = np.zeros(n)
    for v in range(n):
        mask = (val_h0 == v)
        counts_h0[v] = np.sum(P_flat[mask])
        
    # H1: distribution of x + y
    val_h1 = X_flat + Y_flat
    # Values in [0, 18]
    counts_h1 = np.zeros(2*n - 1)
    for v in range(2*n - 1):
        mask = (val_h1 == v)
        counts_h1[v] = np.sum(P_flat[mask])
        
    # H2: distribution of x + 2y
    val_h2 = X_flat + 2*Y_flat
    # Values in [0, 9 + 18 = 27]
    max_val_h2 = 9 + 2*9
    counts_h2 = np.zeros(max_val_h2 + 1)
    for v in range(max_val_h2 + 1):
        mask = (val_h2 == v)
        counts_h2[v] = np.sum(P_flat[mask])

    def shannon_entropy(counts):
        # Filter out zeros
        p = counts[counts > 1e-15]
        if len(p) == 0:
            return 0.0
        # Normalize just in case of floating point errors, though P sums to 1
        p = p / np.sum(p)
        return -np.sum(p * np.log2(p))
    
    D = shannon_entropy(counts_d)
    H0 = shannon_entropy(counts_h0)
    H1 = shannon_entropy(counts_h1)
    H2 = shannon_entropy(counts_h2)
    
    M = max(H0, H1, H2)
    
    return D, M

def _soft_max(values, temperature=0.01):
    """
    Approximate max using log-sum-exp for smoothness.
    values: list of entropies
    """
    # If temperature is very small, it approaches max
    # log(sum(exp(v_i/T))) * T
    # To avoid overflow, subtract max
    v_max = np.max(values)
    if v_max < 1e-15:
        return 0.0
    exps = np.exp((np.array(values) - v_max) / temperature)
    return temperature * (np.log(np.sum(exps)) + v_max/temperature)

def _objective(W):
    """
    Objective function for minimization.
    W is a 10x10 matrix of parameters.
    P = softmax(W)
    Minimizes -log(D/M) = log(M) - log(D)
    """
    # Softmax parameterization
    # Shift to prevent overflow
    W_shifted = W - np.max(W)
    P_flat = np.exp(W_shifted)
    P_flat /= np.sum(P_flat)
    P = P_flat.reshape(10, 10)
    
    D, M = _compute_entropies(P)
    
    if D < 1e-9 or M < 1e-9:
        return 1e10 # Large penalty
    
    # Use soft max for M to make it differentiable everywhere
    # We need H0, H1, H2 separately for soft max
    # Re-calculate or modify _compute_entropies to return individual Hs
    # For speed in objective, we can just use the hard max if the optimizer is robust,
    # but soft max is safer. Let's compute individual Hs.
    
    n = 10
    ys = np.arange(n)
    xs = np.arange(n)
    Y, X = np.meshgrid(ys, xs, indexing='ij')
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    P_flat = P.flatten()
    
    val_d = X_flat - Y_flat
    min_val = -9
    max_val = 9
    counts_d = np.zeros(max_val - min_val + 1)
    for v in range(min_val, max_val + 1):
        mask = (val_d == v)
        counts_d[v - min_val] = np.sum(P_flat[mask])
        
    val_h0 = X_flat
    counts_h0 = np.zeros(n)
    for v in range(n):
        mask = (val_h0 == v)
        counts_h0[v] = np.sum(P_flat[mask])
        
    val_h1 = X_flat + Y_flat
    counts_h1 = np.zeros(2*n - 1)
    for v in range(2*n - 1):
        mask = (val_h1 == v)
        counts_h1[v] = np.sum(P_flat[mask])
        
    val_h2 = X_flat + 2*Y_flat
    max_val_h2 = 9 + 2*9
    counts_h2 = np.zeros(max_val_h2 + 1)
    for v in range(max_val_h2 + 1):
        mask = (val_h2 == v)
        counts_h2[v] = np.sum(P_flat[mask])

    def shannon_entropy(counts):
        p = counts[counts > 1e-15]
        if len(p) == 0:
            return 0.0
        p = p / np.sum(p)
        return -np.sum(p * np.log2(p))
    
    D_val = shannon_entropy(counts_d)
    H0_val = shannon_entropy(counts_h0)
    H1_val = shannon_entropy(counts_h1)
    H2_val = shannon_entropy(counts_h2)
    
    if D_val < 1e-9:
        return 1e10
        
    # Soft max of H0, H1, H2
    M_soft = _soft_max([H0_val, H1_val, H2_val], temperature=0.05)
    
    if M_soft < 1e-9:
        return 1e10
        
    # We want to maximize D / M, so minimize log(M) - log(D)
    return np.log(M_soft) - np.log(D_val)

def arithmetic_kakeya_distribution() -> np.ndarray:
    start_time = time.time()
    timeout = 600.0 # 600s limit, keep safety margin
    best_P = np.ones((10, 10)) / 100.0
    best_score = 0.0
    
    # Initial guesses
    # 1. Uniform
    # 2. Diagonal heavy (favoring x-y spread?)
    # 3. Random
    
    # Let's try a few random restarts
    rng = np.random.default_rng(42)
    
    n_restarts = 20 # Adjust based on time
    
    for i in range(n_restarts):
        if time.time() - start_time > timeout - 30:
            break
            
        # Initialize W
        if i == 0:
            W = np.zeros((10, 10))
        else:
            # Random perturbation or different structure
            # Try to encourage diagonal structures
            W = rng.normal(0, 1, (10, 10))
            # Add a diagonal bias
            diag = np.eye(10) * 2.0
            W += diag
            
        # Optimize
        try:
            res = minimize(_objective, W, method='L-BFGS-B', 
                           options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
            
            if res.success or res.fun < 1e9:
                W_opt = res.x
                W_shifted = W_opt - np.max(W_opt)
                P_flat = np.exp(W_shifted)
                P_flat /= np.sum(P_flat)
                P = P_flat.reshape(10, 10)
                
                # Calculate actual score
                D, M = _compute_entropies(P)
                if M > 1e-9 and D > 1e-9:
                    score = D / M
                    if score > best_score:
                        best_score = score
                        best_P = P.copy()
                        
        except Exception as e:
            continue
            
    return best_P
