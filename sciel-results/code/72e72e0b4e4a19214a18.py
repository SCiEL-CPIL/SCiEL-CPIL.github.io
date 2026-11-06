import numpy as np

def construct(training_x, training_subgrid_tendency):
    x = np.asarray(training_x, dtype=float).flatten()
    t = np.asarray(training_subgrid_tendency, dtype=float).flatten()
    
    mask = np.isfinite(x) & np.isfinite(t)
    x = x[mask]
    t = t[mask]
    
    if len(x) == 0:
        return np.zeros(6)
    
    # 1. Scaling to [-1, 1] with padding for extrapolation robustness
    x_min = np.min(x)
    x_max = np.max(x)
    range_x = x_max - x_min
    if range_x < 1e-10:
        x_min = x_max - 1e-10
        x_max = x_max + 1e-10
        range_x = 2e-10
        
    # Pad by 10% to handle slight extrapolation in validation
    pad = 0.1 * range_x
    x_min_pad = x_min - pad
    x_max_pad = x_max + pad
    
    # z = alpha * x + beta
    alpha = 2.0 / (x_max_pad - x_min_pad)
    beta = -alpha * (x_min_pad + x_max_pad) / 2.0
    
    z = alpha * x + beta
    
    # 2. Construct Chebyshev Design Matrix
    # Degrees 0 to 5
    n_samples = len(x)
    C = np.zeros((n_samples, 6))
    C[:, 0] = 1.0
    if n_samples > 0:
        C[:, 1] = z
        for k in range(2, 6):
            C[:, k] = 2.0 * z * C[:, k-1] - C[:, k-2]
            
    # 3. Solve for Chebyshev coefficients
    A = C.T @ C
    b = C.T @ t
    
    # Small regularization for numerical stability
    mean_diag = np.mean(np.diag(A))
    if mean_diag == 0:
        return np.zeros(6)
        
    lambda_reg = 1e-6 * mean_diag
    I = np.eye(6)
    A_reg = A + lambda_reg * I
    
    try:
        a_cheb = np.linalg.solve(A_reg, b)
    except np.linalg.LinAlgError:
        a_cheb, _, _, _ = np.linalg.lstsq(A_reg, b, rcond=None)
        
    # 4. Convert Chebyshev coefficients to Monomial coefficients
    # We need to express sum_{k=0}^5 a_k T_k(z) as sum_{i=0}^5 c_i x^i
    # where z = alpha * x + beta
    
    # Let P_k(x) be the polynomial in x such that P_k(x) = T_k(alpha*x + beta)
    # We compute the coefficients of P_k(x) recursively.
    # P_0(x) = 1
    # P_1(x) = alpha*x + beta
    # P_k(x) = 2*(alpha*x + beta)*P_{k-1}(x) - P_{k-2}(x)
    
    # coeffs[k] is an array of length 6 representing the polynomial P_k(x)
    # coeffs[k][i] is the coefficient of x^i in P_k(x)
    
    poly_coeffs = np.zeros((6, 6))
    
    # k=0
    poly_coeffs[0, 0] = 1.0
    
    # k=1
    # T_1(z) = z = alpha*x + beta
    poly_coeffs[1, 0] = beta
    poly_coeffs[1, 1] = alpha
    
    # k >= 2
    for k in range(2, 6):
        # P_k(x) = 2 * (alpha*x + beta) * P_{k-1}(x) - P_{k-2}(x)
        # Let Q(x) = (alpha*x + beta) * P_{k-1}(x)
        # Q(x) = alpha * x * P_{k-1}(x) + beta * P_{k-1}(x)
        
        Q = np.zeros(6)
        # alpha * x * P_{k-1}(x)
        # Shift P_{k-1} up by 1 power and multiply by alpha
        Q[1:] = alpha * poly_coeffs[k-1, :-1]
        # beta * P_{k-1}(x)
        Q += beta * poly_coeffs[k-1, :]
        
        # P_k(x) = 2 * Q(x) - P_{k-2}(x)
        poly_coeffs[k, :] = 2.0 * Q - poly_coeffs[k-2, :]
        
    # Final polynomial is sum_{k=0}^5 a_k * P_k(x)
    c = np.zeros(6)
    for k in range(6):
        c += a_cheb[k] * poly_coeffs[k, :]
        
    if not np.all(np.isfinite(c)):
        return np.zeros(6)
        
    return c
