"""Strategy: Orthonormal Legendre basis fit with Tikhonov regularization, avoiding ill-conditioned monomial basis and expensive smoothing."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
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
    
    # 1. Determine the range of x to scale Legendre polynomials to [-1, 1]
    # Using a slightly expanded range to avoid boundary issues, or just min/max
    x_min = np.min(x)
    x_max = np.max(x)
    
    if x_max - x_min < 1e-10:
        # Degenerate case, return zeros or simple mean
        return np.array([np.mean(y), 0, 0, 0, 0, 0])
        
    # Map x to t in [-1, 1]
    # t = (2*x - (x_max + x_min)) / (x_max - x_min)
    # Inverse: x = (t + 1) * (x_max - x_min) / 2 + x_min
    #         x = t * (x_max - x_min) / 2 + (x_min + x_max) / 2
    # Let a = (x_max - x_min) / 2, b = (x_min + x_max) / 2
    a = (x_max - x_min) / 2.0
    b = (x_min + x_max) / 2.0
    
    t = (x - b) / a
    
    # 2. Construct Legendre design matrix P where P[i, j] = P_j(t_i)
    # P_j are standard Legendre polynomials on [-1, 1]
    # We need degree 5, so j=0..5
    # P_0(t) = 1
    # P_1(t) = t
    # P_2(t) = 0.5 * (3*t^2 - 1)
    # P_3(t) = 0.5 * (5*t^3 - 3*t)
    # P_4(t) = (1/8) * (35*t^4 - 30*t^2 + 3)
    # P_5(t) = (1/8) * (63*t^5 - 70*t^3 + 15*t)
    
    t2 = t * t
    t3 = t2 * t
    t4 = t2 * t2
    t5 = t4 * t
    
    P = np.column_stack((
        np.ones_like(t),
        t,
        0.5 * (3 * t2 - 1),
        0.5 * (5 * t3 - 3 * t),
        0.25 * (35 * t4 - 30 * t2 + 3),
        0.125 * (63 * t5 - 70 * t3 + 15 * t)
    ))
    
    # 3. Solve for Legendre coefficients alpha such that y ≈ P @ alpha
    # Use Ridge regression (Tikhonov) to stabilize
    # Regularization strength: small, to allow flexibility but prevent noise fitting
    # Since Legendre basis is orthonormal over continuous [-1,1], 
    # the normal equations P.T @ P are close to identity if data is uniform.
    # With non-uniform data, they are still well-conditioned compared to Vandermonde.
    
    # Let's try a few small lambda values quickly. 
    # Given time constraints, we can pick a sensible fixed lambda or do a very fast CV.
    # The prompt suggests avoiding micro-tuning, but selecting a robust lambda is key.
    # Let's use a simple heuristic or a small grid.
    
    # Calculate P.T @ P and P.T @ y
    PT = P.T
    G = PT @ P
    b_vec = PT @ y
    
    # Add small ridge penalty to higher order terms? Or uniform?
    # Uniform ridge is standard.
    # Let's try lambda = 1e-3 * trace(G)/6
    trace_G = np.trace(G)
    base_lambda = trace_G / 6.0 if trace_G > 0 else 1.0
    
    # Try a small set of lambdas
    lambdas = [0.0, 1e-4 * base_lambda, 1e-3 * base_lambda, 1e-2 * base_lambda]
    
    best_alpha = None
    best_mse = np.inf
    
    # Use a subset for fast MSE check if n is large
    check_size = min(n, 5000)
    check_idx = np.random.RandomState(42).choice(n, size=check_size, replace=False) if n > check_size else np.arange(n)
    y_check = y[check_idx]
    P_check = P[check_idx, :]
    
    for lam in lambdas:
        if time.time() > deadline - 10:
            break
        try:
            A = G + lam * np.eye(6)
            alpha = np.linalg.solve(A, b_vec)
            
            if not np.all(np.isfinite(alpha)):
                continue
                
            y_pred = P_check @ alpha
            mse = np.mean((y_check - y_pred) ** 2)
            
            if mse < best_mse:
                best_mse = mse
                best_alpha = alpha
        except Exception:
            continue
    
    if best_alpha is None:
        # Fallback to simple least squares with very small ridge
        try:
            alpha = np.linalg.lstsq(P, y, rcond=None)[0]
            best_alpha = alpha
        except Exception:
            return np.zeros(6)
            
    # 4. Transform Legendre coefficients back to monomial coefficients in x
    # We have y ≈ sum_{k=0}^5 alpha_k * P_k(t)
    # where t = (x - b) / a
    # We need y ≈ sum_{j=0}^5 c_j * x^j
    
    # First, express P_k(t) in terms of x.
    # P_k(t) is a polynomial in t of degree k.
    # t = (x - b) / a = (1/a)*x - b/a
    # Let u = 1/a, v = -b/a. t = u*x + v
    
    # We can build the monomial coefficients for each P_k(t(x)) and combine.
    
    # Coefficients of P_k(t) in powers of t:
    # P_0(t) = 1
    # P_1(t) = t
    # P_2(t) = 1.5*t^2 - 0.5
    # P_3(t) = 2.5*t^3 - 1.5*t
    # P_4(t) = 4.375*t^4 - 3.75*t^2 + 0.375
    # P_5(t) = 7.875*t^5 - 8.75*t^3 + 1.875*t
    
    # Let's define the coefficients of P_k(t) as arrays p_coeff[k] where index is power of t
    p_coeff = np.zeros((6, 6)) # p_coeff[k][j] is coeff of t^j in P_k(t)
    
    p_coeff[0, 0] = 1.0
    p_coeff[1, 1] = 1.0
    p_coeff[2, 0] = -0.5
    p_coeff[2, 2] = 1.5
    p_coeff[3, 1] = -1.5
    p_coeff[3, 3] = 2.5
    p_coeff[4, 0] = 0.375
    p_coeff[4, 2] = -3.75
    p_coeff[4, 4] = 4.375
    p_coeff[5, 1] = 1.875
    p_coeff[5, 3] = -8.75
    p_coeff[5, 5] = 7.875
    
    # Now, t = u*x + v
    # We need to substitute t = u*x + v into each P_k(t) and expand to powers of x.
    # This is equivalent to multiplying by the transformation matrix.
    
    # Let's do it by iterating through terms.
    # For each k, we have sum_j p_coeff[k][j] * t^j
    # t^j = (u*x + v)^j = sum_{m=0}^j binom(j, m) * (u*x)^m * v^(j-m)
    #     = sum_{m=0}^j binom(j, m) * u^m * v^(j-m) * x^m
    
    # So the coefficient of x^m in P_k(t(x)) is:
    # sum_{j=m}^k p_coeff[k][j] * binom(j, m) * u^m * v^(j-m)
    
    # Total coefficient c_m = sum_{k=0}^5 alpha_k * (coeff of x^m in P_k(t(x)))
    
    c_monomial = np.zeros(6)
    
    u = 1.0 / a
    v = -b / a
    
    for k in range(6):
        if abs(alpha[k]) < 1e-15:
            continue
        for m in range(6):
            coeff_x_m = 0.0
            for j in range(m, 6):
                if abs(p_coeff[k][j]) < 1e-15:
                    continue
                # Binomial(j, m)
                binom_val = 1.0
                # Calculate C(j, m)
                # C(j, m) = j! / (m! (j-m)!)
                # Simple loop for small numbers
                nn = j
                kk = min(m, j-m)
                binom_val = 1.0
                for ii in range(kk):
                    binom_val *= (nn - ii) / (ii + 1)
                    
                term = p_coeff[k][j] * binom_val * (u ** m) * (v ** (j - m))
                coeff_x_m += term
            c_monomial[m] += alpha[k] * coeff_x_m
            
    if not np.all(np.isfinite(c_monomial)):
        return np.zeros(6)
        
    return c_monomial
# EVOLVE-BLOCK-END
