<<<<<<< SEARCH
def _fit_centered(x, y, degree, alpha):
    """
    Fit a degree-`degree` polynomial in centered/scaled x, with ridge regularization.
    Returns coefficients in the original x basis (ascending powers).
    """
    n = len(x)
    if n == 0:
        return np.zeros(degree + 1)
    
    mu = np.mean(x)
    sigma = np.std(x)
    if sigma < 1e-12:
        sigma = 1.0
    
    xc = (x - mu) / sigma
    
    # Build Vandermonde matrix for centered/scaled variable
    V = np.column_stack([xc**k for k in range(degree + 1)])
    
    A = V.T @ V
    b = V.T @ y
    
    if alpha > 0:
        reg = alpha * np.eye(A.shape[0])
        # Small diagonal for numerical safety
        reg[0, 0] += 1e-12
        try:
            c_centered = np.linalg.solve(A + reg, b)
        except np.linalg.LinAlgError:
            # Fallback to lstsq if solve fails
            c_centered, _, _, _ = np.linalg.lstsq(A + reg, b, rcond=None)
    else:
        try:
            c_centered = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            c_centered, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    
    # Convert from centered/scaled basis to original basis
    # p(x) = sum_{k=0}^{degree} c_k * ((x-mu)/sigma)^k
    # Let z = (x-mu)/sigma => x = sigma*z + mu
    # p(x) = sum c_k * z^k
    # We need to express this in powers of x.
    
    coeff_orig = np.zeros(degree + 1)
    a = 1.0 / sigma
    b_val = -mu / sigma
    
    for k in range(degree + 1):
        if c_centered[k] == 0:
            continue
        # Expand (a*x + b_val)^k
        # Coefficients of (a*x + b_val)^k are binom(k, i) * a^i * b_val^(k-i) for x^i
        for i in range(k + 1):
            binom_k_i = comb(k, i)
            term_coeff = binom_k_i * (a ** i) * (b_val ** (k - i))
            coeff_orig[i] += c_centered[k] * term_coeff
            
    return coeff_orig
=======
def _fit_centered(x, y, degree, alpha):
    """
    Fit a degree-`degree` polynomial in centered/scaled x, with ridge regularization.
    Returns coefficients in the original x basis (ascending powers).
    """
    n = len(x)
    if n == 0:
        return np.zeros(degree + 1)
    
    mu = np.mean(x)
    sigma = np.std(x)
    if sigma < 1e-12:
        sigma = 1.0
    
    xc = (x - mu) / sigma
    
    # Build Vandermonde matrix for centered/scaled variable
    V = np.column_stack([xc**k for k in range(degree + 1)])
    
    A = V.T @ V
    b = V.T @ y
    
    if alpha > 0:
        reg = alpha * np.eye(A.shape[0])
        # Small diagonal for numerical safety
        reg[0, 0] += 1e-12
        try:
            c_centered = np.linalg.solve(A + reg, b)
        except np.linalg.LinAlgError:
            # Fallback to lstsq if solve fails
            c_centered, _, _, _ = np.linalg.lstsq(A + reg, b, rcond=None)
    else:
        try:
            c_centered = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            c_centered, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            
    # Convert from centered/scaled basis to original basis
    # z = (x - mu) / sigma
    # P(z) = sum c_k z^k
    # We want Q(x) = P((x-mu)/sigma)
    # Let w = x/sigma. Then z = w - mu/sigma.
    # P(w - mu/sigma) is a shift of P by -mu/sigma.
    c_shifted = np.polynomial.polynomial.polyshift(c_centered, -mu/sigma)
    # Now we have R(w) = P(w - mu/sigma).
    # We want R(x/sigma). This is a scale of R by sigma.
    c_orig = np.polynomial.polynomial.polyscale(c_shifted, sigma)
    
    # Ensure length is degree+1, though polyshift/polyscale should preserve it
    if len(c_orig) < degree + 1:
        c_orig = np.pad(c_orig, (0, degree + 1 - len(c_orig)))
        
    return c_orig
>>>>>>> REPLACE

