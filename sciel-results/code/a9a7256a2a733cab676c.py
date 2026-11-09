import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Fits a polynomial series using Bernstein basis functions.
    Input x is in [-2, 2]. We scale to u in [0, 1].
    params contains 10 coefficients for B_0(u) through B_9(u).
    """
    # Scale x from [-2, 2] to [0, 1]
    u = 0.5 * (x / 2.0 + 1.0)
    
    # Compute Bernstein polynomials B_i(u) for n=9
    # B_i(u) = C(9, i) * u^i * (1-u)^(9-i)
    
    # Precompute binomial coefficients for n=9
    # C(9, 0) = 1
    # C(9, 1) = 9
    # C(9, 2) = 36
    # C(9, 3) = 84
    # C(9, 4) = 126
    # C(9, 5) = 126
    # C(9, 6) = 84
    # C(9, 7) = 36
    # C(9, 8) = 9
    # C(9, 9) = 1
    
    coeffs = np.array([1, 9, 36, 84, 126, 126, 84, 36, 9, 1])
    
    result = np.zeros_like(x)
    
    # Compute powers of u and (1-u)
    # To avoid repeated power calculations, we can iterate
    # B_i(u) = coeffs[i] * u^i * (1-u)^(9-i)
    
    # Let's compute u^k and (1-u)^k
    # Or just loop and update.
    
    # Efficient way:
    # B_0 = (1-u)^9
    # B_i = B_{i-1} * (u / (1-u)) * (9-i+1)/i ... this has division by zero risks.
    
    # Direct calculation is safer for n=9.
    u_powers = np.ones_like(u)
    one_minus_u_powers = np.ones_like(u)
    
    # Precompute powers of u
    u_pow = np.ones_like(u)
    for i in range(1, 10):
        u_pow = u_pow * u
        # Store if needed, or just use in loop
    
    # Precompute powers of (1-u)
    v = 1.0 - u
    v_pow = np.ones_like(u)
    for i in range(1, 10):
        v_pow = v_pow * v
        
    # Now B_i(u) = coeffs[i] * u^i * v^(9-i)
    # We need u^i and v^(9-i)
    
    # Let's just compute the terms directly in the loop
    # Term i: coeffs[i] * (u ** i) * ((1-u) ** (9-i))
    
    # To be efficient and stable:
    # We can compute the ratio recursively if we handle 0 and 1 carefully.
    # But n is small. Direct power is fine.
    
    for i in range(10):
        if i == 0:
            term = (1.0 - u) ** 9
        elif i == 9:
            term = u ** 9
        else:
            term = coeffs[i] * (u ** i) * ((1.0 - u) ** (9 - i))
        
        result += params[i] * term
        
    return result
