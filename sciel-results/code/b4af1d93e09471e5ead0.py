<<<<<<< SEARCH
import numpy as np
from scipy.optimize import minimize
from numpy.polynomial.hermite import herm2poly
import time

def construct():
    start_time = time.time()
    deadline = start_time + 170.0 # Safety margin
    
    # Coefficients of Hermite polynomials H0, H4, H8, H12 in standard basis
    # H_n(x) are physicists' Hermite polynomials
    # herm2poly converts to standard power basis
    h0 = herm2poly([1])
    h4 = herm2poly([0, 0, 0, 0, 1])
    h8 = herm2poly([0, 0, 0, 0, 0, 0, 0, 0, 1])
    h12 = herm2poly([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    
    # Values at x=0
    h0_0 = np.polyval(h0, 0)
    h4_0 = np.polyval(h4, 0)
    h8_0 = np.polyval(h8, 0)
    h12_0 = np.polyval(h12, 0)
    
    def get_c12(c0, c4, c8):
        # P(0) = c0*H0(0) + c4*H4(0) + c8*H8(0) + c12*H12(0) = 0
        return -(c0 * h0_0 + c4 * h4_0 + c8 * h8_0) / h12_0

    def compute_integrals(c0, c4, c8):
        c12 = get_c12(c0, c4, c8)
        
        # Construct P(x) coefficients
        # P(x) = c0*H0 + c4*H4 + c8*H8 + c12*H12
        # herm2poly returns coefficients [a0, a1, ...] for a0 + a1*x + ...
        p_coeff = c0 * h0 + c4 * h4 + c8 * h8 + c12 * h12
        
        # Compute P(x)^2
        p_sq_coeff = np.convolve(p_coeff, p_coeff)
        
        # Compute Q(x) = P''(x) - 2x P'(x) + (x^2 - 1) P(x)
        # P'(x)
        p_prime_coeff = np.polynomial.polynomial.polyder(p_coeff)
        p_double_prime_coeff = np.polynomial.polynomial.polyder(p_prime_coeff)
        
        # -2x P'(x)
        term1 = -2 * np.concatenate([[0], p_prime_coeff])
        
        # (x^2 - 1) P(x)
        term2 = np.convolve(p_coeff, [1, 0, 1]) # (x^2 - 1) * P(x)
        # Wait, (x^2 - 1) is [ -1, 0, 1 ] in poly1d? No, np.convolve uses standard basis.
        # P(x) = sum p_i x^i.
        # x^2 P(x) = sum p_i x^{i+2}. Coeffs: [0, 0, p_0, p_1, ...]
        # -1 P(x) = sum -p_i x^i.
        # So term2 = shift(p_coeff, 2) - p_coeff
        term2 = np.zeros(len(p_coeff) + 2)
        term2[2:] = p_coeff
        term2 -= p_coeff
        
        q_coeff = p_double_prime_coeff + term1 + term2
        # Ensure same length for safety
        max_len = max(len(p_double_prime_coeff), len(term1), len(term2))
        p_double_prime_coeff = np.pad(p_double_prime_coeff, (0, max_len - len(p_double_prime_coeff)))
        term1 = np.pad(term1, (0, max_len - len(term1)))
        term2 = np.pad(term2, (0, max_len - len(term2)))
        q_coeff = p_double_prime_coeff + term1 + term2
        
        # Compute Q(x)^2
        q_sq_coeff = np.convolve(q_coeff, q_coeff)
        
        # Compute integrals
        # I_k = int x^k e^{-x^2} dx
        # I_k = 0 if k odd.
        # I_{2n} = (2n)! / (4^n n!) * sqrt(pi)
        
        def integral_of_poly(coeff):
            # coeff[i] is coefficient of x^i
            total = 0.0
            for i in range(len(coeff)):
                if i % 2 != 0:
                    continue
                n = i // 2
                # (2n)! / (4^n n!)
                # Use gammaln for stability
                log_val = np.math.lgamma(2*n + 1) - n * np.math.log(4) - np.math.lgamma(n + 1)
                val = np.exp(log_val) * np.sqrt(np.pi)
                total += coeff[i] * val
            return total
        
        A = integral_of_poly(p_sq_coeff)
        B = integral_of_poly(np.convolve(p_sq_coeff, [0, 0, 1])) # x^2 P^2
        C = integral_of_poly(q_sq_coeff)
        
        return A, B, C

    def objective(params):
        c4, c8 = params
        c0 = 1.0 # Fix scale
        A, B, C = compute_integrals(c0, c4, c8)
        if A <= 0 or C <= 0:
            return 1e10
        return (B / A) * (C / A)

    # Initial guess from seed
    # Seed: [-113/100, 1/25, 1/3240]
    # c0 = -1.13, c4 = 0.04, c8 = 1/3240
    # Since we fix c0=1, we scale the seed:
    c0_seed = -113 / 100
    c4_seed = 1 / 25
    c8_seed = 1 / 3240
    scale = 1.0 / c0_seed
    c4_init = c4_seed * scale
    c8_init = c8_seed * scale
    
    best_result = None
    best_score = float('inf')
    
    # Try multiple starting points
    starts = [
        [c4_init, c8_init],
        [0.1, 0.01],
        [-0.1, -0.01],
        [0.5, 0.05],
        [-0.5, -0.05],
        [0.0, 0.0]
    ]
    
    # Add some random starts
    np.random.seed(42)
    for _ in range(5):
        starts.append([np.random.uniform(-1, 1), np.random.uniform(-0.1, 0.1)])
        
    for start in starts:
        if time.time() > deadline:
            break
        try:
            res = minimize(objective, start, method='BFGS', options={'gtol': 1e-8, 'maxiter': 1000})
            if res.success and res.fun < best_score:
                best_score = res.fun
                best_result = res.x
        except Exception:
            pass
            
    if best_result is None:
        # Fallback to seed
        return np.array([c0_seed, c4_seed, c8_seed])
        
    # Reconstruct full coefficients
    c4_opt, c8_opt = best_result
    c0_opt = 1.0
    # The problem asks for three coefficients. The scale is arbitrary for the shape, 
    # but the "score" is the bound. The returned coefficients define the function.
    # We return the coefficients that define the optimal shape.
    # We can return [1, c4_opt, c8_opt] or scale it back to the seed's magnitude?
    # The interface says "construct() -> three coefficients".
    # The evaluator likely checks the uncertainty product of the function defined by these coeffs.
    # So any non-zero scalar multiple is equivalent in terms of the product value.
    # We return [1, c4_opt, c8_opt].
    
    return np.array([c0_opt, c4_opt, c8_opt])
=======
import numpy as np
from scipy.optimize import minimize
from numpy.polynomial.hermite import herm2poly
from numpy.polynomial.polynomial import polyder
import time

def _precompute_hermite_coefficients():
    """Precompute Hermite polynomial coefficients in standard power basis."""
    h0 = herm2poly([1])
    h4 = herm2poly([0, 0, 0, 0, 1])
    h8 = herm2poly([0, 0, 0, 0, 0, 0, 0, 0, 1])
    h12 = herm2poly([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    return h0, h4, h8, h12

def _get_c12(c0, c4, c8, h0, h4, h8, h12):
    """Compute c12 from P(0) = 0 constraint."""
    h0_0 = np.polyval(h0, 0)
    h4_0 = np.polyval(h4, 0)
    h8_0 = np.polyval(h8, 0)
    h12_0 = np.polyval(h12, 0)
    return -(c0 * h0_0 + c4 * h4_0 + c8 * h8_0) / h12_0

def _compute_integrals(c0, c4, c8, h0, h4, h8, h12):
    """Compute A, B, C integrals for given coefficients."""
    c12 = _get_c12(c0, c4, c8, h0, h4, h8, h12)
    
    # Construct P(x) coefficients
    p_coeff = c0 * h0 + c4 * h4 + c8 * h8 + c12 * h12
    
    # Compute P(x)^2
    p_sq_coeff = np.convolve(p_coeff, p_coeff)
    
    # Compute Q(x) = P''(x) - 2x P'(x) + (x^2 - 1) P(x)
    p_prime_coeff = polyder(p_coeff)
    p_double_prime_coeff = polyder(p_prime_coeff)
    
    # -2x P'(x)
    term1 = -2 * np.concatenate([[0], p_prime_coeff])
    
    # (x^2 - 1) P(x)
    term2 = np.zeros(len(p_coeff) + 2)
    term2[2:] = p_coeff
    term2 -= p_coeff
    
    # Ensure same length
    max_len = max(len(p_double_prime_coeff), len(term1), len(term2))
    p_double_prime_coeff = np.pad(p_double_prime_coeff, (0, max_len - len(p_double_prime_coeff)))
    term1 = np.pad(term1, (0, max_len - len(term1)))
    term2 = np.pad(term2, (0, max_len - len(term2)))
    
    q_coeff = p_double_prime_coeff + term1 + term2
    
    # Compute Q(x)^2
    q_sq_coeff = np.convolve(q_coeff, q_coeff)
    
    # Compute integrals using I_{2n} = (2n)! / (4^n n!) * sqrt(pi)
    def integral_of_poly(coeff):
        total = 0.0
        for i in range(len(coeff)):
            if i % 2 != 0:
                continue
            n = i // 2
            log_val = np.math.lgamma(2*n + 1) - n * np.math.log(4) - np.math.lgamma(n + 1)
            val = np.exp(log_val) * np.sqrt(np.pi)
            total += coeff[i] * val
        return total
    
    A = integral_of_poly(p_sq_coeff)
    B = integral_of_poly(np.convolve(p_sq_coeff, [0, 0, 1]))
    C = integral_of_poly(q_sq_coeff)
    
    return A, B, C

def _objective_and_grad(params):
    """Compute objective and gradient using numerical differentiation."""
    c4, c8 = params
    c0 = 1.0
    A, B, C = _compute_integrals(c0, c4, c8, h0, h4, h8, h12)
    
    if A <= 0 or C <= 0:
        return 1e10, np.array([0.0, 0.0])
    
    obj = (B / A) * (C / A)
    
    # Numerical gradient
    eps = 1e-7
    A1, B1, C1 = _compute_integrals(c0, c4 + eps, c8, h0, h4, h8, h12)
    A2, B2, C2 = _compute_integrals(c0, c4 - eps, c8, h0, h4, h8, h12)
    A3, B3, C3 = _compute_integrals(c0, c4, c8 + eps, h0, h4, h8, h12)
    A4, B4, C4 = _compute_integrals(c0, c4, c8 - eps, h0, h4, h8, h12)
    
    def safe_obj(A, B, C):
        if A <= 0 or C <= 0:
            return 1e10
        return (B / A) * (C / A)
    
    grad_c4 = (safe_obj(A1, B1, C1) - safe_obj(A2, B2, C2)) / (2 * eps)
    grad_c8 = (safe_obj(A3, B3, C3) - safe_obj(A4, B4, C4)) / (2 * eps)
    
    return obj, np.array([grad_c4, grad_c8])

def _solve_stationary_points():
    """
    Solve the stationary point equations symbolically.
    
    The uncertainty product is U = (B/A) * (C/A) where A, B, C are quadratic forms
    in (c4, c8) (with c0 fixed to 1). At a stationary point, the gradient of U = 0.
    
    We can derive polynomial equations in c4 and c8 by setting partial derivatives to zero.
    """
    # Precompute the quadratic form coefficients
    # A = a00 + a01*c4 + a02*c8 + a11*c4^2 + a12*c4*c8 + a22*c8^2
    # Similarly for B and C
    
    def get_quad_coeffs(c0):
        """Get quadratic form coefficients for A, B, C."""
        # Evaluate at several points to extract coefficients
        # A = sum_{i,j} A_{ij} c_i c_j where c = (1, c4, c8)
        
        # Use finite differences to extract coefficients
        # A(c4, c8) = A(0,0) + A4*c4 + A8*c8 + A44*c4^2 + A48*c4*c8 + A88*c8^2
        
        A00, B00, C00 = _compute_integrals(c0, 0, 0, h0, h4, h8, h12)
        A4, B4, C4 = _compute_integrals(c0, 1, 0, h0, h4, h8, h12)
        A8, B8, C8 = _compute_integrals(c0, 0, 1, h0, h4, h8, h12)
        A44, B44, C44 = _compute_integrals(c0, 1, 0, h0, h4, h8, h12)
        A88, B88, C88 = _compute_integrals(c0, 0, 1, h0, h4, h8, h12)
        
        # A(1,1) = A00 + A4 + A8 + A44 + A48 + A88
        A11, B11, C11 = _compute_integrals(c0, 1, 1, h0, h4, h8, h12)
        A48 = A11 - A00 - A4 - A8 - A44 - A88
        B48 = B11 - B00 - B4 - B8 - B44 - B88
        C48 = C11 - C00 - C4 - C8 - C44 - C88
        
        return {
            'A': (A00, A4, A8, A44, A48, A88),
            'B': (B00, B4, B8, B44, B48, B88),
            'C': (C00, C4, C8, C44, C48, C88)
        }
    
    coeffs = get_quad_coeffs(1.0)
    
    # Now solve the stationary equations:
    # dU/dc4 = 0 and dU/dc8 = 0
    # U = (B/A) * (C/A) = B*C / A^2
    # dU/dc4 = (B_c4 * C * A^2 - B * C_c4 * A^2 - 2 * B * C * A * A_c4) / A^4 = 0
    # => B_c4 * C * A - B * C_c4 * A - 2 * B * C * A_c4 = 0
    # => A * (B_c4 * C - B * C_c4) = 2 * B * C * A_c4
    
    # This gives two polynomial equations in c4, c8.
    # We can use resultants to eliminate one variable.
    
    # For now, use numerical optimization from multiple starting points
    # and verify with high precision
    
    best_result = None
    best_score = float('inf')
    
    # Multiple starting points
    starts = [
        [0.04 / (-1.13), 1/3240 / (-1.13)],  # Scaled seed
        [0.1, 0.01],
        [-0.1, -0.01],
        [0.5, 0.05],
        [-0.5, -0.05],
        [0.0, 0.0],
        [0.2, 0.02],
        [-0.2, -0.02],
        [0.05, 0.005],
        [-0.05, -0.005],
    ]
    
    for start in starts:
        try:
            res = minimize(lambda x: _objective_and_grad(x)[0], start, 
                          method='BFGS', options={'gtol': 1e-12, 'maxiter': 2000})
            if res.success and res.fun < best_score:
                best_score = res.fun
                best_result = res.x
        except Exception:
            pass
    
    if best_result is None:
        return None
    
    # High-precision polish
    try:
        res = minimize(lambda x: _objective_and_grad(x)[0], best_result,
                      method='L-BFGS-B', options={'gtol': 1e-14, 'maxiter': 5000})
        if res.success and res.fun < best_score:
            best_score = res.fun
            best_result = res.x
    except Exception:
        pass
    
    return best_result, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 170.0
    
    # Precompute Hermite coefficients
    h0, h4, h8, h12 = _precompute_hermite_coefficients()
    
    # Try symbolic/analytic approach
    result = _solve_stationary_points()
    
    if result is not None:
        c4_opt, c8_opt, best_score = result
        return np.array([1.0, c4_opt, c8_opt])
    
    # Fallback to seed
    c0_seed = -113 / 100
    c4_seed = 1 / 25
    c8_seed = 1 / 3240
    return np.array([c0_seed, c4_seed, c8_seed])
>>>>>>> REPLACE
