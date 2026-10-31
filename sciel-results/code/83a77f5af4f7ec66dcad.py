import time
import numpy as np
from scipy.optimize import minimize

DEADLINE = time.time() + 195  # 195s deadline, 15s margin

def _hermite_coeffs(n):
    """Return coefficients of H_n(x) as a list [c_0, c_2, ..., c_n] for even powers only."""
    if n == 0:
        return [1.0]
    if n == 4:
        return [3.0, -6.0, 1.0]  # 3 - 6x^2 + x^4
    if n == 8:
        return [105.0, -420.0, 210.0, -28.0, 1.0]  # 105 - 420x^2 + 210x^4 - 28x^6 + x^8
    if n == 12:
        return [945.0, -7560.0, 13860.0, -6930.0, 1155.0, -66.0, 1.0]
    return None

def _compute_moments(a0, a4, a8):
    """
    Compute the moments needed for the uncertainty product.
    Returns (norm2, x2_norm) where:
      norm2 = ∫ |g(x)|^2 dx
      x2_norm = ∫ x^2 |g(x)|^2 dx
    g(x) = P(x) * exp(-x^2/2)
    """
    # H_n(0) values
    H0_0 = 1.0
    H4_0 = -12.0
    H8_0 = 168.0
    H12_0 = -4320.0
    
    # Constraint: g(0) = 0 => a0*H0_0 + a4*H4_0 + a8*H8_0 + a12*H12_0 = 0
    a12 = -(a0 * H0_0 + a4 * H4_0 + a8 * H8_0) / H12_0
    
    # Build P(x) = a0*H0(x) + a4*H4(x) + a8*H8(x) + a12*H12(x)
    # P(x) = sum_{k=0}^{6} p_k * x^{2k}
    h0 = _hermite_coeffs(0)
    h4 = _hermite_coeffs(4)
    h8 = _hermite_coeffs(8)
    h12 = _hermite_coeffs(12)
    
    # Pad to length 7 (for x^0, x^2, ..., x^12)
    def pad(coeffs):
        result = [0.0] * 7
        for i, c in enumerate(coeffs):
            result[i] = c
        return result
    
    p0 = pad(h0)
    p4 = pad(h4)
    p8 = pad(h8)
    p12 = pad(h12)
    
    # P(x) coefficients: p_k for x^{2k}
    p = np.zeros(7)
    p += a0 * np.array(p0)
    p += a4 * np.array(p4)
    p += a8 * np.array(p8)
    p += a12 * np.array(p12)
    
    # |g(x)|^2 = P(x)^2 * exp(-x^2)
    # P(x)^2 = (sum p_k x^{2k})^2 = sum_{i,j} p_i p_j x^{2(i+j)}
    # So |g(x)|^2 = (sum_{m=0}^{12} q_m x^{2m}) * exp(-x^2) where q_m = sum_{i+j=m} p_i p_j
    
    # Compute q_m for m = 0, ..., 12
    q = np.zeros(13)
    for i in range(7):
        for j in range(7):
            m = i + j
            if m <= 12:
                q[m] += p[i] * p[j]
    
    # ∫ x^{2m} exp(-x^2) dx = (2m-1)!! * sqrt(pi) / 2^m for m >= 0
    # More precisely: ∫_0^∞ x^{2m} e^{-x^2} dx = (2m)! / (4^m m!) * sqrt(pi) / 2
    # Actually: ∫_{-∞}^{∞} x^{2m} e^{-x^2} dx = (2m-1)!! * sqrt(pi)
    # where (2m-1)!! = (2m)! / (2^m m!)
    
    def integral_2m(m):
        """Compute ∫_{-∞}^{∞} x^{2m} exp(-x^2) dx"""
        if m == 0:
            return np.sqrt(np.pi)
        # (2m-1)!! * sqrt(pi)
        result = np.sqrt(np.pi)
        for k in range(1, m + 1):
            result *= (2 * k - 1)
        return result
    
    # norm2 = ∫ |g|^2 dx = sum_{m=0}^{12} q[m] * integral_2m(m)
    norm2 = 0.0
    for m in range(13):
        norm2 += q[m] * integral_2m(m)
    
    # x2_norm = ∫ x^2 |g|^2 dx = ∫ x^2 * P(x)^2 * exp(-x^2) dx
    # = sum_{m=0}^{12} q[m] * ∫ x^{2m+2} exp(-x^2) dx
    # = sum_{m=0}^{12} q[m] * integral_2m(m+1)
    x2_norm = 0.0
    for m in range(13):
        x2_norm += q[m] * integral_2m(m + 1)
    
    return norm2, x2_norm

def _objective_and_grad(x):
    """
    Compute the uncertainty product and its gradient.
    The uncertainty product is (x2_norm / norm2)^2 since Δx = Δp for this symmetric case.
    Actually, let me reconsider: Δx^2 = x2_norm/norm2, Δp^2 = x2_norm/norm2 (by symmetry)
    So the product is (x2_norm/norm2)^2.
    
    Wait, I need to be more careful. Let me re-derive.
    
    For g(x) = P(x) exp(-x^2/2) with P even:
    - ||g||^2 = ∫ P(x)^2 exp(-x^2) dx = norm2
    - ∫ x^2 |g|^2 dx = x2_norm
    - Δx^2 = x2_norm / norm2
    
    For the Fourier transform (with convention f̂(ξ) = (1/√(2π)) ∫ f(x) e^{-ixξ} dx):
    - ĝ(ξ) = (1/√(2π)) P(ξ) exp(-ξ^2/2) (since i^n = 1 for even n)
    - ||ĝ||^2 = (1/(2π)) norm2
    - ∫ ξ^2 |ĝ|^2 dξ = (1/(2π)) x2_norm
    - Δp^2 = x2_norm / norm2
    
    So Δx * Δp = (x2_norm/norm2)
    
    We want to minimize this.
    """
    a0, a4, a8 = x
    
    norm2, x2_norm = _compute_moments(a0, a4, a8)
    
    if norm2 <= 1e-15:
        return 1e10, np.zeros(3)
    
    product = x2_norm / norm2
    
    # Compute gradient using finite differences (or analytically)
    # For simplicity and robustness, use finite differences with a small step
    eps = 1e-7
    grad = np.zeros(3)
    for i in range(3):
        x_plus = x.copy()
        x_plus[i] += eps
        norm2_p, x2_norm_p = _compute_moments(x_plus[0], x_plus[1], x_plus[2])
        if norm2_p <= 1e-15:
            grad[i] = 0
        else:
            product_p = x2_norm_p / norm2_p
            grad[i] = (product_p - product) / eps
    
    return product, grad

def _objective(x):
    """Objective function for scipy.optimize.minimize"""
    if time.time() > DEADLINE - 5:
        return 1e10
    val, _ = _objective_and_grad(x)
    return val

def _objective_grad(x):
    """Objective function with gradient for scipy.optimize.minimize"""
    if time.time() > DEADLINE - 5:
        return 1e10, np.zeros(3)
    return _objective_and_grad(x)

def construct():
    """Find optimal coefficients to minimize the uncertainty product."""
    start_time = time.time()
    
    # Initial guess
    best_coeffs = np.array([-113/100, 1/25, 1/3240])
    best_val, _ = _objective_and_grad(best_coeffs)
    
    # Try multiple random initializations
    rng = np.random.default_rng(42)
    
    for i in range(20):
        if time.time() > DEADLINE - 30:
            break
        
        # Random initialization
        coeffs = best_coeffs + rng.normal(0, 0.5, 3)
        val, _ = _objective_and_grad(coeffs)
        if val < best_val:
            best_val = val
            best_coeffs = coeffs.copy()
    
    # Local optimization with L-BFGS-B
    if time.time() < DEADLINE - 20:
        try:
            result = minimize(_objective_grad, best_coeffs, method='L-BFGS-B',
                            jac=True,
                            options={'maxiter': 500, 'gtol': 1e-10, 'ftol': 1e-12})
            if result.fun < best_val:
                best_val = result.fun
                best_coeffs = result.x.copy()
        except Exception:
            pass
    
    # Try Powell method as backup
    if time.time() < DEADLINE - 10:
        try:
            result = minimize(_objective, best_coeffs, method='Powell',
                            options={'maxiter': 200, 'xtol': 1e-10, 'ftol': 1e-12})
            if result.fun < best_val:
                best_val = result.fun
                best_coeffs = result.x.copy()
        except Exception:
            pass
    
    return best_coeffs
