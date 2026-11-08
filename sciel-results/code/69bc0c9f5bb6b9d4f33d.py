import numpy as np
from scipy.optimize import minimize
import time

# Hermite polynomials (physicists' convention)
# H0 = 1
# H2 = 4x^2 - 2
# H4 = 16x^4 - 48x^2 + 12
# H8 = 256x^8 - 3072x^6 + 11520x^4 - 13440x^2 + 3360
# H12 = 4096x^12 - 98304x^10 + 846720x^8 - 3932160x^6 + 8064000x^4 - 6048000x^2 + 1039500

# Coefficients for H12
h12_coeffs = np.array([1039500, 0, -6048000, 0, 8064000, 0, -3932160, 0, 846720, 0, -98304, 0, 4096])  # constant to x^12
# Actually let me define them as arrays of coefficients for x^0, x^2, x^4, ...

# For integration purposes, I need ∫ x^(2n) * exp(-x²) dx = (2n-1)!! * sqrt(pi) / 2^n
# And ∫ x^(2n) * exp(-x²/2) dx = ...

# Let me use a different approach: express everything in terms of moments.

# g(x) = P(x) * exp(-x²/2) where P(x) = sum of c_i * H_{4i}(x)
# |g(x)|² = P(x)² * exp(-x²)
# |ĝ(x)|² = Q(x)² * exp(-x²) where Q is the Fourier transform polynomial

# The Fourier transform of H_n(x)*exp(-x²/2) is (-i)^n * H_n(x) * exp(-x²/2) (up to normalization)

# For the L4 norm: ∫ |g(x)|⁴ dx = ∫ P(x)⁴ * exp(-2x²) dx
# For the L2 norm: ∫ |g(x)|² dx = ∫ P(x)² * exp(-x²) dx

# Similarly for the Fourier side.

# Let me define P(x) explicitly in terms of monomials.

def _hermite_polynomial_coeffs(degree):
    """Return coefficients of physicists' Hermite polynomial H_degree as [c0, c2, c4, ..., c_degree] for even degrees."""
    # H0 = 1
    # H2 = 4x^2 - 2
    # H4 = 16x^4 - 48x^2 + 12
    # H8 = 256x^8 - 3072x^6 + 11520x^4 - 13440x^2 + 3360
    # H12 = 4096x^12 - 98304x^10 + 846720x^8 - 3932160x^6 + 8064000x^4 - 6048000x^2 + 1039500
    if degree == 0:
        return np.array([1.0])
    elif degree == 2:
        return np.array([-2.0, 4.0])
    elif degree == 4:
        return np.array([12.0, -48.0, 16.0])
    elif degree == 8:
        return np.array([3360.0, -13440.0, 11520.0, -3072.0, 256.0])
    elif degree == 12:
        return np.array([1039500.0, -6048000.0, 8064000.0, -3932160.0, 846720.0, -98304.0, 4096.0])
    else:
        raise ValueError(f"Unsupported degree {degree}")

# Precompute Hermite polynomial coefficients
H0_coeffs = _hermite_polynomial_coeffs(0)
H4_coeffs = _hermite_polynomial_coeffs(4)
H8_coeffs = _hermite_polynomial_coeffs(8)
H12_coeffs = _hermite_polynomial_coeffs(12)

# P(x) = c0*H0 + c1*H4 + c2*H8 + c3*H12
# Max degree is 12, so P(x) has terms up to x^12
# P(x) = p0 + p2*x^2 + p4*x^4 + ... + p12*x^12

def _compute_P_coeffs(c0, c1, c2, c3):
    """Compute the monomial coefficients of P(x) = c0*H0 + c1*H4 + c2*H8 + c3*H12."""
    # Each Hermite poly contributes to even powers only
    # H0: [1] (degree 0)
    # H4: [12, -48, 16] (degrees 0, 2, 4)
    # H8: [3360, -13440, 11520, -3072, 256] (degrees 0, 2, 4, 6, 8)
    # H12: [1039500, -6048000, 8064000, -3932160, 846720, -98304, 4096] (degrees 0, 2, 4, 6, 8, 10, 12)
    
    # P has degree 12, so 7 coefficients (x^0, x^2, ..., x^12)
    P = np.zeros(7)
    # H0 contributes to P[0]
    P[0] += c0 * H0_coeffs[0]
    # H4 contributes to P[0], P[1], P[2]
    P[0] += c1 * H4_coeffs[0]
    P[1] += c1 * H4_coeffs[1]
    P[2] += c1 * H4_coeffs[2]
    # H8 contributes to P[0]..P[4]
    for i in range(5):
        P[i] += c2 * H8_coeffs[i]
    # H12 contributes to P[0]..P[6]
    for i in range(7):
        P[i] += c3 * H12_coeffs[i]
    
    return P

def _integral_x2n_exp_minus_x2(n):
    """∫_{-∞}^{∞} x^(2n) * exp(-x²) dx = (2n-1)!! * sqrt(pi) / 2^n"""
    if n == 0:
        return np.sqrt(np.pi)
    result = 1.0
    for k in range(n):
        result *= (2*k + 1)
    return result * np.sqrt(np.pi) / (2.0 ** n)

def _integral_x2n_exp_minus_2x2(n):
    """∫_{-∞}^{∞} x^(2n) * exp(-2x²) dx = (2n-1)!! * sqrt(pi/2) / (2^n * 2^(n+0.5)) = (2n-1)!! * sqrt(pi/2) / 2^(2n+0.5)"""
    # ∫ x^(2n) exp(-ax²) dx = (2n-1)!! * sqrt(pi) / (2^n * a^(n+0.5))
    # For a=2: (2n-1)!! * sqrt(pi) / (2^n * 2^(n+0.5)) = (2n-1)!! * sqrt(pi) / 2^(2n+0.5)
    if n == 0:
        return np.sqrt(np.pi / 2.0)
    result = 1.0
    for k in range(n):
        result *= (2*k + 1)
    return result * np.sqrt(np.pi) / (2.0 ** (2*n + 0.5))

def _poly_square_moment(P, integral_func):
    """Compute ∫ P(x)² * exp(-x²) dx or similar, where P(x) = sum P[i]*x^(2i)."""
    # P(x)² = sum_i sum_j P[i]*P[j]*x^(2i+2j)
    # ∫ P(x)² * exp(-x²) dx = sum_i sum_j P[i]*P[j] * ∫ x^(2(i+j)) * exp(-x²) dx
    n = len(P)
    result = 0.0
    for i in range(n):
        for j in range(n):
            result += P[i] * P[j] * integral_func(i + j)
    return result

def _poly_fourth_moment(P, integral_func):
    """Compute ∫ P(x)⁴ * exp(-2x²) dx, where P(x) = sum P[i]*x^(2i)."""
    # P(x)⁴ = sum_{i,j,k,l} P[i]*P[j]*P[k]*P[l]*x^(2(i+j+k+l))
    # ∫ P(x)⁴ * exp(-2x²) dx = sum_{i,j,k,l} P[i]*P[j]*P[k]*P[l] * ∫ x^(2(i+j+k+l)) * exp(-2x²) dx
    n = len(P)
    result = 0.0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    result += P[i] * P[j] * P[k] * P[l] * integral_func(i + j + k + l)
    return result

def _objective_and_grad(c0, c1, c2):
    """Compute the objective (upper bound on C4) and its gradient."""
    # c3 is determined by g(0) = 0: c0*1 + c1*12 + c2*3360 + c3*1039500 = 0
    c3 = -(c0 + 12*c1 + 3360*c2) / 1039500.0
    
    P = _compute_P_coeffs(c0, c1, c2, c3)
    
    # L2 norm squared: ∫ |g(x)|² dx = ∫ P(x)² * exp(-x²) dx
    L2_sq = _poly_square_moment(P, _integral_x2n_exp_minus_x2)
    
    # L4 norm to the 4th power: ∫ |g(x)|⁴ dx = ∫ P(x)⁴ * exp(-2x²) dx
    L4_4th = _poly_fourth_moment(P, _integral_x2n_exp_minus_2x2)
    
    # For the Fourier side, by Plancherel and the symmetry of Hermite functions:
    # The Fourier transform of H_n(x)*exp(-x²/2) is (-i)^n * H_n(x)*exp(-x²/2) (with proper normalization)
    # So |ĝ(x)|² has the same form as |g(x)|², and |ĝ(x)|⁴ has the same form as |g(x)|⁴
    # Thus the L2 and L4 norms are the same for g and ĝ.
    
    # The quantity C4 = (∫|g|⁴)^(1/4) * (∫|ĝ|⁴)^(1/4) / ((∫|g|²)^(1/2) * (∫|ĝ|²)^(1/2))
    # = (L4_4th)^(1/2) / L2_sq
    # Wait, let me be more careful:
    # ||g||_4 = (∫|g|⁴)^(1/4)
    # ||g||_2 = (∫|g|²)^(1/2)
    # C4 = ||g||_4 * ||ĝ||_4 / (||g||_2 * ||ĝ||_2)
    # = (L4_4th)^(1/4) * (L4_4th)^(1/4) / (L2_sq^(1/2) * L2_sq^(1/2))
    # = (L4_4th)^(1/2) / L2_sq
    
    if L2_sq <= 0:
        return np.inf, np.zeros(3)
    
    C4 = np.sqrt(L4_4th) / L2_sq
    
    # Gradient via finite differences or analytic
    # Let's use analytic gradient by computing dC4/dc_i
    # C4 = sqrt(L4_4th) / L2_sq
    # dC4/dc_i = (0.5/sqrt(L4_4th) * dL4_4th/dc_i * L2_sq - sqrt(L4_4th) * dL2_sq/dc_i) / L2_sq²
    # = (0.5 * dL4_4th/dc_i / sqrt(L4_4th) - dL2_sq/dc_i / L2_sq) * C4
    
    # Need dL2_sq/dc_i and dL4_4th/dc_i
    # P depends linearly on c0, c1, c2 (and c3 which depends on them)
    # dP/dc0 = dP/dc0_direct (from c0*H0) + dc3/dc0 * dP/dc3
    # dc3/dc0 = -1/1039500
    # dc3/dc1 = -12/1039500
    # dc3/dc2 = -3360/1039500
    
    # Let me compute dP/dc_i for i=0,1,2
    # P = c0*H0 + c1*H4 + c2*H8 + c3*H12
    # dP/dc0 = H0 + (dc3/dc0)*H12
    # dP/dc1 = H4 + (dc3/dc1)*H12
    # dP/dc2 = H8 + (dc3/dc2)*H12
    
    dP_dc0 = _compute_P_coeffs(1, 0, 0, 0) + (-1/1039500.0) * _compute_P_coeffs(0, 0, 0, 1)
    dP_dc1 = _compute_P_coeffs(0, 1, 0, 0) + (-12/1039500.0) * _compute_P_coeffs(0, 0, 0, 1)
    dP_dc2 = _compute_P_coeffs(0, 0, 1, 0) + (-3360/1039500.0) * _compute_P_coeffs(0, 0, 0, 1)
    
    # dL2_sq/dc_i = 2 * sum_j P[j] * dP_dc_i[j] * integral(2*(i+j))  ... wait
    # L2_sq = sum_i sum_j P[i]*P[j]*I(i+j) where I(n) = ∫x^(2n)exp(-x²)
    # dL2_sq/dc = 2 * sum_i sum_j (dP/dc)[i] * P[j] * I(i+j)  (by symmetry, this equals sum over all)
    # Actually: d/dc sum_i sum_j P_i P_j I_ij = sum_i sum_j (dP_i/dc * P_j + P_i * dP_j/dc) I_ij
    # = 2 * sum_i sum_j dP_i/dc * P_j * I_ij  (by symmetry of I_ij)
    
    def _dL2_sq(P, dP):
        n = len(P)
        result = 0.0
        for i in range(n):
            for j in range(n):
                result += 2.0 * dP[i] * P[j] * _integral_x2n_exp_minus_x2(i + j)
        return result
    
    # dL4_4th/dc_i = 4 * sum_{j,k,l} (dP/dc)[i] * P[j]*P[k]*P[l] * J(i+j+k+l)
    # where J(n) = ∫x^(2n)exp(-2x²)
    # By symmetry: d/dc sum_{i,j,k,l} P_i P_j P_k P_l J_{ijkl} = 4 * sum_{j,k,l} dP_i/dc * P_j P_k P_l J_{ijkl}
    
    def _dL4_4th(P, dP):
        n = len(P)
        result = 0.0
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    for l in range(n):
                        result += 4.0 * dP[i] * P[j] * P[k] * P[l] * _integral_x2n_exp_minus_2x2(i + j + k + l)
        return result
    
    dL2_sq_dc0 = _dL2_sq(P, dP_dc0)
    dL2_sq_dc1 = _dL2_sq(P, dP_dc1)
    dL2_sq_dc2 = _dL2_sq(P, dP_dc2)
    
    dL4_4th_dc0 = _dL4_4th(P, dP_dc0)
    dL4_4th_dc1 = _dL4_4th(P, dP_dc1)
    dL4_4th_dc2 = _dL4_4th(P, dP_dc2)
    
    sqrt_L4 = np.sqrt(L4_4th)
    dC4_dc0 = (0.5 * dL4_4th_dc0 / sqrt_L4 - dL2_sq_dc0 / L2_sq) * C4
    dC4_dc1 = (0.5 * dL4_4th_dc1 / sqrt_L4 - dL2_sq_dc1 / L2_sq) * C4
    dC4_dc2 = (0.5 * dL4_4th_dc2 / sqrt_L4 - dL2_sq_dc2 / L2_sq) * C4
    
    return C4, np.array([dC4_dc0, dC4_dc1, dC4_dc2])

def construct():
    start_time = time.time()
    deadline = start_time + 180.0  # Safety margin
    
    # Multi-start BFGS optimization
    best_C4 = np.inf
    best_c = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    
    # Generate multiple starting points
    rng = np.random.RandomState(42)
    starts = [best_c.copy()]
    
    # Add perturbed starts around the known good point
    for scale in [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]:
        for _ in range(3):
            perturbation = rng.normal(0, scale, 3)
            starts.append(best_c + perturbation)
    
    # Also add some random starts in a wider range
    for _ in range(20):
        starts.append(rng.uniform(-500, 500, 3))
    
    # Sort by initial objective to try promising ones first
    start_objs = []
    for s in starts:
        try:
            obj, _ = _objective_and_grad(s[0], s[1], s[2])
            start_objs.append((obj, s))
        except:
            start_objs.append((np.inf, s))
    start_objs.sort(key=lambda x: x[0])
    
    for obj, s in start_objs:
        if time.time() > deadline - 5.0:
            break
        if obj >= best_C4:
            continue  # Skip if initial value is already worse
        
        try:
            result = minimize(
                lambda c: _objective_and_grad(c[0], c[1], c[2])[0],
                s,
                jac=lambda c: _objective_and_grad(c[0], c[1], c[2])[1],
                method='BFGS',
                options={'maxiter': 200, 'gtol': 1e-10}
            )
            if result.success and result.fun < best_C4:
                best_C4 = result.fun
                best_c = result.x
        except Exception:
            pass
    
    return np.array(best_c)
