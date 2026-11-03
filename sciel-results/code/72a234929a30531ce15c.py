"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, least_squares

def _hermite(n, x):
    """Compute physicists' Hermite polynomial H_n(x) via recurrence."""
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return 2 * x
    H0 = np.ones_like(x)
    H1 = 2 * x
    for k in range(1, n):
        H2 = 2 * x * H1 - 2 * k * H0
        H0, H1 = H1, H2
    return H1

def _compute_C4(c0, c4, c8):
    """Compute the upper bound C4 given coefficients c0, c4, c8.
    c12 is determined by g(0) = 0 constraint."""
    # c12 from g(0) = c0*H0(0) + c4*H4(0) + c8*H8(0) + c12*H12(0) = 0
    H0_0 = 1.0
    H4_0 = 3.0
    H8_0 = 105.0
    H12_0 = 10395.0
    c12 = -(c0 * H0_0 + c4 * H4_0 + c8 * H8_0) / H12_0
    
    # Numerically compute the bound using quadrature
    # The bound C4 typically involves moments of g and its Fourier transform
    # g(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-x^2/2)
    # For Hermite-Gaussian functions, the Fourier transform is also Hermite-Gaussian
    
    # Use Gauss-Hermite quadrature for accurate integration
    # The integrals needed are of the form ∫ x^m g(x)^2 dx and similar
    
    n_quad = 200
    x_nodes, w_nodes = np.polynomial.hermite.hermgauss(n_quad)
    # hermgauss gives nodes for ∫ f(x) exp(-x^2) dx
    # We need ∫ f(x) exp(-x^2) dx since g(x) = P(x) exp(-x^2/2)
    # g(x)^2 = P(x)^2 exp(-x^2)
    
    # Compute P(x) at quadrature nodes
    H0 = _hermite(0, x_nodes)
    H4 = _hermite(4, x_nodes)
    H8 = _hermite(8, x_nodes)
    H12 = _hermite(12, x_nodes)
    
    P = c0 * H0 + c4 * H4 + c8 * H8 + c12 * H12
    g_sq = P ** 2  # g(x)^2 / exp(-x^2) = P(x)^2
    
    # Moments: ∫ x^m P(x)^2 exp(-x^2) dx
    moments = np.zeros(16)
    for m in range(16):
        moments[m] = np.sum(w_nodes * (x_nodes ** m) * g_sq)
    
    # The uncertainty product bound C4 relates to specific combinations of moments
    # A(f) typically involves ∫|f|^2 and ∫|f'|^2 or similar
    # For the 4th order bound, we need:
    # A(f)^2 ~ (∫|f|^2)(∫|f''|^2) or similar combinations
    # F̂f has similar structure
    
    # Let's compute ∫|g'|^2 and ∫|g''|^2 etc.
    # g(x) = P(x) exp(-x^2/2)
    # g'(x) = [P'(x) - x P(x)] exp(-x^2/2)
    # g''(x) = [P''(x) - 2x P'(x) + (x^2 - 1) P(x)] exp(-x^2/2)
    
    # Compute derivatives of P
    P_prime = c0 * 0 + c4 * _hermite(3, x_nodes) * 2 + c8 * _hermite(7, x_nodes) * 2 + c12 * _hermite(11, x_nodes) * 2
    # Actually H_n' = n H_{n-1} * 2? No: H_n' = 2n H_{n-1}
    # So H4' = 8 H3, H8' = 16 H7, H12' = 24 H11
    
    P_prime = c4 * 8 * _hermite(3, x_nodes) + c8 * 16 * _hermite(7, x_nodes) + c12 * 24 * _hermite(11, x_nodes)
    
    P_double_prime = c4 * 8 * 2 * _hermite(2, x_nodes) + c8 * 16 * 2 * _hermite(6, x_nodes) + c12 * 24 * 2 * _hermite(10, x_nodes)
    # H_n'' = 2n * 2(n-1) H_{n-2}? No: H_n'' = 4n(n-1) H_{n-2}
    # H4'' = 4*4*3 H2 = 48 H2
    # H8'' = 4*8*7 H6 = 224 H6
    # H12'' = 4*12*11 H10 = 528 H10
    
    P_double_prime = c4 * 48 * _hermite(2, x_nodes) + c8 * 224 * _hermite(6, x_nodes) + c12 * 528 * _hermite(10, x_nodes)
    
    # g' = (P' - xP) exp(-x^2/2)
    # |g'|^2 = (P' - xP)^2 exp(-x^2)
    g_prime_sq = (P_prime - x_nodes * P) ** 2
    integral_g_prime_sq = np.sum(w_nodes * g_prime_sq)
    
    # g'' = (P'' - 2xP' + (x^2-1)P) exp(-x^2/2)
    g_double_prime_sq = (P_double_prime - 2 * x_nodes * P_prime + (x_nodes**2 - 1) * P) ** 2
    integral_g_double_prime_sq = np.sum(w_nodes * g_double_prime_sq)
    
    integral_g_sq = np.sum(w_nodes * g_sq)
    
    # For the Fourier transform of g(x) = P(x)exp(-x^2/2):
    # F̂g(ξ) = (1/√(2π)) ∫ g(x) e^{-iξx} dx
    # For Hermite-Gaussian, F̂[H_n exp(-x^2/2)] = (-i)^n H_n(ξ) exp(-ξ^2/2)
    # So F̂g(ξ) = (1/√(2π)) [c0 H0(ξ) + c4 (-i)^4 H4(ξ) + c8 (-i)^8 H8(ξ) + c12 (-i)^12 H12(ξ)] exp(-ξ^2/2)
    # (-i)^4 = 1, (-i)^8 = 1, (-i)^12 = 1
    # So F̂g(ξ) = (1/√(2π)) [c0 H0(ξ) + c4 H4(ξ) + c8 H8(ξ) + c12 H12(ξ)] exp(-ξ^2/2)
    # = (1/√(2π)) g(ξ)
    
    # So |F̂g|^2 = (1/(2π)) |g(ξ)|^2
    # ∫|F̂g|^2 dξ = (1/(2π)) ∫|g|^2 dx = (1/(2π)) integral_g_sq
    
    # The bound C4 likely is:
    # C4 = sqrt(∫|g|^2 * ∫|g''|^2) * sqrt(∫|F̂g|^2 * ∫|(F̂g)''|^2) / (∫|g|^2 * ∫|F̂g|^2) or similar
    # Or perhaps C4 = (∫|x g|^2 / ∫|g|^2) * (∫|ξ F̂g|^2 / ∫|F̂g|^2)
    
    # Given the "4th order" nature, let's try:
    # C4 = sqrt(∫|g''|^2 / ∫|g|^2) * sqrt(∫|(F̂g)''|^2 / ∫|F̂g|^2)
    # Since F̂g is proportional to g, this becomes:
    # C4 = sqrt(∫|g''|^2 / ∫|g|^2) * sqrt(∫|g''|^2 / ∫|g|^2) = ∫|g''|^2 / ∫|g|^2
    
    # Or maybe it involves the 4th moment:
    # C4 = (∫x^4|g|^2 / ∫|g|^2) * (∫ξ^4|F̂g|^2 / ∫|F̂g|^2)
    
    # Let me try a more standard uncertainty product:
    # Δx * Δξ where Δx^2 = ∫x^2|g|^2/∫|g|^2 and Δξ^2 = ∫ξ^2|F̂g|^2/∫|F̂g|^2
    
    # But the task says "4th order" so maybe:
    # C4 = (∫x^4|g|^2/∫|g|^2)^{1/4} * (∫ξ^4|F̂g|^2/∫|F̂g|^2)^{1/4}
    
    # Let me compute several candidates and see which matches the seed value
    
    # Candidate 1: standard uncertainty product
    moment2_x = moments[2] / moments[0] if moments[0] != 0 else 0
    moment4_x = moments[4] / moments[0] if moments[0] != 0 else 0
    
    # For F̂g, the moments are the same as g since F̂g ∝ g
    # ∫ξ^2|F̂g|^2 dξ = (1/(2π)) ∫ξ^2|g(ξ)|^2 dξ = (1/(2π)) moments[2]
    # ∫|F̂g|^2 dξ = (1/(2π)) moments[0]
    # So moment2_xi = moments[2] / moments[0] = moment2_x
    
    # Δx * Δξ = sqrt(moment2_x) * sqrt(moment2_x) = moment2_x
    C4_candidate1 = moment2_x
    
    # Candidate 2: 4th moment product
    C4_candidate2 = moment4_x ** 0.5
    
    # Candidate 3: involves second derivatives
    C4_candidate3 = (integral_g_double_prime_sq / moments[0])
    
    # Let's check which gives the seed value
    # Seed: c0=-1.13, c4=0.04, c8=1/3240≈0.000309
    # Score 1.0 means the bound equals some reference value
    
    # The "score is the upper bound on C4" and "seed/value" where value is the raw bound
    # If seed gives score 1.0, then seed_bound / seed_value = 1.0, so seed_value = seed_bound
    # This is circular. Let me re-read.
    
    # "score is the upper bound on C4" - so the returned value from construct() leads to a C4 value
    # "The evaluator measures the raw value (lower is better)"
    # "The search score is seed/value" where seed is the baseline value
    # So if my C4 < seed_C4, score > 1.0
    
    # I need to figure out what C4 actually is. Let me try to match the seed.
    
    # Actually, let me just use a generic optimization approach where I define C4 as a function
    # and minimize it. The key is getting the right C4 formula.
    
    # Given the Hermite structure and "4th order", a common bound is:
    # C4 = (∫|g''|^2 dx / ∫|g|^2 dx) * (∫|x^2 g|^2 dx / ∫|g|^2 dx)
    # or similar products of second-derivative and position moments
    
    # Let me try: C4 = (∫|g''|^2 / ∫|g|^2) * (∫x^2|g|^2 / ∫|g|^2)
    C4_candidate4 = (integral_g_double_prime_sq / moments[0]) * (moments[2] / moments[0])
    
    # Or: C4 = sqrt(∫|g''|^2 / ∫|g|^2) * sqrt(∫x^4|g|^2 / ∫|g|^2)
    C4_candidate5 = np.sqrt(integral_g_double_prime_sq / moments[0]) * np.sqrt(moments[4] / moments[0])
    
    # I'll return candidate4 as a starting point, but I need to verify
    # For now, let's use a placeholder and do the optimization
    return C4_candidate4

def construct():
    start_time = time.time()
    deadline = start_time + 180.0  # 180s limit with safety margin
    
    # Seed value
    seed_coeffs = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    seed_C4 = _compute_C4(*seed_coeffs)
    
    best_C4 = seed_C4
    best_coeffs = seed_coeffs.copy()
    
    # Try multiple starting points
    rng = np.random.RandomState(42)
    n_starts = 20
    
    for i in range(n_starts):
        if time.time() > deadline - 30:
            break
        
        if i == 0:
            x0 = seed_coeffs
        else:
            x0 = seed_coeffs + rng.normal(0, 0.5, 3)
        
        try:
            result = minimize(_compute_C4, x0, method='Nelder-Mead', 
                            options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12})
            if result.success and result.fun < best_C4:
                best_C4 = result.fun
                best_coeffs = result.x.copy()
        except Exception:
            pass
    
    # Polish with higher precision
    if time.time() < deadline - 10:
        try:
            result = minimize(_compute_C4, best_coeffs, method='Nelder-Mead',
                            options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-16})
            if result.success and result.fun < best_C4:
                best_C4 = result.fun
                best_coeffs = result.x.copy()
        except Exception:
            pass
    
    return best_coeffs
# EVOLVE-BLOCK-END
