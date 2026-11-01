"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _hermite_poly(n, x):
    """Physicist's Hermite polynomial H_n(x)."""
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return 2 * x
    elif n == 2:
        return 4 * x**2 - 2
    elif n == 3:
        return 8 * x**3 - 12 * x
    elif n == 4:
        return 16 * x**4 - 48 * x**2 + 12
    elif n == 8:
        # H_8(x) = 256x^8 - 4480x^6 + 30240x^4 - 67200x^2 + 16800
        return 256 * x**8 - 4480 * x**6 + 30240 * x**4 - 67200 * x**2 + 16800
    elif n == 12:
        # H_12(x) = 4096x^12 - 129024x^10 + 1814400x^8 - 14417280x^6 + 55776000x^4 - 91452000x^2 + 47900160
        return (4096 * x**12 - 129024 * x**10 + 1814400 * x**8 
                - 14417280 * x**6 + 55776000 * x**4 
                - 91452000 * x**2 + 47900160)
    else:
        raise ValueError(f"Hermite polynomial of degree {n} not implemented")


def _uncertainty_product(c0, c4, c8):
    """
    Compute the uncertainty product A(f) * A(F̂) for the test function
    g(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-x^2/2)
    where c12 is chosen so that g(0) = 0.
    
    Returns the upper bound on C4 (the uncertainty product).
    """
    # At x=0: H0(0)=1, H4(0)=12, H8(0)=16800, H12(0)=47900160
    # g(0) = (c0*1 + c4*12 + c8*16800 + c12*47900160) * exp(0) = 0
    # So c12 = -(c0 + 12*c4 + 16800*c8) / 47900160
    c12 = -(c0 + 12 * c4 + 16800 * c8) / 47900160.0
    
    # We need to compute A(f) and A(F̂)
    # A(f) = sqrt(<x^2> - <x>^2) * sqrt(<1> - <1>^2) ... actually for the uncertainty inequality
    # The uncertainty product is A(f) * A(F̂) where
    # A(f) = sqrt(∫ x^2 |f|^2 dx) / sqrt(∫ |f|^2 dx) assuming mean is 0
    # A(F̂) = sqrt(∫ ξ^2 |F̂|^2 dξ) / sqrt(∫ |F̂|^2 dξ)
    
    # For our test function, by symmetry <x> = 0 and <ξ> = 0
    # So A(f) = sqrt(∫ x^2 |g|^2 dx / ∫ |g|^2 dx)
    # A(F̂) = sqrt(∫ ξ^2 |ĝ|^2 dξ / ∫ |ĝ|^2 dξ)
    
    # By Plancherel, ∫ |ĝ|^2 dξ = ∫ |g|^2 dx (up to normalization)
    # And ∫ ξ^2 |ĝ|^2 dξ = ∫ (g')^2 dx (for appropriate normalization)
    
    # Let's use numerical integration for robustness
    # g(x) = P(x) * exp(-x^2/2) where P(x) = c0*H0 + c4*H4 + c8*H8 + c12*H12
    
    # Use Gauss-Hermite quadrature for integrals of form ∫ f(x) exp(-x^2) dx
    # But we have exp(-x^2/2), so let's change variables or use direct quadrature
    
    # Actually, let's use the fact that for g(x) = P(x) exp(-x^2/2):
    # ∫ |g|^2 dx = ∫ P(x)^2 exp(-x^2) dx
    # ∫ x^2 |g|^2 dx = ∫ x^2 P(x)^2 exp(-x^2) dx
    # ∫ (g')^2 dx = ∫ [P'(x) exp(-x^2/2) - x P(x) exp(-x^2/2)]^2 dx
    #            = ∫ [P'(x) - x P(x)]^2 exp(-x^2) dx
    
    # Use Gauss-Hermite quadrature with enough nodes
    # np.polynomial.hermite.hermgauss(n) gives nodes and weights for ∫ f(x) exp(-x^2) dx
    
    n_quad = 200  # 200 nodes should be more than enough
    x_nodes, w_nodes = np.polynomial.hermite.hermgauss(n_quad)
    
    # Compute P(x) and P'(x) at the nodes
    H0 = np.ones_like(x_nodes)
    H4 = 16 * x_nodes**4 - 48 * x_nodes**2 + 12
    H8 = (256 * x_nodes**8 - 4480 * x_nodes**6 + 30240 * x_nodes**4 
          - 67200 * x_nodes**2 + 16800)
    H12 = (4096 * x_nodes**12 - 129024 * x_nodes**10 + 1814400 * x_nodes**8 
           - 14417280 * x_nodes**6 + 55776000 * x_nodes**4 
           - 91452000 * x_nodes**2 + 47900160)
    
    P = c0 * H0 + c4 * H4 + c8 * H8 + c12 * H12
    
    # P'(x): derivative of each Hermite polynomial
    # H0' = 0
    # H4' = 64x^3 - 96x
    # H8' = 2048x^7 - 26880x^5 + 120960x^3 - 134400x
    # H12' = 49152x^11 - 1290240x^9 + 14515200x^7 - 86503680x^5 + 223104000x^3 - 182904000x
    P_prime = (c4 * (64 * x_nodes**3 - 96 * x_nodes) 
               + c8 * (2048 * x_nodes**7 - 26880 * x_nodes**5 
                       + 120960 * x_nodes**3 - 134400 * x_nodes)
               + c12 * (49152 * x_nodes**11 - 1290240 * x_nodes**9 
                        + 14515200 * x_nodes**7 - 86503680 * x_nodes**5 
                        + 223104000 * x_nodes**3 - 182904000 * x_nodes))
    
    # Compute the integrals
    norm_sq = np.sum(w_nodes * P**2)  # ∫ P^2 exp(-x^2) dx
    x2_norm_sq = np.sum(w_nodes * x_nodes**2 * P**2)  # ∫ x^2 P^2 exp(-x^2) dx
    
    # (g')^2 integral: ∫ [P' - x*P]^2 exp(-x^2) dx
    g_prime_sq = (P_prime - x_nodes * P)**2
    freq_norm_sq = np.sum(w_nodes * g_prime_sq)  # This is ∫ (g')^2 exp(-x^2) dx... 
    
    # Wait, I need to be careful with the Fourier transform convention
    # Let me use a different approach: compute numerically with a fine grid
    
    # Actually, let me use the relation:
    # A(f)^2 = ∫ x^2 |g|^2 dx / ∫ |g|^2 dx
    # A(F̂)^2 = ∫ ξ^2 |ĝ|^2 dξ / ∫ |ĝ|^2 dξ
    # By Plancherel: ∫ |ĝ|^2 dξ = (1/2π) ∫ |g|^2 dx (depending on convention)
    # And ∫ ξ^2 |ĝ|^2 dξ = (1/2π) ∫ |g'|^2 dx
    
    # So A(F̂)^2 = ∫ |g'|^2 dx / ∫ |g|^2 dx
    
    # g(x) = P(x) exp(-x^2/2)
    # g'(x) = [P'(x) - x P(x)] exp(-x^2/2)
    # |g'|^2 = [P'(x) - x P(x)]^2 exp(-x^2)
    # |g|^2 = P(x)^2 exp(-x^2)
    
    # So A(F̂)^2 = ∫ [P' - xP]^2 exp(-x^2) dx / ∫ P^2 exp(-x^2) dx
    # And A(f)^2 = ∫ x^2 P^2 exp(-x^2) dx / ∫ P^2 exp(-x^2) dx
    
    if norm_sq <= 0 or not np.isfinite(norm_sq):
        return np.inf
    
    A_f_sq = x2_norm_sq / norm_sq
    A_Fhat_sq = freq_norm_sq / norm_sq
    
    if A_f_sq < 0 or A_Fhat_sq < 0 or not np.isfinite(A_f_sq) or not np.isfinite(A_Fhat_sq):
        return np.inf
    
    A_f = np.sqrt(A_f_sq)
    A_Fhat = np.sqrt(A_Fhat_sq)
    
    return A_f * A_Fhat


def construct():
    """
    Find the three coefficients (c0, c4, c8) that minimize the uncertainty product.
    Returns np.array([c0, c4, c8]).
    """
    start_time = time.time()
    deadline = start_time + 200.0  # 200s deadline, 10s safety margin
    
    # Seed values as initial best
    best_coeffs = np.array([-113.0/100, 1.0/25, 1.0/3240])
    best_value = _uncertainty_product(best_coeffs[0], best_coeffs[1], best_coeffs[2])
    
    if not np.isfinite(best_value):
        best_value = 1e10
    
    # Try multiple optimization strategies
    # Strategy 1: Nelder-Mead from seed
    try:
        result = minimize(
            lambda c: _uncertainty_product(c[0], c[1], c[2]),
            best_coeffs,
            method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12}
        )
        if result.fun < best_value:
            best_value = result.fun
            best_coeffs = result.x
    except Exception:
        pass
    
    if time.time() > deadline:
        return best_coeffs
    
    # Strategy 2: Try different initial points
    initial_points = [
        np.array([-1.0, 0.1, 0.001]),
        np.array([-1.1, 0.05, 0.0005]),
        np.array([-1.05, 0.08, 0.0008]),
        np.array([-0.9, 0.15, 0.002]),
        np.array([-1.2, 0.02, 0.0003]),
    ]
    
    for init in initial_points:
        if time.time() > deadline:
            break
        try:
            result = minimize(
                lambda c: _uncertainty_product(c[0], c[1], c[2]),
                init,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            if result.fun < best_value:
                best_value = result.fun
                best_coeffs = result.x
        except Exception:
            pass
    
    if time.time() > deadline:
        return best_coeffs
    
    # Strategy 3: L-BFGS-B with bounds
    try:
        bounds = [(-5, 5), (-1, 1), (-0.01, 0.01)]
        result = minimize(
            lambda c: _uncertainty_product(c[0], c[1], c[2]),
            best_coeffs,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-15}
        )
        if result.fun < best_value:
            best_value = result.fun
            best_coeffs = result.x
    except Exception:
        pass
    
    if time.time() > deadline:
        return best_coeffs
    
    # Strategy 4: Fine local search around best
    try:
        # Use a smaller step Nelder-Mead
        result = minimize(
            lambda c: _uncertainty_product(c[0], c[1], c[2]),
            best_coeffs,
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14,
                     'initial_simplex': best_coeffs + np.array([1e-6, 1e-7, 1e-9])}
        )
        if result.fun < best_value:
            best_value = result.fun
            best_coeffs = result.x
    except Exception:
        pass
    
    return best_coeffs

# EVOLVE-BLOCK-END
