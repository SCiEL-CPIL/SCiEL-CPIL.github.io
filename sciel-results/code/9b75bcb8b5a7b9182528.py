"""Optimized search for Hermite coefficients minimizing the C4 upper bound."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _hermite_poly(x, n):
    """Compute the nth probabilists' Hermite polynomial He_n(x)."""
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return x
    elif n == 2:
        return x**2 - 1
    elif n == 3:
        return x**3 - 3*x
    elif n == 4:
        return x**4 - 6*x**2 + 3
    elif n == 5:
        return x**5 - 10*x**3 + 15*x
    elif n == 6:
        return x**6 - 15*x**4 + 45*x**2 - 15
    elif n == 7:
        return x**7 - 21*x**5 + 105*x**3 - 105*x
    elif n == 8:
        return x**8 - 28*x**6 + 210*x**4 - 420*x**2 + 105
    elif n == 9:
        return x**9 - 36*x**7 + 378*x**5 - 1260*x**3 + 945*x
    elif n == 10:
        return x**10 - 45*x**8 + 630*x**6 - 3150*x**4 + 4725*x**2 - 945
    elif n == 11:
        return x**11 - 55*x**9 + 990*x**7 - 6930*x**5 + 14175*x**3 - 10395*x
    elif n == 12:
        return x**12 - 66*x**10 + 1485*x**8 - 13860*x**6 + 49545*x**4 - 66795*x**2 + 10395
    else:
        raise ValueError("n too large")

def _compute_bound(c0, c4, c8):
    """
    Compute the upper bound on C4 for the given coefficients.
    The test function is g(x) = (c0*H0(x) + c4*H4(x) + c8*H8(x) + c12*H12(x)) * exp(-x^2/2)
    where c12 is chosen so that g(0) = 0.
    
    The bound involves computing integrals related to the uncertainty inequality.
    """
    # At x=0: H0(0)=1, H4(0)=3, H8(0)=105, H12(0)=10395
    # g(0) = (c0 + 3*c4 + 105*c8 + 10395*c12) * 1 = 0
    c12 = -(c0 + 3*c4 + 105*c8) / 10395.0
    
    # We need to compute the upper bound on A(f)*A(Fourier(f))
    # For a Gaussian-modulated Hermite polynomial, this involves specific integrals
    # The constant C4 relates to the sharp constant in the uncertainty inequality
    
    # Using the framework from the paper, the bound is computed from:
    # 1. The L2 norm of g
    # 2. The L2 norm of x*g
    # 3. The L2 norm of g' (derivative)
    # 4. The L2 norm of x*g'
    
    # For g(x) = P(x) * exp(-x^2/2) where P is a polynomial:
    # ||g||^2 = integral P(x)^2 exp(-x^2) dx
    # ||x*g||^2 = integral x^2 P(x)^2 exp(-x^2) dx
    # ||g'||^2 = integral (P'(x) - x*P(x))^2 exp(-x^2) dx
    # ||x*g'||^2 = integral x^2 (P'(x) - x*P(x))^2 exp(-x^2) dx
    
    # We'll use numerical integration with sufficient accuracy
    # The key quantity for the uncertainty bound is related to:
    # A(g) * A(Fourier(g)) where A(h) = ||h||_2 * ||x*h||_2 / ||h||_2^2 ... 
    
    # Actually, let's use the standard formulation:
    # For the uncertainty inequality, the sharp constant involves:
    # C = sup over g of [ (integral x^2 |g|^2)^(1/2) * (integral |g'|^2)^(1/2) ] / (integral |g|^2)
    
    # Let's compute this properly using Gauss-Hermite quadrature
    
    # Gauss-Hermite quadrature for integral f(x) exp(-x^2) dx
    # Using 100 points for good accuracy
    x_nodes, w_nodes = np.polynomial.hermite.hermgauss(100)
    
    # Build the polynomial P(x) = c0*H0 + c4*H4 + c8*H8 + c12*H12
    H0 = np.ones_like(x_nodes)
    H4 = x_nodes**4 - 6*x_nodes**2 + 3
    H8 = x_nodes**8 - 28*x_nodes**6 + 210*x_nodes**4 - 420*x_nodes**2 + 105
    H12 = x_nodes**12 - 66*x_nodes**10 + 1485*x_nodes**8 - 13860*x_nodes**6 + 49545*x_nodes**4 - 66795*x_nodes**2 + 10395
    
    P = c0*H0 + c4*H4 + c8*H8 + c12*H12
    P_prime = 4*c4*x_nodes**3 - 12*c4*x_nodes + 8*c8*x_nodes**7 - 168*c8*x_nodes**5 + 840*c8*x_nodes**3 - 840*c8*x_nodes \
             + 12*c12*x_nodes**11 - 660*c12*x_nodes**9 + 11880*c12*x_nodes**7 - 83160*c12*x_nodes**5 + 198180*c12*x_nodes**3 - 133590*c12*x_nodes
    
    # g(x) = P(x) * exp(-x^2/2), so |g|^2 = P^2 * exp(-x^2)
    g_sq = P**2 * w_nodes  # Already includes exp(-x^2) from quadrature weights
    
    # ||g||^2 = integral P^2 exp(-x^2) dx
    norm_g_sq = np.sum(g_sq)
    
    if norm_g_sq < 1e-30:
        return np.inf
    
    # ||x*g||^2 = integral x^2 P^2 exp(-x^2) dx
    norm_xg_sq = np.sum(x_nodes**2 * P**2 * w_nodes)
    
    # g'(x) = P'(x) exp(-x^2/2) - x P(x) exp(-x^2/2) = (P' - x*P) exp(-x^2/2)
    # |g'|^2 = (P' - x*P)^2 exp(-x^2)
    g_prime = P_prime - x_nodes * P
    norm_gp_sq = np.sum(g_prime**2 * w_nodes)
    
    # ||x*g'||^2 = integral x^2 (P' - x*P)^2 exp(-x^2) dx
    norm_xgp_sq = np.sum(x_nodes**2 * g_prime**2 * w_nodes)
    
    # The uncertainty product A(g)*A(Fourier(g)) involves:
    # A(g) = sqrt(||g||^2 * ||x*g||^2) / ||g||^2 = sqrt(||x*g||^2 / ||g||^2)
    # Similarly for Fourier transform, using Plancherel and properties
    
    # The sharp constant C4 is the supremum of the product
    # For our test function, the bound is:
    # C_bound = sqrt(norm_xg_sq / norm_g_sq) * sqrt(norm_xgp_sq / norm_gp_sq)
    
    # Wait, let me reconsider. The uncertainty inequality states:
    # (integral x^2 |g|^2 dx / integral |g|^2 dx) * (integral |xi|^2 |G|^2 dxi / integral |G|^2 dxi) >= 1/4
    # where G is the Fourier transform.
    
    # By Plancherel: integral |G|^2 dxi = (1/2pi) integral |g|^2 dx (depending on convention)
    # And integral xi^2 |G|^2 dxi = (1/2pi) integral |g'|^2 dx
    
    # So the product becomes:
    # [integral x^2 |g|^2 / integral |g|^2] * [integral |g'|^2 / integral |g|^2] >= 1/4
    
    # The C4 constant is related to the best constant in this inequality
    # For our purpose, the "upper bound" we're minimizing is:
    # C_bound = [integral x^2 |g|^2 / integral |g|^2] * [integral |g'|^2 / integral |g|^2]
    
    # This should be >= 1/4 for all g, and we want to find g that minimizes this
    # to get the tightest bound.
    
    var_x = norm_xg_sq / norm_g_sq
    var_freq = norm_gp_sq / norm_g_sq
    
    C_bound = var_x * var_freq
    
    return C_bound

def construct():
    """
    Find the three coefficients (c0, c4, c8) that minimize the upper bound on C4.
    The c12 coefficient is derived to ensure g(0) = 0.
    """
    start_time = time.time()
    deadline = start_time + 190.0  # Safety margin before 210s limit
    
    # Initial guess from the prior state of the art
    best_c = np.array([-113.0/100, 1.0/25, 1.0/3240])
    best_bound = _compute_bound(best_c[0], best_c[1], best_c[2])
    
    # Use scipy.optimize for more robust optimization
    try:
        from scipy.optimize import minimize
    except ImportError:
        # Fallback to simple gradient descent if scipy not available
        return best_c
    
    def objective(c):
        return _compute_bound(c[0], c[1], c[2])
    
    # Try multiple starting points
    start_points = [
        best_c,
        np.array([-1.0, 0.1, 0.01]),
        np.array([-0.5, 0.2, 0.05]),
        np.array([-2.0, 0.05, 0.001]),
        np.array([-1.0, 0.0, 0.0]),
    ]
    
    best_result = None
    best_obj = best_bound
    
    for sp in start_points:
        if time.time() > deadline - 30:
            break
        
        try:
            result = minimize(
                objective,
                sp,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if result.fun < best_obj:
                best_obj = result.fun
                best_result = result.x
        
        except Exception:
            continue
    
    if best_result is not None:
        # Final polish with higher precision
        try:
            result = minimize(
                objective,
                best_result,
                method='L-BFGS-B',
                options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if result.fun < best_obj:
                best_obj = result.fun
                best_result = result.x
        except Exception:
            pass
    
    if best_result is not None:
        return np.array(best_result)
    else:
        return best_c

# EVOLVE-BLOCK-END
