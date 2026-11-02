"""Rewritten program: fast numerical optimization of Hermite coefficients for uncertainty bound."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.optimize import differential_evolution
from numpy.polynomial.hermite import hermgauss


def construct():
    """Return three coefficients for H0, H4, H8 Hermite combination."""
    start_time = time.time()
    deadline = start_time + 190.0  # Safety margin for 210s wall-clock limit

    seed = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])

    # Gauss-Hermite quadrature nodes and weights
    # Integrand is P(x)^k e^{-x^2/2}. Change var x = sqrt(2)t.
    # dx = sqrt(2) dt. e^{-x^2/2} = e^{-t^2}.
    # Integral = sqrt(2) * int P(sqrt(2)t)^k e^{-t^2} dt.
    # We need exactness for degree 12 polynomials raised to power 4 -> degree 48.
    # Wait.
    # Norm 2: int P^2 e^{-x^2/2} dx. P is deg 12. P^2 is deg 24.
    # P(sqrt(2)t)^2 is deg 24.
    # GH quadrature with N points is exact for deg 2N-1.
    # 2N-1 >= 24 => 2N >= 25 => N >= 12.5 => N=13.
    # Norm 4: int P^4 e^{-x^2/2} dx. P^4 is deg 48.
    # P(sqrt(2)t)^4 is deg 48.
    # 2N-1 >= 48 => 2N >= 49 => N >= 24.5 => N=25.
    # So we need N=25 for L4 norms.
    
    nodes, weights = hermgauss(25)
    sqrt2 = np.sqrt(2.0)
    x_nodes = sqrt2 * nodes
    # Weights for the original integral:
    # int f(x) e^{-x^2/2} dx = sqrt(2) sum w_i f(x_i)
    # Here f(x) is the polynomial part.
    # So norm2_sq = sqrt(2) * sum(w_i * P(x_i)^2)
    # norm4_4   = sqrt(2) * sum(w_i * P(x_i)^4)
    
    def get_a12(a0, a4, a8):
        """Compute a12 so that g(0) = 0."""
        # H0(0)=1, H4(0)=-3, H8(0)=105, H12(0)=-10395
        # g(0) = (a0*1 + a4*(-3) + a8*105 + a12*(-10395)) * 1 = 0
        # a12 = (a0 - 3*a4 + 105*a8) / 10395
        return (a0 - 3.0 * a4 + 105.0 * a8) / 10395.0

    def hermite_p(x, n):
        """Compute physicists' Hermite polynomial H_n(x) for n in {0,4,8,12}."""
        if n == 0:
            return np.ones_like(x)
        elif n == 4:
            return 16.0 * x**4 - 48.0 * x**2 + 12.0
        elif n == 8:
            return 256.0 * x**8 - 2304.0 * x**6 + 5760.0 * x**4 - 4320.0 * x**2 + 504.0
        elif n == 12:
            return (4096.0 * x**12 - 55296.0 * x**10 + 303104.0 * x**8
                    - 589824.0 * x**6 + 408192.0 * x**4 - 77760.0 * x**2 + 10395.0)
        else:
            raise ValueError(f"Unsupported n={n}")

    def compute_poly_values(coeffs):
        """
        Compute P(x_nodes) and Q(x_nodes) for given coeffs.
        P(x) = a0*H0 + a4*H4 + a8*H8 + a12*H12
        Q(x) = a0*H0 + a4*H4 + a8*H8 - a12*H12  (Fourier transform poly part)
        """
        a0, a4, a8 = coeffs
        a12 = get_a12(a0, a4, a8)
        
        H0 = np.ones_like(x_nodes)
        H4 = hermite_p(x_nodes, 4)
        H8 = hermite_p(x_nodes, 8)
        H12 = hermite_p(x_nodes, 12)
        
        P_vals = a0 * H0 + a4 * H4 + a8 * H8 + a12 * H12
        Q_vals = a0 * H0 + a4 * H4 + a8 * H8 - a12 * H12
        
        return P_vals, Q_vals

    def objective(coeffs):
        """
        Compute the upper bound on C4.
        """
        P_vals, Q_vals = compute_poly_values(coeffs)
        
        # Norms
        # ||g||_2^2 = sqrt(2) * sum(w * P^2)
        norm2_g_sq = sqrt2 * np.sum(weights * P_vals**2)
        norm2_q_sq = sqrt2 * np.sum(weights * Q_vals**2)
        
        # ||g||_4^4 = sqrt(2) * sum(w * P^4)
        norm4_g_4 = sqrt2 * np.sum(weights * P_vals**4)
        norm4_q_4 = sqrt2 * np.sum(weights * Q_vals**4)
        
        if norm2_g_sq < 1e-15 or norm2_q_sq < 1e-15 or norm4_g_4 < 1e-15 or norm4_q_4 < 1e-15:
            return 1e10
            
        norm2_g = np.sqrt(norm2_g_sq)
        norm2_q = np.sqrt(norm2_q_sq)
        norm4_g = norm4_g_4 ** 0.25
        norm4_q = norm4_q_4 ** 0.25
        
        bound = (norm4_g * norm4_q) / (norm2_g * norm2_q)
        return bound

    # Initial evaluation
    best_coeffs = seed.copy()
    best_val = objective(seed)
    
    # Differential Evolution for global search
    # Bounds: The seed is around [-1, 0.04, 0.0003].
    # Let's search in a reasonable range.
    bounds = [(-10.0, 10.0), (-10.0, 10.0), (-10.0, 10.0)]
    
    if time.time() < deadline - 60.0:
        try:
            de_result = differential_evolution(
                objective,
                bounds=bounds,
                seed=42,
                maxiter=500,
                tol=1e-8,
                polish=False, # We will polish later
                init='sobol'
            )
            if de_result.fun < best_val:
                best_val = de_result.fun
                best_coeffs = de_result.x.copy()
        except Exception:
            pass

    # Local refinement with Nelder-Mead
    if time.time() < deadline - 10.0:
        try:
            result = minimize(
                objective,
                best_coeffs,
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_coeffs = result.x.copy()
        except Exception:
            pass
            
    # One more polish if time remains
    if time.time() < deadline - 2.0:
        try:
            result = minimize(
                objective,
                best_coeffs,
                method='Powell',
                options={'maxiter': 5000, 'xtol': 1e-12, 'ftol': 1e-14}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_coeffs = result.x.copy()
        except Exception:
            pass

    return np.array(best_coeffs)
# EVOLVE-BLOCK-END
