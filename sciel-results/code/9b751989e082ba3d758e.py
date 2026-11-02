import numpy as np
from scipy.optimize import differential_evolution, minimize
import time
from math import factorial, comb

def construct():
    """
    Constructs the coefficients for the Hermite-Gaussian test function
    to minimize the uncertainty constant upper bound.
    Uses exact analytic Fourier transform for precision.
    """
    start_time = time.time()
    max_time = 165.0
    
    # Hermite polynomials (physicists): H_n(x)
    # Coefficients stored as list where index i is coefficient of x^i
    # H_0 = 1
    # H_4 = 16x^4 - 48x^2 + 12
    # H_8 = 256x^8 - 3584x^6 + 13440x^4 - 13440x^2 + 1680
    # H_12 = 4096x^12 - 75264x^10 + 704448x^8 - 3184320x^6 + 6220800x^4 - 4636800x^2 + 10395
    
    # Precompute polynomial coefficients (index = power of x)
    H0_coeffs = [1.0]  # 1
    H4_coeffs = [12.0, 0.0, -48.0, 0.0, 16.0]  # 12 - 48x^2 + 16x^4
    H8_coeffs = [1680.0, 0.0, -13440.0, 0.0, 13440.0, 0.0, -3584.0, 0.0, 256.0]
    H12_coeffs = [10395.0, 0.0, -4636800.0, 0.0, 6220800.0, 0.0, -3184320.0, 0.0, 704448.0, 0.0, -75264.0, 0.0, 4096.0]
    
    def get_c12(c0, c4, c8):
        return -(c0 + 12.0 * c4 + 1680.0 * c8) / 665280.0
    
    def poly_eval(coeffs, x):
        """Evaluate polynomial with coefficients (index = power) at x."""
        result = np.zeros_like(x)
        for i, c in enumerate(coeffs):
            if c != 0:
                result += c * (x ** i)
        return result
    
    def objective(coeffs):
        c0, c4, c8 = coeffs
        c12 = get_c12(c0, c4, c8)
        
        # Build combined polynomial P(x) = c0*H0 + c4*H4 + c8*H8 + c12*H12
        # Max degree is 12
        P = np.zeros(13)
        for i in range(len(H0_coeffs)):
            P[i] += c0 * H0_coeffs[i]
        for i in range(len(H4_coeffs)):
            P[i] += c4 * H4_coeffs[i]
        for i in range(len(H8_coeffs)):
            P[i] += c8 * H8_coeffs[i]
        for i in range(len(H12_coeffs)):
            P[i] += c12 * H12_coeffs[i]
        
        # L1 norm: ||g||_1 = integral |P(x) e^{-x^2/2}| dx
        # Use adaptive quadrature on a fine grid
        x = np.linspace(-15, 15, 8192)
        g = poly_eval(P, x) * np.exp(-0.5 * x**2)
        l1_norm = np.trapz(np.abs(g), x)
        
        if l1_norm < 1e-15:
            return 1e10
        
        # Analytic Fourier transform
        # If g(x) = P(x) e^{-x^2/2}, then g_hat(xi) = Q(xi) e^{-xi^2/2}
        # where Q is derived from P via:
        # x^n e^{-x^2/2} has FT proportional to (-i)^n H_n(xi) e^{-xi^2/2}
        # More precisely: F[x^n e^{-x^2/2}](xi) = (-i)^n H_n(xi) e^{-xi^2/2} / (2*pi)^(1/2)
        # But we can work with the polynomial part Q(xi) directly.
        # 
        # The key relation: if P(x) = sum_k a_k x^k, then
        # Q(xi) = sum_k a_k (-i)^k H_k(xi)
        # (up to a constant factor that cancels in the ratio)
        
        # Build Q(xi) coefficients. Q is a polynomial of degree 12.
        # Q(xi) = sum_k P[k] * (-i)^k * H_k(xi)
        # We need H_k(xi) for k=0..12. Since P only has even powers (H0,H4,H8,H12 are even),
        # P[k] = 0 for odd k. So Q is a real polynomial.
        
        # Precompute H_k coefficients for k=0..12
        def hermite_coeffs(n):
            """Return coefficients of H_n(x) (index = power of x)."""
            if n == 0:
                return [1.0]
            if n == 1:
                return [0.0, 2.0]
            # Use recurrence: H_{n+1}(x) = 2x H_n(x) - 2n H_{n-1}(x)
            h_prev = hermite_coeffs(n-1)
            h_curr = hermite_coeffs(n-2)
            # h_next[i] = 2*h_curr[i-1] - 2*(n-1)*h_prev[i]
            h_next = np.zeros(n+1)
            for i in range(n+1):
                term1 = 0.0
                if i >= 1:
                    term1 = 2.0 * h_curr[i-1]
                term2 = -2.0 * (n-1) * h_prev[i] if i < len(h_prev) else 0.0
                h_next[i] = term1 + term2
            return h_next.tolist()
        
        # Precompute all needed Hermite coefficients
        H_coeffs_cache = {}
        for k in range(13):
            H_coeffs_cache[k] = hermite_coeffs(k)
        
        # Build Q(xi) = sum_k P[k] * (-i)^k * H_k(xi)
        # Since P[k] = 0 for odd k, and (-i)^even = (-1)^(k/2),
        # Q(xi) = sum_{k even} P[k] * (-1)^(k/2) * H_k(xi)
        
        Q = np.zeros(13)
        for k in range(0, 13, 2):
            if abs(P[k]) > 1e-15:
                factor = (-1.0) ** (k // 2)
                Hk = H_coeffs_cache[k]
                for i in range(len(Hk)):
                    Q[i] += P[k] * factor * Hk[i]
        
        # Find min |Q(xi) e^{-xi^2/2}| over xi
        # Since e^{-xi^2/2} > 0, minimize |Q(xi)| / e^{xi^2/2} = |Q(xi)| * e^{-xi^2/2}
        # Wait: g_hat(xi) = Q(xi) e^{-xi^2/2} (up to constant)
        # So |g_hat(xi)| = |Q(xi)| * e^{-xi^2/2}
        # We want min of this.
        
        xi = np.linspace(-15, 15, 32768)
        Q_val = poly_eval(Q, xi)
        g_hat_mag = np.abs(Q_val) * np.exp(-0.5 * xi**2)
        
        min_g_hat = np.min(g_hat_mag)
        
        if min_g_hat < 1e-15:
            return 1e10
        
        return (l1_norm ** 2) / min_g_hat
    
    # Initial guess
    initial_guess = np.array([-1.13, 0.04, 1.0/3240.0])
    
    bounds = [(-10, 10), (-1, 1), (-0.1, 0.1)]
    
    best_result = None
    best_score = np.inf
    
    elapsed = time.time() - start_time
    if elapsed < max_time:
        try:
            result = differential_evolution(
                objective, 
                bounds, 
                seed=42, 
                maxiter=30, 
                popsize=12,
                tol=1e-8,
                polish=False
            )
            
            if result.fun < best_score:
                best_score = result.fun
                best_result = result.x
                
            elapsed = time.time() - start_time
            if elapsed < max_time - 15:
                res_local = minimize(
                    objective, 
                    best_result, 
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 50, 'ftol': 1e-14}
                )
                
                if res_local.fun < best_score:
                    best_score = res_local.fun
                    best_result = res_local.x
                    
        except Exception:
            pass

    if best_result is None:
        return initial_guess
    
    return np.array(best_result)

# EVOLVE-BLOCK-END
