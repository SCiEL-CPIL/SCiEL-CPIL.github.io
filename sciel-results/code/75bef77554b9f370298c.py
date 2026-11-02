import numpy as np
import scipy.optimize as opt
import time

def construct():
    """
    Finds the coefficients for the Hermite polynomial combination H0, H4, H8 
    (with H12 fixed by g(0)=0) that minimizes the uncertainty product 
    A(f)*A(Fourier(f)).
    """
    start_time = time.time()
    deadline = 170.0  # Safety margin before 180s limit

    # Coefficients of Hermite polynomials H_n(x)
    # H_0(x) = 1
    # H_4(x) = x^4 - 6x^2 + 3
    # H_8(x) = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    # H_12(x) = x^12 - 66x^10 + 1260x^8 - 9240x^6 + 23100x^4 - 27720x^2 + 10395
    
    # Values at x=0
    H0_0 = 1.0
    H4_0 = 3.0
    H8_0 = 105.0
    H12_0 = 10395.0

    # Normalization constants for Hermite polynomials: 
    # int H_n(x)^2 e^{-x^2} dx = sqrt(pi) 2^n n!
    norm_sq = {
        0: np.sqrt(np.pi) * 1,
        4: np.sqrt(np.pi) * 16 * 24,   # 2^4 * 4!
        8: np.sqrt(np.pi) * 256 * 40320, # 2^8 * 8!
        12: np.sqrt(np.pi) * 4096 * 479001600 # 2^12 * 12!
    }

    def get_c12(c0, c4, c8):
        # g(0) = 0 => P(0) = 0 => c0*H0(0) + c4*H4(0) + c8*H8(0) + c12*H12(0) = 0
        return -(c0 * H0_0 + c4 * H4_0 + c8 * H8_0) / H12_0

    def compute_uncertainty_product(coeffs):
        c0, c4, c8 = coeffs
        c12 = get_c12(c0, c4, c8)
        
        # We need to compute:
        # 1. N = int |g(x)|^2 dx
        # 2. M_x = int x^2 |g(x)|^2 dx
        # 3. M_k = int k^2 |G(k)|^2 dk
        # Uncertainty product = sqrt(M_x/N) * sqrt(M_k/N) = sqrt(M_x * M_k) / N
        
        # Using orthogonality and properties of Hermite polynomials:
        # g(x) = (c0 H0 + c4 H4 + c8 H8 + c12 H12) e^{-x^2/2}
        
        # N = sum_{i,j} c_i c_j int H_i H_j e^{-x^2} dx = sum_i c_i^2 norm_sq[i]
        N = c0**2 * norm_sq[0] + c4**2 * norm_sq[4] + c8**2 * norm_sq[8] + c12**2 * norm_sq[12]
        
        if N <= 1e-12:
            return np.inf

        # M_x = int x^2 |g(x)|^2 dx
        # x H_n(x) = 1/2 H_{n+1}(x) + n H_{n-1}(x)
        # x^2 H_n(x) = ... this gets complicated. 
        # Alternative: Use explicit integration or known moments.
        # Let's use the property that for f(x) = P(x) e^{-x^2/2}, 
        # the moments can be calculated via derivatives or explicit quadrature.
        # Given the low degree (max 12), explicit numerical integration is safe and fast.
        
        # We will use Gauss-Hermite quadrature for efficiency and accuracy.
        # int e^{-x^2} f(x) dx = sum w_i f(x_i)
        # Here we have int e^{-x^2} [P(x)]^2 dx and int x^2 e^{-x^2} [P(x)]^2 dx
        
        # np.polynomial.hermite.hermgauss gives nodes and weights for int e^{-x^2} f(x) dx
        # We need enough points to integrate a polynomial of degree 24 (since P is deg 12, P^2 is deg 24).
        # Gauss-Hermite with n points integrates exactly polynomials up to degree 2n-1.
        # So we need 2n-1 >= 24 => n >= 12.5 => n=13 is enough. Let's use 20 for safety.
        
        x_nodes, w_nodes = np.polynomial.hermite.hermgauss(20)
        
        # P(x) = c0 H0(x) + c4 H4(x) + c8 H8(x) + c12 H12(x)
        # We need H_n(x) for n=0,4,8,12 at x_nodes.
        # We can compute these using recurrence or explicit formulas.
        # Recurrence: H_{n+1}(x) = 2x H_n(x) - 2n H_{n-1}(x)
        
        H_vals = np.zeros((13, len(x_nodes)))
        H_vals[0, :] = 1.0
        if len(x_nodes) > 0:
            H_vals[1, :] = 2 * x_nodes
            for n in range(1, 12):
                H_vals[n+1, :] = 2 * x_nodes * H_vals[n, :] - 2 * n * H_vals[n-1, :]
        
        P_vals = c0 * H_vals[0, :] + c4 * H_vals[4, :] + c8 * H_vals[8, :] + c12 * H_vals[12, :]
        
        # N_quad = sum w_i P(x_i)^2
        N_quad = np.sum(w_nodes * P_vals**2)
        
        # M_x_quad = sum w_i x_i^2 P(x_i)^2
        M_x_quad = np.sum(w_nodes * x_nodes**2 * P_vals**2)
        
        # For frequency space, by Parseval and properties of FT:
        # int k^2 |G(k)|^2 dk = int |g'(x)|^2 dx (for unitary FT)
        # g(x) = P(x) e^{-x^2/2}
        # g'(x) = (P'(x) - x P(x)) e^{-x^2/2}
        # So M_k = int (P'(x) - x P(x))^2 e^{-x^2} dx
        
        # We need P'(x).
        # P'(x) = c0 H0'(x) + c4 H4'(x) + c8 H8'(x) + c12 H12'(x)
        # H_n'(x) = 2n H_{n-1}(x)
        P_prime_vals = c0 * 0 * H_vals[0, :] + c4 * 2*4 * H_vals[3, :] + c8 * 2*8 * H_vals[7, :] + c12 * 2*12 * H_vals[11, :]
        # Note: H_0' = 0. H_4' = 8 H_3. H_8' = 16 H_7. H_12' = 24 H_11.
        
        deriv_term = P_prime_vals - x_nodes * P_vals
        M_k_quad = np.sum(w_nodes * deriv_term**2)
        
        # Check if quadrature is stable
        if N_quad <= 1e-12 or not np.isfinite(N_quad):
            return np.inf
            
        # The uncertainty product is sqrt(M_x/N) * sqrt(M_k/N)
        # = sqrt(M_x * M_k) / N
        # Note: N from quadrature should match N from orthogonality. 
        # We use quadrature values for consistency.
        
        prod = np.sqrt(M_x_quad * M_k_quad) / N_quad
        
        if not np.isfinite(prod) or prod < 0:
            return np.inf
            
        return prod

    # Initial guess from seed
    x0 = np.array([-113 / 100, 1 / 25, 1 / 3240])
    
    best_score = compute_uncertainty_product(x0)
    best_coeffs = x0.copy()
    
    # Optimization
    # Use Nelder-Mead as it doesn't require gradients and is robust for low dimensions
    # We can also try L-BFGS-B if we provide gradients, but NM is sufficient for 3 vars.
    
    rng = np.random.RandomState(42)
    
    # Run a few restarts to avoid local minima
    num_restarts = 10
    
    for i in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        if i == 0:
            x_start = x0
        else:
            # Perturb the best solution found so far
            # Scale of perturbation
            scale = 0.1 * (1 + i/10.0)
            x_start = best_coeffs + rng.normal(0, scale, 3)
            
        try:
            res = opt.minimize(compute_uncertainty_product, x_start, method='Nelder-Mead',
                               options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 200, 'adaptive': True})
            
            if np.isfinite(res.fun) and res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x.copy()
                
        except Exception:
            pass

    return best_coeffs
