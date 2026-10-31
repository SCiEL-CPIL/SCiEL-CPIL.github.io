import numpy as np
from scipy.optimize import minimize
from scipy.special import eval_hermite, factorial
from scipy.integrate import quad

def construct():
    """
    Optimize the three free Hermite coefficients (H0, H4, H8) to minimize
    the normalized uncertainty bound.
    
    New Approach:
    1. Analytical calculation of L2 norm (denominator) using orthogonality of Hermite polynomials.
    2. Numerical integration for L1 norms (numerator) using quad.
    3. Deterministic multi-start local optimization (Nelder-Mead) instead of Differential Evolution.
    """
    # Hermite polynomial values at 0 for n = 0, 4, 8, 12
    H0_0 = 1.0
    H4_0 = 3.0
    H8_0 = 105.0
    H12_0 = 10395.0
    
    # Precompute L2 norm coefficients: || H_n(x) e^{-x^2/2} ||_2^2 = sqrt(pi) * 2^n * n!
    # We use physicist's Hermite polynomials (scipy default).
    # Integral of H_n(x)^2 e^{-x^2} dx = sqrt(pi) * 2^n * n!
    n_vals = [0, 4, 8, 12]
    l2_weights = np.array([np.sqrt(np.pi) * (2**n) * factorial(n) for n in n_vals])

    def get_all_coeffs(a0, a4, a8):
        """Return all four coefficients with a12 determined by g(0)=0."""
        a12 = -(a0 * H0_0 + a4 * H4_0 + a8 * H8_0) / H12_0
        return np.array([a0, a4, a8, a12])

    def compute_l1_norm(coeffs, is_fourier=False):
        """
        Compute L1 norm of g or g_hat.
        g(x) = sum a_n H_n(x) e^{-x^2/2}
        g_hat(xi) = sum a_n (i)^n H_n(xi) e^{-xi^2/2}
        """
        def integrand(x):
            val = 0.0
            for n, a in zip(n_vals, coeffs):
                term = a * eval_hermite(n, x)
                if is_fourier:
                    term *= (1j)**n
                val += term
            return np.abs(val * np.exp(-x**2 / 2))
        
        # Integrate over a sufficient range. e^{-x^2/2} is negligible beyond ~6.
        # Using limit=200 for safety.
        l1_val, _ = quad(integrand, -8.0, 8.0, limit=200, epsabs=1e-12, epsrel=1e-12)
        return l1_val

    def compute_l2_sq(coeffs):
        """
        Compute ||g||_2^2 analytically.
        g(x) = (sum a_n H_n(x)) e^{-x^2/2}
        ||g||_2^2 = sum_{i,j} a_i a_j <H_i, H_j>
        Since H_n are orthogonal w.r.t e^{-x^2}, cross terms vanish.
        """
        return np.sum(coeffs**2 * l2_weights)

    def objective(params):
        """Objective: minimize the normalized uncertainty product."""
        a0, a4, a8 = params
        coeffs = get_all_coeffs(a0, a4, a8)
        
        # Check if coefficients are effectively zero
        if np.allclose(coeffs, 0, atol=1e-12):
            return 1e6
            
        l2_sq_g = compute_l2_sq(coeffs)
        
        if l2_sq_g < 1e-12:
            return 1e6
            
        # Calculate L1 norms
        l1_g = compute_l1_norm(coeffs, is_fourier=False)
        l1_g_hat = compute_l1_norm(coeffs, is_fourier=True)
        
        # The normalized uncertainty bound
        return (l1_g * l1_g_hat) / l2_sq_g

    # Strategy: Multi-start Nelder-Mead
    # We define a grid of starting points to ensure global coverage.
    # The problem is 3-dimensional.
    start_points = []
    # Generate a grid of points in [-5, 5]^3
    # To keep it fast, we use a coarse grid and rely on local optimization.
    grid_vals = np.linspace(-5, 5, 5)
    for x0 in grid_vals:
        for x1 in grid_vals:
            for x2 in grid_vals:
                start_points.append([x0, x1, x2])
                
    # Also add some random points for robustness, but deterministic seed
    rng = np.random.RandomState(42)
    for _ in range(20):
        start_points.append(rng.uniform(-5, 5, 3))

    best_score = 1e6
    best_params = np.array([0.0, 0.0, 0.0])
    
    # To save time, we limit the number of optimizations if the grid is large.
    # 5^3 = 125 points + 20 random = 145 optimizations.
    # Each optimization involves quad calls. Quad is relatively fast for smooth functions.
    # Nelder-Mead is fast for 3D.
    
    for p0 in start_points:
        try:
            res = minimize(
                objective, 
                p0, 
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
            )
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception:
            continue
            
    return np.array(best_params)
