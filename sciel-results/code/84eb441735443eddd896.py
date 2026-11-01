import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad
import time

def construct():
    start_time = time.time()
    deadline = start_time + 170.0  # 10s safety margin

    # Hermite Polynomials (Physicists')
    # H0(x) = 1
    # H4(x) = x^4 - 6x^2 + 3
    # H8(x) = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    # H12(x) = x^12 - 84x^10 + 2520x^8 - 30240x^6 + 166320x^4 - 302400x^2 + 10395
    # H_n(0): H0=1, H4=3, H8=105, H12=10395
    
    def get_c12(c0, c4, c8):
        return -(c0 + 3.0*c4 + 105.0*c8) / 10395.0

    def H0(x):
        return np.ones_like(x) if isinstance(x, np.ndarray) else 1.0

    def H4(x):
        return x**4 - 6.0*x**2 + 3.0

    def H8(x):
        return x**8 - 28.0*x**6 + 210.0*x**4 - 420.0*x**2 + 105.0

    def H12(x):
        return x**12 - 84.0*x**10 + 2520.0*x**8 - 30240.0*x**6 + 166320.0*x**4 - 302400.0*x**2 + 10395.0

    def g(x, c0, c4, c8):
        c12 = get_c12(c0, c4, c8)
        if isinstance(x, np.ndarray):
            P = c0*H0(x) + c4*H4(x) + c8*H8(x) + c12*H12(x)
        else:
            P = c0*1.0 + c4*H4(x) + c8*H8(x) + c12*H12(x)
        return P * np.exp(-x**2 / 2.0)

    # The uncertainty bound for the test function is C4 = (∫|x*g(x)|dx)^2
    # Since g is self-Fourier (up to constants), ||g||_1 * ||ĝ||_1 = (∫|g|dx)^2
    # But the standard uncertainty bound involves ∫|x*g(x)|dx
    # Let's use C4 = (∫|x*g(x)|dx)^2 as the bound to minimize

    def compute_bound(c0, c4, c8):
        """Compute the uncertainty bound C4 = (∫|x*g(x)|dx)^2"""
        def integrand(x):
            return abs(x * g(x, c0, c4, c8))
        
        # Integrate over [-10, 10] where the function is essentially zero outside
        integral, _ = quad(integrand, -10, 10, limit=200, epsabs=1e-10, epsrel=1e-10)
        return integral ** 2

    # For optimization, use a smooth approximation of |x*g(x)|
    # |u| ≈ sqrt(u^2 + eps) for smoothness
    def smooth_bound(c0, c4, c8, eps=1e-6):
        """Smooth approximation of the bound for gradient-based optimization"""
        def integrand(x):
            val = x * g(x, c0, c4, c8)
            return np.sqrt(val**2 + eps)
        
        integral, _ = quad(integrand, -10, 10, limit=200, epsabs=1e-10, epsrel=1e-10)
        return integral ** 2

    # Seed values
    c0_init = -113.0 / 100.0
    c4_init = 1.0 / 25.0
    c8_init = 1.0 / 3240.0

    # Try multiple starting points and optimization methods
    best_result = None
    best_score = np.inf

    starts = [
        [c0_init, c4_init, c8_init],  # Seed
        [1.0, 0.0, 0.0],
        [0.5, 0.1, 0.0],
        [0.5, -0.1, 0.0],
        [1.0, 0.05, 0.0],
        [0.8, 0.0, 0.05],
        [1.2, 0.0, -0.05],
        [0.3, 0.02, 0.001],
    ]

    for x0 in starts:
        if time.time() > deadline:
            break

        try:
            # Use SLSQP with the smooth bound as objective
            def objective(vars):
                c0, c4, c8 = vars
                return smooth_bound(c0, c4, c8)

            res = minimize(objective, x0, method='SLSQP',
                           options={'maxiter': 200, 'ftol': 1e-10})

            if res.success and res.fun < best_score:
                # Verify with exact (non-smooth) bound
                exact_score = compute_bound(res.x[0], res.x[1], res.x[2])
                if exact_score < best_score:
                    best_score = exact_score
                    best_result = res.x
        except Exception as e:
            pass

    # Also try a grid search over a small region around the seed
    if time.time() < deadline - 30:
        c0_grid = np.linspace(c0_init - 0.5, c0_init + 0.5, 20)
        c4_grid = np.linspace(c4_init - 0.1, c4_init + 0.1, 10)
        c8_grid = np.linspace(c8_init - 0.01, c8_init + 0.01, 10)

        for c0_val in c0_grid:
            for c4_val in c4_grid:
                for c8_val in c8_grid:
                    if time.time() > deadline - 20:
                        break
                    try:
                        score = compute_bound(c0_val, c4_val, c8_val)
                        if score < best_score:
                            best_score = score
                            best_result = np.array([c0_val, c4_val, c8_val])
                    except:
                        pass

    if best_result is None:
        return np.array([-113 / 100, 1 / 25, 1 / 3240])

    return np.array(best_result)
