"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution

_DEADLINE = time.time() + 200  # safety margin before 210s kill


def _time_left():
    return _DEADLINE - time.time()


def _hermite_phys(n, x):
    """Physicists' Hermite polynomial H_n(x)."""
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return 2 * x
    h0 = np.ones_like(x)
    h1 = 2 * x
    for k in range(2, n + 1):
        h2 = 2 * x * h1 - 2 * (k - 1) * h0
        h0, h1 = h1, h2
    return h1


def _compute_bound(a0, a4, a8):
    """
    Compute the C4 upper bound for the test function
    g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
    where a12 is chosen so g(0) = 0.
    
    The bound is A(f)*A(F) where A is the uncertainty measure.
    We use numerical quadrature for the norms.
    """
    # g(0) = a0 + a4*3 + a8*105 + a12*945 = 0
    # H0(0)=1, H4(0)=3, H8(0)=105, H12(0)=945
    a12 = -(a0 + 3 * a4 + 105 * a8) / 945.0

    # Use Gauss-Hermite quadrature for integrals of form int f(x) exp(-x^2) dx
    # Our function is g(x) = P(x) * exp(-x^2/2), so g(x)^2 = P(x)^2 * exp(-x^2)
    # We need int g(x)^2 dx and int (x*g(x))^2 dx for A(f)^2
    # And for Fourier: F(g)(xi) = sqrt(2*pi) * P(-i*xi/2... ) -- actually let's use the known result
    # For g(x) = P(x)*exp(-x^2/2), the Fourier transform is related to P evaluated at imaginary args
    # Actually, let's just do numerical integration over a large domain.
    
    # Use Gauss-Hermite for the L2 norm: int |g(x)|^2 dx = int P(x)^2 * exp(-x^2) dx
    # GH quadrature: int f(x)*exp(-x^2) dx ≈ sum w_i * f(x_i)
    n_quad = 200
    x_pts, w_pts = np.polynomial.hermite.hermgauss(n_quad)
    
    P = (a0 * _hermite_phys(0, x_pts) + a4 * _hermite_phys(4, x_pts) + 
         a8 * _hermite_phys(8, x_pts) + a12 * _hermite_phys(12, x_pts))
    
    # A(f)^2 = int |g(x)|^2 dx = sum w_i * P(x_i)^2
    norm_sq = np.sum(w_pts * P**2)
    
    if norm_sq <= 0:
        return 1e10
    
    # A(f) = sqrt( int (x*g)^2 dx / int g^2 dx - (int x*g^2 dx / int g^2 dx)^2 )
    # int (x*g)^2 dx = int x^2 * P^2 * exp(-x^2) dx
    x2_P2 = np.sum(w_pts * x_pts**2 * P**2)
    x_P2 = np.sum(w_pts * x_pts * P**2)
    
    var_f = x2_P2 / norm_sq - (x_P2 / norm_sq)**2
    
    if var_f <= 0:
        return 1e10
    
    # For the Fourier transform: F(g)(xi) = sqrt(2*pi) * integral g(x) exp(-i*xi*x) dx
    # For g(x) = P(x)*exp(-x^2/2), we can use the fact that the Fourier transform of
    # H_n(x)*exp(-x^2/2) is i^n * H_n(xi) * exp(-xi^2/2) * sqrt(2*pi) ... 
    # Actually the standard result: if f(x) = H_n(x)*exp(-x^2/2), then 
    # F(f)(xi) = i^n * H_n(xi) * exp(-xi^2/2) * sqrt(2*pi)
    # Wait, let me be more careful. The Fourier transform convention matters.
    # Let's use: F(f)(xi) = int f(x) exp(-2*pi*i*x*xi) dx
    # For exp(-x^2/2): F = sqrt(2*pi) * exp(-2*pi^2*xi^2)
    # This is getting complicated. Let me use a different approach.
    # 
    # Actually, for the uncertainty product, we can use the fact that for any f:
    # A(f) * A(F(f)) >= 1/(4*pi) (standard Heisenberg)
    # The bound C4 is the specific value of A(f)*A(F(f)) for our test function.
    # 
    # Let me compute A(F(f)) numerically by computing the Fourier transform on a grid.
    
    # Alternative: use the property that for g(x) = P(x)*exp(-x^2/2),
    # the Fourier transform (with convention F(f)(xi) = int f(x) exp(-i*xi*x) dx)
    # is sqrt(2*pi) * Q(xi) * exp(-xi^2/2) where Q is a related polynomial.
    # Specifically, F(H_n(x)*exp(-x^2/2))(xi) = i^n * H_n(xi) * exp(-xi^2/2) * sqrt(2*pi)
    # Wait, I need to verify this. The standard result is:
    # F(H_n(x)*e^{-x^2/2}) = i^n * H_n(xi) * e^{-xi^2/2} * sqrt(2*pi)
    # where F(f)(xi) = (1/sqrt(2*pi)) * int f(x) e^{-ixi x} dx
    # 
    # Let me just use numerical FFT on a fine grid.
    
    # Actually, let me reconsider. The key insight is that the Fourier transform of
    # a Hermite-Gaussian is another Hermite-Gaussian. Specifically:
    # If g(x) = sum_k c_k H_k(x) exp(-x^2/2), then
    # F(g)(xi) = sqrt(2*pi) * sum_k c_k * i^k * H_k(xi) * exp(-xi^2/2)
    # (with the unitary convention)
    
    # So A(F(g)) can be computed the same way as A(g) but with coefficients c_k * i^k
    
    # With the convention F(f)(xi) = int f(x) exp(-i xi x) dx / sqrt(2*pi):
    # F(H_k(x) exp(-x^2/2))(xi) = i^k H_k(xi) exp(-xi^2/2)
    
    # So the Fourier transform has coefficients c_k * i^k
    
    # A(F(g))^2 = var of xi under |F(g)|^2 / |F(g)|^2
    # = [sum w_i * xi_i^2 * |Q(xi_i)|^2] / [sum w_i * |Q(xi_i)|^2] - (sum w_i * xi_i * |Q(xi_i)|^2 / sum w_i * |Q(xi_i)|^2)^2
    # where Q(xi) = sum_k c_k * i^k * H_k(xi)
    
    # Since |i^k|^2 = 1, |Q(xi)|^2 = |sum c_k i^k H_k(xi)|^2
    # This is NOT the same as (sum c_k H_k(xi))^2 in general because of cross terms with i^k factors.
    
    # Let me compute it properly.
    Q_re = np.zeros_like(x_pts)
    Q_im = np.zeros_like(x_pts)
    for k, c in [(0, a0), (4, a4), (8, a8), (12, a12)]:
        Hk = _hermite_phys(k, x_pts)
        phase = 1j ** k
        Q_re += c * Hk * phase.real
        Q_im += c * Hk * phase.imag
    
    Q_abs2 = Q_re**2 + Q_im**2
    fourier_norm_sq = np.sum(w_pts * Q_abs2)
    
    if fourier_norm_sq <= 0:
        return 1e10
    
    xi2_Q2 = np.sum(w_pts * x_pts**2 * Q_abs2)
    xi_Q2 = np.sum(w_pts * x_pts * Q_abs2)
    
    var_F = xi2_Q2 / fourier_norm_sq - (xi_Q2 / fourier_norm_sq)**2
    
    if var_F <= 0:
        return 1e10
    
    # The uncertainty product A(f)*A(F(f))
    # A(f) = sqrt(var_f), A(F) = sqrt(var_F)
    # But we need to account for the normalization convention.
    # With F(f)(xi) = (1/sqrt(2*pi)) int f(x) e^{-ixix} dx, Parseval gives:
    # int |f|^2 dx = int |F(f)|^2 dxi
    # So A(f) = sqrt(int x^2 |f|^2 / int |f|^2 - (int x|f|^2/int|f|^2)^2)
    # A(F) = sqrt(int xi^2 |F|^2 / int |F|^2 - (int xi|F|^2/int|F|^2)^2)
    # And the product is sqrt(var_f * var_F)
    
    return np.sqrt(var_f * var_F)


def _objective_log(log_coeffs):
    """Objective in log-space. Coefficients can be negative, so we need a different approach.
    Actually, the coefficients can be negative (a0 = -113/100). So log-space doesn't directly work.
    Instead, let's use the raw coefficients with differential evolution on a bounded domain."""
    a0, a4, a8 = log_coeffs
    try:
        val = _compute_bound(a0, a4, a8)
        if np.isnan(val) or np.isinf(val):
            return 1e10
        return val
    except Exception:
        return 1e10


def construct():
    t_start = time.time()
    seed = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best_val = _compute_bound(*seed)
    best_coeffs = seed.copy()
    
    # Differential evolution on the 3 coefficients
    # Bounds: based on the seed values, explore a reasonable range
    bounds = [(-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)]
    
    max_iter = 50
    tol = 1e-8
    
    try:
        result = differential_evolution(
            _objective_log,
            bounds=bounds,
            maxiter=max_iter,
            tol=tol,
            seed=42,
            popsize=10,
            init='sobol',
            callback=lambda xk, convergence: (np.any(convergence) or time.time() - t_start > 180)
        )
        
        if result.success or result.fun < best_val:
            if result.fun < best_val:
                best_val = result.fun
                best_coeffs = result.x.copy()
    except Exception:
        pass
    
    # Check time and return best
    if time.time() - t_start > 185:
        return best_coeffs
    
    # Try a few more random starts if time permits
    rng = np.random.RandomState(123)
    for _ in range(5):
        if time.time() - t_start > 190:
            break
        trial = rng.uniform(-3, 3, size=3)
        try:
            val = _compute_bound(*trial)
            if val < best_val:
                best_val = val
                best_coeffs = trial.copy()
        except Exception:
            pass
    
    return best_coeffs
# EVOLVE-BLOCK-END
