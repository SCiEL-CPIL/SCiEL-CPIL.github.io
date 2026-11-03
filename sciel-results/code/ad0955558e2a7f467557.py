"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.integrate import quad

_START_TIME = time.time()
_DEADLINE = 200.0  # 10s margin before 210s kill

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def _hermite_poly(x, n):
    """Compute Hermite polynomial H_n(x) using recurrence."""
    if n == 0:
        return np.ones_like(x, dtype=float)
    if n == 1:
        return 2 * x
    h0 = np.ones_like(x, dtype=float)
    h1 = 2 * x
    for k in range(1, n):
        h2 = 2 * x * h1 - 2 * k * h0
        h0, h1 = h1, h2
    return h1

def _compute_bound(a0, a4, a8):
    """Compute the uncertainty bound analytically/semi-analytically."""
    # a12 determined by g(0) = 0
    # g(0) = (a0 + 12*a4 + 168*a8 + 20901120*a12) * 1 = 0
    a12 = -(a0 + 12*a4 + 168*a8) / 20901120.0
    
    # Coefficients for P(x) = a0*H0 + a4*H4 + a8*H8 + a12*H12
    # H0 = 1
    # H4 = 16x^4 - 48x^2 + 12
    # H8 = 256x^8 - 4480x^6 + 24192x^4 - 3840x^2 + 168
    # H12 = 4096x^12 - 133120x^10 + 2129920x^8 - 13762560x^6 + 41287680x^4 - 51609600x^2 + 20901120
    
    # P(x) = c12*x^12 + c10*x^10 + c8*x^8 + c6*x^6 + c4*x^4 + c2*x^2 + c0
    c0 = a0 + 12*a4 + 168*a8 + 20901120*a12  # This is 0 by construction
    c2 = -48*a4 - 3840*a8 - 51609600*a12
    c4 = 16*a4 + 24192*a8 + 41287680*a12
    c6 = -4480*a8 - 13762560*a12
    c8 = 256*a8 + 2129920*a12
    c10 = -133120*a12
    c12 = 4096*a12
    
    # g(x) = P(x) * exp(-x^2/2)
    # ||g||_2^2 = ∫ P(x)^2 * exp(-x^2) dx
    # For the uncertainty principle, we need A(g) and A(Fourier(g))
    # A(f) is typically the L1 norm or the "area"
    
    # Compute ||g||_1 = ∫ |P(x)| * exp(-x^2/2) dx
    def g_abs(x):
        p = ((c12 * x + c10) * x + c8) * x**2
        p = ((p + c6) * x + c4) * x**2
        p = (p + c2) * x**2
        p = p + c0
        return abs(p) * np.exp(-x*x/2.0)
    
    # Use numerical integration for L1 norm
    l1_val, _ = quad(lambda x: abs(((c12*x**12 + c10*x**10 + c8*x**8 + c6*x**6 + c4*x**4 + c2*x**2 + c0)) * np.exp(-x*x/2.0)), -15, 15, limit=100)
    
    # ||g||_2^2 can be computed via orthogonality or quadrature
    def g_sq(x):
        p = c12*x**12 + c10*x**10 + c8*x**8 + c6*x**6 + c4*x**4 + c2*x**2 + c0
        return (p**2) * np.exp(-x*x)
    
    l2_sq, _ = quad(g_sq, -15, 15, limit=100)
    l2_val = np.sqrt(l2_sq)
    
    # For Fourier transform of P(x)*exp(-x^2/2), use the fact that
    # F[P(x)*exp(-x^2/2)] can be computed via derivatives of Gaussian
    # But for the bound, we need ||F(g)||_1
    # By Plancherel, ||F(g)||_2 = ||g||_2
    # For L1 of Fourier transform, we can use: ||F(g)||_1 = ||g||_1 (for this symmetry)
    # Actually, for the uncertainty principle A(f)*A(F(f)), if A is L1 norm,
    # then by Hausdorff-Young, ||F(g)||_p <= ||g||_q where 1/p + 1/q = 1
    # For p=q=2: ||F(g)||_2 = ||g||_2
    # The product A(g)*A(F(g)) where A is L1:
    # We need ||g||_1 * ||F(g)||_1
    
    # Compute ||F(g)||_1 numerically using FFT on a grid (fast)
    # Or use the fact that for g(x) = P(x)exp(-x^2/2), 
    # F(g)(k) = i^n * d^n/dk^n [sqrt(2*pi)*exp(-k^2/4)] evaluated appropriately
    # This is complex, so let's use a fast numerical approach
    
    # Actually, for the specific uncertainty measure in this task,
    # let's assume A(f) = ||f||_1 and A(F(f)) = ||F(f)||_1
    # The Fourier transform of P(x)*exp(-x^2/2) can be computed via
    # the relation that multiplication by x becomes i*d/dk in Fourier space
    
    # For simplicity and speed, let's compute ||F(g)||_1 using the same quadrature
    # since |F(g)(k)| = |g(-k)| for real even functions... no that's not right.
    # F(g)(k) = ∫ g(x) exp(-ikx) dx
    # |F(g)(k)| is not simply related to g.
    
    # Let me use a fast FFT-based approach
    x_grid = np.linspace(-12, 12, 4096)
    dx = x_grid[1] - x_grid[0]
    p_vals = c12*x_grid**12 + c10*x_grid**10 + c8*x_grid**8 + c6*x_grid**6 + c4*x_grid**4 + c2*x_grid**2 + c0
    g_vals = p_vals * np.exp(-x_grid*x_grid/2.0)
    
    # FFT to get Fourier transform
    ft_vals = np.fft.fftshift(np.fft.fft(g_vals)) * dx
    ft_abs = np.abs(ft_vals)
    
    # Scale k-axis
    k_grid = np.fft.fftshift(np.fft.fftfreq(len(x_grid), d=dx)) * 2 * np.pi
    
    # ||F(g)||_1 ≈ sum |F(g)(k)| * dk
    dk = k_grid[1] - k_grid[0]
    l1_ft = np.sum(ft_abs) * dk
    
    # The uncertainty product
    bound = l1_val * l1_ft
    
    return bound

def construct():
    """Optimize the three coefficients to minimize the uncertainty bound."""
    best_coeffs = np.array([-113.0/100, 1.0/25, 1.0/3240])
    best_bound = _compute_bound(best_coeffs[0], best_coeffs[1], best_coeffs[2])
    
    # Try local optimization
    if _time_left() > 30:
        try:
            def objective(x):
                return _compute_bound(x[0], x[1], x[2])
            
            result = minimize(objective, best_coeffs, method='Nelder-Mead', 
                            options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8,
                                   'adaptive': True})
            if result.fun < best_bound:
                best_bound = result.fun
                best_coeffs = result.x
        except Exception:
            pass
    
    # Try a few random perturbations if time allows
    if _time_left() > 20:
        rng = np.random.RandomState(42)
        for _ in range(50):
            if _time_left() < 5:
                break
            pert = rng.normal(0, 0.1, 3)
            trial = best_coeffs + pert
            try:
                b = _compute_bound(trial[0], trial[1], trial[2])
                if b < best_bound:
                    best_bound = b
                    best_coeffs = trial
            except Exception:
                pass
    
    return best_coeffs
# EVOLVE-BLOCK-END
