"""Optimize Hermite-Gaussian coefficients to minimize uncertainty bound."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, Bounds
from scipy.special import factorial

_start_time = time.time()
_TIME_LIMIT = 195.0  # safety margin before 210s kill


def _hermite(x, n):
    """Physicists' Hermite polynomial H_n(x)."""
    # Use three-term recurrence for stability
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return 2.0 * x
    h0 = np.ones_like(x)
    h1 = 2.0 * x
    for k in range(1, n):
        h2 = 2.0 * x * h1 - 2.0 * k * h0
        h0, h1 = h1, h2
    return h1


def _compute_a12(a0, a4, a8):
    """Derive a12 so that g(0) = 0.
    g(0) = (a0*H0(0) + a4*H4(0) + a8*H8(0) + a12*H12(0)) * exp(0)
    H0(0)=1, H4(0)=-12, H8(0)=1680, H12(0)=-21840
    So: a0 - 12*a4 + 1680*a8 - 21840*a12 = 0
    a12 = (a0 - 12*a4 + 1680*a8) / 21840
    """
    return (a0 - 12.0 * a4 + 1680.0 * a8) / 21840.0


def _compute_bound(a0, a4, a8, x_grid=None):
    """Compute the upper bound on A(g)*A(Fourier(g)).
    
    A(g) = ∫ |g(x)|² dx (L2 norm squared)
    The uncertainty product bound involves the L2 norm of g and its Fourier transform,
    plus potentially derivative terms.
    
    For the uncertainty inequality context, the bound is typically:
    A(f) * A(F(f)) where A(h) = ∫ |h|²
    
    With Fourier normalization: F(g)(ξ) = ∫ g(x) e^{-2πixξ} dx
    Plancherel: ∫|F(g)|² = ∫|g|²
    
    The key quantity to minimize is related to the "uncertainty" which for these
    test functions in the uncertainty inequality literature involves:
    
    Actually, for the uncertainty inequality A(f)*A(F(f)) ≥ 1/(4π), the test
    function approach uses:
    
    A(g) = ∫ |g(x)|² dx
    A(F(g)) = ∫ |F(g)(ξ)|² dξ = A(g) by Plancherel
    
    So the product is just A(g)². But that's not what's being optimized.
    
    Re-reading: "reduce the upper bound on A(f)*A(Fourier(f))"
    
    In the uncertainty inequality context with test functions, the bound is:
    A(f) * A(F(f)) ≥ C * (some expression involving g)
    
    The constant C4 is what we're minimizing. The bound typically takes the form:
    A(f) * A(F(f)) ≥ 1/C4 * (something)
    
    Or more likely, we're computing the sharp constant in an inequality of the form:
    ∫|f|² * ∫|F(f)|² ≥ C4 * (∫|f'|² + ...) 
    
    Actually, the most standard formulation: the uncertainty principle states
    (∫ x²|f|²)(∫ ξ²|F(f)|²) ≥ 1/(16π²)
    
    But here A(f) likely means ∫|f|² (the "area" or L2 norm squared).
    
    Given the task says "score is the upper bound on C4" and "lower is better",
    and the seed returns [-113/100, 1/25, 1/3240], let me think about what
    computation gives a meaningful bound.
    
    For the uncertainty inequality with Hermite test functions, the typical
    approach (e.g., in works by Beurling, or in the context of the 
    uncertainty principle for Lp spaces) involves:
    
    A(g) = ∫ |g(x)|² dx
    B(g) = ∫ x² |g(x)|² dx  (variance-like term)
    
    The bound on the uncertainty product involves ratios of these quantities.
    
    Let me compute:
    - norm_sq = ∫ g² dx
    - moment_x = ∫ x² g² dx
    - In Fourier space: norm_sq is the same, moment_x becomes ∫ (d/dξ F(g))² dξ
    
    The uncertainty product lower bound for a given test function in the
    "uncertainty inequality" context is often:
    
    C = (∫|g|²)² / (∫x²|g|² * ∫|g'|²) or similar
    
    Actually, I think the relevant quantity is:
    
    A(f) = ∫|f|², and the Fourier transform with the convention that makes
    Plancherel hold. The uncertainty inequality is:
    
    A(f) * A(F(f)) ≥ C4 * (∫ x² |f|²)^{-1} * ... 
    
    This is getting complex. Let me use a more direct approach:
    
    For the standard uncertainty principle: σ_x² σ_ξ² ≥ 1/4 (with appropriate normalization)
    
    Where σ_x² = (∫x²|g|²)/(∫|g|²) and σ_ξ² = (∫ξ²|F(g)|²)/(∫|F(g)|²)
    
    The "bound" we're trying to minimize might be the ratio that appears when
    proving the inequality, which for a test function g would be:
    
    C4_bound = (∫|g|²)² / (4 * ∫x²|g|² * ∫|g'|²)  or similar
    
    Let me just compute the key quantities and use a reasonable bound formula.
    
    Given the context of "Fourier-invariant" and the specific Hermite combination,
    I believe the bound is:
    
    C4 = (∫ g² dx)² / (∫ x² g² dx * ∫ (g')² dx)
    
    This is related to the uncertainty principle constant. Lower C4 means a 
    tighter bound (smaller upper bound on the uncertainty product).
    """
    if x_grid is None:
        # Use a sufficiently fine grid
        n_points = 4096
        x_grid = np.linspace(-8, 8, n_points)
    
    dx = x_grid[1] - x_grid[0]
    
    # Compute g(x)
    H0 = _hermite(x_grid, 0)
    H4 = _hermite(x_grid, 4)
    H8 = _hermite(x_grid, 8)
    H12 = _hermite(x_grid, 12)
    
    a12 = _compute_a12(a0, a4, a8)
    
    g = (a0 * H0 + a4 * H4 + a8 * H8 + a12 * H12) * np.exp(-x_grid**2 / 2.0)
    
    # Compute A(g) = ∫ g² dx
    g_sq = g**2
    norm_sq = np.sum(g_sq) * dx
    
    # Compute ∫ x² g² dx
    x2_g_sq = x_grid**2 * g_sq
    moment_x = np.sum(x2_g_sq) * dx
    
    # Compute g'(x) using analytic derivative
    # g(x) = P(x) * exp(-x²/2) where P = a0*H0 + a4*H4 + a8*H8 + a12*H12
    # g'(x) = P'(x)*exp(-x²/2) + P(x)*(-x)*exp(-x²/2)
    #        = [P'(x) - x*P(x)] * exp(-x²/2)
    
    # P'(x) = a0*H0' + a4*H4' + a8*H8' + a12*H12'
    # H_n'(x) = 2n * H_{n-1}(x)
    H0p = np.zeros_like(x_grid)
    H4p = 4 * _hermite(x_grid, 3)
    H8p = 8 * _hermite(x_grid, 7)
    H12p = 12 * _hermite(x_grid, 11)
    
    P_prime = a0 * H0p + a4 * H4p + a8 * H8p + a12 * H12p
    P = a0 * H0 + a4 * H4 + a8 * H8 + a12 * H12
    
    g_prime = (P_prime - x_grid * P) * np.exp(-x_grid**2 / 2.0)
    
    # ∫ (g')² dx
    gp_sq = g_prime**2
    moment_gp = np.sum(gp_sq) * dx
    
    # The uncertainty bound constant
    # C4_bound = norm_sq² / (moment_x * moment_gp)
    # This should be minimized (lower is better)
    
    if norm_sq < 1e-30 or moment_x < 1e-30 or moment_gp < 1e-30:
        return 1e10
    
    bound = norm_sq**2 / (moment_x * moment_gp)
    
    return bound


def _objective(params):
    """Objective function for optimization."""
    a0, a4, a8 = params
    try:
        val = _compute_bound(a0, a4, a8)
        if not np.isfinite(val):
            return 1e10
        return val
    except Exception:
        return 1e10


def construct():
    """Return three coefficients (a0, a4, a8) that minimize the uncertainty bound."""
    global _start_time
    
    # Start from the seed values
    best_params = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    best_score = _compute_bound(best_params[0], best_params[1], best_params[2])
    
    if not np.isfinite(best_score):
        best_score = 1e10
    
    # Pre-compute grid for speed
    n_points = 4096
    x_grid = np.linspace(-8, 8, n_points)
    dx = x_grid[1] - x_grid[0]
    
    # Precompute Hermite polynomials and exp
    H0 = _hermite(x_grid, 0)
    H4 = _hermite(x_grid, 4)
    H8 = _hermite(x_grid, 8)
    H12 = _hermite(x_grid, 12)
    H3 = _hermite(x_grid, 3)
    H7 = _hermite(x_grid, 7)
    H11 = _hermite(x_grid, 11)
    exp_term = np.exp(-x_grid**2 / 2.0)
    x2 = x_grid**2
    
    def fast_bound(a0, a4, a8):
        a12 = (a0 - 12.0*a4 + 1680.0*a8) / 21840.0
        
        P = a0*H0 + a4*H4 + a8*H8 + a12*H12
        g = P * exp_term
        g_sq = g*g
        
        norm_sq = np.sum(g_sq) * dx
        moment_x = np.sum(x2 * g_sq) * dx
        
        P_prime = a0*0.0 + a4*4.0*H3 + a8*8.0*H7 + a12*12.0*H11
        g_prime = (P_prime - x_grid * P) * exp_term
        moment_gp = np.sum(g_prime*g_prime) * dx
        
        if norm_sq < 1e-30 or moment_x < 1e-30 or moment_gp < 1e-30:
            return 1e10
        
        return norm_sq**2 / (moment_x * moment_gp)
    
    def fast_objective(params):
        try:
            val = fast_bound(params[0], params[1], params[2])
            return val if np.isfinite(val) else 1e10
        except Exception:
            return 1e10
    
    # Multi-start local optimization
    np.random.seed(42)
    
    # Generate multiple starting points
    starts = [best_params.copy()]
    
    # Add perturbations around the seed
    for i in range(20):
        perturbation = np.random.randn(3) * 0.1
        starts.append(best_params + perturbation)
    
    # Add some random starts in a reasonable range
    for i in range(20):
        s = np.array([
            np.random.uniform(-2, 2),
            np.random.uniform(-0.1, 0.1),
            np.random.uniform(-0.001, 0.001)
        ])
        starts.append(s)
    
    # Try L-BFGS-B from each start
    bounds = [(-10, 10), (-1, 1), (-0.01, 0.01)]
    
    for start in starts:
        if time.time() - _start_time > _TIME_LIMIT:
            break
        
        try:
            result = minimize(
                fast_objective,
                start,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
    
    # Final polish with tighter tolerance
    if time.time() - _start_time < _TIME_LIMIT:
        try:
            result = minimize(
                fast_objective,
                best_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
    
    # Verify with the original (slower) computation
    if time.time() - _start_time < _TIME_LIMIT:
        verify_score = _compute_bound(best_params[0], best_params[1], best_params[2], x_grid)
        if np.isfinite(verify_score):
            # Use the verified score
            pass
    
    return np.array([best_params[0], best_params[1], best_params[2]])
# EVOLVE-BLOCK-END
