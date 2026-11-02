"""Optimized program for uncertainty inequality task.
Uses analytic Fourier transform of Hermite-Gaussian functions."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

def _hermite_values(x):
    """Evaluate H0, H4, H8, H12 at points x (vectorized)."""
    x2 = x * x
    H0 = np.ones_like(x)
    H4 = x2 * x2 - 6.0 * x2 + 3.0
    H8 = x2*x2*x2*x2 - 28.0*x2*x2*x2 + 210.0*x2*x2 - 420.0*x2 + 105.0
    x4 = x2 * x2
    x6 = x4 * x2
    x8 = x4 * x4
    x10 = x8 * x2
    x12 = x6 * x6
    H12 = x12 - 42.0*x10 + 840.0*x8 - 7560.0*x6 + 30240.0*x4 - 45360.0*x2 + 10395.0
    return H0, H4, H8, H12

def _compute_bound_analytic(a0, a4, a8, npts=2000, xrange=8.0):
    """
    Compute the uncertainty bound using analytic Fourier transform property.
    
    For f(x) = g(x) * exp(-x^2/2), the Fourier transform is:
    F(f)(x) = i^4 * g(x) * exp(-x^2/2) for the even Hermite components
    More precisely, FT[H_n(x) * exp(-x^2/2)] = i^n * H_n(x) * exp(-x^2/2)
    
    Since all our Hermite degrees are even (0,4,8,12), i^n = 1 for n=0,4,8,12.
    So F(f)(x) = f(x) = g(x) * exp(-x^2/2).
    
    The bound is: ||f||_1 * ||F(f)||_1 / (||f||_2 * ||F(f)||_2)
    Since F(f) = f (up to phase which is 1 for even n), this becomes:
    bound = ||f||_1^2 / ||f||_2^2
    """
    x = np.linspace(-xrange, xrange, npts)
    dx = x[1] - x[0]
    
    H0, H4, H8, H12 = _hermite_values(x)
    
    # a12 from constraint g(0) = 0
    a12 = -(a0 + 3.0*a4 + 105.0*a8) / 10395.0
    
    g = a0*H0 + a4*H4 + a8*H8 + a12*H12
    
    # f(x) = g(x) * exp(-x^2/2)
    f = g * np.exp(-0.5 * x * x)
    
    # Compute norms
    norm1 = np.sum(np.abs(f)) * dx
    norm2 = np.sqrt(np.sum(f * f) * dx)
    
    if norm2 < 1e-15:
        return 1e10
    
    # Since FT is essentially identity for even Hermite-Gaussians (up to i^n = 1),
    # ||F(f)||_1 = ||f||_1 and ||F(f)||_2 = ||f||_2
    # So bound = ||f||_1 * ||f||_1 / (||f||_2 * ||f||_2) = (||f||_1 / ||f||_2)^2
    bound = (norm1 / norm2) ** 2
    
    return bound

def _compute_bound_refined(a0, a4, a8, npts=4000, xrange=10.0):
    """Higher precision bound computation for final evaluation."""
    return _compute_bound_analytic(a0, a4, a8, npts=npts, xrange=xrange)

def construct():
    """
    Find optimal coefficients (a0, a4, a8) that minimize the uncertainty bound.
    Uses differential evolution + local refinement with analytic bound.
    """
    start_time = time.time()
    deadline = start_time + 175.0  # 5s safety margin
    
    best_bound = 1e10
    best_coeffs = np.array([1.0, 0.0, 0.0])
    
    # Phase 1: Differential evolution for global search
    # The bound is scale-invariant, so we constrain to unit sphere
    def objective_sphere(params):
        # params are angles on sphere: theta, phi
        theta, phi = params
        a0 = np.sin(theta) * np.cos(phi)
        a4 = np.sin(theta) * np.sin(phi)
        a8 = np.cos(theta)
        return _compute_bound_analytic(a0, a4, a8, npts=800, xrange=6.0)
    
    def objective_cartesian(params):
        a0, a4, a8 = params
        return _compute_bound_analytic(a0, a4, a8, npts=800, xrange=6.0)
    
    try:
        if time.time() < deadline - 60:
            # Use differential evolution on the sphere
            bounds = [(0.01, np.pi - 0.01), (0, 2*np.pi)]
            de_result = differential_evolution(
                objective_sphere, bounds,
                maxiter=100, popsize=20, tol=1e-8,
                seed=42, polish=True,
                init='sobol'
            )
            if de_result.fun < best_bound:
                best_bound = de_result.fun
                theta, phi = de_result.x
                best_coeffs = np.array([
                    np.sin(theta) * np.cos(phi),
                    np.sin(theta) * np.sin(phi),
                    np.cos(theta)
                ])
    except Exception as e:
        pass
    
    # Phase 2: Local refinement with Nelder-Mead in Cartesian space
    if time.time() < deadline - 30:
        def objective_fast(params):
            a0, a4, a8 = params
            return _compute_bound_analytic(a0, a4, a8, npts=1500, xrange=7.0)
        
        # Refine from best found so far
        try:
            nm_result = minimize(
                objective_fast, best_coeffs, method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            if nm_result.fun < best_bound:
                best_bound = nm_result.fun
                best_coeffs = nm_result.x
        except:
            pass
        
        # Try additional starting points
        if time.time() < deadline - 20:
            # Generate diverse starting points
            rng = np.random.default_rng(123)
            for _ in range(10):
                if time.time() > deadline - 20:
                    break
                # Random point on sphere
                v = rng.normal(size=3)
                v /= np.linalg.norm(v)
                try:
                    nm_result = minimize(
                        objective_fast, v, method='Nelder-Mead',
                        options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12}
                    )
                    if nm_result.fun < best_bound:
                        best_bound = nm_result.fun
                        best_coeffs = nm_result.x
                except:
                    pass
    
    # Phase 3: Final high-precision evaluation
    if time.time() < deadline - 5:
        try:
            final_bound = _compute_bound_refined(best_coeffs[0], best_coeffs[1], best_coeffs[2])
            # Don't update best_coeffs based on final bound since it's the same function
        except:
            pass
    
    return np.array(best_coeffs)
# EVOLVE-BLOCK-END
