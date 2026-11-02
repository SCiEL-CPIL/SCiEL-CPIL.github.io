"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Precompute constants for the Hermite-Gaussian basis
# g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
# H_n are physicists' Hermite polynomials
# The constraint g(0) = 0 determines a12 in terms of a0, a4, a8
# H0(0)=1, H4(0)=-12, H8(0)=1680, H12(0)=-145152

def _compute_bound(a0, a4, a8):
    """
    Compute the C4 uncertainty bound for the given coefficients.
    
    The test function is g(x) = (a0*H0(x) + a4*H4(x) + a8*H8(x) + a12*H12(x)) * exp(-x^2/2)
    where a12 is chosen so that g(0) = 0.
    
    The uncertainty product bound involves:
    - ||g||_1^2 (L1 norm squared)
    - min |hat{g}(k)| (minimum of the Fourier transform magnitude)
    
    The bound on A(f)*A(F(f)) is proportional to ||g||_1 / min|hat{g}|
    or similar ratio. Let's use the standard formulation.
    """
    # Compute a12 from g(0) = 0
    # g(0) = (a0*1 + a4*(-12) + a8*1680 + a12*(-145152)) * 1 = 0
    # a12 = (a0 - 12*a4 + 1680*a8) / 145152
    a12 = (a0 - 12.0*a4 + 1680.0*a8) / 145152.0
    
    # We need to compute ||g||_1 and min|hat{g}|
    # For efficiency, use numerical integration on a grid
    
    # The Fourier transform of H_n(x)*exp(-x^2/2) is (-i)^n * H_n(k) * exp(-k^2/2)
    # So hat{g}(k) = (a0*(-i)^0*H0(k) + a4*(-i)^4*H4(k) + a8*(-i)^8*H8(k) + a12*(-i)^12*H12(k)) * exp(-k^2/2)
    # (-i)^0 = 1, (-i)^4 = 1, (-i)^8 = 1, (-i)^12 = 1
    # So hat{g}(k) = (a0*H0(k) + a4*H4(k) + a8*H8(k) + a12*H12(k)) * exp(-k^2/2)
    # Note: the Fourier transform is the same polynomial structure!
    
    # For the L1 norm, we need to integrate |g(x)| over R
    # For min|hat{g}|, we need to find the minimum of |hat{g}(k)| over R
    
    # Use a reasonable grid
    x = np.linspace(-15, 15, 3001)
    dx = x[1] - x[0]
    
    # Compute Hermite polynomials
    H0 = np.ones_like(x)
    H4 = x**4 - 6*x**2 + 3
    H8 = x**8 - 28*x**6 + 210*x**4 - 420*x**2 + 105
    H12 = x**12 - 42*x**10 + 630*x**8 - 3780*x**6 + 7560*x**4 - 5040*x**2 + 945
    
    gauss = np.exp(-x**2 / 2.0)
    
    g = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * gauss
    
    # L1 norm
    l1_norm = np.sum(np.abs(g)) * dx
    
    # Fourier transform magnitude
    # hat{g}(k) has the same polynomial structure
    hat_g = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * gauss  # same as g since FT is symmetric here
    
    # Find minimum of |hat{g}|, but avoid points where it's exactly 0
    # We want min over k of |hat{g}(k)|, but if it touches 0 the bound is infinite
    # So we need to find the minimum away from zeros, or the bound is infinite
    
    abs_hat_g = np.abs(hat_g)
    
    # Find the minimum, but if it's very close to 0, the bound is large
    # Use a small threshold to avoid division by zero
    min_val = np.min(abs_hat_g)
    
    if min_val < 1e-12:
        return np.inf
    
    # The C4 bound is related to l1_norm / min_val or (l1_norm)^2 / min_val
    # Based on the problem statement: "upper bound on A(f)*A(Fourier(f))"
    # and the score being "the upper bound on C4"
    # The standard uncertainty inequality bound using a test function g is:
    # A(f)*A(F(f)) <= C * ||g||_1 / min|hat{g}|
    # or similar. Let's use l1_norm / min_val as the objective.
    
    return l1_norm / min_val


def construct():
    """
    Find the three coefficients (a0, a4, a8) that minimize the C4 uncertainty bound.
    """
    start_time = time.time()
    deadline = 210.0 - 10.0  # 10 second safety margin
    
    # Start with the known prior
    best_a0, best_a4, best_a8 = -113/100.0, 1/25.0, 1/3240.0
    best_bound = _compute_bound(best_a0, best_a4, best_a8)
    
    # Grid search in a reasonable range
    # The coefficients are expected to be O(1) or smaller
    # Let's search in a range around the initial guess
    
    # Phase 1: Coarse grid search
    n_coarse = 15  # 15^3 = 3375 evaluations
    a0_range = np.linspace(-2, 2, n_coarse)
    a4_range = np.linspace(-1, 1, n_coarse)
    a8_range = np.linspace(-0.1, 0.1, n_coarse)
    
    for a0 in a0_range:
        if time.time() > deadline:
            break
        for a4 in a4_range:
            if time.time() > deadline:
                break
            for a8 in a8_range:
                if time.time() > deadline:
                    break
                bound = _compute_bound(a0, a4, a8)
                if bound < best_bound:
                    best_bound = bound
                    best_a0, best_a4, best_a8 = a0, a4, a8
    
    # Phase 2: Local refinement using Nelder-Mead or similar
    # Use scipy if available, otherwise do a simple local search
    try:
        from scipy.optimize import minimize
        from scipy.optimize import differential_evolution
        
        # Use differential evolution for global search
        def objective(params):
            a0, a4, a8 = params
            return _compute_bound(a0, a4, a8)
        
        bounds = [(best_a0 - 0.5, best_a0 + 0.5),
                  (best_a4 - 0.5, best_a4 + 0.5),
                  (best_a8 - 0.1, best_a8 + 0.1)]
        
        result = differential_evolution(
            objective, bounds, maxiter=100, tol=1e-10, 
            seed=42, polish=True, timeout=60
        )
        
        if result.fun < best_bound:
            best_a0, best_a4, best_a8 = result.x
            best_bound = result.fun
        
        # Additional local refinement
        result2 = minimize(
            objective, [best_a0, best_a4, best_a8],
            method='Nelder-Mead', 
            options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-12}
        )
        
        if result2.fun < best_bound:
            best_a0, best_a4, best_a8 = result2.x
            best_bound = result2.fun
            
    except Exception:
        # Fallback: simple local search
        step_sizes = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
        for step in step_sizes:
            if time.time() > deadline:
                break
            improved = True
            while improved and time.time() < deadline:
                improved = False
                for da0 in [-step, 0, step]:
                    for da4 in [-step, 0, step]:
                        for da8 in [-step, 0, step]:
                            if da0 == da4 == da8 == 0:
                                continue
                            new_a0 = best_a0 + da0
                            new_a4 = best_a4 + da4
                            new_a8 = best_a8 + da8
                            new_bound = _compute_bound(new_a0, new_a4, new_a8)
                            if new_bound < best_bound:
                                best_bound = new_bound
                                best_a0, best_a4, best_a8 = new_a0, new_a4, new_a8
                                improved = True
    
    return np.array([best_a0, best_a4, best_a8])
# EVOLVE-BLOCK-END
