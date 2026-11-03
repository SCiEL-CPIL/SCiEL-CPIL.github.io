"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import eval_hermite
from scipy.optimize import differential_evolution, minimize
import time


def _compute_objective(coeffs):
    """
    Compute the upper bound on A(f)*A(Fourier(f)) for the Hermite-Gaussian test function.
    
    coeffs: array of 3 values [a0, a1, a2] for H0, H4, H8.
    The H12 coefficient is determined by the constraint f(0) = 0.
    
    Returns the product of L1 norms (the upper bound on C4).
    """
    a0, a1, a2 = coeffs
    
    # Hermite polynomials at x=0:
    # H0(0) = 1
    # H4(0) = 12
    # H8(0) = 1680
    # H12(0) = 40320
    h0_0 = 1.0
    h4_0 = 12.0
    h8_0 = 1680.0
    h12_0 = 40320.0
    
    # Constraint: f(0) = 0 means a0*h0_0 + a1*h4_0 + a2*h8_0 + a3*h12_0 = 0
    # So a3 = -(a0*h0_0 + a1*h4_0 + a2*h8_0) / h12_0
    a3 = -(a0 * h0_0 + a1 * h4_0 + a2 * h8_0) / h12_0
    
    # Define the function f(x) = (a0*H0(x) + a1*H4(x) + a2*H8(x) + a3*H12(x)) * exp(-x^2/2)
    # For the Fourier transform with unitary convention:
    # F[H_n(x) * exp(-x^2/2)] = i^n * H_n(x) * exp(-x^2/2)
    # So F̂f(ξ) = (a0*i^0*H0(ξ) + a1*i^1*H4(ξ) + a2*i^2*H8(ξ) + a3*i^3*H12(ξ)) * exp(-ξ^2/2)
    #            = (a0*H0(ξ) - a1*H4(ξ) - a2*H8(ξ) + a3*H12(ξ)) * exp(-ξ^2/2)
    
    # We need to compute:
    # A(f) = ∫|f(x)|dx
    # A(F̂f) = ∫|F̂f(ξ)|dξ
    
    # Use numerical integration over a sufficiently large domain
    x = np.linspace(-15, 15, 20001)
    dx = x[1] - x[0]
    
    # Compute Hermite polynomials
    H0 = np.ones_like(x)
    H4 = eval_hermite(4, x)
    H8 = eval_hermite(8, x)
    H12 = eval_hermite(12, x)
    
    gaussian = np.exp(-x**2 / 2.0)
    
    # f(x)
    f = (a0 * H0 + a1 * H4 + a2 * H8 + a3 * H12) * gaussian
    
    # F̂f(ξ) - note the signs change due to i^n factors
    # i^0 = 1, i^1 = i, i^2 = -1, i^3 = -i
    # But since we're taking absolute values, and the Hermite polynomials are real,
    # the Fourier transform is: (a0*H0 - a1*H4 - a2*H8 + a3*H12) * exp(-x^2/2)
    # Wait, let me reconsider. The Fourier transform gives:
    # F[f](ξ) = (1/√(2π)) ∫ (a0*H0(x) + a1*H4(x) + a2*H8(x) + a3*H12(x)) * exp(-x^2/2) * exp(-ixξ) dx
    # = (1/√(2π)) * [a0 * F[exp(-x^2/2)] * ... ]
    
    # Actually, for the Hermite functions φ_n(x) = (2^n n! √π)^(-1/2) H_n(x) exp(-x^2/2),
    # F[φ_n](ξ) = i^n φ_n(ξ)
    
    # But we're using H_n(x) exp(-x^2/2) without normalization. Let me just compute numerically.
    
    # For the L1 norm of the Fourier transform, I'll use the property that:
    # If f(x) = Σ c_n H_n(x) exp(-x^2/2), then F̂f(ξ) = Σ c_n i^n H_n(ξ) exp(-ξ^2/2) / √(2π)
    # The 1/√(2π) factor affects both A(f) and A(F̂f), so it contributes 1/2π to the product.
    # But actually, let me just compute both L1 norms numerically.
    
    # A(f)
    A_f = np.sum(np.abs(f)) * dx
    
    # For F̂f, using the eigenfunction property:
    # F̂f(ξ) ∝ (a0*H0(ξ) + a1*i*H4(ξ) + a2*(-1)*H8(ξ) + a3*(-i)*H12(ξ)) * exp(-ξ^2/2)
    # The magnitude is: |a0*H0(ξ) + a1*i*H4(ξ) - a2*H8(ξ) - a3*i*H12(ξ)| * exp(-ξ^2/2)
    # = |a0*H0(ξ) - a2*H8(ξ) + i*(a1*H4(ξ) - a3*H12(ξ))| * exp(-ξ^2/2)
    # = sqrt((a0*H0(ξ) - a2*H8(ξ))^2 + (a1*H4(ξ) - a3*H12(ξ))^2) * exp(-ξ^2/2)
    
    real_part = a0 * H0 - a2 * H8
    imag_part = a1 * H4 - a3 * H12
    f_hat_mag = np.sqrt(real_part**2 + imag_part**2) * gaussian
    
    A_f_hat = np.sum(f_hat_mag) * dx
    
    # The product A(f) * A(F̂f) - note that the Fourier transform convention may include
    # a normalization factor. The upper bound on C4 is proportional to this product.
    # We minimize this product.
    return A_f * A_f_hat


def construct():
    """
    Return three coefficients for the Hermite combination (degrees 0, 4, 8).
    The degree-12 weight is fixed so g(0) = 0.
    """
    start_time = time.time()
    deadline = start_time + 170  # Safety margin before 180s limit
    
    # Initial guess from the seed program
    initial_guess = np.array([-113 / 100, 1 / 25, 1 / 3240])
    
    best_coeffs = initial_guess.copy()
    best_score = _compute_objective(best_coeffs)
    
    # Use differential evolution for global optimization
    # Bounds for the coefficients
    bounds = [(-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)]
    
    try:
        # First, try differential evolution with a reasonable number of iterations
        result = differential_evolution(
            _compute_objective,
            bounds,
            seed=42,
            maxiter=200,
            popsize=15,
            tol=1e-8,
            polish=True,
            init='sobol'
        )
        
        if time.time() < deadline:
            if result.fun < best_score:
                best_score = result.fun
                best_coeffs = result.x
        
        # Refine with local optimization
        if time.time() < deadline:
            result2 = minimize(
                _compute_objective,
                best_coeffs,
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-12}
            )
            
            if result2.fun < best_score:
                best_score = result2.fun
                best_coeffs = result2.x
        
    except Exception:
        pass
    
    # Try multiple random restarts if we have time
    if time.time() < deadline - 30:
        rng = np.random.default_rng(123)
        for i in range(50):
            if time.time() > deadline - 10:
                break
            
            # Random initialization
            random_coeffs = rng.uniform(-3.0, 3.0, size=3)
            
            try:
                # Local optimization from random start
                res = minimize(
                    _compute_objective,
                    random_coeffs,
                    method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10}
                )
                
                if res.fun < best_score:
                    best_score = res.fun
                    best_coeffs = res.x
                    
            except Exception:
                continue
    
    # Final polish with very tight tolerance
    if time.time() < deadline - 5:
        try:
            final_res = minimize(
                _compute_objective,
                best_coeffs,
                method='Nelder-Mead',
                options={'maxiter': 50000, 'xatol': 1e-14, 'fatol': 1e-14}
            )
            
            if final_res.fun < best_score:
                best_score = final_res.fun
                best_coeffs = final_res.x
                
        except Exception:
            pass
    
    return np.array(best_coeffs)
# EVOLVE-BLOCK-END
