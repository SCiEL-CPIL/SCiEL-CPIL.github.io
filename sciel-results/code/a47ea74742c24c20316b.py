"""Optimized program for the uncertainty inequality task."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from scipy.special import factorial
import warnings


def _compute_bound_analytic(c0, c4, c8):
    """
    Compute the uncertainty product analytically using Hermite function expansion.
    
    g(x) = (c0 H0 + c4 H4 + c8 H8 + c12 H12) exp(-x^2/2)
    Let phi_n(x) be orthonormal Hermite functions.
    H_n(x) exp(-x^2/2) = A_n phi_n(x) where A_n = sqrt(2^n n! sqrt(pi)).
    
    g(x) = sum a_n phi_n(x) with a_n = c_n A_n.
    
    Norm squared N = sum a_n^2.
    M_g = int x^2 g^2 dx.
    M_F = int x^2 |F[g]|^2 dx.
    
    Using x phi_n = 1/sqrt(2) (sqrt(n+1) phi_{n+1} + sqrt(n) phi_{n-1}):
    <phi_n | x^2 | phi_m> is non-zero for m = n, n+2, n-2.
    D_n = n + 0.5
    U_n = sqrt((n+1)(n+2)/2)  # <phi_n | x^2 | phi_{n+2}>
    L_n = sqrt(n(n-1)/2)      # <phi_n | x^2 | phi_{n-2}>
    
    M_g = sum_{n,m} a_n a_m <phi_n | x^2 | phi_m>
    F[g] = sum a_n (-i)^n phi_n.
    M_F = sum_{n,m} a_n a_m (-i)^n (i)^m <phi_n | x^2 | phi_m>
         = sum_{n,m} a_n a_m i^{m-n} <phi_n | x^2 | phi_m>
    
    Objective = (M_g * M_F) / (N * N)
    """
    # Constants
    h0_0 = 1.0
    h4_0 = 12.0
    h8_0 = 1680.0
    h12_0 = 665280.0
    
    c12 = -(c0 * h0_0 + c4 * h4_0 + c8 * h8_0) / h12_0
    
    # Coefficients a_n = c_n * A_n
    # A_n = sqrt(2^n * n! * sqrt(pi))
    A0 = np.sqrt(1.0 * 1.0 * np.sqrt(np.pi))
    A4 = np.sqrt(16.0 * 24.0 * np.sqrt(np.pi))
    A8 = np.sqrt(256.0 * 40320.0 * np.sqrt(np.pi))
    A12 = np.sqrt(4096.0 * factorial(12) * np.sqrt(np.pi))
    
    a = np.zeros(13) # indices 0 to 12
    a[0] = c0 * A0
    a[4] = c4 * A4
    a[8] = c8 * A8
    a[12] = c12 * A12
    
    N = np.sum(a**2)
    if N < 1e-15:
        return 1e10
        
    # Matrix elements for x^2
    # We only need indices 0, 4, 8, 12
    # Non-zero interactions:
    # n=0: m=0, 2
    # n=4: m=2, 4, 6
    # n=8: m=6, 8, 10
    # n=12: m=10, 12
    
    # Let's compute M_g and M_F by summing over pairs (n,m) in {0,4,8,12}
    # Note: <phi_n | x^2 | phi_m> = 0 if |n-m| > 2.
    # So for our set {0,4,8,12}, the only non-zero diagonal terms are n=m.
    # Off-diagonal terms require |n-m|=2.
    # 0 and 4 differ by 4 -> 0.
    # 0 and 8 differ by 8 -> 0.
    # 4 and 8 differ by 4 -> 0.
    # Thus, M_g and M_F are purely diagonal sums?
    # Wait. x^2 connects n to n+2 and n-2.
    # So <phi_0 | x^2 | phi_4> = 0.
    # <phi_4 | x^2 | phi_8> = 0.
    # Yes, the Hermite functions of different parity (even/odd) don't mix, but here all are even.
    # However, they are separated by 4 in index.
    # <phi_n | x^2 | phi_m> is zero unless m = n, n+2, n-2.
    # Since our indices are 0, 4, 8, 12, no two distinct indices differ by 2.
    # Therefore, M_g = sum a_n^2 D_n.
    # M_F = sum a_n^2 D_n as well?
    # Let's check M_F.
    # M_F = sum_{n,m} a_n a_m i^{m-n} <phi_n | x^2 | phi_m>.
    # Since <phi_n | x^2 | phi_m> is zero for n != m in our set,
    # M_F = sum a_n^2 i^0 D_n = sum a_n^2 D_n.
    # So M_g = M_F?
    # If M_g = M_F, then Objective = M_g^2 / N^2 = (M_g/N)^2.
    # This implies we just minimize the average D_n weighted by a_n^2.
    # D_n = n + 0.5.
    # D_0 = 0.5, D_4 = 4.5, D_8 = 8.5, D_12 = 12.5.
    # We want to minimize weighted average of D_n.
    # This means we want to put weight on low n (n=0).
    # But we have constraint g(0)=0.
    
    # Let's re-verify the "zero off-diagonal" claim.
    # x phi_0 = 1/sqrt(2) phi_1.
    # x^2 phi_0 = x (1/sqrt(2) phi_1) = 1/sqrt(2) (1/sqrt(2) (phi_2 + phi_0)) = 0.5 phi_2 + 0.5 phi_0.
    # <phi_0 | x^2 | phi_0> = 0.5. Correct.
    # <phi_0 | x^2 | phi_4> = 0. Correct.
    
    # So M_g = 0.5 a_0^2 + 4.5 a_4^2 + 8.5 a_8^2 + 12.5 a_12^2.
    # M_F = same.
    # Objective J = (M_g / N)^2.
    # To minimize J, we minimize M_g / N.
    # This is a Rayleigh quotient.
    
    M_g = 0.5 * a[0]**2 + 4.5 * a[4]**2 + 8.5 * a[8]**2 + 12.5 * a[12]**2
    J = (M_g / N) ** 2
    
    return J


def _objective(coeffs):
    """Objective function for optimization."""
    c0, c4, c8 = coeffs
    try:
        bound = _compute_bound_analytic(c0, c4, c8)
        if not np.isfinite(bound) or bound < 0:
            return 1e10
        return bound
    except Exception:
        return 1e10


def construct():
    """
    Find coefficients that minimize the upper bound on A(f)*A(Fourier(f)).
    
    Returns an array of three coefficients [c0, c4, c8] for the Hermite combination.
    The degree-12 coefficient is determined to ensure g(0) = 0.
    """
    # Start from the prior state of the art and refine
    initial = np.array([-113 / 100, 1 / 25, 1 / 3240])
    seed_val = _objective(initial)
    
    # Use differential evolution for global optimization
    # Tighter bounds as coefficients are expected to be small
    bounds = [(-10, 10), (-10, 10), (-10, 10)]
    
    best_x = initial
    best_val = seed_val
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        # Try a few different seeds for DE to ensure robustness
        for s in [42, 123, 999]:
            res = differential_evolution(_objective, bounds, seed=s, maxiter=1000, popsize=50, tol=1e-10, polish=True)
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
                
    # Local refinement from the best point found
    from scipy.optimize import minimize
    try:
        res_local = minimize(_objective, best_x, method='L-BFGS-B', options={'maxiter': 1000})
        if res_local.fun < best_val:
            best_val = res_local.fun
            best_x = res_local.x
    except Exception:
        pass

    return best_x
# EVOLVE-BLOCK-END
