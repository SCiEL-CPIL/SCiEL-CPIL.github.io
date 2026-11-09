"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def construct():
    """
    Construct 12 positive double-root locations for the Laguerre combination (m=12, n=1)
    such that the upper bound on C4 is minimized.
    """
    m = 12
    n = 1
    # The roots are x_1, ..., x_m.
    # The polynomial is P(x) = prod_{i=1}^m (x - x_i)^2.
    # The bound involves the integral of |P(x)|^2 e^{-x} or similar, but specifically 
    # for Cohn-Goncalves, the constant C is related to the norm of the polynomial.
    # A common proxy for the "sharpness" or the value of the constant in such inequalities 
    # is the value of the polynomial at specific points or its norm.
    # However, the task says "score is the upper bound on C4".
    # In the context of uncertainty principles, the constant is often derived from 
    # min_{f} ||f|| ||f'|| / ||f||_1 ||f'||_1 etc.
    # For the Laguerre construction, the bound is typically of the form:
    # C >= K * (something involving roots)
    # Or rather, we want to TUNE the construction to get a BETTER (lower) upper bound.
    # The "upper bound on C4" likely refers to the constant derived from the specific polynomial choice.
    # Let's assume the objective function to minimize is the value of the constant associated with the polynomial.
    # A standard quantity is the norm of the polynomial in the Laguerre space, or the value at 0, or the max value.
    # Given the "double roots" and "Laguerre", the polynomial is likely related to L_m^{(2m-1)} or similar.
    # But the task says "construct() -> list of 12 positive double-root locations".
    # This implies we are choosing the roots of a polynomial Q(x) = \prod (x-x_i)^2.
    # The bound on C4 is likely proportional to \sqrt{\int_0^\infty Q(x)^2 e^{-x} dx} / (\int_0^\infty Q(x) e^{-x} dx) or similar ratio.
    # Let's define the objective as the ratio of the L2 norm (weighted) to the L1 norm (weighted) or similar.
    # Actually, for uncertainty, the constant is often 1 / (something).
    # Let's look at the seed: [4, 8, ..., 48].
    # Let's try to minimize the "spread" or a specific norm ratio.
    # A robust proxy for "best constant" in these polynomial approximations is the condition number or the norm ratio.
    # Let's define the objective function as the value of the polynomial's norm ratio.
    # Specifically, for a polynomial P(x) = \prod (x-x_i)^2, the constant in the uncertainty principle 
    # involving the sign change is often bounded by the reciprocal of the integral of the polynomial.
    # Let's minimize: J(x) = \frac{\int_0^\infty P(x)^2 e^{-x} dx}{(\int_0^\infty P(x) e^{-x} dx)^2}
    # This is a standard Rayleigh quotient type functional. Minimizing this sharpens the bound.
    
    start_time = time.time()
    deadline = start_time + 600.0
    
    # Initial guess: spread out roots
    # The seed uses 4, 8, ..., 48.
    # Let's start with a slightly perturbed version or the seed itself.
    x0 = np.array([4.0 * (i + 1) for i in range(m)])
    
    def objective(x):
        # x is a vector of length m
        # Compute P(x) coefficients? No, direct integration is better.
        # P(t) = \prod_{i=1}^m (t - x_i)^2
        # We need I2 = \int_0^\infty P(t)^2 e^{-t} dt
        # and I1 = \int_0^\infty P(t) e^{-t} dt
        # Note: P(t) is a polynomial of degree 2m.
        # We can compute the integrals using the moments of the exponential distribution.
        # \int_0^\infty t^k e^{-t} dt = k!
        # P(t) = \sum_{j=0}^{2m} c_j t^j
        # I1 = \sum c_j j!
        # I2 = \sum_{j,k} c_j c_k (j+k)!
        
        # Compute coefficients of P(t)
        # Start with 1
        coeffs = np.array([1.0])
        for xi in x:
            # Multiply by (t - xi)^2 = t^2 - 2xi t + xi^2
            # coeffs = coeffs * [1, -2xi, xi^2]
            new_coeffs = np.zeros(len(coeffs) + 2)
            new_coeffs[:-2] += coeffs
            new_coeffs[1:-1] += -2 * xi * coeffs
            new_coeffs[2:] += (xi ** 2) * coeffs
            coeffs = new_coeffs
            
        # coeffs[j] is the coefficient of t^j
        # Compute I1
        # I1 = sum_{j=0}^{2m} coeffs[j] * j!
        # Precompute factorials up to 2m
        max_deg = 2 * m
        factorials = np.array([np.math.factorial(i) for i in range(max_deg + 1)])
        
        I1 = np.dot(coeffs, factorials)
        
        # Compute I2
        # I2 = sum_{j=0}^{2m} sum_{k=0}^{2m} coeffs[j] coeffs[k] (j+k)!
        # This can be computed as coeffs^T M coeffs where M_{jk} = (j+k)!
        # To avoid building the full matrix, we can use the fact that it's a quadratic form.
        # Or just loop, since m=12, 2m=24, matrix is 25x25.
        # M is 25x25.
        M = np.array([[np.math.factorial(i + j) for j in range(max_deg + 1)] for i in range(max_deg + 1)])
        I2 = coeffs @ M @ coeffs
        
        if I1 == 0:
            return 1e100
        return I2 / (I1 ** 2)

    # We want to minimize this objective.
    # Use L-BFGS-B with bounds to ensure positive roots.
    # Bounds: [1e-6, None] for each root.
    # Also, we should probably sort the roots to maintain a consistent ordering for the optimizer?
    # The objective is symmetric with respect to permutation of x.
    # L-BFGS-B might struggle with symmetries.
    # We can enforce x_1 < x_2 < ... < x_m by penalizing violations or using bounds.
    # Simpler: just let it optimize and sort at the end?
    # But if it swaps, the gradient might be misleading.
    # Let's add a small penalty for disorder to help the optimizer.
    def objective_with_penalty(x):
        base_val = objective(x)
        # Penalty for non-increasing order
        diffs = np.diff(x)
        penalty = np.sum(np.maximum(0, -diffs) ** 2)
        return base_val + 1e-5 * penalty

    # Initial guess
    x0 = np.array([4.0 * (i + 1) for i in range(m)])
    
    # Optimization
    best_x = x0
    best_val = objective_with_penalty(x0)
    
    try:
        # Use L-BFGS-B
        res = minimize(objective_with_penalty, x0, method='L-BFGS-B', 
                       bounds=[(1e-6, None)] * m, 
                       options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12})
        if res.success or res.fun < best_val:
            best_x = res.x
            best_val = res.fun
    except Exception:
        pass

    # Sort the roots to be safe
    best_x = np.sort(best_x)
    
    # Ensure they are positive
    best_x = np.maximum(best_x, 1e-6)
    
    return list(best_x)
# EVOLVE-BLOCK-END
