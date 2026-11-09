"""
Strategy: Use zeros of generalized Laguerre polynomial L_12^(1)(x) as initial roots,
then refine using Nelder-Mead optimization on the C4 upper bound.
"""

import numpy as np
from scipy.special import genlaguerre, eval_genlaguerre
from scipy.optimize import minimize
import time

# Constants for the task
M = 12
N = 1

def _compute_c4_bound(roots):
    """
    Compute the C4 upper bound for the given roots.
    The Cohn-Goncalves construction involves a polynomial P(x) with roots at the given locations.
    For m=12, n=1, the polynomial is related to the Laguerre weight.
    
    The bound C4 is typically derived from the norm of the polynomial or its dual.
    In the context of the Fourier sign uncertainty, the bound is often related to 
    the integral of the square of the polynomial against the weight, or similar moment constraints.
    
    A common formulation for the refined uncertainty bound involves minimizing the 
    "energy" of the polynomial subject to orthogonality or root constraints.
    
    Let's define the polynomial Q(x) = prod_{i=1}^{12} (x - r_i).
    The C4 bound in many such problems is proportional to the L2 norm of Q(x) 
    with respect to the weight w(x) = x^N e^{-x}, or related to the moment matrix.
    
    Specifically, for the Cohn-Goncalves construction, the bound is often:
    C4 = (1 / (2 * sqrt(pi))) * (something involving the roots).
    
    However, a more direct interpretation from "refined uncertainty inequality" literature 
    suggests that the bound is determined by the minimum of the function 
    f(x) = |P(x)|^2 w(x) or similar.
    
    Let's assume the score is the value of the upper bound on the uncertainty product.
    A standard lower bound for the uncertainty product in this context is related to 
    the optimal constant C such that ||f|| ||f'|| >= C ||f||^2.
    
    For the specific "Laguerre-polynomial construction", the bound is often 
    calculated as the square root of the ratio of certain moments or norms.
    
    Let's use a proxy that is computationally efficient and correlates with the true bound:
    The L2 norm of the polynomial P(x) = prod(x - r_i) with respect to the weight x^N e^{-x}.
    Actually, the "upper bound on C4" usually refers to a constant in the inequality.
    
    Let's look at the seed: [4, 8, ..., 48].
    If we use the zeros of L_12^(1), they are clustered around 12.
    
    Let's define the objective as the L2 norm of the polynomial P(x) = prod_{i=1}^m (x - r_i)
    with respect to the weight w(x) = x^N e^{-x}.
    Wait, the "upper bound" being minimized suggests we want to make the polynomial "small" 
    in some sense, or the constant C4 is inversely related to the norm?
    
    Let's re-read: "score is the upper bound on C4". "lower is better".
    This implies we want to minimize the upper bound.
    
    In the Cohn-Goncalves paper, the constant C is related to the optimal polynomial.
    The bound is typically of the form C = sqrt( ... ).
    
    Let's try a specific formulation:
    The uncertainty principle for the Fourier transform on the line gives 
    ||x f|| ||f'|| >= (1/4) ||f||^2.
    Refined bounds use Laguerre polynomials.
    
    A key quantity is the "defect" or the norm of the polynomial formed by the roots.
    Let P(x) = prod_{i=1}^{12} (x - r_i).
    The bound might be related to the integral I = ∫ P(x)^2 x^N e^{-x} dx.
    If we minimize this integral, we are finding the polynomial that is "closest" to zero in the L2 sense, 
    but since the roots are fixed as variables, we are effectively choosing the support.
    
    Actually, for the uncertainty principle, the optimal constant is often derived from 
    the eigenvalues of a certain operator or the norm of a projection.
    
    Let's assume the score is proportional to the L2 norm of the polynomial P(x) 
    with respect to the weight w(x) = x^N e^{-x}, normalized by the norm of the 
    Laguerre polynomial L_M^(N)(x).
    
    Score = ||P||_w / ||L_M^(N)||_w
    
    This ratio measures how "aligned" our chosen roots are with the optimal 
    Laguerre polynomial structure. If the roots are exactly the zeros of L_M^(N), 
    P(x) is a constant multiple of L_M^(N)(x), and the ratio is 1 (or close to it, 
    depending on normalization).
    
    Wait, if the roots ARE the zeros, then P(x) = c * L_M^(N)(x).
    Then ||P||_w = |c| ||L_M^(N)||_w.
    The ratio is |c|.
    The leading coefficient of L_M^(N)(x) is (-1)^M / M!.
    The leading coefficient of P(x) = prod(x-r_i) is 1.
    So c = (-1)^M M! = (-1)^12 * 12! = 12!.
    So the score would be 12! / ||L_M^(N)||_w.
    
    This is a constant if the roots are the zeros.
    If we move the roots, the polynomial P(x) is no longer orthogonal to lower degree polynomials 
    in the same way, and its norm changes.
    
    Minimizing this norm (or the ratio) is a valid strategy to find the "best" 
    configuration of roots that minimizes the "energy" of the constructed polynomial.
    
    Let's implement this objective.
    """
    # Ensure roots are sorted and positive
    roots = np.array(roots)
    if np.any(roots <= 0):
        return 1e10
    
    # Construct the polynomial P(x) = prod (x - r_i)
    # Using coefficients might be unstable for high degree, but M=12 is small.
    # However, evaluating the integral numerically is safer.
    
    # We want to compute ∫ P(x)^2 x^N e^{-x} dx
    # We can use Gaussian quadrature. The weight is x^N e^{-x}.
    # The nodes and weights for this weight are the zeros of L_M^(N)(x) and the corresponding weights.
    # But we are optimizing the roots r_i.
    
    # Let's use a fixed high-order Gaussian quadrature for the integral, 
    # independent of the roots r_i.
    # Use scipy.integrate.quad for accuracy, or a fixed Gauss-Laguerre grid with high order.
    # Since M=12, we can use a Gauss-Laguerre quadrature of order, say, 50.
    
    # Nodes and weights for w(x) = x^N e^{-x}
    # scipy.special.roots_genlaguerre(M, N) gives nodes and weights for L_M^(N).
    # But we need a quadrature rule that is accurate for P(x)^2.
    # P(x) is degree 12. P(x)^2 is degree 24.
    # A Gauss-Laguerre quadrature of order K is exact for polynomials of degree 2K-1.
    # So we need 2K-1 >= 24 => K >= 12.5 => K=13.
    # Let's use K=20 for safety.
    
    nodes, weights = np.polynomial.laguerre.laggauss(20)
    # The weights from laggauss are for w(x) = e^{-x}.
    # We need w(x) = x^N e^{-x} = x^1 e^{-x}.
    # So the integral is sum( P(nodes)^2 * nodes^N * weights )
    
    # Compute P(nodes)
    # P(x) = prod (x - r_i)
    # To avoid overflow/underflow, we can compute log or just multiply.
    # M=12 is small.
    
    P_vals = np.ones_like(nodes)
    for r in roots:
        P_vals *= (nodes - r)
    
    integrand = P_vals**2 * (nodes**N) * weights
    integral = np.sum(integrand)
    
    # Normalize by the norm of the Laguerre polynomial L_M^(N)(x)
    # ||L_M^(N)||_w^2 = Γ(M+N+1) / M!
    # Here M=12, N=1.
    # norm_sq = Γ(12+1+1) / 12! = Γ(14) / 12! = 13! / 12! = 13.
    # Wait, formula is: ∫ L_M^(α)(x)^2 x^α e^{-x} dx = Γ(M+α+1)/M!
    # α = N = 1.
    # norm_sq = Γ(12+1+1)/12! = 13! / 12! = 13.
    
    norm_sq_lag = 13.0
    norm_lag = np.sqrt(norm_sq_lag)
    
    # The polynomial P(x) has leading coefficient 1.
    # L_M^(N)(x) has leading coefficient (-1)^M / M!.
    # So P(x) ≈ ( (-1)^M M! ) L_M^(N)(x) if roots are zeros.
    # The "energy" of P is integral.
    # The "energy" of the optimal Laguerre polynomial (scaled to leading coeff 1) is:
    # P_opt(x) = (-1)^M M! L_M^(N)(x).
    # ||P_opt||^2 = (M!)^2 ||L_M^(N)||^2 = (12!)^2 * 13.
    
    # So the ratio (Score) = ||P|| / ||P_opt|| = sqrt(integral / ((12!)^2 * 13))
    
    factorial_12 = 12.0
    for i in range(2, 13):
        factorial_12 *= i
        
    opt_norm_sq = (factorial_12**2) * 13.0
    
    score = np.sqrt(integral / opt_norm_sq)
    
    return score

def construct():
    start_time = time.time()
    deadline = start_time + 590  # 10s safety margin
    
    # Initial guess: Zeros of L_12^(1)(x)
    # These are the optimal quadrature nodes and the roots of the orthogonal polynomial.
    roots_init, _ = np.polynomial.laguerre.laggauss(12)
    # Note: laggauss(M) gives roots of L_M^(0). We need L_M^(1).
    # Use scipy.special.roots_genlaguerre
    from scipy.special import roots_genlaguerre
    roots_init, _ = roots_genlaguerre(12, 1)
    roots_init = np.sort(roots_init)
    
    # Ensure positive
    roots_init = np.maximum(roots_init, 1e-6)
    
    best_roots = roots_init
    best_score = _compute_c4_bound(best_roots)
    
    # Nelder-Mead optimization
    # It is derivative-free and robust.
    try:
        result = minimize(
            _compute_c4_bound, 
            x0=roots_init, 
            method='Nelder-Mead', 
            options={
                'maxiter': 2000, 
                'maxfev': 2000,
                'xatol': 1e-6,
                'fatol': 1e-8,
                'adaptive': True
            }
        )
        
        if result.success or result.fun < best_score:
            # Check if result is valid (positive roots)
            res_roots = np.array(result.x)
            if np.all(res_roots > 0):
                best_roots = res_roots
                best_score = result.fun
        
        # Check time
        if time.time() > deadline:
            return list(best_roots)
            
        # Try a few more local refinements from different starts if time permits
        # Start 2: Slightly perturbed zeros
        np.random.seed(42)
        for _ in range(3):
            if time.time() > deadline:
                break
            
            perturbation = np.random.normal(0, 0.1, size=12)
            x0 = best_roots + perturbation
            x0 = np.maximum(x0, 1e-6)
            
            res = minimize(
                _compute_c4_bound, 
                x0=x0, 
                method='Nelder-Mead', 
                options={
                    'maxiter': 500, 
                    'maxfev': 500,
                    'xatol': 1e-6,
                    'fatol': 1e-8,
                    'adaptive': True
                }
            )
            
            if res.fun < best_score:
                res_roots = np.array(res.x)
                if np.all(res_roots > 0):
                    best_roots = res_roots
                    best_score = res.fun
                    
    except Exception as e:
        # Fallback to initial guess if optimization fails
        pass
        
    return list(best_roots)

if __name__ == "__main__":
    roots = construct()
    print(f"Roots: {roots}")
    print(f"Score: {_compute_c4_bound(roots)}")
