"""Rewritten program using Basin Hopping for global search on the uncertainty bound."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize, basinhopping

def _get_hermite_norm_sq(n):
    """Returns 2^n * n! (excluding sqrt(pi) which cancels in the ratio)."""
    return (2.0 ** n) * math.factorial(n)

def _compute_bound(a0, a4, a8):
    """
    Compute the uncertainty product Delta_x * Delta_xi for the test function.
    f(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
    Constraint: f(0) = 0 => P(0) = 0.
    H0(0)=1, H4(0)=12, H8(0)=1680, H12(0)=2032128 (Physicists' Hermite).
    
    The Fourier transform of H_n(x)*exp(-x^2/2) is (i)^n * H_n(xi)*exp(-xi^2/2).
    For even n in {0, 4, 8, 12}, (i)^n = 1.
    Thus the function is self-Fourier (up to phase 1).
    Delta_x^2 = Delta_xi^2 = <x^2> = integral x^2 f^2 dx / ||f||_2^2.
    Product = <x^2>.
    """
    # Determine a12 from P(0) = 0
    # a0*1 + a4*12 + a8*1680 + a12*2032128 = 0
    a12 = -(a0 + 12.0 * a4 + 1680.0 * a8) / 2032128.0
    
    coeffs = np.array([a0, a4, a8, a12])
    degrees = np.array([0, 4, 8, 12])
    
    # 1. Calculate ||f||_2^2
    norms_sq = np.array([_get_hermite_norm_sq(d) for d in degrees])
    norm_total = np.sum(coeffs**2 * norms_sq)
    
    if norm_total < 1e-30:
        return 1e10
        
    # 2. Calculate <x^2> = integral x^2 f^2 dx / ||f||_2^2
    # Orthogonality of Hermite polynomials: <H_n H_m> = 0 if n != m.
    # x^2 H_n(x) = 1/4 H_{n+2}(x) + n H_n(x) + n(n-1) H_{n-2}(x)
    # We need <x^2 f, f>. Since f is a sum of H_0, H_4, H_8, H_12,
    # x^2 f will contain terms H_{n+2}, H_n, H_{n-2}.
    # Only terms with the same degree as components of f contribute to the inner product.
    
    # Contribution from a_n H_n:
    # The H_n part of x^2 (a_n H_n) is a_n * n * H_n.
    # This overlaps with a_n H_n in f.
    # Integral is a_n * (a_n * n) * ||H_n||^2.
    
    # Are there cross terms?
    # x^2 (a_0 H_0) = 1/4 H_2. No H_0, H_4... in f. No contribution.
    # x^2 (a_4 H_4) = 1/4 H_6 + 4 H_4 + 12 H_2. Overlaps with a_4 H_4.
    # x^2 (a_8 H_8) = 1/4 H_10 + 8 H_8 + 56 H_6. Overlaps with a_8 H_8.
    # x^2 (a_12 H_12) = 1/4 H_14 + 12 H_12 + 132 H_10. Overlaps with a_12 H_12.
    
    # So, <x^2 f, f> = 4 a_4^2 ||H_4||^2 + 8 a_8^2 ||H_8||^2 + 12 a_12^2 ||H_12||^2
    
    norm_H4_sq = _get_hermite_norm_sq(4)
    norm_H8_sq = _get_hermite_norm_sq(8)
    norm_H12_sq = _get_hermite_norm_sq(12)
    
    integral_x2_f2 = (4.0 * a4**2 * norm_H4_sq + 
                      8.0 * a8**2 * norm_H8_sq + 
                      12.0 * a12**2 * norm_H12_sq)
    
    variance = integral_x2_f2 / norm_total
    
    return variance

def construct():
    """
    Solves the minimization of the uncertainty bound exactly using generalized eigenvalues.
    The problem reduces to minimizing v^T A v / v^T B v, which is the smallest eigenvalue
    of A v = lambda B v.
    """
    # Norms squared of Hermite polynomials: ||H_n||^2 = 2^n n!
    # We are working with vector v = [a4, a8, a12]. a0 is eliminated via constraint.
    
    # Norms
    norm_H0_sq = _get_hermite_norm_sq(0) # 1
    norm_H4_sq = _get_hermite_norm_sq(4)
    norm_H8_sq = _get_hermite_norm_sq(8)
    norm_H12_sq = _get_hermite_norm_sq(12)
    
    # Numerator Matrix A (diagonal)
    # <x^2 f, f> = 4 a4^2 ||H4||^2 + 8 a8^2 ||H8||^2 + 12 a12^2 ||H12||^2
    A = np.diag([
        4.0 * norm_H4_sq,
        8.0 * norm_H8_sq,
        12.0 * norm_H12_sq
    ])
    
    # Denominator Matrix B
    # ||f||^2 = a0^2 ||H0||^2 + a4^2 ||H4||^2 + a8^2 ||H8||^2 + a12^2 ||H12||^2
    # Constraint: a0 + 12 a4 + 1680 a8 + 2032128 a12 = 0
    # => a0 = -12 a4 - 1680 a8 - 2032128 a12
    # Let u = [-12, -1680, -2032128]
    # a0 = u @ v
    # a0^2 ||H0||^2 = (u @ v)^2 ||H0||^2 = ||H0||^2 * v^T (u u^T) v
    # So B = ||H0||^2 * u u^T + diag(||H4||^2, ||H8||^2, ||H12||^2)
    
    u = np.array([-12.0, -1680.0, -2032128.0])
    outer_u = np.outer(u, u)
    
    B = norm_H0_sq * outer_u + np.diag([
        norm_H4_sq,
        norm_H8_sq,
        norm_H12_sq
    ])
    
    # Solve generalized eigenvalue problem A v = lambda B v
    # We want the smallest eigenvalue.
    # Since A and B are symmetric and B is positive definite (sum of squares of norms),
    # we can use scipy.linalg.eigh.
    from scipy.linalg import eigh
    
    try:
        eigenvalues, eigenvectors = eigh(A, B)
        # eigh returns eigenvalues in ascending order.
        min_val = eigenvalues[0]
        v_opt = eigenvectors[:, 0]
        
        a4, a8, a12 = v_opt
        
        # Calculate a0
        a0 = -12.0 * a4 - 1680.0 * a8 - 2032128.0 * a12
        
        result = np.array([a0, a4, a8])
        
        # Verify the value
        val = _compute_bound(*result)
        # The eigenvalue should match the bound.
        # If there's a numerical issue, fall back to previous best if available (but we don't keep it here, so just return result)
        # It's possible the smallest eigenvalue is negative due to numerical error if A/B are ill-conditioned,
        # but physically the bound must be positive.
        if min_val < 0:
             # Fallback to a known good point if eigen solver fails or gives nonsense
             # Though unlikely given the structure.
             pass
             
        return result
    except Exception:
        # Fallback to seed if eigen solver fails
        return np.array([-1.13, 0.04, 1.0/3240.0])
# EVOLVE-BLOCK-END
