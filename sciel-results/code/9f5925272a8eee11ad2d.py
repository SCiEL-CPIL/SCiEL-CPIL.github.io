"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import sys

# EVOLVE-BLOCK-START
def construct():
    """
    Returns 12 double-root locations z_1..z_12.
    Strategy: Use the positive roots of the derivative of L_{13}^{(-1/2)}(x)
    as initial guesses, then refine via a simple optimization to minimize B.
    However, given the 600s budget and the complexity of the exact rational
    solve, we must be very careful.
    
    The task says: "Each z_i is converted exactly to a rational... and the list is sorted."
    "The 26 coefficients are solved exactly (sympy LUsolve) from g(0)=0, g'(0)=1 and g(z_i)=g'(z_i)=0"
    "B = max(0, largest counted r)" where r is a real root of q(x) = g(x) / (x * prod (x-z_i)^2)
    "Score = B / (2*pi), lower is better."
    
    To minimize B, we want the largest real root of q(x) to be as small as possible.
    Ideally, we want q(x) to have no positive real roots, or very small ones.
    
    Let's analyze the structure.
    g(x) is a linear combination of L_{2k}^{(-1/2)}(x) for k=0..25.
    g(x) has double roots at z_1..z_12.
    g(0) = 0, g'(0) = 1.
    
    The roots z_i are the variables we control.
    The system of equations for c_k is linear.
    g(z_i) = 0 and g'(z_i) = 0 for i=1..12.
    Plus g(0)=0, g'(0)=1.
    Total 26 equations for 26 unknowns c_k.
    
    If the matrix is singular, it's invalid.
    
    q(x) = g(x) / (x * prod_{i=1}^{12} (x-z_i)^2).
    Since g has double roots at z_i and a simple root at 0 (since g(0)=0 and g'(0)=1 != 0),
    the division is exact.
    deg(g) <= 50.
    deg(denominator) = 1 + 2*12 = 25.
    So deg(q) <= 25.
    
    We want to minimize the largest positive real root of q(x).
    
    The directive suggests: "Place the 12 double roots z_i as the positive roots of the derivative of the generalized Laguerre polynomial L_{13}^{(-1/2)}(x)".
    
    Let's try this specific placement first. It might yield a valid and good score.
    L_n^{(alpha)}(x) has n positive roots.
    L_{13}^{(-1/2)}(x) has 13 positive roots.
    Its derivative L'_{13}^{(-1/2)}(x) has 12 roots, which are the critical points of L_{13}^{(-1/2)}.
    These roots are interlaced with the roots of L_{13}^{(-1/2)}.
    They are strictly positive and distinct.
    
    Let's compute these roots using numpy/scipy if possible, or sympy.
    Since we need to return floats, we can use scipy.special.roots_genlaguerre to get the roots of L_{13}^{(-1/2)}, and then find the roots of its derivative?
    Actually, the roots of the derivative of a polynomial with distinct roots are not simply related to the roots of the polynomial in a way that scipy directly provides for generalized Laguerre derivatives.
    However, we can compute the roots of L_{13}^{(-1/2)}(x) and then use Newton's method or companion matrix to find the roots of the derivative.
    Or, we can use the fact that the roots of the derivative of L_n^{(alpha)} are the eigenvalues of a certain tridiagonal matrix (Jacobi matrix).
    
    Alternatively, we can just use the roots of L_{12}^{(-1/2)}(x)? No, the directive says derivative of L_{13}^{(-1/2)}.
    
    Let's try to implement this.
    1. Compute the 13 roots of L_{13}^{(-1/2)}(x).
    2. Compute the derivative polynomial? Or use the Jacobi matrix for the derivative?
    The roots of the derivative of a monic polynomial p(x) are the eigenvalues of the companion matrix of p'(x).
    But constructing p'(x) from roots is easy.
    
    Let's use numpy.
    roots_13 = scipy.special.roots_genlaguerre(13, -0.5)
    p = numpy.poly(roots_13)
    p_prime = numpy.polyder(p)
    roots_prime = numpy.roots(p_prime)
    Filter for real positive roots.
    
    Let's try this.
    """
    start_time = time.time()
    
    import numpy as np
    import scipy.special
    
    # Get roots of L_{13}^{(-1/2)}(x)
    try:
        roots_13 = scipy.special.roots_genlaguerre(13, -0.5)
        # Convert to numpy array
        roots_13 = np.real(roots_13)
        # Construct polynomial
        p = np.poly(roots_13)
        # Derivative
        p_prime = np.polyder(p)
        # Roots of derivative
        roots_prime = np.roots(p_prime)
        # Filter real positive
        z_candidates = []
        for r in roots_prime:
            if np.isreal(r) and np.real(r) > 0:
                z_candidates.append(np.real(r))
        
        if len(z_candidates) != 12:
            # Fallback: use roots of L_{12}^{(-1/2)}
            roots_12 = scipy.special.roots_genlaguerre(12, -0.5)
            z_candidates = np.real(roots_12).tolist()
            
        z_candidates = sorted(z_candidates)
        
        # Ensure they are distinct and > 0
        if len(z_candidates) != 12:
            # Fallback to simple distinct values
            return [4.0 * (i + 1) for i in range(12)]
            
        return z_candidates
        
    except Exception:
        # Fallback
        return [4.0 * (i + 1) for i in range(12)]
# EVOLVE-BLOCK-END
