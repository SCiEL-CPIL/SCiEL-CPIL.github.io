"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys
import numpy as np
import sympy as sp
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    """
    Find 12 double-root locations z_1..z_12 to minimize B / (2*pi).
    
    Strategy: Use a log-spaced geometric progression for the double roots.
    We optimize the base z_1 and ratio r via a coarse 2D grid search.
    For each candidate set of roots, we:
    1. Solve the linear system for coefficients c_k of g(x) = sum c_k L_{2k}^{(-1/2)}(x).
    2. Perform exact polynomial division to get q(x).
    3. Find the largest real root r of q(x) that satisfies the sign-change condition.
    4. Return the roots that minimize B = max(0, largest counted r).
    
    Due to time constraints (600s limit), we perform a limited grid search.
    """
    start_time = time.time()
    deadline = 600.0
    safety_margin = 30.0  # seconds before deadline to stop
    
    # We'll do a coarse grid search over z_1 and r
    # z_i = z_1 * r^(i-1) for i=1..12
    # z_1 should be small, r should be > 1 to spread roots
    # Let's try z_1 in [0.1, 0.5, 1.0, 2.0] and r in [1.5, 2.0, 2.5, 3.0, 4.0]
    
    # Precompute the generalized Laguerre polynomials L_{2k}^{(-1/2)}(x) for k=0..25
    # These are exact symbolic expressions
    x = sp.Symbol('x')
    alpha = sp.Rational(-1, 2)
    
    # L_n^{(alpha)}(x) for n = 0, 2, 4, ..., 50
    # k goes from 0 to 25, so degree 2k goes from 0 to 50
    laguerre_pols = []
    for k in range(26):
        n = 2 * k
        # Use sympy's laguerre polynomial
        L = sp.laguerre(n, x, alpha)
        laguerre_pols.append(L)
    
    # The polynomial g(x) = sum_{k=0}^{25} c_k * L_{2k}^{(-1/2)}(x)
    # c_0, ..., c_25 are the unknowns
    
    # Conditions:
    # g(0) = 0
    # g'(0) = 1
    # For each z_i: g(z_i) = 0 and g'(z_i) = 0
    
    # That's 2 + 12*2 = 26 equations for 26 unknowns.
    
    # We need to build the system exactly.
    # Let's precompute the values of L_{2k}(0) and L'_{2k}(0)
    
    # L_n^{(alpha)}(0) = binomial(n + alpha, n) = (n+alpha)! / (n! alpha!)
    # But we can just evaluate symbolically and then substitute.
    
    # To save time, let's precompute the Laguerre polynomials as polynomials in x
    # and then evaluate them at 0 and take derivatives.
    
    # Actually, let's just build the full system each time since the Laguerre polys are fixed.
    
    # Let's precompute the Laguerre polynomials once
    laguerre_pols_sym = []
    for k in range(26):
        n = 2 * k
        L = sp.laguerre(n, x, alpha)
        # Expand it
        L_expanded = sp.expand(L)
        laguerre_pols_sym.append(L_expanded)
    
    # Now, for a given set of roots z_1..z_12, we build the 26x26 system.
    
    def evaluate_candidate(roots):
        """
        Given 12 roots (as floats or rationals), solve for coefficients and compute B.
        Returns B or None if invalid.
        """
        # Convert roots to rationals for exact arithmetic
        z_rationals = [sp.Rational(float(z)) for z in roots]
        
        # Build the system of equations
        # Unknowns: c_0, c_1, ..., c_25
        c = sp.symbols('c0:26')
        
        # g(x) = sum c_k * L_{2k}(x)
        g = sum(c[k] * laguerre_pols_sym[k] for k in range(26))
        
        # g'(x)
        g_prime = sp.diff(g, x)
        
        # Equations:
        # 1. g(0) = 0
        # 2. g'(0) = 1
        # 3-26. g(z_i) = 0 and g'(z_i) = 0 for i=0..11
        
        equations = []
        
        # g(0) = 0
        eq0 = g.subs(x, 0)
        equations.append(eq0)
        
        # g'(0) = 1
        eq1 = g_prime.subs(x, 0) - 1
        equations.append(eq1)
        
        # For each root
        for i, z_i in enumerate(z_rationals):
            # g(z_i) = 0
            eq_g = g.subs(x, z_i)
            equations.append(eq_g)
            
            # g'(z_i) = 0
            eq_gp = g_prime.subs(x, z_i)
            equations.append(eq_gp)
        
        # Solve the linear system
        # The system is linear in c_0..c_25
        # We have 26 equations and 26 unknowns
        
        # Build the matrix and vector
        # A * c = b, where b is [0, 1, 0, 0, ..., 0]
        
        # Extract coefficients of c_j in each equation
        A = sp.zeros(26, 26)
        b = sp.zeros(26, 1)
        
        for i, eq in enumerate(equations):
            # eq is a linear expression in c_0..c_25
            for j in range(26):
                A[i, j] = sp.coeff(eq, c[j])
        
        # The right-hand side
        b[0, 0] = 0  # g(0) = 0
        b[1, 0] = 1  # g'(0) = 1
        # All others are 0
        
        # Solve using LU decomposition
        try:
            # Check if the system is singular
            det_A = A.det()
            if det_A == 0:
                return None  # Singular system
            
            # Solve A * c = b
            c_sol = sp.LUsolve(A, b)
        except Exception:
            return None
        
        # Now we have the coefficients c_0..c_25
        # Construct g(x) exactly
        g_exact = sum(c_sol[k] * laguerre_pols_sym[k] for k in range(26))
        g_exact = sp.expand(g_exact)
        
        # Construct the denominator: x * prod_i (x - z_i)^2
        denom = x
        for z_i in z_rationals:
            denom = denom * (x - z_i)**2
        denom = sp.expand(denom)
        
        # Perform exact polynomial division: q(x) = g(x) / denom
        # g(x) should be divisible by denom
        try:
            q, remainder = sp.div(g_exact, denom, x)
            if remainder != 0:
                # This shouldn't happen if the system was solved correctly
                # But due to numerical issues in converting floats to rationals, it might
                return None
        except Exception:
            return None
        
        # Find real roots of q(x)
        try:
            real_roots = sp.real_roots(q, x)
        except Exception:
            return None
        
        # For each real root r, check if q has strictly opposite signs at r - 1e-198 and r + 1e-198
        # Note: 1e-198 is a very small number, so we're checking if r is a simple root
        # of q (i.e., q changes sign at r)
        
        counted_roots = []
        for r in real_roots:
            # r is a rational or algebraic number
            # We need to evaluate q at r - 1e-198 and r + 1e-198
            # But 1e-198 is a float, and we're working with exact rationals
            # Let's convert 1e-198 to a rational
            eps = sp.Rational(1, 10)**198
            
            q_minus = q.subs(x, r - eps)
            q_plus = q.subs(x, r + eps)
            
            # Check if they have strictly opposite signs
            # Convert to float for sign check (since exact sign of large rationals can be slow)
            try:
                q_minus_float = float(q_minus)
                q_plus_float = float(q_plus)
            except Exception:
                continue
            
            if q_minus_float * q_plus_float < 0:
                # Strictly opposite signs
                counted_roots.append(float(r))
        
        if not counted_roots:
            return 0.0
        
        B = max(0.0, max(counted_roots))
        return B
    
    # Grid search over z_1 and r
    # z_i = z_1 * r^(i-1) for i=1..12
    # Let's try a coarse grid
    
    z1_values = [0.1, 0.2, 0.5, 1.0, 2.0]
    r_values = [1.3, 1.5, 1.8, 2.0, 2.5, 3.0]
    
    best_B = float('inf')
    best_roots = None
    
    for z1 in z1_values:
        for r in r_values:
            # Check time
            if time.time() - start_time > deadline - safety_margin:
                break
            
            # Generate roots
            roots = [z1 * (r ** (i)) for i in range(12)]
            
            # Check if all roots are positive and distinct
            if any(z <= 0 for z in roots):
                continue
            if len(set(roots)) != 12:
                continue
            
            # Evaluate this candidate
            B = evaluate_candidate(roots)
            
            if B is not None and B < best_B:
                best_B = B
                best_roots = roots.copy()
        
        if time.time() - start_time > deadline - safety_margin:
            break
    
    # If we found a valid candidate, return it
    if best_roots is not None:
        # Ensure they are floats
        return [float(z) for z in best_roots]
    
    # Fallback: return the initial program's roots
    return [4.0 * (i + 1) for i in range(12)]
# EVOLVE-BLOCK-END
