"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def _evaluate_exact(z_list):
    """Evaluate the exact score for given z_list. Returns B value or None if invalid."""
    import sympy as sp
    
    x = sp.Symbol('x')
    alpha = sp.Rational(-1, 2)
    
    # Convert z_i to exact rationals
    z_rats = [sp.Rational(float(zi)) for zi in z_list]
    
    # Build the system of equations
    # g(x) = sum_{k=0}^{25} c_k * L_2k^(alpha)(x)
    # Conditions: g(0)=0, g'(0)=1, g(z_i)=0, g'(z_i)=0 for i=1..12
    # Total: 2 + 24 = 26 conditions, 26 unknowns c_0..c_25
    
    # Precompute Laguerre polynomials L_2k^(alpha)(x) for k=0..25
    c_symbols = sp.symbols('c0:26')
    
    # Build g(x)
    g_expr = sp.Integer(0)
    for k in range(26):
        L = sp.genlaguerre(2*k, alpha, x)
        g_expr = g_expr + c_symbols[k] * L
    
    g_prime = sp.diff(g_expr, x)
    
    # Build equations
    eqs = []
    # g(0) = 0
    eqs.append(g_expr.subs(x, 0) - 0)
    # g'(0) = 1
    eqs.append(g_prime.subs(x, 0) - 1)
    
    # g(z_i) = 0 and g'(z_i) = 0 for each z_i
    for zi in z_rats:
        eqs.append(g_expr.subs(x, zi) - 0)
        eqs.append(g_prime.subs(x, zi) - 0)
    
    # Solve the linear system
    A = sp.Matrix([eq.subs(c_symbols[i], 0) + c_symbols[i] * 1 for eq in eqs])  # This is wrong
    # Better: build matrix properly
    A = sp.zeros(26, 26)
    b_vec = sp.zeros(26, 1)
    
    for row, eq in enumerate(eqs):
        # eq is a linear expression in c_0..c_25
        for col, c_sym in enumerate(c_symbols):
            A[row, col] = eq.coeff(c_sym)
        b_vec[row] = -eq.subs({c_sym: 0 for c_sym in c_symbols})
    
    try:
        sol = A.LUsolve(b_vec)
    except Exception:
        return None  # Singular system
    
    # Substitute solution into g
    g_sol = g_expr.subs({c_symbols[i]: sol[i] for i in range(26)})
    
    # Compute denominator: x * prod_i (x - z_i)^2
    denom = x
    for zi in z_rats:
        denom = denom * (x - zi)**2
    
    # Polynomial division
    try:
        q_poly, rem = sp.div(g_sol, denom, x)
        if rem != 0:
            return None  # Not exact division
    except Exception:
        return None
    
    # Find real roots of q
    try:
        real_roots = sp.real_roots(q_poly)
    except Exception:
        return None
    
    # Check sign change at each root with 1e-198 offset
    eps = sp.Rational(1, 10**198)
    max_valid_r = sp.Integer(0)
    
    for r in real_roots:
        q_minus = q_poly.subs(x, r - eps)
        q_plus = q_poly.subs(x, r + eps)
        if q_minus * q_plus < 0:  # Strictly opposite signs
            if r > max_valid_r:
                max_valid_r = r
    
    B = max(sp.Integer(0), max_valid_r)
    return float(B)

def construct():
    start_time = time.time()
    deadline = start_time + 600  # Safety margin
    
    # Try several candidate configurations
    candidates = []
    
    # Candidate 1: small evenly spaced
    candidates.append([0.05 * (i + 1) for i in range(12)])
    
    # Candidate 2: very small
    candidates.append([0.01 * (i + 1) for i in range(12)])
    
    # Candidate 3: geometric spacing
    candidates.append([0.01 * (2.0 ** i) for i in range(12)])
    
    # Candidate 4: clustered near 1
    candidates.append([0.5 + 0.05 * (i + 1) for i in range(12)])
    
    # Candidate 5: near 0
    candidates.append([0.001 * (i + 1) for i in range(12)])
    
    best_B = float('inf')
    best_z = None
    
    for z_cand in candidates:
        if time.time() > deadline - 60:  # Need 60s margin
            break
        
        # Check validity
        if any(zi <= 0 or not np.isfinite(zi) for zi in z_cand):
            continue
        if len(set(round(zi, 15) for zi in z_cand)) < 12:
            continue
        
        try:
            B = _evaluate_exact(z_cand)
            if B is not None and B < best_B:
                best_B = B
                best_z = z_cand[:]
        except Exception:
            continue
    
    if best_z is None:
        # Fallback: return the first candidate
        return candidates[0]
    
    return best_z
# EVOLVE-BLOCK-END
