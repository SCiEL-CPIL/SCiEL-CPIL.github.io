"""
Strategy:
Use the roots of the generalized Laguerre polynomial L_12^(-1/2)(x) as initial double-root locations,
then attempt local optimization if time permits. The Laguerre nodes minimize interpolation error
in the Laguerre basis, which should minimize the quotient q(x) and thus B.
"""
import time
import math
import sympy

def _compute_B(z_list, deadline):
    """
    Compute B for given z_list using exact rational arithmetic.
    Returns B (float) or None if timeout/error.
    """
    if time.time() > deadline:
        return None
    
    x = sympy.Symbol('x')
    alpha = sympy.Rational(-1, 2)
    
    # Convert z to rationals
    z_rats = [sympy.Rational(float(zi)) for zi in z_list]
    
    # Build basis: L_2k(x) for k=0..25
    basis = [sympy.genlaguerre(2*k, alpha, x) for k in range(26)]
    
    # Build system: g(0)=0, g'(0)=1, g(z_i)=0, g'(z_i)=0 for each i
    # 2 + 2*12 = 26 equations, 26 unknowns c_0..c_25
    
    # Compute derivatives
    basis_deriv = [b.diff(x) for b in basis]
    
    # Build matrix A and vector b
    A = sympy.Matrix(26, 26, lambda i, j: 0)
    b_vec = sympy.Matrix(26, 1, lambda i, j: 0)
    
    # g(0) = 0
    A[0, :] = [sympy.Rational(basis[j].subs(x, 0)) for j in range(26)]
    b_vec[0] = sympy.Rational(0)
    
    # g'(0) = 1
    A[1, :] = [sympy.Rational(basis_deriv[j].subs(x, 0)) for j in range(26)]
    b_vec[1] = sympy.Rational(1)
    
    # g(z_i) = 0 and g'(z_i) = 0
    for i, zi in enumerate(z_rats):
        row_g = 2 + 2*i
        row_gp = 3 + 2*i
        A[row_g, :] = [sympy.Rational(basis[j].subs(x, zi)) for j in range(26)]
        b_vec[row_g] = sympy.Rational(0)
        A[row_gp, :] = [sympy.Rational(basis_deriv[j].subs(x, zi)) for j in range(26)]
        b_vec[row_gp] = sympy.Rational(0)
    
    if time.time() > deadline:
        return None
    
    # Solve
    try:
        c = A.LUsolve(b_vec)
    except Exception:
        return None
    
    if time.time() > deadline:
        return None
    
    # Build g(x)
    g = sum(c[k] * basis[k] for k in range(26))
    g = sympy.expand(g)
    
    # Build denominator: x * prod_i (x - z_i)^2
    denom = x
    for zi in z_rats:
        denom *= (x - zi)**2
    denom = sympy.expand(denom)
    
    # Polynomial division: q = g / denom
    try:
        q, rem = sympy.div(g, denom, x)
    except Exception:
        return None
    
    if rem != 0:
        # Should not happen if system is correct, but be safe
        return None
    
    if time.time() > deadline:
        return None
    
    # Find real roots of q
    try:
        real_roots = sympy.real_roots(q, x)
    except Exception:
        return None
    
    if not real_roots:
        return 0.0
    
    # Check sign change at each root with epsilon = 1e-198
    eps = sympy.Rational(1, 10**198)
    max_r = 0.0
    
    for r in real_roots:
        r_val = float(r)
        if r_val <= 0:
            continue
        
        # Check sign at r - eps and r + eps
        try:
            val_minus = q.subs(x, r - eps)
            val_plus = q.subs(x, r + eps)
        except Exception:
            continue
        
        if val_minus == 0 or val_plus == 0:
            continue
        
        if (val_minus > 0 and val_plus < 0) or (val_minus < 0 and val_plus > 0):
            if r_val > max_r:
                max_r = r_val
    
    return max_r

def construct():
    """
    Returns 12 double-root locations z_1..z_12.
    Uses roots of L_12^(-1/2)(x) as primary candidate.
    """
    t0 = time.time()
    DEADLINE = 600
    SAFETY_MARGIN = 30
    deadline = t0 + DEADLINE - SAFETY_MARGIN
    
    # Deadline guard
    if time.time() > deadline:
        return [4.0 * (i + 1) for i in range(12)]
    
    try:
        # Compute roots of L_12^(-1/2)(x)
        x = sympy.Symbol('x')
        L_poly = sympy.genlaguerre(12, sympy.Rational(-1, 2), x)
        roots = L_poly.roots()
        
        z_list = sorted([float(r) for r in roots])
        
        if len(z_list) != 12:
            return [4.0 * (i + 1) for i in range(12)]
        if not all(zi > 0 for zi in z_list):
            return [4.0 * (i + 1) for i in range(12)]
        if len(set(z_list)) != 12:
            return [4.0 * (i + 1) for i in range(12)]
    except Exception:
        return [4.0 * (i + 1) for i in range(12)]
    
    # We have valid Laguerre roots. Try to evaluate B and possibly improve.
    # First, check if we have enough time for at least one B evaluation
    time_left = deadline - time.time()
    
    if time_left < 45:  # Need at least ~45s for one evaluation + safety
        return z_list
    
    # Try to compute B for the Laguerre roots
    B_base = _compute_B(z_list, deadline)
    
    if time.time() > deadline:
        return z_list
    
    best_z = z_list
    best_B = B_base if B_base is not None else float('inf')
    
    # Try a few small perturbations if time allows
    if best_B is not None and time.time() < deadline - 50:
        import random
        random.seed(42)
        
        # Try perturbing each z_i slightly
        for attempt in range(3):
            if time.time() > deadline - 60:
                break
            
            z_trial = list(z_list)
            # Perturb a random subset
            for idx in random.sample(range(12), 4):
                z_trial[idx] *= (1.0 + random.uniform(-0.05, 0.05))
            
            # Ensure positive and distinct
            if not all(zi > 0 for zi in z_trial):
                continue
            if len(set(z_trial)) != 12:
                continue
            
            B_trial = _compute_B(z_trial, deadline)
            if B_trial is not None and B_trial < best_B:
                best_B = B_trial
                best_z = z_trial
    
    return best_z
