import scipy.special
import numpy as np
import time
import sys

def _compute_B(z_list):
    """Compute B for a given list of 12 positive distinct values."""
    import sympy
    
    if not all(z > 0 and np.isfinite(z) for z in z_list):
        return None
    if len(set(round(z, 15) for z in z_list)) != 12:
        return None
    
    # Convert to rationals
    z_rat = [sympy.Rational(float(z)) for z in z_list]
    z_rat = sorted(z_rat)
    
    x = sympy.Symbol('x')
    alpha = sympy.Rational(-1, 2)
    
    # Build the system: g(x) = sum_{k=0}^{25} c_k * L_{2k}^{alpha}(x)
    # Conditions: g(0)=0, g'(0)=1, g(z_i)=0, g'(z_i)=0 for i=1..12
    # Total: 2 + 2*12 = 26 equations, 26 unknowns
    
    # Precompute Laguerre polynomials L_{2k}^{alpha}(x) for k=0..25
    L = []
    for k in range(26):
        L.append(sympy.laguerre(2*k, alpha, x))
    
    # Build coefficient matrix and RHS
    # Unknowns: c_0, c_1, ..., c_25
    A = sympy.zeros(26, 26)
    b = sympy.zeros(26, 1)
    
    # Equation 0: g(0) = 0
    for k in range(26):
        A[0, k] = L[k].subs(x, 0)
    b[0] = 0
    
    # Equation 1: g'(0) = 1
    for k in range(26):
        A[1, k] = sympy.diff(L[k], x).subs(x, 0)
    b[1] = 1
    
    # Equations 2-25: g(z_i) = 0 and g'(z_i) = 0 for i=0..11
    for i, zi in enumerate(z_rat):
        row_val = 2 + 2*i
        row_der = 3 + 2*i
        for k in range(26):
            A[row_val, k] = L[k].subs(x, zi)
            A[row_der, k] = sympy.diff(L[k], x).subs(x, zi)
        b[row_val] = 0
        b[row_der] = 0
    
    # Solve the linear system
    try:
        sol = sympy.LUsolve(A, b)
    except Exception:
        return None
    
    if sol is None or sol.has(sympy.oo) or sol.has(sympy.zoo):
        return None
    
    # Build g(x)
    g = sum(sol[k] * L[k] for k in range(26))
    g = sympy.expand(g)
    
    # Build denominator: x * prod_i (x - z_i)^2
    denom = x
    for zi in z_rat:
        denom *= (x - zi)**2
    
    # Polynomial division: q(x) = g(x) / denom
    try:
        q, rem = sympy.div(g, denom, x)
    except Exception:
        return None
    
    if rem != 0:
        # Not a clean division - this shouldn't happen if the system is correct
        return None
    
    # Find real roots of q
    try:
        roots = sympy.real_roots(q)
    except Exception:
        return None
    
    if not roots:
        return 0
    
    # Check sign change at each root
    B = 0
    for r in roots:
        r_val = float(r)
        if not np.isfinite(r_val):
            continue
        eps = sympy.Rational(1, 10)**198
        r_sym = sympy.Rational(r_val)
        try:
            q_minus = q.subs(x, r_sym - eps)
            q_plus = q.subs(x, r_sym + eps)
            if q_minus * q_plus < 0:
                if r_val > B:
                    B = r_val
        except Exception:
            continue
    
    return B


def construct():
    start_time = time.time()
    deadline = start_time + 580  # Safety margin
    
    # Initial guess: squares of J_{-1/2} zeros
    # J_{-1/2}(x) = sqrt(2/(pi*x)) * cos(x)
    # Zeros of cos(x) are at (k+1/2)*pi for k=0,1,2,...
    # So J_{-1/2} zeros are at (k+1/2)*pi
    # Squaring: ((k+1/2)*pi)^2
    
    initial_z = [((k + 0.5) * np.pi)**2 for k in range(12)]
    
    # Convert to list of floats
    z = [float(z) for z in initial_z]
    
    # Evaluate initial B
    B_current = _compute_B(z)
    if B_current is None:
        # Fallback to J0 zeros squared
        j0_zeros = scipy.special.jn_zeros(0, 12)
        z = [float(x**2) for x in j0_zeros]
        B_current = _compute_B(z)
        if B_current is None:
            return [4.0 * (i + 1) for i in range(12)]
    
    best_z = z[:]
    best_B = B_current
    
    # Local optimization: perturb each z_i and see if B decreases
    # Use a simple coordinate descent / random perturbation approach
    
    np.random.seed(42)
    
    for iteration in range(100):
        if time.time() > deadline:
            break
        
        # Try perturbing each coordinate
        improved = False
        for i in range(12):
            if time.time() > deadline:
                break
            
            # Try several perturbation sizes
            for scale in [0.1, 0.05, 0.01, 0.005, 0.001]:
                for direction in [-1, 1]:
                    if time.time() > deadline:
                        break
                    
                    z_test = best_z[:]
                    z_test[i] = best_z[i] + direction * scale * max(1.0, best_z[i])
                    
                    # Ensure positivity and distinctness
                    if z_test[i] <= 0:
                        continue
                    if any(abs(z_test[i] - z_j) < 1e-10 for j in range(12) if j != i):
                        continue
                    
                    B_test = _compute_B(z_test)
                    if B_test is not None and B_test < best_B - 1e-12:
                        best_z = z_test[:]
                        best_B = B_test
                        improved = True
                        break  # Restart perturbation search from new point
        
        if not improved:
            # Try random perturbations
            for _ in range(5):
                if time.time() > deadline:
                    break
                
                z_test = best_z[:]
                for i in range(12):
                    z_test[i] *= (1.0 + np.random.uniform(-0.05, 0.05))
                
                if all(z > 0 for z in z_test) and len(set(round(z, 10) for z in z_test)) == 12:
                    B_test = _compute_B(z_test)
                    if B_test is not None and B_test < best_B - 1e-12:
                        best_z = z_test[:]
                        best_B = B_test
                        break
    
    return best_z
