import time
import sympy as sp

def _evaluate_z_set(z_list):
    """Evaluate a candidate z set, return B or None if invalid."""
    x = sp.Symbol('x')
    alpha = sp.Rational(-1, 2)
    
    # Build Laguerre polynomials L_0, L_2, ..., L_50
    L = [sp.genlaguerre(2*k, alpha, x) for k in range(26)]
    
    c = sp.symbols('c0:26')
    g = sum(c[i] * L[i] for i in range(26))
    gp = sp.diff(g, x)
    
    b_vec = sp.zeros(26, 1)
    b_vec[1] = 1
    
    coeff_matrix = sp.zeros(26, 26)
    
    g0 = g.subs(x, 0)
    for i in range(26):
        coeff_matrix[0, i] = g0.coeff(c[i])
    
    gp0 = gp.subs(x, 0)
    for i in range(26):
        coeff_matrix[1, i] = gp0.coeff(c[i])
    
    row = 2
    for zi in z_list:
        zi_r = sp.Rational(float(zi))
        gzi = g.subs(x, zi_r)
        gpzi = gp.subs(x, zi_r)
        for i in range(26):
            coeff_matrix[row, i] = gzi.coeff(c[i])
        row += 1
        for i in range(26):
            coeff_matrix[row, i] = gpzi.coeff(c[i])
        row += 1
    
    try:
        sol = coeff_matrix.LUsolve(b_vec)
    except Exception:
        return None
    
    g_sol = sp.expand(sum(sol[i] * L[i] for i in range(26)))
    
    denom = x
    for zi in z_list:
        zi_r = sp.Rational(float(zi))
        denom = denom * (x - zi_r)**2
    denom = sp.expand(denom)
    
    try:
        q, rem = sp.div(g_sol, denom, x)
    except Exception:
        return None
    
    if rem != 0:
        return None
    
    try:
        real_roots = sp.real_roots(q, x)
    except Exception:
        return None
    
    B = 0
    for r in real_roots:
        r_val = float(r)
        if r_val <= 0:
            continue
        qm = float(q.subs(x, r_val - 1e-198))
        qp = float(q.subs(x, r_val + 1e-198))
        if qm * qp < 0:
            if r_val > B:
                B = r_val
    
    return B

def construct():
    start_time = time.time()
    deadline = 630.0
    
    candidates = [
        [4.0 * (i + 1) for i in range(12)],           # seed
        [0.1 * (i + 1) for i in range(12)],           # small even
        [0.05 * (i + 1) for i in range(12)],          # smaller even
        [0.5 + 0.01*(i-5.5) for i in range(12)],      # tight near 0.5
        [1.0 + 0.01*i for i in range(12)],            # tight near 1
        [0.01 * (10**(i/11)) for i in range(12)],     # log spaced
        [0.05 * (1.15**i) for i in range(12)],        # geometric
    ]
    
    best_B = float('inf')
    best_z = candidates[0]
    
    for z_set in candidates:
        if time.time() - start_time > deadline - 150:
            break
        try:
            B = _evaluate_z_set(z_set)
            if B is not None and B < best_B:
                best_B = B
                best_z = z_set
        except Exception:
            continue
    
    return best_z
