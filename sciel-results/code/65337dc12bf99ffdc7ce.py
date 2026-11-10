import numpy as np
from sympy import symbols, Rational, Poly, L, solve, Matrix, LUdecomp, real_roots, diff, simplify, nsimplify
import time

def construct():
    deadline = time.time() + 600.0
    best_z = None
    best_B = float('inf')
    
    x = symbols('x')
    alpha = Rational(-1, 2)
    
    # Precompute the basis: L_{2k}^{(-1/2)}(x) for k=0..25
    basis = [L(2*k, alpha, x) for k in range(26)]
    
    def try_z(z_list):
        nonlocal best_z, best_B
        if time.time() > deadline - 60:
            return
        # Convert to rationals
        z_rat = [Rational(float(zi)) for zi in z_list]
        # Sort
        z_rat = sorted(z_rat)
        # Check distinct and positive
        for i in range(len(z_rat)):
            if z_rat[i] <= 0:
                return
        for i in range(len(z_rat)-1):
            if z_rat[i] == z_rat[i+1]:
                return
        
        # Build system: 26 unknowns c_0..c_25
        # Equations:
        # g(0) = 0
        # g'(0) = 1
        # g(z_i) = 0, g'(z_i) = 0 for each i
        
        c = symbols('c0:26')
        g = sum(c[k] * basis[k] for k in range(26))
        gp = diff(g, x)
        
        eqs = []
        eqs.append(g.subs(x, 0))  # = 0
        eqs.append(gp.subs(x, 0) - 1)  # = 0
        for zi in z_rat:
            eqs.append(g.subs(x, zi))  # = 0
            eqs.append(gp.subs(x, zi))  # = 0
        
        # Solve linear system
        A = Matrix([[eq.subs({c[k]: 0 for k in range(26)}) for k in range(26)] for eq in eqs])
        b = Matrix([0]*26)
        # Wait, the equations are linear in c. Let me build A properly.
        # A_{row, col} = coefficient of c_col in eq_row
        # b_row = RHS (move non-c terms to RHS, but all terms are linear in c)
        
        # Actually each eq is a linear combination of c's. Let me extract coefficients.
        A = Matrix(26, 26, lambda i, j: 0)
        b = Matrix(26, 1, lambda i, j: 0)
        
        for i, eq in enumerate(eqs):
            for j in range(26):
                A[i, j] = eq.coeff(c[j])
            # The equation is sum_j A[i,j]*c[j] = 0 (for most) or = 1 for the g'(0)=1 case
            # Actually eq is already the full expression. We need sum A[i,j]*c[j] = target
            # For g(0)=0: sum A[i,j]*c[j] = 0
            # For g'(0)=1: sum A[i,j]*c[j] = 1
            # For g(z_i)=0: sum A[i,j]*c[j] = 0
            # For g'(z_i)=0: sum A[i,j]*c[j] = 0
            
            if i == 1:  # g'(0) = 1
                b[i] = 1
            else:
                b[i] = 0
        
        try:
            lu = LUdecomp(A)
            sol = lu.solve(b)
        except Exception:
            return
        
        # Build g with solution
        g_sol = sum(Rational(sol[k]) * basis[k] for k in range(26))
        
        # Build denominator: x * prod_i (x - z_i)^2
        denom = x
        for zi in z_rat:
            denom = denom * (x - zi)**2
        
        # Polynomial division
        g_poly = Poly(g_sol, x)
        denom_poly = Poly(denom, x)
        
        try:
            q_poly, rem_poly = g_poly.div(denom_poly)
        except Exception:
            return
        
        if rem_poly != 0:
            return
        
        # Find real roots of q
        q_expr = q_poly.as_expr()
        try:
            roots = real_roots(q_expr, x)
        except Exception:
            return
        
        # Check sign change at each root
        counted_roots = []
        for r in roots:
            r_val = float(r)
            if r_val <= 0:
                continue
            # Check sign at r - 1e-198 and r + 1e-198
            # Use rational arithmetic
            r_rat = Rational(r_val)
            eps = Rational(10, 10**198)
            left = q_expr.subs(x, r_rat - eps)
            right = q_expr.subs(x, r_rat + eps)
            if left > 0 and right < 0 or left < 0 and right > 0:
                counted_roots.append(r_val)
        
        if not counted_roots:
            B = 0.0
        else:
            B = max(counted_roots)
        
        if B < best_B:
            best_B = B
            best_z = list(z_list)
    
    # Try several candidate sets
    # 1. Geometric spacing
    z1 = [0.1 * (2.0**i) for i in range(12)]
    try_z(z1)
    
    # 2. Roots of L_12^{(-1/2)}
    try:
        from sympy import laguerre
        L12 = laguerre(12, alpha, x)
        roots12 = real_roots(L12, x)
        z2 = [float(r) for r in roots12]
        if all(z > 0 for z in z2) and len(set(z2)) == 12:
            try_z(z2)
    except Exception:
        pass
    
    # 3. Small clustered near origin
    z3 = [0.01 * (i + 1)**2 for i in range(12)]
    try_z(z3)
    
    # 4. Roots of L_12 (alpha=0)
    try:
        L12_0 = L(12, 0, x)
        roots12_0 = real_roots(L12_0, x)
        z4 = [float(r) for r in roots12_0]
        if all(z > 0 for z in z4) and len(set(z4)) == 12:
            try_z(z4)
    except Exception:
        pass
    
    # 5. Uniform small
    z5 = [0.5 * (i + 1) for i in range(12)]
    try_z(z5)
    
    # 6. Very small
    z6 = [0.001 * (i + 1) for i in range(12)]
    try_z(z6)
    
    if best_z is None:
        # Fallback: return something valid
        return [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
    
    return best_z
