import numpy as np
import sympy
from sympy import Rational, Laguerre, symbols, diff, solve, Poly, real_roots
import time

def construct():
    start_time = time.time()
    deadline = 630.0  # seconds, leave margin
    
    # Precompute symbolic setup
    x = symbols('x')
    alpha = Rational(-1, 2)
    
    # Build the 26 Laguerre polynomials L_{2k}^{-1/2}(x) for k=0..25
    laguerres = []
    for k in range(26):
        L = Laguerre(2*k, x, alpha)
        laguerres.append(L)
    
    # Precompute derivatives
    laguerres_deriv = [diff(L, x) for L in laguerres]
    
    def evaluate_B(z_list):
        """Given 12 z values, compute B/(2*pi). Returns inf if invalid."""
        z_rationals = [Rational(float(z)) for z in z_list]
        
        # Build the 26x26 system
        A = sympy.Matrix(26, 26, lambda r, c: 0)
        b = sympy.Matrix(26, 1, lambda r, c: 0)
        
        # Row 0: g(0) = 0
        for k in range(26):
            A[0, k] = laguerres[k].subs(x, 0)
        b[0] = 0
        
        # Row 1: g'(0) = 1
        for k in range(26):
            A[1, k] = laguerres_deriv[k].subs(x, 0)
        b[1] = 1
        
        # Rows 2..25: for each z_i, g(z_i)=0 and g'(z_i)=0
        for i in range(12):
            zi = z_rationals[i]
            row_g = 2 + 2*i
            row_gp = 3 + 2*i
            for k in range(26):
                A[row_g, k] = laguerres[k].subs(x, zi)
                A[row_gp, k] = laguerres_deriv[k].subs(x, zi)
            b[row_g] = 0
            b[row_gp] = 0
        
        # Solve the system
        try:
            c_vec = A.LUsolve(b)
        except Exception:
            return float('inf')
        
        # Build g(x)
        g = sum(c_vec[k] * laguerres[k] for k in range(26))
        
        # Build denominator: x * prod (x - z_i)^2
        denom = x
        for zi in z_rationals:
            denom = denom * (x - zi)**2
        
        # q(x) = g(x) / denom, must be exact polynomial
        try:
            q_poly = sympy.Poly(g, x) / sympy.Poly(denom, x)
            q_expr = q_poly.as_expr()
        except Exception:
            return float('inf')
        
        # Find real roots of q
        try:
            roots = real_roots(q_expr, x)
        except Exception:
            return float('inf')
        
        # Check sign change at r ± 1e-198
        B = 0
        for r in roots:
            r_val = float(r)
            if not np.isfinite(r_val):
                continue
            try:
                q_left = float(q_expr.subs(x, Rational(r_val - 1e-198)))
                q_right = float(q_expr.subs(x, Rational(r_val + 1e-198)))
            except Exception:
                continue
            if q_left * q_right < 0:
                if r_val > B:
                    B = r_val
        
        return B / (2 * np.pi)
    
    # Initial guesses
    initial_guesses = [
        [4.0 * (i + 1) for i in range(12)],       # Seed: 4,8,...,48
        [1.0 + i for i in range(12)],              # 1,2,...,12
        [0.5 * (i + 1) for i in range(12)],        # 0.5,1,...,6
        [2.0 * (i + 1) for i in range(12)],        # 2,4,...,24
        [0.1 * (i + 1) for i in range(12)],        # 0.1,0.2,...,1.2
        [10.0 * (i + 1) for i in range(12)],       # 10,20,...,120
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5],
    ]
    
    best_B = float('inf')
    best_z = None
    
    # Evaluate initial guesses
    for guess in initial_guesses:
        if time.time() > deadline - 60:
            break
        try:
            B_val = evaluate_B(guess)
            if B_val < best_B:
                best_B = B_val
                best_z = guess[:]
        except Exception:
            pass
    
    # Simple local search: perturb the best solution
    if best_z is not None and time.time() < deadline - 90:
        # Try small perturbations
        for scale in [0.1, 0.5, 1.0, 2.0]:
            if time.time() > deadline - 90:
                break
            for _ in range(3):
                if time.time() > deadline - 90:
                    break
                perturbed = [z * (1 + scale * (2*np.random.random() - 1)) for z in best_z]
                # Ensure positive
                perturbed = [max(0.01, p) for p in perturbed]
                # Ensure distinct
                for i in range(len(perturbed)):
                    for j in range(i+1, len(perturbed)):
                        if abs(perturbed[i] - perturbed[j]) < 1e-4:
                            perturbed[j] = perturbed[i] + 1e-3
                try:
                    B_val = evaluate_B(perturbed)
                    if B_val < best_B:
                        best_B = B_val
                        best_z = perturbed[:]
                except Exception:
                    pass
    
    # Return best_z
    if best_z is None:
        return [4.0 * (i + 1) for i in range(12)]
    
    result = [float(z) for z in best_z]
    
    # Ensure validity
    for i in range(len(result)):
        if not np.isfinite(result[i]) or result[i] <= 0:
            result[i] = 1.0 + i
    for i in range(len(result)):
        for j in range(i+1, len(result)):
            if abs(result[i] - result[j]) < 1e-4:
                result[j] = result[i] + 1e-3
    
    return result
