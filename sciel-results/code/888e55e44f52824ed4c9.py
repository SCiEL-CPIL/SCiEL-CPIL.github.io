import time
import math
from fractions import Fraction
import numpy as np
import sympy as sp

# EVOLVE-BLOCK-START

def construct():
    """
    Returns 12 double-root locations z_1..z_12 for the Laguerre double-root problem.
    
    Strategy: Use the zeros of the generalized Laguerre polynomial L_12^(-1/2)(x)
    as the initial nodes. These are the natural nodes for this type of problem
    and should minimize the largest root B of the quotient polynomial q(x).
    
    We then perform a local optimization to further reduce B.
    """
    deadline = time.time() + 620  # Safety margin before 630s limit
    
    # Step 1: Get zeros of L_12^(-1/2)(x)
    x = sp.Symbol('x')
    L12 = sp.genlaguerre(12, -sp.Rational(1, 2), x)
    
    # Find the zeros numerically first
    zeros = sp.nroots(L12, n=50)
    zeros = [z.re for z in zeros if abs(z.im) < 1e-10]
    zeros.sort()
    
    # Convert to floats
    z_init = np.array([float(z) for z in zeros])
    
    # Verify they are positive and distinct
    if len(z_init) != 12 or np.any(z_init <= 0) or len(set(z_init)) != 12:
        # Fallback: use simple positive values
        z_init = np.array([4.0 * (i + 1) for i in range(12)])
    
    def evaluate(z_values):
        """
        Evaluate the score for given z values.
        Returns B or None if invalid.
        """
        try:
            # Convert to rationals
            z_rationals = [sp.Rational(float(z)) for z in z_values]
            z_rationals.sort()
            
            # Check validity
            if len(z_rationals) != 12:
                return None
            if any(z <= 0 for z in z_rationals):
                return None
            if len(set(z_rationals)) != 12:
                return None
            
            # Build the system of equations
            # g(x) = sum_k c_k L_{2k}(x), k = 0..25
            # Unknowns: c_0, c_1, ..., c_25 (26 unknowns)
            
            c = sp.symbols('c0:26')
            
            # Build g(x)
            g = sp.Integer(0)
            for k in range(26):
                L_2k = sp.genlaguerre(2*k, -sp.Rational(1, 2), x)
                g += c[k] * L_2k
            
            # Conditions:
            # 1. g(0) = 0
            # 2. g'(0) = 1
            # 3. g(z_i) = 0 for each i
            # 4. g'(z_i) = 0 for each i
            
            g_prime = sp.diff(g, x)
            
            eqs = []
            eqs.append(sp.Eq(g.subs(x, 0), 0))
            eqs.append(sp.Eq(g_prime.subs(x, 0), 1))
            
            for z_i in z_rationals:
                eqs.append(sp.Eq(g.subs(x, z_i), 0))
                eqs.append(sp.Eq(g_prime.subs(x, z_i), 0))
            
            # Solve the linear system
            # Extract coefficients
            A = []
            b = []
            for eq in eqs:
                lhs = eq.lhs
                rhs = eq.rhs
                # lhs - rhs = 0
                expr = sp.simplify(lhs - rhs)
                # Extract coefficients of c_0, ..., c_25
                row = []
                for cj in c:
                    row.append(expr.coeff(cj))
                A.append(row)
                b.append(-sp.simplify(expr.coeff(c, 0) * 0))  # constant term
                # Actually, we need to handle this more carefully
                # Let's use a different approach
            
            # Better approach: use linear system directly
            A_mat = sp.Matrix(len(eqs), 26, lambda i, j: eqs[i].lhs.coeff(c[j]))
            b_vec = sp.Matrix([eqs[i].rhs for i in range(len(eqs))])
            
            # Check if system is singular
            if A_mat.det() == 0:
                return None
            
            # Solve
            try:
                c_vals = A_mat.LUsolve(b_vec)
            except Exception:
                return None
            
            # Substitute c values back into g
            g_sol = g
            for i, ci in enumerate(c):
                g_sol = g_sol.subs(ci, c_vals[i])
            
            # Build q(x) = g(x) / (x * prod_i (x - z_i)^2)
            denom = x
            for z_i in z_rationals:
                denom *= (x - z_i)**2
            
            # Polynomial division
            try:
                q_poly = sp.div(g_sol, denom, x)
                # q_poly is (quotient, remainder)
                quotient, remainder = q_poly
            except Exception:
                return None
            
            # Check if remainder is zero (or very small)
            if remainder != 0:
                # If remainder is not zero, the division doesn't work out
                # This might mean the system is inconsistent
                return None
            
            # Find real roots of quotient
            try:
                real_roots = sp.real_roots(quotient, x)
            except Exception:
                return None
            
            # Count roots where q has opposite signs at r ± 1e-198
            eps = sp.Rational(1, 10**198)
            counted_roots = []
            for r in real_roots:
                if r <= 0:
                    continue
                q_left = quotient.subs(x, r - eps)
                q_right = quotient.subs(x, r + eps)
                if q_left * q_right < 0:
                    counted_roots.append(r)
            
            if not counted_roots:
                B = 0
            else:
                B = max(counted_roots)
            
            return float(B)
        
        except Exception as e:
            return None
    
    # Evaluate initial solution
    B_init = evaluate(z_init)
    best_z = z_init.copy()
    best_B = B_init if B_init is not None else float('inf')
    
    # Local optimization: try small perturbations
    # Use a simple coordinate descent or random perturbation approach
    np.random.seed(42)
    
    for iteration in range(50):
        if time.time() > deadline - 60:
            break
        
        # Try random perturbations
        for trial in range(100):
            if time.time() > deadline - 60:
                break
            
            # Perturb each z by a small amount
            z_trial = best_z.copy()
            for i in range(12):
                # Keep z positive and maintain ordering
                delta = np.random.normal(0, 0.1)
                z_trial[i] += delta
            
            # Ensure positivity and distinctness
            if np.any(z_trial <= 0):
                continue
            z_trial.sort()
            if len(set(z_trial)) != 12:
                continue
            
            B_trial = evaluate(z_trial)
            if B_trial is not None and B_trial < best_B:
                best_B = B_trial
                best_z = z_trial.copy()
    
    # Return the best solution
    result = [float(z) for z in best_z]
    result.sort()
    
    # Final validity check
    if len(result) != 12:
        return [4.0 * (i + 1) for i in range(12)]
    if any(z <= 0 for z in result):
        return [4.0 * (i + 1) for i in range(12)]
    if len(set(result)) != 12:
        return [4.0 * (i + 1) for i in range(12)]
    
    return result

# EVOLVE-BLOCK-END
