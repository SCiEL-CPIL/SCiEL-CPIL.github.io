import sympy
import time
import sys

def construct():
    start_time = time.time()
    time_limit = 580  # Safety margin
    
    def evaluate(Z):
        """
        Evaluates the score for a given list of z_i.
        Returns (score, valid)
        """
        try:
            # Check validity of Z
            if len(Z) != 12:
                return (float('inf'), False)
            for z in Z:
                if not (z > 0 and z == z): # Check finite and > 0
                    return (float('inf'), False)
            
            # Check distinctness
            Z_sorted = sorted([float(z) for z in Z])
            for i in range(11):
                if Z_sorted[i] == Z_sorted[i+1]:
                    return (float('inf'), False)
            
            # Convert to exact rationals
            z_rats = [sympy.Rational(float(z)) for z in Z]
            
            # Define alpha
            alpha = sympy.Rational(-1, 2)
            
            # Build the system of equations
            # g(x) = sum_{k=0}^{25} c_k L_{2k}^{alpha}(x)
            # Conditions:
            # 1. g(0) = 0
            # 2. g'(0) = 1
            # 3. g(z_i) = 0 for i=1..12
            # 4. g'(z_i) = 0 for i=1..12
            
            # Precompute Laguerre polynomials L_{2k}^{alpha}(x) and their derivatives
            # We need L_{2k}(x) and L'_{2k}(x)
            
            x = sympy.Symbol('x')
            
            # Coefficients c_k are unknowns
            c = [sympy.Symbol(f'c_{k}') for k in range(26)]
            
            # Define g(x)
            # L_n^{alpha}(x) is generalized Laguerre
            # sympy.laguerre(n, alpha, x)
            
            # We need to evaluate at x=0 and x=z_i
            # And derivatives at x=0 and x=z_i
            
            # It is more efficient to build the matrix A and vector b
            # A[0, k] = L_{2k}(0)
            # A[1, k] = L'_{2k}(0)
            # A[2i, k] = L_{2k}(z_i)
            # A[2i+1, k] = L'_{2k}(z_i)
            
            # b[0] = 0
            # b[1] = 1
            # b[2i] = 0
            # b[2i+1] = 0
            
            A = sympy.zeros(26, 26)
            b = sympy.zeros(26, 1)
            
            # Compute L_{2k}(x) and derivatives
            # Note: L_n(x) is a polynomial.
            # We can compute the polynomial once and then evaluate.
            
            polys = []
            dpolys = []
            for k in range(26):
                n = 2 * k
                L_n = sympy.laguerre(n, alpha, x)
                dL_n = sympy.diff(L_n, x)
                polys.append(L_n)
                dpolys.append(dL_n)
            
            # Fill A and b
            # Row 0: g(0) = 0
            for k in range(26):
                A[0, k] = polys[k].subs(x, 0)
            b[0] = 0
            
            # Row 1: g'(0) = 1
            for k in range(26):
                A[1, k] = dpolys[k].subs(x, 0)
            b[1] = 1
            
            # Rows for z_i
            for i, z_val in enumerate(z_rats):
                row_g = 2 * i + 2
                row_dg = 2 * i + 3
                
                for k in range(26):
                    A[row_g, k] = polys[k].subs(x, z_val)
                    A[row_dg, k] = dpolys[k].subs(x, z_val)
                
                b[row_g] = 0
                b[row_dg] = 0
            
            # Solve A c = b
            # Check for singularity
            if A.det() == 0:
                return (float('inf'), False)
                
            c_sol = sympy.LUsolve(A, b)
            
            # Construct g(x)
            g = sum(c_sol[k] * polys[k] for k in range(26))
            
            # Construct denominator: x * prod (x - z_i)^2
            denom = x
            for z_val in z_rats:
                denom = denom * (x - z_val)**2
            
            # q(x) = g(x) / denom
            # Exact polynomial division
            q, r = sympy.div(g, denom, x)
            
            # Check remainder
            if r != 0:
                # This should not happen if the system is consistent
                return (float('inf'), False)
                
            # Find real roots of q
            # q is a polynomial of degree 25
            roots = sympy.real_roots(q)
            
            # Filter roots:
            # 1. Must be real (real_roots ensures this)
            # 2. Must be positive (r > 0)
            # 3. Must have sign change at r
            #    Check q(r - 1e-198) and q(r + 1e-198)
            
            max_r = 0
            for r in roots:
                if r > 0:
                    # Evaluate q at r - eps and r + eps
                    # Use a very small epsilon
                    eps = sympy.Rational(1, 10**198)
                    val_left = q.subs(x, r - eps)
                    val_right = q.subs(x, r + eps)
                    
                    # Check if they have opposite signs
                    # val_left * val_right < 0
                    if val_left * val_right < 0:
                        if r > max_r:
                            max_r = r
            
            B = max_r
            score = B / (2 * sympy.pi)
            return (float(score), True)
            
        except Exception as e:
            return (float('inf'), False)

    # Candidate sets
    candidates = []
    
    # 1. Small linear
    candidates.append([i / 12.0 for i in range(1, 13)])
    
    # 2. Very small linear
    candidates.append([i / 100.0 for i in range(1, 13)])
    
    # 3. Linear 1..12
    candidates.append([float(i) for i in range(1, 13)])
    
    # 4. Geometric
    candidates.append([2.0**i for i in range(12)])
    
    # 5. Clustered near 1
    candidates.append([1.0 + 0.1 * i for i in range(12)])
    
    best_score = float('inf')
    best_Z = None
    
    for Z in candidates:
        if time.time() - start_time > time_limit:
            break
            
        score, valid = evaluate(Z)
        
        if valid and score < best_score:
            best_score = score
            best_Z = Z
            
        # If we found a perfect score (0), we can stop
        if best_score == 0.0:
            break
            
    if best_Z is None:
        # Fallback
        return [4.0 * (i + 1) for i in range(12)]
        
    return best_Z
