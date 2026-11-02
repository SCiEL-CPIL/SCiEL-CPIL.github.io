"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction
from itertools import product
from collections import defaultdict

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 130.0  # safety margin before 150s kill

    # Schoolbook decomposition of rank n*m*p
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    best = schoolbook()
    best_rank = n * m * p

    # Nonzero entries of the tensor
    nz_set = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                nz_set.add((i*m+j, j*p+k, k*n+i))
    nz = list(nz_set)

    # Verify a decomposition: check that sum_t U[a,t]*V[b,t]*W[c,t] = 1 for (a,b,c) in nz, 0 otherwise
    def verify(U, V, W):
        r = len(U[0])
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    s = Fraction(0)
                    for t in range(r):
                        s += U[a][t] * V[b][t] * W[c][t]
                    expected = 1 if (a, b, c) in nz_set else 0
                    if s != expected:
                        return False
        return True

    # Try alternating bilinear solve for a given W
    def solve_given_W(W, max_iters=50):
        r = len(W[0])
        # Initialize U and V randomly with small rationals
        U = [[Fraction(1, 2) if (a + t) % 3 == 0 else Fraction(1, 4) for t in range(r)] for a in range(n*m)]
        V = [[Fraction(1, 2) if (b + t) % 3 == 0 else Fraction(1, 4) for t in range(r)] for b in range(m*p)]
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                return None
            
            # Given W and V, solve for U:
            # For each (a, b, c): sum_t U[a,t] * V[b,t] * W[c,t] = T[a,b,c]
            # This is linear in U[a,t] for fixed a
            # For each a, we have equations: sum_t U[a,t] * (V[b,t] * W[c,t]) = T[a,b,c] for all (b,c)
            # This is a system: M_a * U_a = rhs_a where M_a[b,c,t] = V[b,t]*W[c,t]
            
            # Actually, let's do it differently. For fixed W, the equations are:
            # sum_t (U[a,t] * V[b,t]) * W[c,t] = T[a,b,c]
            # Let X[a,t] = U[a,t], Y[b,t] = V[b,t]
            # Then sum_t X[a,t] * Y[b,t] * W[c,t] = T[a,b,c]
            
            # Given Y (i.e., V), for each a, the equations become:
            # sum_t X[a,t] * (sum_b' Y[b',t] * W[c,t] * delta(b',b)) ... no that's not right.
            
            # Let me think again. The full system is:
            # For all (a,b,c): sum_t U[a,t] * V[b,t] * W[c,t] = T[a,b,c]
            
            # Fix V. Then for each a, we need to find U[a,0..r-1] such that:
            # For all (b,c): sum_t U[a,t] * V[b,t] * W[c,t] = T[a,b,c]
            # This is a linear system in U[a,t] with (m*p)*(p*n) equations and r unknowns.
            # Generally overdetermined. We solve least squares.
            
            # Build the linear system for U given V:
            # Unknowns: U[a,0], U[a,1], ..., U[a,r-1] for each a
            # Equation (b,c,a): sum_t U[a,t] * V[b,t] * W[c,t] = T[a,b,c]
            
            # Let's use numpy for speed, then convert back
            # For each a, build matrix A_a of shape (m*p*p*n, r) where row (b,c) = [V[b,0]*W[c,0], ..., V[b,r-1]*W[c,r-1]]
            # rhs_a[(b,c)] = T[a,b,c]
            
            import numpy as np
            
            U_new = [[Fraction(0)] * r for _ in range(n*m)]
            for a in range(n*m):
                # Build system
                rows = []
                rhs = []
                for b in range(m*p):
                    for c in range(p*n):
                        row = [V[b][t] * W[c][t] for t in range(r)]
                        rows.append(row)
                        T_val = 1 if (a, b, c) in nz_set else 0
                        rhs.append(T_val)
                
                # Convert to numpy for least squares
                A_np = np.array(rows, dtype=float)
                b_np = np.array(rhs, dtype=float)
                
                if A_np.shape[0] < r:
                    # Underdetermined, use pinv
                    sol = np.linalg.pinv(A_np) @ b_np
                else:
                    sol, residuals, rank, sv = np.linalg.lstsq(A_np, b_np, rcond=None)
                
                # Check residual
                residual = np.linalg.norm(A_np @ sol - b_np)
                if residual > 1e-6:
                    # Bad solution, try next
                    break
                
                # Convert solution to Fractions (round to nearest simple fraction)
                for t in range(r):
                    val = sol[t]
                    # Try to find a simple rational approximation
                    best_frac = Fraction(0)
                    for den in [1, 2, 3, 4, 5, 6, 8, 10, 12]:
                        num = round(val * den)
                        if abs(val - num/den) < 1e-9:
                            best_frac = Fraction(num, den)
                            break
                    U_new[a][t] = best_frac
            
            if time.time() > deadline - 5:
                return None
            
            # Now given U_new and W, solve for V similarly
            V_new = [[Fraction(0)] * r for _ in range(m*p)]
            for b in range(m*p):
                rows = []
                rhs = []
                for a in range(n*m):
                    for c in range(p*n):
                        row = [U_new[a][t] * W[c][t] for t in range(r)]
                        rows.append(row)
                        T_val = 1 if (a, b, c) in nz_set else 0
                        rhs.append(T_val)
                
                A_np = np.array(rows, dtype=float)
                b_np = np.array(rhs, dtype=float)
                
                if A_np.shape[0] < r:
                    sol = np.linalg.pinv(A_np) @ b_np
                else:
                    sol, residuals, rank, sv = np.linalg.lstsq(A_np, b_np, rcond=None)
                
                residual = np.linalg.norm(A_np @ sol - b_np)
                if residual > 1e-6:
                    break
                
                for t in range(r):
                    val = sol[t]
                    best_frac = Fraction(0)
                    for den in [1, 2, 3, 4, 5, 6, 8, 10, 12]:
                        num = round(val * den)
                        if abs(val - num/den) < 1e-9:
                            best_frac = Fraction(num, den)
                            break
                    V_new[b][t] = best_frac
            
            U = U_new
            V = V_new
            
            # Check convergence
            converged = True
            for a in range(n*m):
                for t in range(r):
                    if U[a][t] != U_new[a][t]:
                        converged = False
                        break
                if not converged:
                    break
            
            if converged:
                break
        
        # Final verification
        if verify(U, V, W):
            return {"U": U, "V": V, "W": W}
        return None

    # Try to find a decomposition with rank r using alternating solve
    def try_rank(r):
        if time.time() > deadline - 5:
            return None
        
        # Try several structured W initializations
        W_patterns = []
        
        # Pattern 1: Identity-like
        W1 = [[Fraction(0)]*r for _ in range(p*n)]
        for t in range(min(r, p*n)):
            W1[t][t] = 1
        W_patterns.append(W1)
        
        # Pattern 2: Cyclic
        W2 = [[Fraction(0)]*r for _ in range(p*n)]
        for t in range(r):
            c = t % (p*n)
            W2[c][t] = 1
        W_patterns.append(W2)
        
        # Pattern 3: Random sparse
        import random
        random.seed(42)
        W3 = [[Fraction(0)]*r for _ in range(p*n)]
        for t in range(r):
            c = random.randint(0, p*n-1)
            W3[c][t] = Fraction(1)
        W_patterns.append(W3)
        
        # Pattern 4: Structured based on (i,k) pairs
        W4 = [[Fraction(0)]*r for _ in range(p*n)]
        count = 0
        for i in range(n):
            for k in range(p):
                if count < r:
                    W4[k*n+i][count] = 1
                    count += 1
        W_patterns.append(W4)
        
        for W in W_patterns:
            if time.time() > deadline - 5:
                break
            result = solve_given_W(W)
            if result is not None:
                return result
        
        return None

    # Try each candidate rank from lower bound up
    lb = max(n*m, m*p, p*n)
    
    for r in range(lb, n*m*p + 1):
        if time.time() > deadline - 10:
            break
        result = try_rank(r)
        if result is not None:
            if verify(result['U'], result['V'], result['W']):
                if r < best_rank:
                    best = result
                    best_rank = r
                break

    return best
# EVOLVE-BLOCK-END
