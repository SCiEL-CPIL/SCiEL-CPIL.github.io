"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import itertools
from fractions import Fraction

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    total = n * m * p
    
    # Build the tensor T[i,j,k] = 1 for all i,j,k
    # We want: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Start with schoolbook rank
    best_rank = n * m * p
    best_result = None
    
    # Schoolbook
    U_school = np.zeros((nm, best_rank))
    V_school = np.zeros((mp, best_rank))
    W_school = np.zeros((pn, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    def to_frac_matrix(mat):
        """Convert numpy matrix to list of lists of Fractions"""
        result = []
        for row in mat:
            result.append([Fraction(int(round(x*2)), 2) if abs(x*2 - round(x*2)) < 1e-9 else Fraction(int(round(x)), 1) for x in row])
        return result
    
    def verify(U, V, W):
        """Verify that the decomposition is exact"""
        n, m, p = DIMENSIONS
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(len(U[0])):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    # Try to find lower rank decompositions
    # Use iterative refinement: fix two factors, solve for third
    
    def try_rank(r, max_iters=50):
        """Try to find a decomposition of rank r"""
        # Initialize with random values
        np.random.seed(42)
        
        for attempt in range(3):
            U = np.random.randn(nm, r)
            V = np.random.randn(mp, r)
            W = np.random.randn(pn, r)
            
            for it in range(max_iters):
                # Fix U and V, solve for W
                # For each (k,i): sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all j
                # This is a system: for each (k,i), we have m equations in r unknowns (W[k*n+i, :])
                # Actually, let's fix U and W, solve for V, or use alternating
                
                # Fix U, V -> solve for W
                # For each (k, i): W[k*n+i, :] needs to satisfy:
                # sum_c V[j*p+k, c] * W[k*n+i, c] * U[i*m+j, c] = 1 for all j
                # Let A_{j,c} = V[j*p+k, c] * U[i*m+j, c]
                # Then sum_c A_{j,c} * W[k*n+i, c] = 1 for all j
                # This is A @ W[k*n+i, :] = ones(m)
                
                W_new = np.zeros((pn, r))
                for k in range(p):
                    for i in range(n):
                        # Build A: m x r
                        A = np.zeros((m, r))
                        for j in range(m):
                            for c in range(r):
                                A[j, c] = V[j*p+k, c] * U[i*m+j, c]
                        # Solve A @ w = ones(m)
                        try:
                            w, _, _, _ = np.linalg.lstsq(A, np.ones(m), rcond=None)
                            W_new[k*n+i, :] = w
                        except:
                            pass
                W = W_new
                
                # Fix V, W -> solve for U
                U_new = np.zeros((nm, r))
                for i in range(n):
                    for j in range(m):
                        # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all k
                        # Let B_{k,c} = V[j*p+k, c] * W[k*n+i, c]
                        # B @ u = ones(p)
                        B = np.zeros((p, r))
                        for k in range(p):
                            for c in range(r):
                                B[k, c] = V[j*p+k, c] * W[k*n+i, c]
                        try:
                            u, _, _, _ = np.linalg.lstsq(B, np.ones(p), rcond=None)
                            U_new[i*m+j, :] = u
                        except:
                            pass
                U = U_new
                
                # Fix U, W -> solve for V
                V_new = np.zeros((mp, r))
                for j in range(m):
                    for k in range(p):
                        # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i
                        # Let C_{i,c} = U[i*m+j, c] * W[k*n+i, c]
                        # C @ v = ones(n)
                        C = np.zeros((n, r))
                        for i in range(n):
                            for c in range(r):
                                C[i, c] = U[i*m+j, c] * W[k*n+i, c]
                        try:
                            v, _, _, _ = np.linalg.lstsq(C, np.ones(n), rcond=None)
                            V_new[j*p+k, :] = v
                        except:
                            pass
                V = V_new
                
                # Check convergence
                err = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0
                            for c in range(r):
                                s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                            err += abs(s - 1)
                if err < 1e-6:
                    # Try to snap to rational values
                    U_frac = [[Fraction(int(round(x*4)), 4) for x in row] for row in U]
                    V_frac = [[Fraction(int(round(x*4)), 4) for x in row] for row in V]
                    W_frac = [[Fraction(int(round(x*4)), 4) for x in row] for row in W]
                    if verify(U_frac, V_frac, W_frac):
                        return (U_frac, V_frac, W_frac)
                    # Try with 2x precision
                    U_frac = [[Fraction(int(round(x*2)), 2) for x in row] for row in U]
                    V_frac = [[Fraction(int(round(x*2)), 2) for x in row] for row in V]
                    W_frac = [[Fraction(int(round(x*2)), 2) for x in row] for row in W]
                    if verify(U_frac, V_frac, W_frac):
                        return (U_frac, V_frac, W_frac)
                    # Try integer
                    U_frac = [[Fraction(int(round(x)), 1) for x in row] for row in U]
                    V_frac = [[Fraction(int(round(x)), 1) for x in row] for row in V]
                    W_frac = [[Fraction(int(round(x)), 1) for x in row] for row in W]
                    if verify(U_frac, V_frac, W_frac):
                        return (U_frac, V_frac, W_frac)
                    # Return as is if close enough
                    return (U_frac, V_frac, W_frac)
        
        return None
    
    # Try ranks from 60 down
    start_time = time.time()
    for r in range(best_rank, 30, -1):
        if time.time() - start_time > 100:
            break
        result = try_rank(r)
        if result is not None:
            U_frac, V_frac, W_frac = result
            if verify(U_frac, V_frac, W_frac):
                best_rank = r
                best_result = {"U": U_frac, "V": V_frac, "W": W_frac}
                break
    
    if best_result is None:
        # Return schoolbook
        U_frac = [[Fraction(int(round(x)), 1) for x in row] for row in U_school]
        V_frac = [[Fraction(int(round(x)), 1) for x in row] for row in V_school]
        W_frac = [[Fraction(int(round(x)), 1) for x in row] for row in W_school]
        best_result = {"U": U_frac, "V": V_frac, "W": W_frac}
    
    return best_result
# EVOLVE-BLOCK-END
