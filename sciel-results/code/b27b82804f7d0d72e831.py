"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 3
    deadline = time.time() + 130  # safety margin before 150s limit
    
    # The multiplication tensor: C[i,k] = sum_j A[i,j] B[j,k]
    # T[i,j,k] = 1 for all i,j,k
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Try rank 16 first (known to be achievable for 2x3x3)
    # Use numerical optimization to find a solution, then round to rationals
    
    def try_rank(r, max_time=120):
        start_time = time.time()
        random.seed(42)
        
        # Initialize with random values
        U = np.random.randn(n * m, r)
        V = np.random.randn(m * p, r)
        W = np.random.randn(p * n, r)
        
        # Scale to help convergence
        U /= np.sqrt(r)
        V /= np.sqrt(r)
        W /= np.sqrt(r)
        
        best_err = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        # Gradient-free optimization (Nelder-Mead style on flattened params)
        # But that's too slow for 21r variables. Use coordinate descent or random restarts.
        
        # Alternative: use the structure. For each (i,j,k), we need:
        # sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1
        # This is like a 3-way tensor decomposition.
        
        # Let's try a different approach: fix some entries and solve linearly.
        # For rank r, we have 21r unknowns and 18 equations (one per (i,j,k)).
        # For r >= 7, we have more unknowns than equations.
        
        # Strategy: Use iterative refinement. Start with a random init, then
        # alternately update each factor while keeping others fixed.
        
        for iteration in range(50000):
            if time.time() - start_time > max_time:
                break
            
            # Update U: for each (i,j), solve for U[i,j,:] to minimize error
            # The contribution to equation (i',j',k') from U[i,j,:] is:
            # U[i,j,c] * V[j,k',c] * W[k',i',c] for i'=i
            # So for fixed i,j, the equations involving U[i,j,:] are those with i'=i:
            # sum_k' sum_c U[i,j,c] * V[j,k',c] * W[k',i,c] = 1 for all k'
            # Wait, the equation is for (i,j,k): sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1
            # For fixed (i,j), varying k gives 3 equations:
            # sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1 for k=0,1,2
            
            # This is linear in U[i,j,:] if we define for each k:
            # a_k[c] = V[j,k,c] * W[k,i,c]
            # Then sum_c U[i,j,c] * a_k[c] = 1 for k=0,1,2
            # So we need U[i,j,:] to satisfy: A @ U[i,j,:] = ones(3)
            # where A[k,c] = V[j,k,c] * W[k,i,c]
            
            # This is an underdetermined system (r unknowns, 3 equations for r>=3)
            # Least squares solution
            
            for i in range(n):
                for j in range(m):
                    # Build the 3 x r matrix
                    A_mat = np.zeros((p, r))
                    for k in range(p):
                        A_mat[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    
                    # Solve A_mat @ U[i*m+j, :] = ones(p) in least squares sense
                    # But we want exact, so use pseudo-inverse
                    try:
                        U_new = np.linalg.lstsq(A_mat, np.ones(p), rcond=None)[0]
                        U[i * m + j, :] = U_new
                    except:
                        pass
            
            # Update V: for each (j,k), equations with j fixed, varying i:
            # sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1 for i=0,1
            # For fixed (j,k): a_i[c] = U[i,j,c] * W[k,i,c]
            # sum_c V[j,k,c] * a_i[c] = 1 for i=0,1
            
            for j in range(m):
                for k in range(p):
                    A_mat = np.zeros((n, r))
                    for i in range(n):
                        A_mat[i, :] = U[i * m + j, :] * W[k * n + i, :]
                    try:
                        V_new = np.linalg.lstsq(A_mat, np.ones(n), rcond=None)[0]
                        V[j * p + k, :] = V_new
                    except:
                        pass
            
            # Update W: for each (k,i), equations with i fixed, varying j:
            # sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1 for j=0,1,2
            # For fixed (k,i): a_j[c] = U[i,j,c] * V[j,k,c]
            # sum_c W[k,i,c] * a_j[c] = 1 for j=0,1,2
            
            for k in range(p):
                for i in range(n):
                    A_mat = np.zeros((m, r))
                    for j in range(m):
                        A_mat[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    try:
                        W_new = np.linalg.lstsq(A_mat, np.ones(m), rcond=None)[0]
                        W[k * n + i, :] = W_new
                    except:
                        pass
            
            # Check error
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                        err += (val - 1.0) ** 2
            err = np.sqrt(err)
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if err < 1e-10:
                break
        
        if best_err < 1e-8:
            # Try to find rational approximation
            U_rat = _find_rational(best_U, max_den=100)
            V_rat = _find_rational(best_V, max_den=100)
            W_rat = _find_rational(best_W, max_den=100)
            
            # Verify
            if U_rat is not None and V_rat is not None and W_rat is not None:
                if _verify(U_rat, V_rat, W_rat, n, m, p):
                    return U_rat, V_rat, W_rat, r
        
        return None
    
    def _find_rational(mat, max_den=100):
        """Try to find rational approximation with small denominators."""
        result = []
        for row in mat:
            r = []
            for val in row:
                if abs(val) < 1e-12:
                    r.append(Fraction(0, 1))
                    continue
                # Try small denominators
                found = False
                for den in range(1, max_den + 1):
                    num = round(val * den)
                    if abs(val - num / den) < 1e-10:
                        r.append(Fraction(num, den))
                        found = True
                        break
                if not found:
                    return None
            result.append(r)
        return result
    
    def _verify(U, V, W, n, m, p):
        """Verify exact decomposition using Fraction arithmetic."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = Fraction(0, 1)
                    for c in range(len(U[0])):
                        total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if total != Fraction(1, 1):
                        return False
        return True
    
    # Try ranks from 16 upward (16 is the known lower bound for 2x3x3)
    for r in [16, 17, 18]:
        if time.time() > deadline - 10:
            break
        result = try_rank(r, max_time=min(60, deadline - time.time() - 5))
        if result is not None:
            U, V, W, rank = result
            # Convert to lists of Fractions
            U_list = [[float(x) if x.denominator == 1 else float(x) for x in row] for row in U]
            # Actually, keep as Fractions for exactness
            return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook rank 18
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
