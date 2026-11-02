import time
import numpy as np
from itertools import product

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    rank = len(U[0]) if U else 0
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    for r in range(rank):
        for i in range(n):
            if len(U[i * m]) != rank:
                return False
        for j in range(m):
            if len(V[j * p]) != rank:
                return False
        for k in range(p):
            if len(W[k * n]) != rank:
                return False
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(rank):
                    total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if total != 1:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = 3, 5, 7
    
    # Trivial decomposition: rank = n*m*p = 105
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
    
    if not _verify_decomposition(U, V, W, n, m, p):
        raise ValueError("Trivial decomposition failed verification")
    
    best_U, best_V, best_W = U, V, W
    best_rank = rank
    
    # Try to find lower rank using structured Kronecker product approach
    # The 3x5x7 multiplication tensor can be viewed as having structure where
    # the 3 and 7 dimensions are "output" indices and 5 is "intermediate"
    # Try rank = 3*7 = 21 by exploiting that each (i,k) pair is independent
    # and using a rank-1 decomposition per (i,k) pair with shared intermediate structure
    
    # Approach: For each (i,k), we need sum_j A[i,j]*B[j,k] = C[i,k]
    # This is a rank-1 bilinear form in the j dimension for fixed (i,k)
    # But we can share terms across (i,k) pairs
    
    # Try greedy: find a set of terms that cover all (i,j,k) triples
    # Each term r has U[i*m+j][r], V[j*p+k][r], W[k*n+i][r]
    # We need: for each (i,j,k), sum_r U[i*m+j][r]*V[j*p+k][r]*W[k*n+i][r] = 1
    
    # Use a structured approach: assume W[k*n+i][r] is nonzero only for specific (k,i) pairs
    # and try to minimize the number of terms
    
    # Try rank values from low to high
    for target_rank in range(15, 106):
        if time.time() > deadline - 10:
            break
        
        # Initialize with zeros
        U_t = [[0] * target_rank for _ in range(n * m)]
        V_t = [[0] * target_rank for _ in range(m * p)]
        W_t = [[0] * target_rank for _ in range(p * n)]
        
        # Greedy: for each (i,j,k) not yet covered, add a term
        # Track which (i,j,k) are satisfied
        covered = [[False] * p for _ in range(n)]  # covered[i][k] = all j done? No, we need per (i,j,k)
        
        # Actually, let's track residuals
        # residual[i][j][k] = 1 - sum_{r so far} U[i*m+j][r]*V[j*p+k][r]*W[k*n+i][r]
        residual = [[[0] * p for _ in range(m)] for _ in range(n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    residual[i][j][k] = 1
        
        for r in range(target_rank):
            if time.time() > deadline - 10:
                break
            
            # Find the (i,j,k) with largest residual
            max_res = -1
            best_ijk = None
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if abs(residual[i][j][k]) > max_res:
                            max_res = abs(residual[i][j][k])
                            best_ijk = (i, j, k)
            
            if best_ijk is None or max_res < 1e-10:
                break
            
            i, j, k = best_ijk
            # Add a term that contributes to this (i,j,k)
            U_t[i * m + j][r] = 1
            V_t[j * p + k][r] = 1
            W_t[k * n + i][r] = 1
            
            # Update residuals
            for ii in range(n):
                for jj in range(m):
                    for kk in range(p):
                        residual[ii][jj][kk] -= U_t[ii * m + jj][r] * V_t[jj * p + kk][r] * W_t[kk * n + ii][r]
        
        # Check if all residuals are zero
        all_zero = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if abs(residual[i][j][k]) > 1e-10:
                        all_zero = False
                        break
                if not all_zero:
                    break
            if not all_zero:
                break
        
        if all_zero:
            # Convert to integers
            U_int = [[round(U_t[i][r]) for r in range(target_rank)] for i in range(n * m)]
            V_int = [[round(V_t[i][r]) for r in range(target_rank)] for i in range(m * p)]
            W_int = [[round(W_t[i][r]) for r in range(target_rank)] for i in range(p * n)]
            
            if _verify_decomposition(U_int, V_int, W_int, n, m, p):
                best_U, best_V, best_W = U_int, V_int, W_int
                best_rank = target_rank
                break
    
    return {"U": best_U, "V": best_V, "W": best_W}
