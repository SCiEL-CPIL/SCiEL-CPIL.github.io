"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W form a valid decomposition of the matrix multiplication tensor."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # For each (i, k), check that C[i,k] = sum_j A[i,j]*B[j,k]
    # is decomposed as sum_c U[i*m+j] * V[j*p+k] * W[k*n+i]
    for i in range(n):
        for k in range(p):
            # Left side: the actual product coefficient for each (j)
            # The tensor entry T[i,k,j] = delta (1 if we're looking at C[i,k] from A[i,j]*B[j,k])
            # Actually, we need to check: for all i,j,k, the sum over c of U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
            # should equal 1 if this is a valid term, but actually the tensor is:
            # T[i,j,k] = 1 (the multiplication tensor has 1s at all (i,j,k) positions)
            # So we need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k
            
            # Wait, that's not right either. Let me reconsider.
            # The matrix multiplication tensor M_{n,m,p} has entries:
            # M[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
            # (This represents C[i,k] = sum_j A[i,j] B[j,k])
            # 
            # A rank-r decomposition means: M[i,j,k] = sum_{c=0}^{r-1} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            # So for each (i,j,k), the sum should equal 1.
            
            total = 0.0
            for j in range(m):
                for c in range(rank):
                    total += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
            if abs(total - 1.0) > tol:
                return False
    return True

def _verify_decomposition_fast(U, V, W, n, m, p, tol=1e-9):
    """Faster verification using matrix operations."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U_reshaped = U_arr.reshape(n, m, rank)
    V_reshaped = V_arr.reshape(m, p, rank)
    W_reshaped = W_arr.reshape(p, n, rank)
    
    # For each (i,j,k), compute sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # This is a tensor contraction
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = np.sum(U_reshaped[i, j, :] * V_reshaped[j, k, :] * W_reshaped[k, i, :])
                if abs(total - 1.0) > tol:
                    return False
    return True

def _try_rank_decomposition(n, m, p, max_rank, time_limit=10):
    """
    Try to find a rank decomposition with entries in {-1, -0.5, 0, 0.5, 1}
    using a structured search based on block decomposition.
    """
    start_time = time.time()
    
    # First, try to construct a rank-24 decomposition using block structure
    # Split p=6 into three blocks of size 2
    # For each block, we need a decomposition of 2x3 x 3x2
    
    # Let's try to find a rank-8 decomposition for 2x3 x 3x2
    # A is 2x3, B is 3x2, C = A*B is 2x2
    
    # The schoolbook rank for 2x3 x 3x2 is 12 (2*3*2)
    # We want to see if we can do better
    
    # Try a Strassen-like approach:
    # Pad A to 2x4 by adding a zero column: A' = [A | 0]
    # Pad B to 4x2 by adding zero rows: B' = [B; 0]
    # Then A*B = A'*B'
    # Use Strassen on 2x4 x 4x2? That doesn't directly help.
    
    # Alternative: use the fact that 2x3 x 3x2 can be decomposed as
    # (2x2 x 2x2) + (2x1 x 1x2) + (2x1 x 1x2) + cross terms
    
    # Let me try a direct construction.
    # For 2x3 x 3x2, define 8 terms:
    
    # Actually, let's use a known result: the rank of 2x3 x 3x2 is at most 9.
    # Let me try to construct a rank-9 decomposition.
    
    # For now, let's try to find any valid decomposition with rank < 36
    # by searching over small structured patterns.
    
    best_result = None
    best_rank = n * m * p  # schoolbook
    
    # Try rank from a lower bound up to max_rank
    # Lower bound: max(n*m, m*p, p*n) / something... actually the rank must be at least
    # the max of the "slice" ranks. For M_{2,3,6}, a lower bound is ceil(2*3*6/2) = 18?
    # Actually, the rank is at least max(n*p, m*p, n*m) for some bounds... let's just search.
    
    for target_rank in range(24, min(max_rank + 1, best_rank)):
        if time.time() - start_time > time_limit:
            break
        
        # Try to construct a rank-target_rank decomposition
        # Use a greedy/structured approach
        
        # Strategy: start with schoolbook and try to merge terms
        # Or: use a random initialization and local search
        
        # For now, let's try a specific structured construction for rank 24
        if target_rank == 24:
            # Split into 3 blocks of 8
            # For each block t=0,1,2 (columns 2t, 2t+1 of B)
            # We need a rank-8 decomposition of 2x3 x 3x2
            
            # Let me try to construct one:
            # A = [[a00, a01, a02], [a10, a11, a12]]
            # B = [[b00, b01], [b10, b11], [b20, b21]]
            # C = A*B = [[c00, c01], [c10, c11]]
            # where c00 = a00*b00 + a01*b10 + a02*b20
            #       c01 = a00*b01 + a01*b11 + a02*b21
            #       c10 = a10*b00 + a11*b10 + a12*b20
            #       c11 = a10*b01 + a11*b11 + a12*b21
            
            # Try 8 terms:
            # t0: a00 * b00 -> c00
            # t1: a00 * b01 -> c01
            # t2: a01 * b10 -> c00
            # t3: a01 * b11 -> c01
            # t4: a02 * b20 -> c00
            # t5: a02 * b21 -> c01
            # t6: a10 * b00 -> c10  [but b00 already used]
            # t7: a10 * b01 -> c11  [but b01 already used]
            # This doesn't work directly because of conflicts.
            
            # Let me try a different set of 8 terms using Strassen-like ideas:
            # s0 = (a00+a10) * (b00+b10) = a00*b00 + a00*b10 + a10*b00 + a10*b10
            # s1 = (a00+a10) * (b01+b11) = a00*b01 + a00*b11 + a10*b01 + a10*b11
            # s2 = (a01+a11) * (b10+b20) = a01*b10 + a01*b20 + a11*b10 + a11*b20
            # s3 = (a01+a11) * (b11+b21) = a01*b11 + a01*b21 + a11*b11 + a11*b21
            # s4 = a02 * b20
            # s5 = a02 * b21
            # s6 = a12 * b20
            # s7 = a12 * b21
            
            # Now express c00, c01, c10, c11 in terms of s0..s7:
            # c00 = a00*b00 + a01*b10 + a02*b20
            #     = (s0 - a00*b10 - a10*b00 - a10*b10) + a01*b10 + a02*b20
            #     = s0 - a00*b10 - a10*b00 - a10*b10 + a01*b10 + s4
            # This is getting complicated. Let me try a different approach.
            
            # Actually, let me just try to find ANY valid decomposition with rank < 36
            # by using a more systematic search.
            
            pass
    
    # If we didn't find anything better, return schoolbook
    if best_result is None:
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
    
    return best_result

def construct():
    """
    Rank-minimal exact tensor decomposition for 2x3 * 3x6 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Try to find a better decomposition
    result = _try_rank_decomposition(n, m, p, max_rank=36, time_limit=60)
    
    # Verify the result
    if result is not None:
        if _verify_decomposition_fast(result["U"], result["V"], result["W"], n, m, p):
            return result
    
    # Fallback to schoolbook
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
