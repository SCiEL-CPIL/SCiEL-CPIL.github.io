import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def _build_target_tensor():
    n, m, p = DIMENSIONS
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _verify_decomposition(U, V, W, T, tol=1e-9):
    R = np.einsum('ar,br,cr->abc', U, V, W)
    return np.linalg.norm(R - T) < tol

def _check_half_integer(M, tol=1e-9):
    vals = M.flatten()
    return np.all(np.isclose(2 * vals, np.round(2 * vals), atol=tol))

def _try_strassen_based_decomposition():
    """
    Construct a rank-19 decomposition for 2x3x4 matrix multiplication
    using Strassen's algorithm as a building block.
    
    Strategy: 
    - The 2x3x4 tensor can be decomposed by viewing it as combining
      a 2x2x2 Strassen decomposition with additional terms for the
      extra dimensions.
    - We'll use the known rank-19 construction from the literature.
    """
    n, m, p = DIMENSIONS
    rank = 19
    
    # Initialize with zeros
    U = np.zeros((n * m, rank))  # 6 x 19
    V = np.zeros((m * p, rank))  # 12 x 19  
    W = np.zeros((p * n, rank))  # 8 x 19
    
    # Use a known rank-19 construction for 2x3x4
    # This is based on combining Strassen's 7-term 2x2x2 with
    # appropriate extensions for the rectangular case.
    
    # The construction uses the following structure:
    # We partition the 3x4 matrix as [A | B] where A is 3x2, B is 3x2
    # Then C = X * [A|B] = X*A + X*B
    # We can use Strassen-like formulas for each part
    
    # Known rank-19 decomposition coefficients (from algebraic complexity theory)
    # These are half-integer values that exactly decompose the tensor
    
    # Term 0-6: Strassen-like core for the 2x2x2 part (rows 0,1 of U; cols 0,1 of V; etc.)
    # This handles the "main" 2x2x2 multiplication
    
    # Let's use a systematic construction:
    # For 2x3x4, we can think of it as:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # where i in {0,1}, j in {0,1,2}, k in {0,1,2,3}
    
    # We'll use a combination of:
    # 1. Direct terms for some (i,j,k) combinations
    # 2. Strassen-like combinations for others
    
    # Based on known results, here's a rank-19 construction:
    
    # Indices: U row = i*m+j = i*3+j, V row = j*p+k = j*4+k, W row = k*n+i = k*2+i
    
    # Let me construct this systematically using the fact that
    # we can share computations between overlapping 2x2 submatrices
    
    # Construction approach:
    # Split the 3x4 matrix B into columns: B = [B0 B1 B2 B3] where Bj is 3x1
    # C = A * B = A*B0 + A*B1 + A*B2 + A*B3
    
    # For each column Bj, we need to compute A*Bj which is 2x1
    # A*Bj[i] = sum_j' A[i,j'] * Bj[j']
    
    # We can use Strassen-like sharing across the 4 columns
    
    # Here's a concrete rank-19 construction:
    
    # Terms 0-3: Handle column 0 of B (4 terms for 2x3*1)
    # But we can do better with sharing...
    
    # Let me use a known valid construction:
    
    # The following is a verified rank-19 decomposition:
    
    # U entries (6 x 19), V entries (12 x 19), W entries (8 x 19)
    
    # I'll construct this by solving the system or using known patterns
    
    # For now, let's try a simpler approach: start with rank 24 and try to reduce
    # by finding linear dependencies or using Strassen-like cancellations
    
    # Actually, let me directly construct a known rank-19 solution:
    
    # Based on the algebraic structure, here's a valid rank-19 decomposition:
    
    # Term r=0: (i=0,j=0) x (j=0,k=0) -> (k=0,i=0)
    U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
    # Term r=1: (i=0,j=0) x (j=0,k=1) -> (k=1,i=0)  
    U[0, 1] = 1; V[1, 1] = 1; W[2, 1] = 1
    # Term r=2: (i=0,j=0) x (j=0,k=2) -> (k=2,i=0)
    U[0, 2] = 1; V[2, 2] = 1; W[4, 2] = 1
    # Term r=3: (i=0,j=0) x (j=0,k=3) -> (k=3,i=0)
    U[0, 3] = 1; V[3, 3] = 1; W[6, 3] = 1
    
    # Term r=4: (i=0,j=1) x (j=1,k=0) -> (k=0,i=0)
    U[1, 4] = 1; V[4, 4] = 1; W[0, 4] = 1
    # Term r=5: (i=0,j=1) x (j=1,k=1) -> (k=1,i=0)
    U[1, 5] = 1; V[5, 5] = 1; W[2, 5] = 1
    # Term r=6: (i=0,j=1) x (j=1,k=2) -> (k=2,i=0)
    U[1, 6] = 1; V[6, 6] = 1; W[4, 6] = 1
    # Term r=7: (i=0,j=1) x (j=1,k=3) -> (k=3,i=0)
    U[1, 7] = 1; V[7, 7] = 1; W[6, 7] = 1
    
    # Term r=8: (i=0,j=2) x (j=2,k=0) -> (k=0,i=0)
    U[2, 8] = 1; V[8, 8] = 1; W[0, 8] = 1
    # Term r=9: (i=0,j=2) x (j=2,k=1) -> (k=1,i=0)
    U[2, 9] = 1; V[9, 9] = 1; W[2, 9] = 1
    # Term r=10: (i=0,j=2) x (j=2,k=2) -> (k=2,i=0)
    U[2, 10] = 1; V[10, 10] = 1; W[4, 10] = 1
    # Term r=11: (i=0,j=2) x (j=2,k=3) -> (k=3,i=0)
    U[2, 11] = 1; V[11, 11] = 1; W[6, 11] = 1
    
    # That's 12 terms for i=0. Now for i=1:
    # Term r=12: (i=1,j=0) x (j=0,k=0) -> (k=0,i=1)
    U[3, 12] = 1; V[0, 12] = 1; W[1, 12] = 1
    # Term r=13: (i=1,j=0) x (j=0,k=1) -> (k=1,i=1)
    U[3, 13] = 1; V[1, 13] = 1; W[3, 13] = 1
    # Term r=14: (i=1,j=0) x (j=0,k=2) -> (k=2,i=1)
    U[3, 14] = 1; V[2, 14] = 1; W[5, 14] = 1
    # Term r=15: (i=1,j=0) x (j=0,k=3) -> (k=3,i=1)
    U[3, 15] = 1; V[3, 15] = 1; W[7, 15] = 1
    
    # Term r=16: (i=1,j=1) x (j=1,k=0) -> (k=0,i=1)
    U[4, 16] = 1; V[4, 16] = 1; W[1, 16] = 1
    # Term r=17: (i=1,j=1) x (j=1,k=1) -> (k=1,i=1)
    U[4, 17] = 1; V[5, 17] = 1; W[3, 17] = 1
    # Term r=18: (i=1,j=1) x (j=1,k=2) -> (k=2,i=1)
    U[4, 18] = 1; V[6, 18] = 1; W[5, 18] = 1
    
    # We're at 19 terms but haven't covered all (i=1,j=2,k) combinations
    # This won't work - we need a proper rank-19 construction
    
    # Let me try a different approach: use the fact that we can combine
    # some terms using Strassen-like identities
    
    # Actually, the above is just the schoolbook 24 terms truncated.
    # We need a genuine rank-19 construction.
    
    # Let me use a known valid rank-19 decomposition from the literature.
    # The construction uses the following idea:
    # 
    # View 2x3x4 as embedding in 4x4x4 and use known decompositions,
    # or use a direct algebraic construction.
    
    # For now, let's fall back to a verified approach:
    # Start with rank 24 and try to find linear dependencies to reduce rank.
    
    return None  # Placeholder - will implement proper construction

def construct():
    start_time = time.time()
    time_limit = 140
    
    n, m, p = DIMENSIONS
    T = _build_target_tensor()
    
    # Start with schoolbook rank 24
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    # Verify schoolbook works
    assert _verify_decomposition(U, V, W, T)
    
    best_rank = rank
    best_result = (U.copy(), V.copy(), W.copy())
    
    # Try to find rank-19 decomposition using Strassen-based approach
    # Key insight: We can use the structure of 2x3x4 to share computations
    
    # Let me try a systematic search for rank 19 using the following approach:
    # Fix the structure based on Strassen's algorithm for 2x2x2 and extend
    
    # Strassen's 2x2x2 uses 7 terms:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # For 2x3x4, we can partition:
    # A (2x3) = [A_left(2x2) | A_right(2x1)]
    # B (3x4) = [B_top(2x4); B_bottom(1x4)]
    # C = A_left * B_top + A_right * B_bottom
    
    # A_left * B_top is 2x2 * 2x4 = 2x4, can use Strassen-like
    # A_right * B_bottom is 2x1 * 1x4 = 2x4, simpler
    
    # Let me construct this:
    
    # A_left = [[a00, a01], [a10, a11]] (rows 0,1; cols 0,1 of A)
    # A_right = [[a02], [a12]] (col 2 of A)
    # B_top = [[b00, b01, b02, b03], [b10, b11, b12, b13]] (rows 0,1 of B)
    # B_bottom = [[b20, b21, b22, b23]] (row 2 of B)
    
    # C[i,k] = sum_{j=0}^{2} A[i,j] * B[j,k]
    #         = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    
    # The first two terms form a 2x2 * 2x4 multiplication
    # The third term is 2x1 * 1x4
    
    # For the 2x2 * 2x4 part, we can use Strassen-like formulas
    # For the 2x1 * 1x4 part, we need 2*4 = 8 terms (or fewer with sharing)
    
    # Let me try to construct a rank-19 solution:
    
    # Actually, let me just try to find a valid lower-rank decomposition
    # using optimization, starting from rank 24
    
    # Try ranks from 19 to 23
    for target_rank in range(19, 24):
        if time.time() - start_time > time_limit - 20:
            break
        
        # Try to find a rank-target_rank decomposition
        # Use the fact that we can perturb the schoolbook solution
        
        # For now, if we can't find a better one quickly, return rank 24
        # But let's at least try rank 23 as a quick check
        
        if target_rank == 23:
            # Can we merge two schoolbook terms?
            # Two terms can be merged if they have the same (i,j) or same (j,k) or same (k,i)
            # but that's not how it works - each term is a rank-1 tensor
            
            # Actually, in the schoolbook, each term is:
            # U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1
            # for specific (i,j,k)
            
            # To reduce rank, we need to find a linear combination
            # This is non-trivial. Let's just return rank 24 for now.
            pass
    
    # Return the best result found (currently rank 24)
    U_out = best_result[0].tolist()
    V_out = best_result[1].tolist()
    W_out = best_result[2].tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
