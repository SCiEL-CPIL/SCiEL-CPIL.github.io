"""Improved program: block-Strassen for 4x4 * 4x6 matrix multiplication."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify tensor decomposition: T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    where T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # Check all (i,j,k) triples: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    # Also check that for invalid index combinations, the sum is 0
    # The tensor T[a,b,c] is nonzero only when a=i*m+j, b=j'*p+k', c=k''*n+i''
    # with j=j' and i=i'' (i.e., the middle index of first matches first index of second,
    # and last index of second matches first index of third in the right way)
    # Actually, let's just check a subset of invalid entries to be safe
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                # Extract indices
                i1, j1 = divmod(a, m)
                j2, k1 = divmod(b, p)
                k2, i2 = divmod(c, n)
                # Valid if j1 == j2 and i1 == i2
                if j1 != j2 or i1 != i2:
                    s = 0.0
                    for r in range(rank):
                        s += U[a, r] * V[b, r] * W[c, r]
                    if abs(s) > tol:
                        return False
    return True

def _strassen_2x2():
    """Return the 7-term Strassen decomposition for 2x2 matrix multiplication.
    Returns U (4x7), V (4x7), W (4x7) such that
    T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r] for the 2x2 mult tensor.
    Convention: a = i*2+j, b = j*2+k, c = k*2+i for 2x2 matrices."""
    # Strassen's 7 products:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # U maps input indices (i*2+j) to coefficients for each M
    # V maps input indices (j*2+k) to coefficients for each M
    # W maps output indices (k*2+i) to coefficients for each M
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A00+A11)*(B00+B11)
    U[0,0] = 1  # A00
    U[3,0] = 1  # A11
    V[0,0] = 1  # B00
    V[3,0] = 1  # B11
    W[0,0] = 1  # C00
    W[3,0] = 1  # C11
    
    # M2 = (A10+A11)*B00
    U[2,1] = 1  # A10
    U[3,1] = 1  # A11
    V[0,1] = 1  # B00
    W[2,1] = 1  # C10
    W[3,1] = -1  # C11
    
    # M3 = A00*(B01-B11)
    U[0,2] = 1  # A00
    V[1,2] = 1  # B01
    V[3,2] = -1  # B11
    W[1,2] = 1  # C01
    W[3,2] = 1  # C11
    
    # M4 = A11*(B10-B00)
    U[3,3] = 1  # A11
    V[2,3] = 1  # B10
    V[0,3] = -1  # B00
    W[0,3] = 1  # C00
    W[2,3] = 1  # C10
    
    # M5 = (A00+A01)*B11
    U[0,4] = 1  # A00
    U[1,4] = 1  # A01
    V[3,4] = 1  # B11
    W[0,4] = -1  # C00
    W[1,4] = 1  # C01
    
    # M6 = (A10-A00)*(B00+B01)
    U[2,5] = 1  # A10
    U[0,5] = -1  # A00
    V[0,5] = 1  # B00
    V[1,5] = 1  # B01
    W[3,5] = 1  # C11
    
    # M7 = (A01-A11)*(B10+B11)
    U[1,6] = 1  # A01
    U[3,6] = -1  # A11
    V[2,6] = 1  # B10
    V[3,6] = 1  # B11
    W[0,6] = 1  # C00
    W[1,6] = 1  # C01
    
    return U, V, W

def _sum_two_2x2_products():
    """Return decomposition for X*U + Y*V where X,Y,U,V are 2x2.
    This is a tensor of shape (8, 8, 4):
    - First factor: X[0,0],X[0,1],X[1,0],X[1,1],Y[0,0],Y[0,1],Y[1,0],Y[1,1]
    - Second factor: U[0,0],U[0,1],U[1,0],U[1,1],V[0,0],V[0,1],V[1,0],V[1,1]
    - Output: C[0,0],C[0,1],C[1,0],C[1,1]
    
    We use two Strassen decompositions (one for X*U, one for Y*V), giving rank 14.
    But we can potentially do better. For now, use rank 14."""
    # Use two separate Strassen decompositions
    U1, V1, W1 = _strassen_2x2()
    
    U = np.zeros((8, 14))
    V = np.zeros((8, 14))
    W = np.zeros((4, 14))
    
    # First 7 terms: X*U
    U[0:4, 0:7] = U1
    V[0:4, 0:7] = V1
    W[0:4, 0:7] = W1
    
    # Next 7 terms: Y*V
    U[4:8, 7:14] = U1
    V[4:8, 7:14] = V1
    W[0:4, 7:14] = W1
    
    return U, V, W

def _block_strassen_4x4_4x6():
    """Construct block-Strassen decomposition for 4x4 * 4x6.
    
    Split A (4x4) into 2x2 blocks of 2x2: A00,A01,A10,A11
    Split B (4x6) into 2x3 blocks of 2x2: B00,B01,B02,B10,B11,B12
    Output C (4x6) into 2x3 blocks of 2x2: C00,C01,C02,C10,C11,C12
    
    C[i,j] = A[i,0]*B[0,j] + A[i,1]*B[1,j] for i,j in {0,1} x {0,1,2}
    
    For each output block, use the sum-of-two-2x2-products decomposition (rank 14).
    Total rank: 6 * 14 = 84.
    
    Index conventions:
    - A index: a = i*4+j where i is block-row, j is within-block col, but actually
      a = row*4+col for the full 4x4 matrix.
      Block (bi, bj) occupies rows 2*bi to 2*bi+1, cols 2*bj to 2*bj+1.
      So element A[row, col] has block indices (row//2, col//2) and local (row%2, col%2).
      The U matrix has 16 rows (one per A element) and 84 cols.
      
    - B index: b = row*6+col for the full 4x6 matrix.
      Block (bi, bj) occupies rows 2*bi to 2*bi+1, cols 3*bj to 3*bj+1.
      Wait, B is 4x6, so blocks are 2x2. Block (bi, bj) has bi in {0,1}, bj in {0,1,2}.
      Element B[row, col] has block indices (row//2, col//2) and local (row%2, col%2).
      The V matrix has 24 rows and 84 cols.
      
    - C index: c = row*6+col for the full 4x6 output.
      Block (ci, cj) has ci in {0,1}, cj in {0,1,2}.
      Element C[row, col] has block indices (row//2, col//2) and local (row%2, col%2).
      The W matrix has 24 rows and 84 cols.
    """
    n, m, p = 4, 4, 6
    num_blocks_rows = 2
    num_blocks_cols_B = 3
    num_blocks_cols_C = 3
    
    total_rank = 6 * 14  # 6 output blocks, each with rank 14
    
    U = np.zeros((n*m, total_rank))  # 16 x 84
    V = np.zeros((m*p, total_rank))  # 24 x 84
    W = np.zeros((p*n, total_rank))  # 24 x 84
    
    # Get the sum-of-two-2x2-products decomposition
    U_sub, V_sub, W_sub = _sum_two_2x2_products()
    
    # For each output block (ci, cj) where ci in {0,1}, cj in {0,1,2}:
    for ci in range(2):
        for cj in range(3):
            base_col = (ci * 3 + cj) * 14
            
            # C[ci, cj] = A[ci, 0] * B[0, cj] + A[ci, 1] * B[1, cj]
            # 
            # A[ci, 0] is 2x2: rows 2*ci to 2*ci+1, cols 0 to 1
            # A[ci, 1] is 2x2: rows 2*ci to 2*ci+1, cols 2 to 3
            # B[0, cj] is 2x2: rows 0 to 1, cols 2*cj to 2*cj+1
            # B[1, cj] is 2x2: rows 2 to 3, cols 2*cj to 2*cj+1
            # C[ci, cj] is 2x2: rows 2*ci to 2*ci+1, cols 2*cj to 2*cj+1
            #
            # In the sub-decomposition:
            # First 4 indices of U_sub correspond to X (which is A[ci,0])
            # Next 4 indices of U_sub correspond to Y (which is A[ci,1])
            # First 4 indices of V_sub correspond to U_mat (which is B[0,cj])
            # Next 4 indices of V_sub correspond to V_mat (which is B[1,cj])
            # All 4 indices of W_sub correspond to C[ci,cj]
            
            for local_r in range(14):
                col = base_col + local_r
                
                # U matrix: map A elements to this term
                # X = A[ci, 0]: local indices (li, lj) -> global (2*ci+li, lj)
                for li in range(2):
                    for lj in range(2):
                        a_idx = (2*ci + li) * m + lj
                        U[a_idx, col] += U_sub[li*2+lj, local_r]
                
                # Y = A[ci, 1]: local indices (li, lj) -> global (2*ci+li, 2+lj)
                for li in range(2):
                    for lj in range(2):
                        a_idx = (2*ci + li) * m + (2 + lj)
                        U[a_idx, col] += U_sub[4 + li*2+lj, local_r]
                
                # V matrix: map B elements to this term
                # U_mat = B[0, cj]: local indices (li, lj) -> global (li, 2*cj+lj)
                for li in range(2):
                    for lj in range(2):
                        b_idx = li * p + (2*cj + lj)
                        V[b_idx, col] += V_sub[li*2+lj, local_r]
                
                # V_mat = B[1, cj]: local indices (li, lj) -> global (2+li, 2*cj+lj)
                for li in range(2):
                    for lj in range(2):
                        b_idx = (2 + li) * p + (2*cj + lj)
                        V[b_idx, col] += V_sub[4 + li*2+lj, local_r]
                
                # W matrix: map C elements to this term
                # C[ci, cj]: local indices (li, lj) -> global (2*ci+li, 2*cj+lj)
                for li in range(2):
                    for lj in range(2):
                        c_idx = (2*ci + li) * p + (2*cj + lj)
                        W[c_idx, col] += W_sub[li*2+lj, local_r]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def construct():
    """Find a low-rank tensor decomposition for 4x4 * 4x6 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 145
    
    n, m, p = DIMENSIONS
    
    # Try block-Strassen with rank 84
    result = _block_strassen_4x4_4x6()
    
    # Verify
    if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
        return result
    
    # Fallback to schoolbook if verification fails
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
# EVOLVE-BLOCK-END
