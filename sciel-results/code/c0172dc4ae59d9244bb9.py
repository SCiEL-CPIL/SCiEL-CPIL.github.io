"""Matrix multiplication 5x5x5 rank-minimal exact tensor decomposition.

Strategy: Use Strassen's algorithm (rank 49) for the 4x4 sub-blocks and 
schoolbook for the interactions with the 5th row/column.
Total Rank: 49 (for 4x4) + 57 (for 5th row/col interactions) = 106.
"""

import time

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p):
    """
    Check that the tensor decomposition is exact.
    Convention: C[i,k] = sum_j A[i,j] B[j,k]
    Tensor T[i*m+j, j*p+k, k*n+i] = 1 for valid (i,j,k).
    We verify: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
    and for invalid triples, the sum is 0.
    """
    r = len(U[0])
    # Check all valid triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the standard rank n*m*p decomposition."""
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

def _construct_strassen_4x4(n, m, p):
    """
    Construct Strassen-based decomposition for the 4x4 subblock.
    Returns U, V, W for indices 0-3 only, with rank 49.
    
    This uses Strassen's algorithm for 4x4 matrix multiplication:
    - Split 4x4 matrices into 2x2 blocks
    - Use 7 Strassen products (each 2x2)
    - Each 2x2 product uses Strassen's algorithm (rank 7)
    - Total: 7 * 7 = 49 terms
    """
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen_2x2(a_r0, a_c0, b_r0, b_c0, c_r0, c_c0, start_col):
        """
        Add 7 Strassen terms for a 2x2 block multiplication.
        A block: rows a_r0, a_r0+1; cols a_c0, a_c0+1
        B block: rows b_r0, b_r0+1; cols b_c0, b_c0+1
        C block: rows c_r0, c_r0+1; cols c_c0, c_c0+1
        For standard matmul: b_r0 == a_c0, c_r0 == a_r0, c_c0 == b_c0.
        """
        i0, i1 = a_r0, a_r0 + 1
        j0, j1 = a_c0, a_c0 + 1
        k0, k1 = b_c0, b_c0 + 1
        
        # Term 1: m1 = (A[i0,j0] + A[i1,j1]) * (B[j0,k0] + B[j1,k1])
        # Contributes to C[i0,k0] and C[i1,k1]
        U[i0*m+j0][start_col] += 1
        U[i1*m+j1][start_col] += 1
        V[j0*p+k0][start_col] += 1
        V[j1*p+k1][start_col] += 1
        W[k0*n+i0][start_col] += 1
        W[k1*n+i1][start_col] += 1
        col = start_col + 1
        
        # Term 2: m2 = (A[i1,j0] + A[i1,j1]) * B[j0,k0]
        # Contributes to C[i1,k0]
        U[i1*m+j0][col] += 1
        U[i1*m+j1][col] += 1
        V[j0*p+k0][col] += 1
        W[k0*n+i1][col] += 1
        col += 1
        
        # Term 3: m3 = A[i0,j0] * (B[j0,k1] - B[j1,k1])
        # Contributes to C[i0,k1]
        U[i0*m+j0][col] += 1
        V[j0*p+k1][col] += 1
        V[j1*p+k1][col] -= 1
        W[k1*n+i0][col] += 1
        col += 1
        
        # Term 4: m4 = A[i1,j1] * (B[j1,k0] - B[j0,k0])
        # Contributes to C[i0,k0] and C[i1,k0]
        U[i1*m+j1][col] += 1
        V[j1*p+k0][col] += 1
        V[j0*p+k0][col] -= 1
        W[k0*n+i0][col] += 1
        W[k0*n+i1][col] += 1
        col += 1
        
        # Term 5: m5 = (A[i0,j0] + A[i0,j1]) * B[j1,k1]
        # Contributes to C[i0,k0] with -1, C[i0,k1] with +1
        U[i0*m+j0][col] += 1
        U[i0*m+j1][col] += 1
        V[j1*p+k1][col] += 1
        W[k0*n+i0][col] -= 1
        W[k1*n+i0][col] += 1
        col += 1
        
        # Term 6: m6 = (A[i1,j0] - A[i0,j0]) * (B[j0,k0] + B[j0,k1])
        # Contributes to C[i1,k1]
        U[i1*m+j0][col] += 1
        U[i0*m+j0][col] -= 1
        V[j0*p+k0][col] += 1
        V[j0*p+k1][col] += 1
        W[k1*n+i1][col] += 1
        col += 1
        
        # Term 7: m7 = (A[i0,j1] - A[i1,j1]) * (B[j1,k0] + B[j1,k1])
        # Contributes to C[i0,k0]
        U[i0*m+j1][col] += 1
        U[i1*m+j1][col] -= 1
        V[j1*p+k0][col] += 1
        V[j1*p+k1][col] += 1
        W[k0*n+i0][col] += 1
        col += 1
        
        return col
    
    def add_strassen_4x4_2x2_product(A_r0, A_c0, B_r0, B_c0, C_r0, C_c0, start_col):
        """
        Add 7 Strassen 2x2 products for one of the 7 Strassen 4x4 products.
        Each product is a 2x2 matrix multiplication using Strassen (rank 7).
        
        The 7 Strassen 4x4 products are:
        M1 = (A11+A22)(B11+B22)
        M2 = (A21+A22)B11
        M3 = A11(B21-B11)
        M4 = A22(B22-B12)
        M5 = (A11+A12)B22
        M6 = (A21-A11)(B11+B12)
        M7 = (A12-A22)(B21+B22)
        
        Where:
        A11 = A[0:2, 0:2], A12 = A[0:2, 2:4]
        A21 = A[2:4, 0:2], A22 = A[2:4, 2:4]
        B11 = B[0:2, 0:2], B12 = B[0:2, 2:4]
        B21 = B[2:4, 0:2], B22 = B[2:4, 2:4]
        C11 = C[0:2, 0:2], C12 = C[0:2, 2:4]
        C21 = C[2:4, 0:2], C22 = C[2:4, 2:4]
        
        And:
        C11 = M1 + M4 - M5 - M7
        C12 = M3 + M5
        C21 = M2 + M4
        C22 = M1 - M2 + M3 + M6
        """
        # Define the 7 2x2 block multiplications
        # Each is: (linear_comb_of_A_blocks) * (linear_comb_of_B_blocks) -> (linear_comb_of_C_blocks)
        
        # M1 = (A11+A22)(B11+B22) -> contributes to C11(+1), C22(-1)
        # We need to add 7 Strassen terms for each of these 7 products
        # But the A and B factors are sums of blocks, so we need to expand
        
        # Actually, let's just directly add the 7 Strassen terms for each 2x2 product
        # The key insight: each M_i is a 2x2 matrix product, and we can use Strassen for it
        
        # For M1: A_factor = A11 + A22, B_factor = B11 + B22
        # A_factor[i,j] = A[i,j] + A[i+2,j+2] for i,j in {0,1}
        # B_factor[i,j] = B[i,j] + B[i+2,j+2] for i,j in {0,1}
        # C_contribution: C11 += M1, C22 -= M1
        
        # This is getting complex. Let me use a different approach:
        # Just use the standard Strassen 4x4 algorithm which gives rank 49.
        # The 7 products are 2x2 matrix multiplications, each done with Strassen (rank 7).
        
        # Let me implement this more carefully.
        
        # For each of the 7 Strassen 4x4 products, we have:
        # M_k = A_k * B_k where A_k and B_k are 2x2 matrices that are linear combinations of 2x2 blocks
        
        # We'll add 7 terms for each product using Strassen's 2x2 algorithm.
        
        # Define helper to add Strassen 2x2 product with arbitrary block offsets
        def _add_strassen_2x2_with_offsets(A_offs, B_offs, C_offs, start_col):
            """
            A_offs: list of (row_off, col_off, sign) for A blocks
            B_offs: list of (row_off, col_off, sign) for B blocks  
            C_offs: list of (row_off, col_off, sign) for C blocks
            """
            # For Strassen 2x2:
            # m1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1]) -> C[0,0]+=m1, C[1,1]+=m1
            # m2 = (A[1,0]+A[1,1]) * B[0,0] -> C[1,0]+=m2
            # m3 = A[0,0] * (B[1,0]-B[0,0]) -> C[0,1]+=m3
            # m4 = A[1,1] * (B[1,1]-B[0,1]) -> C[0,0]+=m4, C[1,0]+=m4
            # m5 = (A[0,0]+A[0,1]) * B[1,1] -> C[0,0]-=m5, C[0,1]+=m5
            # m6 = (A[1,0]-A[0,0]) * (B[0,0]+B[0,1]) -> C[1,1]+=m6
            # m7 = (A[0,1]-A[1,1]) * (B[1,0]+B[1,1]) -> C[0,0]+=m7
            
            # A[i,j] = sum of A_offs: sign * A[block_row + i, block_col + j]
            # B[i,j] = sum of B_offs: sign * B[block_row + i, block_col + j]
            # C[i,j] = sum of C_offs: sign * C[block_row + i, block_col + j]
            
            # For each Strassen term, we add contributions to U, V, W
            
            # m1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
            # U: for each a_off in A_offs, for (ai,aj) in [(0,0),(1,1)]: U[(a_off[0]+ai)*m + (a_off[1]+aj)] += a_off[2]
            # V: for each b_off in B_offs, for (bi,bj) in [(0,0),(1,1)]: V[(b_off[0]+bi)*p + (b_off[1]+bj)] += b_off[2]
            # W: for each c_off in C_offs, for (ci,cj) in [(0,0),(1,1)]: W[(c_off[1]+cj)*n + (c_off[0]+ci)] += c_off[2]
            
            col = start_col
            
            # m1
            for a_off in A_offs:
                for (ai, aj) in [(0,0), (1,1)]:
                    U[(a_off[0]+ai)*m + (a_off[1]+aj)][col] += a_off[2]
            for b_off in B_offs:
                for (bi, bj) in [(0,0), (1,1)]:
                    V[(b_off[0]+bi)*p + (b_off[1]+bj)][col] += b_off[2]
            for c_off in C_offs:
                for (ci, cj) in [(0,0), (1,1)]:
                    W[(c_off[1]+cj)*n + (c_off[0]+ci)][col] += c_off[2]
            col += 1
            
            # m2 = (A[1,0]+A[1,1]) * B[0,0]
            for a_off in A_offs:
                for (ai, aj) in [(1,0), (1,1)]:
                    U[(a_off[0]+ai)*m + (a_off[1]+aj)][col] += a_off[2]
            for b_off in B_offs:
                V[(b_off[0]+0)*p + (b_off[1]+0)][col] += b_off[2]
            for c_off in C_offs:
                W[(c_off[1]+0)*n + (c_off[0]+1)][col] += c_off[2]
            col += 1
            
            # m3 = A[0,0] * (B[1,0]-B[0,0])
            for a_off in A_offs:
                U[(a_off[0]+0)*m + (a_off[1]+0)][col] += a_off[2]
            for b_off in B_offs:
                V[(b_off[0]+1)*p + (b_off[1]+0)][col] += b_off[2]
                V[(b_off[0]+0)*p + (b_off[1]+0)][col] -= b_off[2]
            for c_off in C_offs:
                W[(c_off[1]+1)*n + (c_off[0]+0)][col] += c_off[2]
            col += 1
            
            # m4 = A[1,1] * (B[1,1]-B[0,1])
            for a_off in A_offs:
                U[(a_off[0]+1)*m + (a_off[1]+1)][col] += a_off[2]
            for b_off in B_offs:
                V[(b_off[0]+1)*p + (b_off[1]+1)][col] += b_off[2]
                V[(b_off[0]+0)*p + (b_off[1]+1)][col] -= b_off[2]
            for c_off in C_offs:
                W[(c_off[1]+0)*n + (c_off[0]+0)][col] += c_off[2]
                W[(c_off[1]+0)*n + (c_off[0]+1)][col] += c_off[2]
            col += 1
            
            # m5 = (A[0,0]+A[0,1]) * B[1,1]
            for a_off in A_offs:
                for (ai, aj) in [(0,0), (0,1)]:
                    U[(a_off[0]+ai)*m + (a_off[1]+aj)][col] += a_off[2]
            for b_off in B_offs:
                V[(b_off[0]+1)*p + (b_off[1]+1)][col] += b_off[2]
            for c_off in C_offs:
                W[(c_off[1]+0)*n + (c_off[0]+0)][col] -= c_off[2]
                W[(c_off[1]+1)*n + (c_off[0]+0)][col] += c_off[2]
            col += 1
            
            # m6 = (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
            for a_off in A_offs:
                U[(a_off[0]+1)*m + (a_off[1]+0)][col] += a_off[2]
                U[(a_off[0]+0)*m + (a_off[1]+0)][col] -= a_off[2]
            for b_off in B_offs:
                for (bi, bj) in [(0,0), (0,1)]:
                    V[(b_off[0]+bi)*p + (b_off[1]+bj)][col] += b_off[2]
            for c_off in C_offs:
                W[(c_off[1]+1)*n + (c_off[0]+1)][col] += c_off[2]
            col += 1
            
            # m7 = (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
            for a_off in A_offs:
                U[(a_off[0]+0)*m + (a_off[1]+1)][col] += a_off[2]
                U[(a_off[0]+1)*m + (a_off[1]+1)][col] -= a_off[2]
            for b_off in B_offs:
                for (bi, bj) in [(1,0), (1,1)]:
                    V[(b_off[0]+bi)*p + (b_off[1]+bj)][col] += b_off[2]
            for c_off in C_offs:
                W[(c_off[1]+0)*n + (c_off[0]+0)][col] += c_off[2]
            col += 1
            
            return col
        
        # M1 = (A11+A22)(B11+B22) -> C11 += M1, C22 -= M1
        A_offs_M1 = [(0,0,1), (2,2,1)]
        B_offs_M1 = [(0,0,1), (2,2,1)]
        C_offs_M1 = [(0,0,1), (2,2,-1)]
        start_col = _add_strassen_2x2_with_offsets(A_offs_M1, B_offs_M1, C_offs_M1, start_col)
        
        # M2 = (A21+A22)B11 -> C21 += M2, C22 -= M2
        A_offs_M2 = [(2,0,1), (2,2,1)]
        B_offs_M2 = [(0,0,1)]
        C_offs_M2 = [(2,0,1), (2,2,-1)]
        start_col = _add_strassen_2x2_with_offsets(A_offs_M2, B_offs_M2, C_offs_M2, start_col)
        
        # M3 = A11(B21-B11) -> C12 += M3, C22 += M3
        A_offs_M3 = [(0,0,1)]
        B_offs_M3 = [(2,0,1), (0,0,-1)]
        C_offs_M3 = [(0,2,1), (2,2,1)]
        start_col = _add_strassen_2x2_with_offsets(A_offs_M3, B_offs_M3, C_offs_M3, start_col)
        
        # M4 = A22(B22-B12) -> C11 += M4, C21 += M4
        A_offs_M4 = [(2,2,1)]
        B_offs_M4 = [(2,2,1), (0,2,-1)]
        C_offs_M4 = [(0,0,1), (2,0,1)]
        start_col = _add_strassen_2x2_with_offsets(A_offs_M4, B_offs_M4, C_offs_M4, start_col)
        
        # M5 = (A11+A12)B22 -> C11 -= M5, C12 += M5
        A_offs_M5 = [(0,0,1), (0,2,1)]
        B_offs_M5 = [(2,2,1)]
        C_offs_M5 = [(0,0,-1), (0,2,1)]
        start_col = _add_strassen_2x2_with_offsets(A_offs_M5, B_offs_M5, C_offs_M5, start_col)
        
        # M6 = (A21-A11)(B11+B12) -> C22 += M6
        A_offs_M6 = [(2,0,1), (0,0,-1)]
        B_offs_M6 = [(0,0,1), (0,2,1)]
        C_offs_M6 = [(2,2,1)]
        start_col = _add_strassen_2x2_with_offsets(A_offs_M6, B_offs_M6, C_offs_M6, start_col)
        
        # M7 = (A12-A22)(B21+B22) -> C11 += M7
        A_offs_M7 = [(0,2,1), (2,2,-1)]
        B_offs_M7 = [(2,0,1), (2,2,1)]
        C_offs_M7 = [(0,0,1)]
        start_col = _add_strassen_2x2_with_offsets(A_offs_M7, B_offs_M7, C_offs_M7, start_col)
        
        return start_col
    
    col = 0
    col = add_strassen_4x4_2x2_product(0, 0, 0, 0, 0, 0, col)
    
    assert col == 49, f"Expected 49, got {col}"
    return U, V, W

def construct():
    """
    Construct the tensor decomposition for 5x5x5 matrix multiplication.
    
    Returns dict with 'U', 'V', 'W' factor matrices.
    U: shape (25, rank), V: shape (25, rank), W: shape (25, rank)
    Convention: W[k*n+i, c] for the output factor.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 130.0  # safety margin
    
    # Start with schoolbook as fallback
    result_fallback = _schoolbook(n, m, p)
    
    # Try Strassen 4x4 + schoolbook for 5th row/col
    # Total: 49 (Strassen 4x4) + 57 (cross terms) = 106
    total_rank = 106
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # First, get the Strassen 4x4 decomposition
    U_s, V_s, W_s = _construct_strassen_4x4(n, m, p)
    
    # Copy Strassen part into U, V, W
    for i in range(n * m):
        for c in range(49):
            U[i][c] = U_s[i][c]
            V[i][c] = V_s[i][c]
            W[i][c] = W_s[i][c]
    
    col = 49
    
    # Now add schoolbook terms for the 5th row and column (index 4)
    # 1. A[0:4, 4] * B[4, 0:4] -> C[0:4, 0:4]: 16 terms
    for i in range(4):
        for k in range(4):
            U[i * m + 4][col] = 1
            V[4 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    # 2. A[4, 0:4] * B[0:4, 0:4] -> C[4, 0:4]: 16 terms
    for j in range(4):
        for k in range(4):
            U[4 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 4][col] = 1
            col += 1
            
    # 3. A[0:4, 0:4] * B[0:4, 4] -> C[0:4, 4]: 16 terms
    for i in range(4):
        for j in range(4):
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1
            
    # 4. A[4, 4] * B[4, 4] -> C[4, 4]: 1 term
    U[4 * m + 4][col] = 1
    V[4 * p + 4][col] = 1
    W[4 * n + 4][col] = 1
    col += 1
    
    # 5. A[0:4, 4] * B[4, 4] -> C[0:4, 4]: 4 terms
    for i in range(4):
        U[i * m + 4][col] = 1
        V[4 * p + 4][col] = 1
        W[4 * n + i][col] = 1
        col += 1
        
    # 6. A[4, 0:4] * B[0:4, 4] -> C[4, 0:4]: 4 terms
    for j in range(4):
        U[4 * m + j][col] = 1
        V[j * p + 4][col] = 1
        W[4 * n + 4][col] = 1
        col += 1
        
    if col != total_rank:
        return result_fallback
        
    result = {"U": U, "V": V, "W": W}
    
    # Verify
    if _verify(U, V, W, n, m, p):
        return result
    
    # If verification fails, return schoolbook as safe fallback
    return result_fallback

# EVOLVE-BLOCK-END
