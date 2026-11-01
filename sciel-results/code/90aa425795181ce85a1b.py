"""Matrix multiplication 2,5,6 tensor decomposition."""
import time

DIMENSIONS = [2, 5, 6]

def _schoolbook(n, m, p):
    """Standard rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return U, V, W, rank

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is correct."""
    rank = len(U[0]) if U else 0
    # Check dimensions
    if len(U) != n * m or len(U[0]) != rank:
        return False
    if len(V) != m * p or len(V[0]) != rank:
        return False
    if len(W) != p * n or len(W[0]) != rank:
        return False
    # Verify: for each (i,k), the sum over j of A[i,j]*B[j,k] should equal
    # sum over c of (sum over j of U[i*m+j][c]*V[j*p+k][c]) * W[k*n+i][c]
    # More precisely: M[i,k] = sum_j A[i,j] B[j,k] = sum_c W[k*n+i][c] * (sum_j U[i*m+j][c] * V[j*p+k][c])
    # We can verify by checking that the tensor decomposes correctly.
    # The matrix multiplication tensor T[i,j,k] = delta_{i,k via j} has the property that:
    # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]  ... no, that's not right either.
    
    # Actually, the convention is:
    # (A*B)[i,k] = sum_j A[i,j] B[j,k]
    # In tensor form: T[i,j,k] = 1 for the multiplication operation
    # The decomposition says: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Wait, that's not standard. Let me re-read the task.
    
    # "output factor follows the official trace convention W[k*n+i,r]"
    # So W is indexed by (k, i) -> W[k*n+i, r]
    # The standard decomposition: (AB)_{ik} = sum_j A_{ij} B_{jk} = sum_r U_{ir} V_{rj} ... no.
    
    # Let me think about this more carefully.
    # The tensor for matrix multiplication C = A * B where C_{ik} = sum_j A_{ij} B_{jk}
    # has entries T_{i,j,k} = delta (it's the identity in the middle index pairing)
    # The rank-r decomposition: T_{i,j,k} = sum_{c=0}^{r-1} U_{i,j,c} V_{j,k,c} W_{k,i,c}
    # where U has shape (n*m, r), V has shape (m*p, r), W has shape (p*n, r)
    # with U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    
    # For the schoolbook: U[i*m+j, c]=1, V[j*p+k, c]=1, W[k*n+i, c]=1 for matching (i,j,k)
    # This gives T_{i,j,k} = 1 if (i,j,k) matches, 0 otherwise. But T should be 1 for all valid (i,j,k)!
    # Wait no - the tensor T_{i,j,k} for matrix multiplication is indeed 1 for every triple (i,j,k)
    # because C_{ik} = sum_j A_{ij} B_{jk}, so each term A_{ij} B_{jk} contributes with coefficient 1.
    
    # So we need: for all i in [0,n), j in [0,m), k in [0,p):
    #   1 = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Let's verify this.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > 1e-10:
                    return False
    return True

def _strassen_2x2x2():
    """
    Rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    Returns U, V, W each of shape (4, 7) with entries in {-1, 0, 1}.
    
    Strassen's 7 products for computing C = A*B where A, B are 2x2:
    M1 = (A00 + A11)(B00 + B11)
    M2 = (A10 + A11) B11
    M3 = A00 (B01 - B11)
    M4 = A11 (B01 + B10)
    M5 = (A00 - A10)(B00 + B10)
    M6 = (A01 - A00)(B00 + B10)  -- wait, let me get this right
    M7 = A01 B00
    
    Actually the standard Strassen formulas:
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)B11
    M3 = A00(B01-B11)
    M4 = A11(B01+B10)
    M5 = (A00-A10)(B00+B10)
    M6 = (A01-A00)(B00+B10)  -- I keep getting confused. Let me use a known reference.
    
    Standard Strassen:
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)B11
    M3 = A00(B01-B11)
    M4 = A11(B01+B10)
    M5 = (A00-A10)(B00+B10)
    M6 = (A01-A00)(B00+B10) -- no
    
    Let me just use the explicit coefficient table.
    
    For 2x2x2, the 7 rank-1 terms have coefficients:
    Each term c contributes U[i,j,c], V[j,k,c], W[k,i,c] where i,j,k in {0,1}.
    
    The 7 products are:
    c=0: (A00+A11)(B00+B11) -> U[0,0]=1,U[1,1]=1; V[0,0]=1,V[1,1]=1; W[0,0]=1,W[1,1]=1
         This gives: C00 += A00*B00 + A00*B11 + A11*B00 + A11*B11
    c=1: (A10+A11)B11 -> U[1,0]=1,U[1,1]=1; V[1,1]=1; W[1,1]=1
         C11 += A10*B11 + A11*B11
    c=2: A00(B01-B11) -> U[0,0]=1; V[0,1]=1,V[1,1]=-1; W[0,1]=1
         C01 += A00*B01 - A00*B11
    c=3: A11(B01+B10) -> U[1,1]=1; V[0,1]=1,V[1,0]=1; W[1,1]=1
         C11 += A11*B01 + A11*B10
    c=4: (A00-A10)(B00+B10) -> U[0,0]=1,U[1,0]=-1; V[0,0]=1,V[1,0]=1; W[0,0]=1
         C00 += A00*B00 + A00*B10 - A10*B00 - A10*B10
    c=5: (A01-A00)(B00+B10) -- hmm this doesn't look right for C10
    c=6: ...
    
    I think I'm overcomplicating this. Let me just hardcode the known Strassen coefficients.
    """
    # I'll use a verified Strassen decomposition.
    # The 7 factors for 2x2x2:
    # Factor 0: M1 = (A00+A11)(B00+B11)
    # Factor 1: M2 = (A10+A11)B11  
    # Factor 2: M3 = A00(B01-B11)
    # Factor 3: M4 = A11(B01+B10)
    # Factor 4: M5 = (A00-A10)(B00+B10)
    # Factor 5: M6 = (A01-A00)(B00+B10)
    # Factor 6: M7 = A01B00
    
    # Let me just define them explicitly by their contributions.
    # U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    
    U = [[0.0]*7 for _ in range(4)]
    V = [[0.0]*7 for _ in range(4)]
    W = [[0.0]*7 for _ in range(4)]
    
    # c=0: (A00+A11)(B00+B11)
    # U: A00->1, A11->1 => U[0*2+0]=1, U[1*2+1]=1
    # V: B00->1, B11->1 => V[0*2+0]=1, V[1*2+1]=1
    # W: C00 gets +1, C11 gets +1 => W[0*2+0]=1, W[1*2+1]=1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # c=1: (A10+A11)B11
    # U: A10->1, A11->1 => U[1*2+0]=1, U[1*2+1]=1
    # V: B11->1 => V[1*2+1]=1
    # W: C11 gets +1 => W[1*2+1]=1
    U[2][1] = 1; U[3][1] = 1
    V[3][1] = 1
    W[3][1] = 1
    
    # c=2: A00(B01-B11)
    # U: A00->1 => U[0*2+0]=1
    # V: B01->1, B11->-1 => V[0*2+1]=1, V[1*2+1]=-1
    # W: C01 gets +1 => W[0*2+0]=1
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[0][2] = 1
    
    # c=3: A11(B01+B10)
    # U: A11->1 => U[1*2+1]=1
    # V: B01->1, B10->1 => V[0*2+1]=1, V[1*2+0]=1
    # W: C11 gets +1 => W[1*2+1]=1
    U[3][3] = 1
    V[1][3] = 1; V[2][3] = 1
    W[3][3] = 1
    
    # c=4: (A00-A10)(B00+B10)
    # U: A00->1, A10->-1 => U[0*2+0]=1, U[1*2+0]=-1
    # V: B00->1, B10->1 => V[0*2+0]=1, V[1*2+0]=1
    # W: C00 gets +1 => W[0*2+0]=1
    U[0][4] = 1; U[2][4] = -1
    V[0][4] = 1; V[2][4] = 1
    W[0][4] = 1
    
    # c=5: (A01-A00)(B00+B10)
    # U: A01->1, A00->-1 => U[0*2+1]=1, U[0*2+0]=-1
    # V: B00->1, B10->1 => V[0*2+0]=1, V[1*2+0]=1
    # W: C10 gets +1 => W[0*2+1]=1
    U[1][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[2][5] = 1
    W[2][5] = 1
    
    # c=6: A01*B00
    # U: A01->1 => U[0*2+1]=1
    # V: B00->1 => V[0*2+0]=1
    # W: C10 gets +1 => W[0*2+1]=1
    U[1][6] = 1
    V[0][6] = 1
    W[2][6] = 1
    
    return U, V, W

def _test_strassen():
    """Test the Strassen decomposition."""
    n, m, p = 2, 2, 2
    U, V, W = _strassen_2x2x2()
    return _verify(U, V, W, n, m, p)

def _build_block_decomp(n, m, p, sub_n, sub_m, sub_p, sub_U, sub_V, sub_W):
    """
    Build a decomposition for n x m x p from a sub-block decomposition.
    This is for the case where n = sub_n * a, m = sub_m * b, p = sub_p * c
    and we tile the sub-decomposition.
    
    Actually, let me think about this differently.
    For 2x5x6, I can think of it as:
    - Split the 5 columns of A (and rows of B) into groups
    - Use the Strassen 2x2x2 where applicable
    
    A simpler approach: just use the schoolbook for now but try to find a better one.
    """
    pass

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n, m, p = DIMENSIONS
    # Schoolbook rank: 60
    U_sb, V_sb, W_sb, rank_sb = _schoolbook(n, m, p)
    best = (U_sb, V_sb, W_sb, rank_sb)
    
    if time.time() > deadline:
        return {"U": best[0], "V": best[1], "W": best[2]}
    
    # Test Strassen
    if _test_strassen():
        # Strassen works. Now build 2x5x6 using Strassen blocks.
        # Strategy: 
        # - The 2x2x2 Strassen can handle a 2x2 submatrix multiplication.
        # - For 2x5x6: think of A as 2x5, B as 5x6.
        # - Split the 5 into 2+3, or use 2x2 blocks where possible.
        # 
        # Alternative: use the fact that rank(2x5x6) <= rank(2x2x2) * ceil(5/2) * ... no that's not how it works.
        #
        # Actually, a cleaner approach: 
        # The multiplication 2x5x6 can be seen as multiplying by a 5x6 matrix.
        # We can decompose B into 2x2 blocks (plus remainders) and use Strassen for each.
        # But the rank would be sum of ranks of the block multiplications.
        #
        # Let me think of it as: C[i,k] = sum_{j=0}^{4} A[i,j] B[j,k]
        # Split j into {0,1}, {2,3}, {4}:
        # C[i,k] = sum_{j in {0,1}} A[i,j] B[j,k] + sum_{j in {2,3}} A[i,j] B[j,k] + A[i,4] B[4,k]
        # The first two terms are 2x2x2 multiplications (Strassen, rank 7 each)
        # The last term is 2x1x2 (rank 2)
        # Total rank: 7 + 7 + 2 = 16
        #
        # Let me verify: for the first block (j=0,1), we multiply A[:,0:2] (2x2) by B[0:2,:] (2x6)
        # But Strassen is for 2x2x2, not 2x2x6.
        #
        # Hmm, I need to be more careful. Strassen 2x2x2 computes C = A*B where A is 2x2, B is 2x2, C is 2x2.
        # For 2x2x6, I'd need to do 3 separate 2x2x2 multiplications (for columns 0-1, 2-3, 4-5 of B)
        # That gives rank 7*3 = 21 for the 2x2x6 part.
        #
        # So: split 5 rows of B into {0,1}, {2,3}, {4}:
        # - Block 1: A[:,0:2] * B[0:2,:] = 2x2 * 2x6 -> 3 Strassen 2x2x2, rank 21
        # - Block 2: A[:,2:4] * B[2:4,:] = 2x2 * 2x6 -> 3 Strassen 2x2x2, rank 21
        # - Block 3: A[:,4:5] * B[4:5,:] = 2x1 * 1x6 -> rank 2
        # Total: 21 + 21 + 2 = 44
        #
        # That's better than 60! Let me implement this.
        
        def _expand_strassen_block(sub_U, sub_V, sub_W, sub_n, sub_m, sub_p, 
                                    i_off, j_off, k_off, n, m, p):
            """
            Place a sub-block decomposition into the full n x m x p space.
            sub_U has shape (sub_n*sub_m, rank_sub)
            sub_V has shape (sub_m*sub_p, rank_sub)
            sub_W has shape (sub_p*sub_n, rank_sub)
            Returns contributions to U, V, W for the full space.
            """
            rank_sub = sub_U[0].__len__()
            U_full = [[0.0] * rank_sub for _ in range(n * m)]
            V_full = [[0.0] * rank_sub for _ in range(m * p)]
            W_full = [[0.0] * rank_sub for _ in range(p * n)]
            
            for ii in range(sub_n):
                for jj in range(sub_m):
                    for c in range(rank_sub):
                        val = sub_U[ii * sub_m + jj][c]
                        if val != 0:
                            U_full[(i_off + ii) * m + (j_off + jj)][c] = val
            
            for jj in range(sub_m):
                for kk in range(sub_p):
                    for c in range(rank_sub):
                        val = sub_V[jj * sub_p + kk][c]
                        if val != 0:
                            V_full[(j_off + jj) * p + (k_off + kk)][c] = val
            
            for kk in range(sub_p):
                for ii in range(sub_n):
                    for c in range(rank_sub):
                        val = sub_W[kk * sub_n + ii][c]
                        if val != 0:
                            W_full[(k_off + kk) * n + (i_off + ii)][c] = val
            
            return U_full, V_full, W_full
        
        # Build 2x5x6 using Strassen blocks
        # Block structure for the 5 middle dimension:
        # j in [0,1]: Strassen 2x2x2 for each pair of k columns
        # j in [2,3]: Strassen 2x2x2 for each pair of k columns  
        # j in [4,4]: trivial 2x1x2 for each k column
        
        sU, sV, sW = _strassen_2x2x2()
        
        # We'll accumulate all factors
        all_U = []
        all_V = []
        all_W = []
        
        # Block 1: j in {0,1}, k in {0,1}
        # This is a 2x2x2 subproblem with i_off=0, j_off=0, k_off=0
        # sub_n=2, sub_m=2, sub_p=2
        block_U, block_V, block_W = _expand_strassen_block(sU, sV, sW, 2, 2, 2, 0, 0, 0, 2, 5, 6)
        all_U.append(block_U)
        all_V.append(block_V)
        all_W.append(block_W)
        
        # Block 2: j in {0,1}, k in {2,3}
        block_U, block_V, block_W = _expand_strassen_block(sU, sV, sW, 2, 2, 2, 0, 0, 2, 2, 5, 6)
        all_U.append(block_U)
        all_V.append(block_V)
        all_W.append(block_W)
        
        # Block 3: j in {0,1}, k in {4,5}
        block_U, block_V, block_W = _expand_strassen_block(sU, sV, sW, 2, 2, 2, 0, 0, 4, 2, 5, 6)
        all_U.append(block_U)
        all_V.append(block_V)
        all_W.append(block_W)
        
        # Block 4: j in {2,3}, k in {0,1}
        block_U, block_V, block_W = _expand_strassen_block(sU, sV, sW, 2, 2, 2, 0, 2, 0, 2, 5, 6)
        all_U.append(block_U)
        all_V.append(block_V)
        all_W.append(block_W)
        
        # Block 5: j in {2,3}, k in {2,3}
        block_U, block_V, block_W = _expand_strassen_block(sU, sV, sW, 2, 2, 2, 0, 2, 2, 2, 5, 6)
        all_U.append(block_U)
        all_V.append(block_V)
        all_W.append(block_W)
        
        # Block 6: j in {2,3}, k in {4,5}
        block_U, block_V, block_W = _expand_strassen_block(sU, sV, sW, 2, 2, 2, 0, 2, 4, 2, 5, 6)
        all_U.append(block_U)
        all_V.append(block_V)
        all_W.append(block_W)
        
        # Block 7: j in {4,4}, k in {0,1} - trivial 2x1x2
        # For 2x1x2: C[i,k] = A[i,4] * B[4,k]
        # Rank 2: one factor for k=0, one for k=1
        # Actually 2x1x2 has rank min(2,1,2)*... no, rank is at most 2*1*2=4 but actually it's 2.
        # C[i,k] = A[i,4] * B[4,k]. This is an outer product of A[:,4] and B[4,:].
        # Rank = 2 (one for each k, or we can do it in 2 factors)
        # Factor for k=0: U[i*5+4, c]=1 for both i, V[4*6+0, c]=1, W[0*2+i, c]=1
        # Factor for k=1: U[i*5+4, c]=1 for both i, V[4*6+1, c]=1, W[1*2+i, c]=1
        
        block_U7 = [[0.0] * 2 for _ in range(10)]  # n*m = 2*5 = 10
        block_V7 = [[0.0] * 2 for _ in range(30)]  # m*p = 5*6 = 30
        block_W7 = [[0.0] * 2 for _ in range(12)]  # p*n = 6*2 = 12
        
        for i in range(2):
            block_U7[i * 5 + 4][0] = 1.0
            block_U7[i * 5 + 4][1] = 1.0
        block_V7[4 * 6 + 0][0] = 1.0
        block_V7[4 * 6 + 1][1] = 1.0
        for i in range(2):
            block_W7[0 * 2 + i][0] = 1.0
            block_W7[1 * 2 + i][1] = 1.0
        
        all_U.append(block_U7)
        all_V.append(block_V7)
        all_W.append(block_W7)
        
        # Block 8: j in {4,4}, k in {2,3}
        block_U8 = [[0.0] * 2 for _ in range(10)]
        block_V8 = [[0.0] * 2 for _ in range(30)]
        block_W8 = [[0.0] * 2 for _ in range(12)]
        
        for i in range(2):
            block_U8[i * 5 + 4][0] = 1.0
            block_U8[i * 5 + 4][1] = 1.0
        block_V8[4 * 6 + 2][0] = 1.0
        block_V8[4 * 6 + 3][1] = 1.0
        for i in range(2):
            block_W8[2 * 2 + i][0] = 1.0
            block_W8[3 * 2 + i][1] = 1.0
        
        all_U.append(block_U8)
        all_V.append(block_V8)
        all_W.append(block_W8)
        
        # Block 9: j in {4,4}, k in {4,5}
        block_U9 = [[0.0] * 2 for _ in range(10)]
        block_V9 = [[0.0] * 2 for _ in range(30)]
        block_W9 = [[0.0] * 2 for _ in range(12)]
        
        for i in range(2):
            block_U9[i * 5 + 4][0] = 1.0
            block_U9[i * 5 + 4][1] = 1.0
        block_V9[4 * 6 + 4][0] = 1.0
        block_V9[4 * 6 + 5][1] = 1.0
        for i in range(2):
            block_W9[4 * 2 + i][0] = 1.0
            block_W9[5 * 2 + i][1] = 1.0
        
        all_U.append(block_U9)
        all_V.append(block_V9)
        all_W.append(block_W9)
        
        # Combine all blocks
        total_rank = sum(len(bU[0]) for bU in all_U)
        U_combined = [[0.0] * total_rank for _ in range(n * m)]
        V_combined = [[0.0] * total_rank for _ in range(m * p)]
        W_combined = [[0.0] * total_rank for _ in range(p * n)]
        
        col_offset = 0
        for bU, bV, bW in zip(all_U, all_V, all_W):
            r = len(bU[0])
            for i in range(n * m):
                for c in range(r):
                    if bU[i][c] != 0:
                        U_combined[i][col_offset + c] = bU[i][c]
            for j in range(m * p):
                for c in range(r):
                    if bV[j][c] != 0:
                        V_combined[j][col_offset + c] = bV[j][c]
            for k in range(p * n):
                for c in range(r):
                    if bW[k][c] != 0:
                        W_combined[k][col_offset + c] = bW[k][c]
            col_offset += r
        
        # Verify
        if _verify(U_combined, V_combined, W_combined, n, m, p):
            best = (U_combined, V_combined, W_combined, total_rank)
        
        if time.time() > deadline:
            return {"U": best[0], "V": best[1], "W": best[2]}
        
        # Try to improve further if time permits
        # For now, return the best we have
    
    # Convert to integer values if possible
    U_final = best[0]
    V_final = best[1]
    W_final = best[2]
    
    # Try to convert to ints
    def to_int_if_close(mat):
        out = []
        for row in mat:
            new_row = []
            for v in row:
                if abs(v - round(v)) < 1e-10:
                    new_row.append(int(round(v)))
                else:
                    new_row.append(v)
            out.append(new_row)
        return out
    
    U_final = to_int_if_close(U_final)
    V_final = to_int_if_close(V_final)
    W_final = to_int_if_close(W_final)
    
    return {"U": U_final, "V": V_final, "W": W_final}
