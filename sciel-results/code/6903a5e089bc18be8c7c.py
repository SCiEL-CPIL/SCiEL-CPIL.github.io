"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    import time
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # n=2, m=2, p=8
    
    def verify(U, V, W, rank):
        """Verify the tensor decomposition exactly."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if s != 1:
                        return False
        return True
    
    # Try to construct rank p+6 = 14 decomposition
    # Using Strassen-based sharing across columns
    rank_target = p + 6  # 14
    
    U = [[0] * rank_target for _ in range(n * m)]  # (4, 14)
    V = [[0] * rank_target for _ in range(m * p)]  # (16, 14)
    W = [[0] * rank_target for _ in range(p * n)]  # (16, 14)
    
    # Construction for (2,2,p) with rank p+6:
    # We use 6 "shared" terms that exploit Strassen's structure across all columns,
    # plus p "diagonal" terms for each column.
    #
    # Strassen's 7 products for (2,2,2):
    #   M1 = (A00+A11)(B00+B11)
    #   M2 = (A10+A11)(B00)
    #   M3 = (A00)(B10-B11)
    #   M4 = (A11)(B01-B00)
    #   M5 = (A00+A01)(B11)
    #   M6 = (A10-A00)(B00+B10)
    #   M7 = (A01-A11)(B01+B11)
    #
    # For (2,2,p), we can structure it as:
    # - 6 terms that are "shared" in the sense they combine multiple columns
    # - p terms that handle individual column contributions
    #
    # Let me use a known explicit construction for rank p+6.
    # The idea: 
    #   Terms 0-5: Strassen-like shared terms
    #   Terms 6-6+p-1: Column-specific terms
    #
    # Actually, let me use a direct algebraic construction.
    # For (2,2,p), C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    #
    # We can write this using 6+p terms as follows:
    # Let's define the 6 shared terms to capture the "Strassen" structure,
    # and p terms for the remaining contributions.
    
    # Alternative approach: Use the fact that we can factor the computation.
    # For each output column k, we compute:
    #   C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
    #   C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
    #
    # With 6+p terms, we can share some multiplications.
    # Let me try an explicit construction and verify.
    
    # Try construction 1: Based on known results for (2,2,p) rank = p+6
    # The construction uses:
    # - 6 terms for the "off-diagonal" Strassen-like parts
    # - p terms for the "diagonal" parts
    
    # Let me define the factor matrices explicitly.
    # For rank = p + 6, we index terms as:
    #   r = 0, 1, ..., 5: shared terms
    #   r = 6, 7, ..., 5+p: column-specific terms (one per column)
    
    # Shared term 0: (A[0,0] + A[1,1]) * (B[0,k] + B[1,k]) for all k
    # This contributes to C[0,k] and C[1,k]
    U[0 * m + 0][0] = 1  # A[0,0]
    U[1 * m + 1][0] = 1  # A[1,1]
    for k in range(p):
        V[0 * p + k][0] = 1  # B[0,k]
        V[1 * p + k][0] = 1  # B[1,k]
        W[k * n + 0][0] = 1  # contributes to C[0,k]
        W[k * n + 1][0] = 1  # contributes to C[1,k]
    
    # Shared term 1: (A[1,0] + A[1,1]) * B[0,k]
    U[1 * m + 0][1] = 1  # A[1,0]
    U[1 * m + 1][1] = 1  # A[1,1]
    for k in range(p):
        V[0 * p + k][1] = 1  # B[0,k]
        W[k * n + 1][1] = 1  # contributes to C[1,k]
    
    # Shared term 2: A[0,0] * (B[1,k] - B[0,k])
    U[0 * m + 0][2] = 1  # A[0,0]
    for k in range(p):
        V[1 * p + k][2] = 1  # B[1,k]
        V[0 * p + k][2] = -1  # -B[0,k]
        W[k * n + 0][2] = 1  # contributes to C[0,k]
    
    # Shared term 3: A[1,1] * (B[0,k] - B[1,k])
    U[1 * m + 1][3] = 1  # A[1,1]
    for k in range(p):
        V[0 * p + k][3] = 1  # B[0,k]
        V[1 * p + k][3] = -1  # -B[1,k]
        W[k * n + 1][3] = 1  # contributes to C[1,k]
    
    # Shared term 4: (A[0,0] + A[0,1]) * B[1,k]
    U[0 * m + 0][4] = 1  # A[0,0]
    U[0 * m + 1][4] = 1  # A[0,1]
    for k in range(p):
        V[1 * p + k][4] = 1  # B[1,k]
        W[k * n + 0][4] = 1  # contributes to C[0,k]
    
    # Shared term 5: (A[1,0] - A[0,0]) * (B[0,k] + B[1,k])
    U[1 * m + 0][5] = 1  # A[1,0]
    U[0 * m + 0][5] = -1  # -A[0,0]
    for k in range(p):
        V[0 * p + k][5] = 1  # B[0,k]
        V[1 * p + k][5] = 1  # B[1,k]
        W[k * n + 1][5] = 1  # contributes to C[1,k]
    
    # Now the p column-specific terms (r = 6 to 6+p-1)
    # These handle the remaining contributions.
    # After the 6 shared terms, what's left?
    # Let me check by computing what each C[i,k] gets from the 6 shared terms.
    #
    # C[0,k] from shared terms:
    #   Term 0: 1 (from A[0,0]*B[0,k] + A[0,0]*B[1,k] + A[1,1]*B[0,k] + A[1,1]*B[1,k])
    #   Wait, I need to be more careful. The W factor tells us which output gets the contribution.
    #
    # Let me re-examine. The decomposition requires:
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    #
    # This means for each (i,j,k), the sum over r of the product must equal 1.
    #
    # Let me verify my construction by checking a few (i,j,k) values.
    #
    # For i=0, j=0, k=0:
    #   r=0: U[0,0]*V[0,0]*W[0,0] = 1*1*1 = 1
    #   r=1: U[0,0]*V[0,0]*W[0,1] = 1*1*0 = 0 (W[0,1]=0 since W[0*n+0][1]=W[0][1]=0)
    #   Wait, W is indexed as W[k*n+i]. So W[0*2+0] = W[0], W[0*2+1] = W[1].
    #   W[0][0] = 1, W[1][0] = 1 (from term 0)
    #   W[0][1] = 0, W[1][1] = 1 (from term 1)
    #   W[0][2] = 1, W[1][2] = 0 (from term 2)
    #   W[0][3] = 0, W[1][3] = 1 (from term 3)
    #   W[0][4] = 1, W[1][4] = 0 (from term 4)
    #   W[0][5] = 0, W[1][5] = 1 (from term 5)
    #
    # For (i=0, j=0, k=0): need sum_r U[0,0]*V[0,0]*W[0,0] = 1
    #   r=0: U[0][0]=1, V[0][0]=1, W[0][0]=1 => 1
    #   r=1: U[0][1]=0, V[0][1]=1, W[0][1]=0 => 0
    #   r=2: U[0][2]=1, V[0][2]=-1, W[0][2]=1 => -1
    #   r=3: U[0][3]=0, V[0][3]=1, W[0][3]=0 => 0
    #   r=4: U[0][4]=1, V[0][4]=0, W[0][4]=1 => 0
    #   r=5: U[0][5]=-1, V[0][5]=1, W[0][5]=0 => 0
    #   Sum = 1 + 0 + (-1) + 0 + 0 + 0 = 0. Not 1!
    #
    # My construction is wrong. Let me try a different approach.
    
    # Let me use a known correct construction for (2,2,p) rank p+6.
    # Based on the literature, the construction is more subtle.
    #
    # Let me try to use a different strategy: start from the schoolbook and
    # see if I can merge terms using linearity.
    
    # For now, let me try to verify if rank 14 is achievable by a different construction.
    # If not, I'll fall back to schoolbook.
    
    # Let me try a simpler approach: use the fact that for (2,2,2), rank is 7,
    # and see if I can extend this to (2,2,8) with rank 14.
    
    # Actually, let me just return schoolbook for now to ensure correctness,
    # and try to improve in future iterations.
    
    if time.time() > DEADLINE - 10:
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
    
    # Try to find a valid rank 14 construction
    # Let me use a systematic approach with the known Strassen extension.
    
    # For (2,2,p), the rank p+6 construction works as follows:
    # We have 6 "global" terms and p "local" terms.
    #
    # Let me define them properly:
    # Global terms (r = 0 to 5):
    #   These involve combinations of all A entries and all B entries for each column.
    #
    # Local terms (r = 6 to 6+p-1):
    #   One for each column k, involving specific A and B entries.
    
    # Let me try a specific known construction:
    # For (2,2,p), use the following 6+p terms:
    
    U = [[0] * rank_target for _ in range(n * m)]
    V = [[0] * rank_target for _ in range(m * p)]
    W = [[0] * rank_target for _ in range(p * n)]
    
    # Term 0: A[0,0] * B[0,k] for all k -> contributes to C[0,k]
    U[0 * m + 0][0] = 1
    for k in range(p):
        V[0 * p + k][0] = 1
        W[k * n + 0][0] = 1
    
    # Term 1: A[0,1] * B[1,k] for all k -> contributes to C[0,k]
    U[0 * m + 1][1] = 1
    for k in range(p):
        V[1 * p + k][1] = 1
        W[k * n + 0][1] = 1
    
    # Term 2: A[1,0] * B[0,k] for all k -> contributes to C[1,k]
    U[1 * m + 0][2] = 1
    for k in range(p):
        V[0 * p + k][2] = 1
        W[k * n + 1][2] = 1
    
    # Term 3: A[1,1] * B[1,k] for all k -> contributes to C[1,k]
    U[1 * m + 1][3] = 1
    for k in range(p):
        V[1 * p + k][3] = 1
        W[k * n + 1][3] = 1
    
    # These 4 terms give us the schoolbook for the "diagonal" parts.
    # Now we need 2 more global terms and p-4 local terms? That doesn't add up to p+6.
    
    # I think I'm overcomplicating this. Let me just return schoolbook.
    # The score will be 1.0, same as parent, but at least it's correct.
    
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
