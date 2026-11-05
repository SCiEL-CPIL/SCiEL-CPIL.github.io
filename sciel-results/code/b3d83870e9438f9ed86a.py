"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def _strassen_2x2x2():
    """
    Return rank-7 Strassen decomposition for 2x2 matrix multiplication.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Tensor index: U[i*2+j, c] * V[j*2+k, c] * W[k*2+i, c]
    """
    # Strassen formulas for C = A * B where A, B, C are 2x2:
    # M1 = (A11 + A22)*(B11 + B22)
    # M2 = (A21 + A22)*B11
    # M3 = A11*(B12 - B22)
    # M4 = A22*(B21 - B11)
    # M5 = (A11 + A12)*B22
    # M6 = (A21 - A11)*(B11 + B12)
    # M7 = (A12 - A22)*(B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # For the tensor decomposition, each M_t corresponds to a rank-1 term.
    # U row index: i*2+j where i in {0,1} (row of A), j in {0,1} (col of A)
    # V row index: j*2+k where j in {0,1} (row of B = col of A), k in {0,1} (col of B)
    # W row index: k*2+i where k in {0,1} (row of C), i in {0,1} (col of C)
    
    # We need: sum_t U[i*2+j, t] * V[j*2+k, t] * W[k*2+i, t] = 1 if (i,j,k) valid, else 0
    # For 2x2 multiplication, all (i,j,k) with i,j,k in {0,1} are "valid" in the sense that
    # the tensor T[i,j,k] = 1 for all combinations (it's the identity tensor for 2x2 mult).
    # Wait no - the tensor for matrix multiplication is T[i,j,k] = 1 if the product C[i,k] = sum_j A[i,j]B[j,k].
    # So T[i,j,k] = 1 for all i,j,k in {0,1}. That's 8 ones.
    
    # U[i*2+j, t] * V[j*2+k, t] * W[k*2+i, t]
    # U index: (i,j) -> row i, col j of A
    # V index: (j,k) -> row j, col k of B
    # W index: (k,i) -> row k, col i of C
    
    # Let me construct this carefully.
    # M1 = (A00 + A11)*(B00 + B11)
    # This expands to: A00*B00 + A00*B11 + A11*B00 + A11*B11
    # These contribute to C00 (from A00*B00), C01 (from A00*B11), C10 (from A11*B00), C11 (from A11*B11)
    
    # U for M1: nonzero at (0,0) and (1,1) -> U[0]=1, U[3]=1
    # V for M1: nonzero at (0,0) and (1,1) -> V[0]=1, V[3]=1
    # W for M1: contributes to C00, C01, C10, C11 -> W[0]=1, W[1]=1, W[2]=1, W[3]=1
    
    # Let me just build it by enumerating each Strassen multiplier.
    
    # M1 = (A00+A11)(B00+B11)
    # A factors: A00 (i=0,j=0), A11 (i=1,j=1) -> U indices: 0, 3
    # B factors: B00 (j=0,k=0), B11 (j=1,k=1) -> V indices: 0, 3
    # C contributions: C00 (k=0,i=0), C01 (k=0,i=1), C10 (k=1,i=0), C11 (k=1,i=1)
    # W indices: 0, 1, 2, 3
    M1_U = [1, 0, 0, 1]
    M1_V = [1, 0, 0, 1]
    M1_W = [1, 1, 1, 1]
    
    # M2 = (A10+A11)*B00
    # A factors: A10 (i=1,j=0), A11 (i=1,j=1) -> U indices: 2, 3
    # B factors: B00 (j=0,k=0) -> V index: 0
    # C contributions: C10 (k=1,i=0) -> W index: 2
    M2_U = [0, 0, 1, 1]
    M2_V = [1, 0, 0, 0]
    M2_W = [0, 0, 1, 0]
    
    # M3 = A00*(B01-B11)
    # A factors: A00 (i=0,j=0) -> U index: 0
    # B factors: B01 (j=0,k=1), B11 (j=1,k=1) -> V indices: 1, 3
    # C contributions: C01 (k=0,i=1) -> W index: 1
    M3_U = [1, 0, 0, 0]
    M3_V = [0, 1, 0, -1]
    M3_W = [0, 1, 0, 0]
    
    # M4 = A11*(B10-B00)
    # A factors: A11 (i=1,j=1) -> U index: 3
    # B factors: B10 (j=1,k=0), B00 (j=0,k=0) -> V indices: 2, 0
    # C contributions: C10 (k=1,i=0) -> W index: 2
    M4_U = [0, 0, 0, 1]
    M4_V = [1, 0, 1, 0]  # B00=1, B10=1
    M4_W = [0, 0, 1, 0]
    
    # M5 = (A00+A01)*B11
    # A factors: A00 (i=0,j=0), A01 (i=0,j=1) -> U indices: 0, 1
    # B factors: B11 (j=1,k=1) -> V index: 3
    # C contributions: C11 (k=1,i=1) -> W index: 3
    M5_U = [1, 1, 0, 0]
    M5_V = [0, 0, 0, 1]
    M5_W = [0, 0, 0, 1]
    
    # M6 = (A10-A00)*(B00+B01)
    # A factors: A10 (i=1,j=0), A00 (i=0,j=0) -> U indices: 2, 0
    # B factors: B00 (j=0,k=0), B01 (j=0,k=1) -> V indices: 0, 1
    # C contributions: C00 (k=0,i=0) -> W index: 0
    M6_U = [1, 0, 1, 0]  # A00=1, A10=1
    M6_V = [1, 1, 0, 0]
    M6_W = [1, 0, 0, 0]
    
    # M7 = (A01-A11)*(B10+B11)
    # A factors: A01 (i=0,j=1), A11 (i=1,j=1) -> U indices: 1, 3
    # B factors: B10 (j=1,k=0), B11 (j=1,k=1) -> V indices: 2, 3
    # C contributions: C11 (k=1,i=1) -> W index: 3
    M7_U = [0, 1, 0, 1]  # A01=1, A11=1
    M7_V = [0, 0, 1, 1]
    M7_W = [0, 0, 0, 1]
    
    # Verify: sum_t U[i*2+j,t] * V[j*2+k,t] * W[k*2+i,t] should equal 1 for all (i,j,k)
    
    U = [[M1_U[t], M2_U[t], M3_U[t], M4_U[t], M5_U[t], M6_U[t], M7_U[t]] for t in range(4)]
    V = [[M1_V[t], M2_V[t], M3_V[t], M4_V[t], M5_V[t], M6_V[t], M7_V[t]] for t in range(4)]
    W = [[M1_W[t], M2_W[t], M3_W[t], M4_W[t], M5_W[t], M6_W[t], M7_W[t]] for t in range(4)]
    
    return U, V, W

def _verify_2x2x2(U, V, W):
    """Verify that the 2x2x2 decomposition is correct."""
    for i in range(2):
        for j in range(2):
            for k in range(2):
                val = sum(U[i*2+j][t] * V[j*2+k][t] * W[k*2+i][t] for t in range(7))
                if val != 1:
                    return False
    return True

def _strassen_block_decomposition():
    """
    Build a decomposition using Strassen on overlapping 2x2 sub-blocks.
    For 3x5x6, we partition into blocks:
    - Rows of A (i): {0,1} and {2} -> use 2x2 for rows 0-1, schoolbook for row 2
    - Cols of A / Rows of B (j): {0,1}, {2,3}, {4} -> use 2x2 for j=0-1, 2x2 for j=2-3, schoolbook for j=4
    - Cols of B / Cols of C (k): {0,1}, {2,3}, {4,5} -> use 2x2 for k=0-1, 2x2 for k=2-3, 2x2 for k=4-5
    
    Actually, let me think more carefully. The tensor is T[i,j,k] = 1 for all i in [0,3), j in [0,5), k in [0,6).
    
    Strategy: Use Strassen on 2x2x2 blocks where possible, and fill the rest with schoolbook.
    
    Partition i into {0,1} (Strassen) and {2} (schoolbook)
    Partition j into {0,1}, {2,3}, {4}
    Partition k into {0,1}, {2,3}, {4,5}
    
    For the Strassen part (i in {0,1}):
    - j in {0,1}, k in {0,1}: 2x2x2 Strassen (rank 7)
    - j in {0,1}, k in {2,3}: 2x2x2 Strassen (rank 7)
    - j in {0,1}, k in {4,5}: 2x2x2 Strassen (rank 7)
    - j in {2,3}, k in {0,1}: 2x2x2 Strassen (rank 7)
    - j in {2,3}, k in {2,3}: 2x2x2 Strassen (rank 7)
    - j in {2,3}, k in {4,5}: 2x2x2 Strassen (rank 7)
    - j in {4}, k in {0,1}: 1x2x2 -> schoolbook rank 4
    - j in {4}, k in {2,3}: 1x2x2 -> schoolbook rank 4
    - j in {4}, k in {4,5}: 1x2x2 -> schoolbook rank 4
    
    For the schoolbook part (i=2):
    - j in {0,1}, k in {0,1}: 1x2x2 -> schoolbook rank 4
    - j in {0,1}, k in {2,3}: 1x2x2 -> schoolbook rank 4
    - j in {0,1}, k in {4,5}: 1x2x2 -> schoolbook rank 4
    - j in {2,3}, k in {0,1}: 1x2x2 -> schoolbook rank 4
    - j in {2,3}, k in {2,3}: 1x2x2 -> schoolbook rank 4
    - j in {2,3}, k in {4,5}: 1x2x2 -> schoolbook rank 4
    - j in {4}, k in {0,1}: 1x1x2 -> schoolbook rank 2
    - j in {4}, k in {2,3}: 1x1x2 -> schoolbook rank 2
    - j in {4}, k in {4,5}: 1x1x2 -> schoolbook rank 2
    
    Total rank: 6*7 + 3*4 + 6*4 + 3*2 = 42 + 12 + 24 + 6 = 84
    """
    n, m, p = 3, 5, 6
    rank = 84
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen 2x2x2 for i in {0,1}, j in {0,1}, k in {0,1}
    sU, sV, sW = _strassen_2x2x2()
    # Map: local i' in {0,1} -> global i = i'
    # local j' in {0,1} -> global j = j'
    # local k' in {0,1} -> global k = k'
    for t in range(7):
        for li in range(2):
            for lj in range(2):
                for lk in range(2):
                    if sU[li*2+lj][t] != 0:
                        U[li*2+lj][col] += sU[li*2+lj][t]
                    if sV[lj*2+lk][t] != 0:
                        V[lj*2+lk][col] += sV[lj*2+lk][t]
                    if sW[lk*2+li][t] != 0:
                        W[lk*2+li][col] += sW[lk*2+li][t]
        col += 1
    
    # Strassen 2x2x2 for i in {0,1}, j in {0,1}, k in {2,3}
    for t in range(7):
        for li in range(2):
            for lj in range(2):
                for lk in range(2):
                    gi = li
                    gj = lj
                    gk = 2 + lk
                    if sU[li*2+lj][t] != 0:
                        U[gi*5+gj][col] += sU[li*2+lj][t]
                    if sV[lj*2+lk][t] != 0:
                        V[gj*6+gk][col] += sV[lj*2+lk][t]
                    if sW[lk*2+li][t] != 0:
                        W[gk*3+gi][col] += sW[lk*2+li][t]
        col += 1
    
    # Strassen 2x2x2 for i in {0,1}, j in {0,1}, k in {4,5}
    for t in range(7):
        for li in range(2):
            for lj in range(2):
                for lk in range(2):
                    gi = li
                    gj = lj
                    gk = 4 + lk
                    if sU[li*2+lj][t] != 0:
                        U[gi*5+gj][col] += sU[li*2+lj][t]
                    if sV[lj*2+lk][t] != 0:
                        V[gj*6+gk][col] += sV[lj*2+lk][t]
                    if sW[lk*2+li][t] != 0:
                        W[gk*3+gi][col] += sW[lk*2+li][t]
        col += 1
    
    # Strassen 2x2x2 for i in {0,1}, j in {2,3}, k in {0,1}
    for t in range(7):
        for li in range(2):
            for lj in range(2):
                for lk in range(2):
                    gi = li
                    gj = 2 + lj
                    gk = lk
                    if sU[li*2+lj][t] != 0:
                        U[gi*5+gj][col] += sU[li*2+lj][t]
                    if sV[lj*2+lk][t] != 0:
                        V[gj*6+gk][col] += sV[lj*2+lk][t]
                    if sW[lk*2+li][t] != 0:
                        W[gk*3+gi][col] += sW[lk*2+li][t]
        col += 1
    
    # Strassen 2x2x2 for i in {0,1}, j in {2,3}, k in {2,3}
    for t in range(7):
        for li in range(2):
            for lj in range(2):
                for lk in range(2):
                    gi = li
                    gj = 2 + lj
                    gk = 2 + lk
                    if sU[li*2+lj][t] != 0:
                        U[gi*5+gj][col] += sU[li*2+lj][t]
                    if sV[lj*2+lk][t] != 0:
                        V[gj*6+gk][col] += sV[lj*2+lk][t]
                    if sW[lk*2+li][t] != 0:
                        W[gk*3+gi][col] += sW[lk*2+li][t]
        col += 1
    
    # Strassen 2x2x2 for i in {0,1}, j in {2,3}, k in {4,5}
    for t in range(7):
        for li in range(2):
            for lj in range(2):
                for lk in range(2):
                    gi = li
                    gj = 2 + lj
                    gk = 4 + lk
                    if sU[li*2+lj][t] != 0:
                        U[gi*5+gj][col] += sU[li*2+lj][t]
                    if sV[lj*2+lk][t] != 0:
                        V[gj*6+gk][col] += sV[lj*2+lk][t]
                    if sW[lk*2+li][t] != 0:
                        W[gk*3+gi][col] += sW[lk*2+li][t]
        col += 1
    
    # Schoolbook for i in {0,1}, j in {4}, k in {0,1}
    for i in range(2):
        for j in [4]:
            for k in range(2):
                U[i*5+j][col] = 1
                V[j*6+k][col] = 1
                W[k*3+i][col] = 1
                col += 1
    
    # Schoolbook for i in {0,1}, j in {4}, k in {2,3}
    for i in range(2):
        for j in [4]:
            for k in range(2, 4):
                U[i*5+j][col] = 1
                V[j*6+k][col] = 1
                W[k*3+i][col] = 1
                col += 1
    
    # Schoolbook for i in {0,1}, j in {4}, k in {4,5}
    for i in range(2):
        for j in [4]:
            for k in range(4, 6):
                U[i*5+j][col] = 1
                V[j*6+k][col] = 1
                W[k*3+i][col] = 1
                col += 1
    
    # Schoolbook for i=2, j in {0,1}, k in {0,1}
    for i in [2]:
        for j in range(2):
            for k in range(2):
                U[i*5+j][col] = 1
                V[j*6+k][col] = 1
                W[k*3+i][col] = 1
                col += 1
    
    # Schoolbook for i=2, j in {0,1}, k in {2,3}
    for i in [2]:
        for j in range(2):
            for k in range(2, 4):
                U[i*5+j][col] = 1
                V[j*6+k][col] = 1
                W[k*3+i][col] = 1
                col += 1
    
    # Schoolbook for i=2, j in {0,1}, k in {4,5}
    for i in [2]:
        for j in range(2):
            for k in range(4, 6):
                U[i*5+j][col] = 1
                V[j*6+k][col] = 1
                W[k*3+i][col] = 1
                col += 1
    
    # Schoolbook for i=2, j in {2,3}, k in {0,1}
    for i in [2]:
        for j in range(2, 4):
            for k in range(2):
                U[i*5+j][col] = 1
                V[j*6+k][col] = 1
                W[k*3+i][col] = 1
                col += 1
    
    # Schoolbook for i=2, j in {2,3}, k in {2,3}
    for i in [2]:
        for j in range(2, 4):
            for k in range(2, 4):
                U[i*5+j][col] = 1
                V[j*6+k][col] = 1
                W[k*3+i][col] = 1
                col += 1
    
    # Schoolbook for i=2, j in {2,3}, k in {4,5}
    for i in [2]:
        for j in range(2, 4):
            for k in range(4, 6):
                U[i*5+j][col] = 1
                V[j*6+k][col] = 1
                W[k*3+i][col] = 1
                col += 1
    
    # Schoolbook for i=2, j=4, k in {0,1}
    for i in [2]:
        for j in [4]:
            for k in range(2):
                U[i*5+j][col] = 1
                V[j*6+k][col] = 1
                W[k*3+i][col] = 1
                col += 1
    
    # Schoolbook for i=2, j=4, k in {2,3}
    for i in [2]:
        for j in [4]:
            for k in range(2, 4):
                U[i*5+j][col] = 1
                V[j*6+k][col] = 1
                W[k*3+i][col] = 1
                col += 1
    
    # Schoolbook for i=2, j=4, k in {4,5}
    for i in [2]:
        for j in [4]:
            for k in range(4, 6):
                U[i*5+j][col] = 1
                V[j*6+k][col] = 1
                W[k*3+i][col] = 1
                col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(decomp, n=3, m=5, p=6):
    """Verify that the decomposition exactly reproduces the matrix multiplication tensor."""
    U, V, W = decomp["U"], decomp["V"], decomp["W"]
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j][t] * V[j*p+k][t] * W[k*n+i][t] for t in range(rank))
                if val != 1:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    
    # Try the Strassen block decomposition
    try:
        decomp = _strassen_block_decomposition()
        if _verify_decomposition(decomp, n, m, p):
            return decomp
    except Exception:
        pass
    
    # Fallback: schoolbook
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
