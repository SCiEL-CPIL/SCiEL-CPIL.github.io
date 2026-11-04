"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We'll build the decomposition using tensor product of Strassen for 4x4x4 part
    # plus schoolbook for the extra column.
    
    # First, define the 2x2x2 Strassen decomposition (rank 7)
    # Multiplying A(2x2) * B(2x2) = C(2x2)
    # A indices: (i,j) with i,j in {0,1}
    # B indices: (j,k) with j,k in {0,1}  
    # C indices: (i,k) with i,k in {0,1}
    # Convention: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r] for n=m=p=2
    # So U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    
    # Strassen terms (using standard formulation):
    # Let A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # Standard Strassen:
    # P1 = (a+b)(e+f)
    # P2 = (d+b)(e+f)
    # P3 = a(d-c)
    # P4 = g(a+b)
    # P5 = e(d-a)
    # P6 = h(b+c)
    # P7 = (a-c)(e-f)
    # C00 = P1 + P4 - P6 - P7
    # C01 = P3 + P5
    # C10 = P2 + P7
    # C11 = P1 + P3 - P4 - P5
    
    # For each term, we need U, V, W vectors of length 4 (for 2x2)
    # U[i*2+j, r] = coefficient of A[i,j]
    # V[j*2+k, r] = coefficient of B[j,k]
    # W[k*2+i, r] = coefficient of C[i,k]
    
    strassen_U = []
    strassen_V = []
    strassen_W = []
    
    # Term 1: P1 = (a+b)(e+f)
    # U: a=1, b=1, c=0, d=0 -> [1, 1, 0, 0]
    # V: e=1, f=1, g=0, h=0 -> [1, 1, 0, 0]
    # W: all 0 (it's an intermediate, not directly C)
    # Actually, for tensor decomposition, each term r has:
    # U[:,r], V[:,r], W[:,r] such that C = sum_r W[:,r] * (U[:,r] outer V[:,r]) ... 
    # Wait, the convention is different. Let me re-read.
    
    # The task says: U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank)
    # And it decomposes the matrix multiplication tensor.
    # The standard tensor decomposition: T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
    # where T[i,j,k] = 1 if the output element (i,k) depends on input A[i,j]*B[j,k]
    # So for matrix mult: T[i,j,k] = delta_{j, output_row} ... 
    # Actually: C[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor T has T[i,j,k] = 1, meaning A[i,j] * B[j,k] contributes to C[i,k]
    # The decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # For Strassen, we have 7 terms. Each term r corresponds to one of the P's.
    # For term r, U[i*2+j, r] is the coefficient of A[i,j] in the first factor of P_r
    # V[j*2+k, r] is the coefficient of B[j,k] in the second factor of P_r
    # W[k*2+i, r] is the coefficient of C[i,k] in the combination
    
    # This is getting complex. Let me just verify the full 4x4x4 decomposition numerically.
    
    # Simpler approach: just verify the schoolbook (rank 80) works, and try to use
    # a known smaller rank. But to be safe, let me use a verified approach.
    
    # Actually, let me just use the schoolbook but with a twist: we can exploit that
    # some entries might be zero or use linear dependencies. But for 4x4x5, the 
    # schoolbook is already minimal for this naive approach.
    
    # Let me try a different approach: use the fact that 4x4x4 can be done in rank 49
    # by tensor product of 2x2x2 rank 7. I'll implement this carefully and verify.
    
    # For 2x2x2, the multiplication tensor has rank 7. The decomposition is:
    # T[i,j,k] = sum_{r=0}^{6} U[i*2+j, r] * V[j*2+k, r] * W[k*2+i, r]
    
    # The 7 rank-1 tensors correspond to the Strassen decomposition. Let me define them:
    
    # I'll use a known correct rank-7 decomposition for 2x2x2.
    # One standard representation (from literature):
    
    rank7_U = [
        [1, 0, 0, 1, 1, 0, 1],   # i=0,j=0: a
        [1, 0, 0, 1, 0, 1, 0],   # i=0,j=1: b  
        [0, 1, 1, 0, 0, 1, 1],   # i=1,j=0: c
        [0, 1, -1, 1, -1, 0, -1] # i=1,j=1: d
    ]
    rank7_V = [
        [1, 1, 0, 0, 1, 0, 1],   # j=0,k=0: e
        [1, 1, 0, 0, 0, 1, -1],  # j=0,k=1: f
        [0, 0, 1, 1, 0, 1, 0],   # j=1,k=0: g
        [0, 0, 1, 1, 1, 0, 1]    # j=1,k=1: h
    ]
    rank7_W = [
        [0, 0, 0, 1, 0, 0, 0],   # k=0,i=0: C00
        [0, 1, 0, 0, 0, 1, 0],   # k=1,i=0: C01
        [0, 0, 0, 0, 0, 0, 1],   # k=0,i=1: C10
        [0, 0, 0, 0, 1, 0, 0]    # k=1,i=1: C11
    ]
    
    # Verify 2x2x2 decomposition
    def verify_2x2x2():
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    lhs = 1.0  # T[i,j,k] = 1 for all (since every A[i,j]*B[j,k] contributes)
                    rhs = 0.0
                    for r in range(7):
                        rhs += rank7_U[i*2+j][r] * rank7_V[j*2+k][r] * rank7_W[k*2+i][r]
                    if abs(lhs - rhs) > 1e-10:
                        return False
        return True
    
    if not verify_2x2x2():
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Now build 4x4x4 using tensor product of 2x2x2
    # 4x4x4: n=m=p=4
    # Index A[i,j] where i = i1*2+i2, j = j1*2+j2
    # Index B[j,k] where j = j1*2+j2, k = k1*2+k2
    # Index C[i,k] where i = i1*2+i2, k = k1*2+k2
    # For the tensor product: r = r1*7 + r2 where r1,r2 in 0..6
    # U4[i*4+j, r] = rank7_U[i1*2+j1, r1] * rank7_U[i2*2+j2, r2]
    # V4[j*4+k, r] = rank7_V[j1*2+k1, r1] * rank7_V[j2*2+k2, r2]
    # W4[k*4+i, r] = rank7_W[k1*2+i1, r1] * rank7_W[k2*2+i2, r2]
    
    rank444 = 7 * 7
    U444 = [[0] * rank444 for _ in range(16)]
    V444 = [[0] * rank444 for _ in range(16)]
    W444 = [[0] * rank444 for _ in range(16)]
    
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            for i1 in range(2):
                for i2 in range(2):
                    for j1 in range(2):
                        for j2 in range(2):
                            i = i1 * 2 + i2
                            j = j1 * 2 + j2
                            U444[i*4+j][r] = rank7_U[i1*2+j1][r1] * rank7_U[i2*2+j2][r2]
                            for k1 in range(2):
                                for k2 in range(2):
                                    k = k1 * 2 + k2
                                    V444[j*4+k][r] = rank7_V[j1*2+k1][r1] * rank7_V[j2*2+k2][r2]
                                    W444[k*4+i][r] = rank7_W[k1*2+i1][r1] * rank7_W[k2*2+i2][r2]
    
    # Verify 4x4x4
    def verify_4x4x4():
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    lhs = 1.0
                    rhs = 0.0
                    for r in range(rank444):
                        rhs += U444[i*4+j][r] * V444[j*4+k][r] * W444[k*4+i][r]
                    if abs(lhs - rhs) > 1e-10:
                        return False
        return True
    
    if not verify_4x4x4():
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Now extend to 4x4x5: add 16 schoolbook terms for k=4 (the 5th column, index 4)
    # For k=4, we need: A[i,j] * B[j,4] contributes to C[i,4]
    # U[i*4+j, new_r] = 1
    # V[j*5+4, new_r] = 1
    # W[4*4+i, new_r] = 1
    
    total_rank = rank444 + 16
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Copy 4x4x4 part (for k=0,1,2,3)
    for row in range(16):
        for r in range(rank444):
            U[row][r] = U444[row][r]
            V[row][r] = V444[row][r]
            W[row][r] = W444[row][r]
    
    # Add schoolbook for k=4
    col = rank444
    for i in range(n):
        for j in range(m):
            U[i*m+j][col] = 1
            V[j*p+4][col] = 1
            W[4*n+i][col] = 1
            col += 1
    
    # Final verification
    def verify_full():
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    lhs = 1.0
                    rhs = 0.0
                    for r in range(total_rank):
                        rhs += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(lhs - rhs) > 1e-10:
                        return False
        return True
    
    if not verify_full():
        # Fallback to pure schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
