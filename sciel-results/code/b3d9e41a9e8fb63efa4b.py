"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    # Rank-49 decomposition via Kronecker product of two rank-7 Strassen 2x2x2 decompositions.
    # The 4x4x4 matrix multiplication tensor is isomorphic to (2x2x2) x (2x2x2).
    # We use the standard Strassen decomposition for the base case.
    
    # Strassen factors for 2x2x2 (rank 7)
    # U[i, c] = A input factor: U[i, c] * A[i, j] ... actually U is for A, V for B, W for C
    # Convention: C[i, k] = sum_c U[i, c] * V[j, c] * W[k, c] ? No.
    # The task says: W[k*n+i, r]. This suggests the output C[i, k] corresponds to W index k*n+i.
    # Let's check the schoolbook code:
    # U[i*m+j][column] = 1  -> A[i, j]
    # V[j*p+k][column] = 1  -> B[j, k]
    # W[k*n+i][column] = 1  -> C[i, k] (since index is k*n+i, row i, col k)
    # So the tensor is T[i, j, k] = sum_c U[i, c] * V[j, c] * W[k, c] ?
    # In schoolbook: U[i,c]=1 if i matches, V[j,c]=1 if j matches, W[k,c]=1 if k matches.
    # Sum over c: U[i,c] V[j,c] W[k,c] is 1 only if i,j,k match the same c.
    # This corresponds to C[i, k] = sum_j A[i, j] B[j, k].
    # So the factorization is: C[i, k] = sum_j sum_c (A[i, j] U[i, c]) (B[j, k] V[j, c]) W[k, c] ??
    # No, the standard tensor decomposition is T_{i,j,k} = sum_r U_{i,r} V_{j,r} W_{k,r}.
    # Here T_{i,j,k} is the tensor for matrix mult: T_{i,j,k} = delta_{i, i} delta_{j, j} ... no.
    # The multiplication tensor is defined such that <T, A \otimes B> = A B.
    # T_{i, j, k} = 1 if the term A[i,j] B[j,k] contributes to C[i,k].
    # Actually, C[i,k] = sum_j A[i,j] B[j,k].
    # So T_{i, j, k} = 1 for all i,j,k? No.
    # The tensor T is such that sum_{i,j,k} T_{i,j,k} A[i,j] B[j,k] = sum_{i,k} C[i,k] (something).
    # Usually, we define the tensor T with indices (i, j, k) where i is row of A/C, j is inner, k is col of B/C.
    # T_{i, j, k} = 1. Then sum_{j} T_{i, j, k} A[i, j] B[j, k] = sum_j A[i,j] B[j,k] = C[i,k].
    # So the tensor is just the identity in the middle? No, it's the structure of the bilinear map.
    # The rank of this tensor is the bilinear complexity.
    # The factors U, V, W satisfy: T_{i, j, k} = sum_r U_{i, r} V_{j, r} W_{k, r}.
    # For 4x4x4, i in 0..3 (row of A), j in 0..3 (col of A / row of B), k in 0..3 (col of B).
    # U has shape (4*4, rank)?? No, U has shape (n*m, rank) = (16, rank).
    # Wait, the task says: U (n*m, rank), V (m*p, rank), W (p*n, rank).
    # Indices for U: i*m+j. So U[i*m+j, r] corresponds to A[i,j].
    # Indices for V: j*p+k. So V[j*p+k, r] corresponds to B[j,k].
    # Indices for W: k*n+i. So W[k*n+i, r] corresponds to C[i,k].
    # The decomposition is: C[i,k] = sum_j sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r].
    # This matches the bilinear form.
    
    # Strassen 2x2:
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # C = [[c11, c12], [c21, c22]]
    # m1 = (a11+a22)(b11+b22)
    # m2 = (a21+a22)b11
    # m3 = a11(b12-b22)
    # m4 = a22(b21-b11)
    # m5 = (a11+a12)b22
    # m6 = (a21-a11)(b11+b12)
    # m7 = (a12-a22)(b21+b22)
    # c11 = m1 - m2 + m4 + m6
    # c12 = m3 + m5 + m7
    # c21 = m2 + m4 - m7
    # c22 = m1 - m3 + m5 + m6
    
    # Let's map this to U, V, W for 2x2.
    # U rows: (0,0), (0,1), (1,0), (1,1) -> indices 0,1,2,3.
    # V rows: (0,0), (0,1), (1,0), (1,1) -> indices 0,1,2,3.
    # W rows: (0,0), (0,1), (1,0), (1,1) -> indices 0,1,2,3. (k*n+i: k=0,i=0->0; k=0,i=1->1; k=1,i=0->2; k=1,i=1->3)
    
    # m1: U[0,0]=1, U[1,1]=1. V[0,0]=1, V[1,1]=1. W[0,0]=1, W[1,1]=1?
    # m1 contributes to c11 (W[0,0]), c12 (W[0,1]), c21 (W[1,0]), c22 (W[1,1])?
    # c11 = m1 - m2 + m4 + m6
    # c12 = m3 + m5 + m7
    # c21 = m2 + m4 - m7
    # c22 = m1 - m3 + m5 + m6
    
    # Let's build U, V, W for Strassen.
    # r=0 (m1):
    #   A part: a11+a22 -> U[0,0]=1, U[3,3]=1? No, U index is i*m+j.
    #   U[0,0] (a11) = 1, U[3,3] (a22) = 1.
    #   B part: b11+b22 -> V[0,0] (b11) = 1, V[3,3] (b22) = 1.
    #   C part: c11 (W[0,0]) = 1, c12 (W[0,1]) = 1, c21 (W[1,0]) = 1, c22 (W[1,1]) = 1.
    
    # r=1 (m2):
    #   A: a21+a22 -> U[2,2]=1, U[3,3]=1.
    #   B: b11 -> V[0,0]=1.
    #   C: c11 (W[0,0]) = -1, c21 (W[1,0]) = 1.
    
    # r=2 (m3):
    #   A: a11 -> U[0,0]=1.
    #   B: b12-b22 -> V[0,1]=1, V[3,3]=-1.
    #   C: c12 (W[0,1]) = 1, c22 (W[1,1]) = -1.
    
    # r=3 (m4):
    #   A: a22 -> U[3,3]=1.
    #   B: b21-b11 -> V[2,2]=1, V[0,0]=-1.
    #   C: c11 (W[0,0]) = 1, c21 (W[1,0]) = 1.
    
    # r=4 (m5):
    #   A: a11+a12 -> U[0,0]=1, U[0,1]=1.
    #   B: b22 -> V[3,3]=1.
    #   C: c12 (W[0,1]) = 1, c22 (W[1,1]) = 1.
    
    # r=5 (m6):
    #   A: a21-a11 -> U[2,2]=1, U[0,0]=-1.
    #   B: b11+b12 -> V[0,0]=1, V[0,1]=1.
    #   C: c11 (W[0,0]) = 1, c22 (W[1,1]) = 1.
    
    # r=6 (m7):
    #   A: a12-a22 -> U[0,1]=1, U[3,3]=-1.
    #   B: b21+b22 -> V[2,2]=1, V[3,3]=1.
    #   C: c12 (W[0,1]) = 1, c21 (W[1,0]) = -1.

    # Let's verify c11: m1(1) - m2(-1) + m4(1) + m6(1) = 1+1+1+1 = 4? No.
    # W[0,0] for r=0 is 1. r=1 is -1. r=3 is 1. r=5 is 1. Sum = 1-1+1+1 = 2.
    # But c11 should be m1 - m2 + m4 + m6.
    # So W[0,0] should be 1 for r=0, -1 for r=1, 1 for r=3, 1 for r=5.
    # My W assignments:
    # r=0: W[0,0]=1
    # r=1: W[0,0]=-1
    # r=2: W[0,0]=0
    # r=3: W[0,0]=1
    # r=4: W[0,0]=0
    # r=5: W[0,0]=1
    # r=6: W[0,0]=0
    # Sum = 1 - 1 + 0 + 1 + 0 + 1 + 0 = 2.
    # Wait, the bilinear form is sum_r U V W.
    # For c11, we sum over r.
    # The coefficient of m_r in c11 is W[0,0, r] * (U part) * (V part).
    # But U and V are already factored into m_r.
    # So yes, W[0,0, r] should be the coefficient of m_r in c11.
    # c11 = m1 - m2 + m4 + m6.
    # So W[0,0, 0] = 1, W[0,0, 1] = -1, W[0,0, 3] = 1, W[0,0, 5] = 1.
    # This matches my assignment.
    
    # Now for 4x4x4.
    # We use Kronecker product.
    # A is 4x4, B is 4x4.
    # A = [[A11, A12], [A21, A22]] where Aij are 2x2.
    # B = [[B11, B12], [B21, B22]] where Bij are 2x2.
    # C = A B = [[A11B11 + A12B21, A11B12 + A12B22], [A21B11 + A22B21, A21B12 + A22B22]]
    # This is exactly the Strassen structure applied to the 2x2 blocks.
    # So we can use the same rank-7 decomposition, but applied to the 2x2 blocks.
    # The total rank is 7 * 7 = 49.
    
    # The indices for 4x4:
    # A[i, j] where i, j in 0..3.
    # i = 2*i1 + i2, j = 2*j1 + j2.
    # U_4[i, j, r] where r = 7*r1 + r2.
    # U_4[i, j, r] = U_2[i1, j1, r1] * U_2[i2, j2, r2].
    # Similarly for V and W.
    
    # Let's build the 2x2 factors first.
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # r=0
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[1][0] = 1; W2[2][0] = 1; W2[3][0] = 1
    
    # r=1
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[0][1] = -1; W2[2][1] = 1
    
    # r=2
    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    W2[1][2] = 1; W2[3][2] = -1
    
    # r=3
    U2[3][3] = 1
    V2[2][3] = 1; V2[0][3] = -1
    W2[0][3] = 1; W2[2][3] = 1
    
    # r=4
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[1][4] = 1; W2[3][4] = 1
    
    # r=5
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    W2[0][5] = 1; W2[3][5] = 1
    
    # r=6
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    W2[1][6] = 1; W2[2][6] = -1

    # Now build 4x4 factors.
    rank = 49
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]
    
    for i in range(4):
        for j in range(4):
            i1, i2 = i // 2, i % 2
            j1, j2 = j // 2, j % 2
            for r1 in range(7):
                for r2 in range(7):
                    r = r1 * 7 + r2
                    U[i * 4 + j][r] = U2[i1 * 2 + j1][r1] * U2[i2 * 2 + j2][r2]
                    
    for j in range(4):
        for k in range(4):
            j1, j2 = j // 2, j % 2
            k1, k2 = k // 2, k % 2
            for r1 in range(7):
                for r2 in range(7):
                    r = r1 * 7 + r2
                    V[j * 4 + k][r] = V2[j1 * 2 + k1][r1] * V2[j2 * 2 + k2][r2]
                    
    for k in range(4):
        for i in range(4):
            k1, k2 = k // 2, k % 2
            i1, i2 = i // 2, i % 2
            for r1 in range(7):
                for r2 in range(7):
                    r = r1 * 7 + r2
                    W[k * 4 + i][r] = W2[k1 * 2 + i1][r1] * W2[k2 * 2 + i2][r2]
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
