"""Strassen-based decomposition for 2x3x10 matrix multiplication."""

def construct():
    # For multiplying (2,3) by (3,10):
    # A is 2x3, B is 3x10, C = A*B is 2x10
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # We can split B into 5 blocks of 3x2 each: B = [B_0, B_1, B_2, B_3, B_4]
    # where B_t is 3x2 for t=0..4
    # Then C = [A*B_0, A*B_1, A*B_2, A*B_3, A*B_4]
    #
    # Each A*B_t is 2x2. We use Strassen's algorithm for each 2x2x2 block.
    # Strassen reduces 2x2x2 from rank 8 to rank 7.
    # Total rank = 5 * 7 = 35
    #
    # Strassen's algorithm for 2x2 matrices:
    # M1 = (A00 + A11)*(B00 + B11)
    # M2 = (A10 + A11)*B00
    # M3 = A00*(B01 - B11)
    # M4 = A11*(B10 - B00)
    # M5 = (A00 + A01)*B11
    # M6 = (A10 - A00)*(B00 + B01)
    # M7 = (A01 - A11)*(B10 + B11)
    #
    # C00 = M1 - M4 + M6
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M7

    n, m, p = 2, 3, 10
    # U has shape (n*m, rank) = (6, rank)
    # V has shape (m*p, rank) = (30, rank)
    # W has shape (p*n, rank) = (20, rank)
    # Indexing: U[i*m + j, c], V[j*p + k, c], W[k*n + i, c]

    rank = 35
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # For each 3x2 block t (t=0..4), we have 7 Strassen terms
    # Block t corresponds to columns k = 2*t, 2*t+1 of B (and C)
    # B[j, 2*t] and B[j, 2*t+1] for j=0,1,2
    # A[i, j] for i=0,1 and j=0,1,2
    
    # Strassen indices within block t:
    # Term s (s=0..6):
    # We need to express each term as sum of products U * V * W
    
    # For a 2x2x2 Strassen term, the matrices are:
    # A is 2x2 (but our A is 2x3, so we need to handle the 3rd column)
    # Actually, wait - we're multiplying 2x3 by 3x2, not 2x2 by 2x2.
    # So we need to think more carefully.
    
    # Let me reconsider. A is 2x3, B_t is 3x2.
    # A = [[a00, a01, a02], [a10, a11, a12]]
    # B_t = [[b00, b01], [b10, b11], [b20, b21]]
    # C_t = A * B_t = [[sum_j a0j*bj0, sum_j a0j*bj1], [sum_j a1j*bj0, sum_j a1j*bj1]]
    
    # This is a 2x3x2 tensor, not 2x2x2. The rank of 2x3x2 is not simply 7.
    # The trivial rank is 2*3*2 = 12.
    
    # The knowledge says: "Using Strassen's algorithm for the 2x2 part of the 2x3x2 block 
    # and trivial multiplication for the 2x1 part reduces the rank per p=2 block from 12 to 11"
    
    # So for each 3x2 block, we can achieve rank 11 instead of 12.
    # Total rank = 5 * 11 = 55.
    
    # Let me decompose the 2x3x2 problem:
    # C[0,0] = a00*b00 + a01*b10 + a02*b20
    # C[0,1] = a00*b01 + a01*b11 + a02*b21
    # C[1,0] = a10*b00 + a11*b10 + a12*b20
    # C[1,1] = a10*b01 + a11*b11 + a12*b21
    
    # Idea: Split A into A_2 (2x2, first two columns) and A_1 (2x1, third column)
    # A = [A_2 | A_1] where A_2 is 2x2, A_1 is 2x1
    # B_t = [B_2; B_1] where B_2 is 2x2, B_1 is 1x2
    # C_t = A_2 * B_2 + A_1 * B_1
    # 
    # A_2 * B_2 is 2x2x2: use Strassen (rank 7)
    # A_1 * B_1 is 2x1x2: trivial rank 4
    # Total: 7 + 4 = 11
    
    # Strassen for A_2 * B_2:
    # A_2 = [[a00, a01], [a10, a11]]
    # B_2 = [[b00, b01], [b10, b11]]
    # 
    # M1 = (a00 + a11)*(b00 + b11)
    # M2 = (a10 + a11)*b00
    # M3 = a00*(b01 - b11)
    # M4 = a11*(b10 - b00)
    # M5 = (a00 + a01)*b11
    # M6 = (a10 - a00)*(b00 + b01)
    # M7 = (a01 - a11)*(b10 + b11)
    #
    # C00 = M1 - M4 + M6
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M7
    
    # Trivial for A_1 * B_1:
    # A_1 = [[a02], [a12]]
    # B_1 = [[b20, b21]]
    # N1 = a02 * b20
    # N2 = a02 * b21
    # N3 = a12 * b20
    # N4 = a12 * b21
    #
    # C00 += N1
    # C01 += N2
    # C10 += N3
    # C11 += N4

    col = 0
    for t in range(5):
        # k indices for this block: k0 = 2*t, k1 = 2*t+1
        k0 = 2 * t
        k1 = 2 * t + 1
        
        # Strassen terms (7 terms)
        # Term 0: M1 = (a00 + a11)*(b00 + b11)
        # U: a00 and a11 -> U[0, col]=1, U[3, col]=1
        # V: b00 and b11 -> V[0, col]=1, V[1, col]=1 (j=0,k0 and j=1,k1)
        # W: C00 and C11 -> W[k0*2+0, col]=1, W[k1*2+1, col]=1
        U[0][col] = 1
        U[3][col] = 1
        V[0 * p + k0][col] = 1
        V[1 * p + k1][col] = 1
        W[k0 * n + 0][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
        
        # Term 1: M2 = (a10 + a11)*b00
        # U: a10 and a11 -> U[3, col]=1, U[4, col]=1
        # V: b00 -> V[0, col]=1
        # W: C10 -> W[k0*2+1, col]=1
        U[3][col] = 1
        U[4][col] = 1
        V[0 * p + k0][col] = 1
        W[k0 * n + 1][col] = 1
        col += 1
        
        # Term 2: M3 = a00*(b01 - b11)
        # U: a00 -> U[0, col]=1
        # V: b01 and -b11 -> V[0, col]=1, V[1, col]=-1
        # W: C01 -> W[k1*2+0, col]=1
        U[0][col] = 1
        V[0 * p + k1][col] = 1
        V[1 * p + k1][col] = -1
        W[k1 * n + 0][col] = 1
        col += 1
        
        # Term 3: M4 = a11*(b10 - b00)
        # U: a11 -> U[4, col]=1
        # V: b10 and -b00 -> V[1, col]=1, V[0, col]=-1
        # W: C00 and C10 -> W[k0*2+0, col]=-1, W[k0*2+1, col]=1
        U[4][col] = 1
        V[1 * p + k0][col] = 1
        V[0 * p + k0][col] = -1
        W[k0 * n + 0][col] = -1
        W[k0 * n + 1][col] = 1
        col += 1
        
        # Term 4: M5 = (a00 + a01)*b11
        # U: a00 and a01 -> U[0, col]=1, U[1, col]=1
        # V: b11 -> V[1, col]=1
        # W: C01 -> W[k1*2+0, col]=1
        U[0][col] = 1
        U[1][col] = 1
        V[1 * p + k1][col] = 1
        W[k1 * n + 0][col] = 1
        col += 1
        
        # Term 5: M6 = (a10 - a00)*(b00 + b01)
        # U: a10 and -a00 -> U[3, col]=1, U[0, col]=-1
        # V: b00 and b01 -> V[0, col]=1, V[0, col]=1 (same j=0)
        # W: C00 -> W[k0*2+0, col]=1
        U[3][col] = 1
        U[0][col] = -1
        V[0 * p + k0][col] = 1
        V[0 * p + k1][col] = 1
        W[k0 * n + 0][col] = 1
        col += 1
        
        # Term 6: M7 = (a01 - a11)*(b10 + b11)
        # U: a01 and -a11 -> U[1, col]=1, U[4, col]=-1
        # V: b10 and b11 -> V[1, col]=1, V[1, col]=1 (same j=1)
        # W: C11 -> W[k1*2+1, col]=1
        U[1][col] = 1
        U[4][col] = -1
        V[1 * p + k0][col] = 1
        V[1 * p + k1][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
        
        # Trivial terms for A_1 * B_1 (4 terms)
        # Term 7: N1 = a02 * b20
        # U: a02 -> U[2, col]=1
        # V: b20 -> V[2, col]=1
        # W: C00 -> W[k0*2+0, col]=1
        U[2][col] = 1
        V[2 * p + k0][col] = 1
        W[k0 * n + 0][col] = 1
        col += 1
        
        # Term 8: N2 = a02 * b21
        # U: a02 -> U[2, col]=1
        # V: b21 -> V[2, col]=1
        # W: C01 -> W[k1*2+0, col]=1
        U[2][col] = 1
        V[2 * p + k1][col] = 1
        W[k1 * n + 0][col] = 1
        col += 1
        
        # Term 9: N3 = a12 * b20
        # U: a12 -> U[5, col]=1
        # V: b20 -> V[2, col]=1
        # W: C10 -> W[k0*2+1, col]=1
        U[5][col] = 1
        V[2 * p + k0][col] = 1
        W[k0 * n + 1][col] = 1
        col += 1
        
        # Term 10: N4 = a12 * b21
        # U: a12 -> U[5, col]=1
        # V: b21 -> V[2, col]=1
        # W: C11 -> W[k1*2+1, col]=1
        U[5][col] = 1
        V[2 * p + k1][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1

    return {"U": U, "V": V, "W": W}
