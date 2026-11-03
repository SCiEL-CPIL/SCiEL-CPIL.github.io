import numpy as np

DIMENSIONS = [2, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    # Rank 22: 8 for j=0, 14 for j=1,2
    
    rank = 22
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Part 1: j=0 (2x1x8) rank 8
    # C[i,k] += A[i,0] B[0,k]
    # Use 8 terms, one for each k
    for k in range(8):
        # U[i*m + j, col]
        # j=0
        U[0 * m + 0][col] = 1
        U[1 * m + 0][col] = 1
        # V[j*p + k, col]
        # j=0
        V[0 * p + k][col] = 1
        # W[k*n + i, col]
        # i=0,1
        W[k * n + 0][col] = 1
        W[k * n + 1][col] = 1
        col += 1
        
    # Part 2: j=1,2 (2x2x8) rank 14
    # Split into two 2x2x4
    # B[1:3, 0:4] and B[1:3, 4:8]
    
    # Strassen for 2x2x4
    # A is 2x2, B is 2x4
    # C is 2x4
    # Rank 7
    
    def strassen_2x2x4(offset_b, offset_w):
        # offset_b: column offset in B for the 2x4 block
        # offset_w: index offset in W for the 2x4 block?
        # W index is k*n + i. k is 0..3 for first block, 4..7 for second.
        # So we just use the actual k indices.
        nonlocal col
        # A indices: i=0,1; j=1,2
        # B indices: j=1,2; k=0..3 (relative)
        
        # Strassen products for 2x2:
        # m1 = (a11 + a22)(b11 + b22)
        # m2 = (a21 + a22)b11
        # m3 = a11(b12 - b22)
        # m4 = a22(b21 - b11)
        # m5 = (a11 + a12)b22
        # m6 = (a21 - a11)(b11 + b12)
        # m7 = (a12 - a22)(b21 + b22)
        # c11 = m1 + m4 - m5 + m7
        # c12 = m3 + m5
        # c21 = m2 + m4
        # c22 = m1 - m2 + m3 + m6
        
        # We need to map this to the tensor decomposition.
        # A is 2x2 (rows i=0,1; cols j=1,2)
        # B is 2x4 (rows j=1,2; cols k=0..3)
        
        # For each of the 7 Strassen terms, we add a rank-1 term.
        # The term r corresponds to one of the m's.
        # U, V, W entries are determined by the coefficients in the Strassen formulas.
        
        # Let's define the 7 terms.
        # Term 0: m1
        # A part: (A[0,1] + A[1,2]) -> U[0*m+1]=1, U[1*m+2]=1
        # B part: (B[1,0] + B[1,1] + ...)? No.
        # B is 2x4. b11 is B[1,0], b22 is B[2,3]?
        # Strassen is for 2x2. Here B is 2x4.
        # We can apply Strassen to the 2x2 blocks of B?
        # B = [B_left, B_right] where B_left is 2x2, B_right is 2x2.
        # C = [A B_left, A B_right].
        # We need 7 terms for B_left and 7 for B_right. Total 14.
        
        # So we call strassen_2x2x2 twice?
        pass

    # Actually, let's just implement the 2x2x2 Strassen and apply it to 2x2x4 by processing 2 columns at a time.
    
    def strassen_2x2x2(block_k_start):
        nonlocal col
        # block_k_start is 0 or 4
        # B cols are block_k_start, block_k_start+1
        # B rows are 1, 2
        
        # m1 = (A01 + A12) * (B1k1 + B2k2)
        # A01 is A[0,1], A12 is A[1,2]
        # B1k1 is B[1, k1], B2k2 is B[2, k2]
        # k1 = block_k_start, k2 = block_k_start + 1
        
        # Term 1: m1
        U[0*m+1][col] = 1
        U[1*m+2][col] = 1
        V[1*p + block_k_start][col] = 1
        V[2*p + block_k_start+1][col] = 1
        # W: c11, c12, c21, c22 contributions
        # c11 += m1 -> W[k1*n+0][col] += 1
        # c22 += m1 -> W[k2*n+1][col] += 1
        W[block_k_start*n + 0][col] = 1
        W[(block_k_start+1)*n + 1][col] = 1
        col += 1
        
        # Term 2: m2 = (A11 + A12) * B1k1
        U[0*m+1][col] = 1
        U[1*m+2][col] = 1
        V[1*p + block_k_start][col] = 1
        # c21 += m2 -> W[k1*n+1][col] += 1
        W[block_k_start*n + 1][col] = 1
        col += 1
        
        # Term 3: m3 = A01 * (B1k2 - B2k2)
        U[0*m+1][col] = 1
        V[1*p + block_k_start+1][col] = 1
        V[2*p + block_k_start+1][col] = -1
        # c12 += m3 -> W[k2*n+0][col] += 1
        W[(block_k_start+1)*n + 0][col] = 1
        col += 1
        
        # Term 4: m4 = A12 * (B2k1 - B1k1)
        U[1*m+2][col] = 1
        V[2*p + block_k_start][col] = 1
        V[1*p + block_k_start][col] = -1
        # c11 += m4 -> W[k1*n+0][col] += 1
        # c21 += m4 -> W[k1*n+1][col] += 1
        W[block_k_start*n + 0][col] = 1
        W[block_k_start*n + 1][col] = 1
        col += 1
        
        # Term 5: m5 = (A01 + A02) * B2k2
        U[0*m+1][col] = 1
        U[0*m+2][col] = 1
        V[2*p + block_k_start+1][col] = 1
        # c11 -= m5 -> W[k1*n+0][col] -= 1
        # c12 += m5 -> W[k2*n+0][col] += 1
        W[block_k_start*n + 0][col] = -1
        W[(block_k_start+1)*n + 0][col] = 1
        col += 1
        
        # Term 6: m6 = (A11 - A01) * (B1k1 + B1k2)
        U[1*m+1][col] = 1
        U[0*m+1][col] = -1
        V[1*p + block_k_start][col] = 1
        V[1*p + block_k_start+1][col] = 1
        # c22 += m6 -> W[k2*n+1][col] += 1
        W[(block_k_start+1)*n + 1][col] = 1
        col += 1
        
        # Term 7: m7 = (A02 - A12) * (B2k1 + B2k2)
        U[0*m+2][col] = 1
        U[1*m+2][col] = -1
        V[2*p + block_k_start][col] = 1
        V[2*p + block_k_start+1][col] = 1
        # c11 += m7 -> W[k1*n+0][col] += 1
        # c22 -= m7 -> W[k2*n+1][col] -= 1
        W[block_k_start*n + 0][col] = 1
        W[(block_k_start+1)*n + 1][col] = -1
        col += 1

    strassen_2x2x2(0)
    strassen_2x2x2(4)
    
    return {"U": U, "V": V, "W": W}
