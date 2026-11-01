"""Matrix multiplication 2x2x15 using block-wise Strassen decomposition."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Split p=15 into seven 2-column blocks and one 1-column block
    # 7 blocks of size 2x2, 1 block of size 2x1
    # Strassen for 2x2x2: rank 7
    # Standard for 2x2x1: rank 4
    # Total rank: 7*7 + 4 = 53
    
    rank = 53
    U = [[0] * rank for _ in range(n * m)]  # (4, 53)
    V = [[0] * rank for _ in range(m * p)]  # (30, 53)
    W = [[0] * rank for _ in range(p * n)]  # (30, 53)
    
    # Strassen 2x2x2 decomposition
    # For matrices A (2x2), B (2x2), C (2x2):
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # Strassen coefficient table: (U_coeff, V_coeff, W_coeff)
    # U indices: 0=A11, 1=A12, 2=A21, 3=A22
    # V indices: 0=B11, 1=B12, 2=B21, 3=B22
    # W indices: 0=C11, 1=C12, 2=C21, 3=C22 (using W[k*n+i] convention)
    # W[0] = C11 (k=0,i=0), W[1] = C21 (k=0,i=1), W[2] = C12 (k=1,i=0), W[3] = C22 (k=1,i=1)
    
    strassen_terms = [
        # M1: U=[1,0,0,1], V=[1,0,0,1], W=[1,0,0,0] -> C11
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 0]),
        # M2: U=[0,0,1,1], V=[1,0,0,0], W=[0,1,0,0] -> C21
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, 0]),
        # M3: U=[1,0,0,0], V=[0,0,1,-1], W=[0,0,1,0] -> C12
        ([1, 0, 0, 0], [0, 0, 1, -1], [0, 0, 1, 0]),
        # M4: U=[0,0,0,1], V=[0,1,-1,0], W=[1,0,1,0] -> C11 and C21
        ([0, 0, 0, 1], [0, 1, -1, 0], [1, 0, 1, 0]),
        # M5: U=[1,1,0,0], V=[0,0,0,1], W=[-1,0,0,0] -> C11
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 0, 0]),
        # M6: U=[-1,0,1,0], V=[1,1,0,0], W=[0,0,0,1] -> C22
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        # M7: U=[0,1,0,-1], V=[0,0,1,1], W=[1,0,0,1] -> C11 and C22
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 1]),
    ]
    
    # Build U, V, W for all 7 Strassen blocks and 1 standard block
    col = 0
    for block_idx in range(7):
        # Each block covers columns 2*block_idx and 2*block_idx+1 of B and C
        k_offset = 2 * block_idx
        for u_coeffs, v_coeffs, w_coeffs in strassen_terms:
            for i in range(2):
                for j in range(2):
                    if u_coeffs[i * 2 + j] != 0:
                        U[i * m + j][col] = u_coeffs[i * 2 + j]
            for j in range(2):
                for k in range(2):
                    if v_coeffs[j * 2 + k] != 0:
                        V[j * p + (k_offset + k)][col] = v_coeffs[j * 2 + k]
            for k in range(2):
                for i in range(2):
                    if w_coeffs[k * 2 + i] != 0:
                        W[(k_offset + k) * n + i][col] = w_coeffs[k * 2 + i]
            col += 1
    
    # Standard 2x2x1 decomposition for the last column (k=14)
    # C11 = A11*B11 + A12*B21
    # C21 = A21*B11 + A22*B21
    k_offset = 14
    # Term 1: A11 * B11 -> C11
    U[0][col] = 1
    V[0 * p + k_offset][col] = 1
    W[k_offset * n + 0][col] = 1
    col += 1
    # Term 2: A12 * B21 -> C11
    U[1][col] = 1
    V[1 * p + k_offset][col] = 1
    W[k_offset * n + 0][col] = 1
    col += 1
    # Term 3: A21 * B11 -> C21
    U[2][col] = 1
    V[0 * p + k_offset][col] = 1
    W[k_offset * n + 1][col] = 1
    col += 1
    # Term 4: A22 * B21 -> C21
    U[3][col] = 1
    V[1 * p + k_offset][col] = 1
    W[k_offset * n + 1][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
