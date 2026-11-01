import time

DIMENSIONS = [2, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    # We split m=3 into 1 and 2.
    # A is 2x3, B is 3x8.
    # A1 is 2x1 (col 0), A2 is 2x2 (cols 1,2).
    # B1 is 1x8 (row 0), B2 is 2x8 (rows 1,2).
    # C = A1 B1 + A2 B2.
    
    # Part 1: A1 B1 (2x1 * 1x8) -> Rank 16
    # Part 2: A2 B2 (2x2 * 2x8) -> Split B2 into two 2x4 blocks.
    # B2a is 2x4 (cols 0-3), B2b is 2x4 (cols 4-7).
    # A2 B2a and A2 B2b.
    # Each 2x2 * 2x4 uses Strassen (Rank 14).
    
    rank = 16 + 14 + 14
    U = [[0] * rank for _ in range(n * m)] # 6 x rank
    V = [[0] * rank for _ in range(m * p)] # 24 x rank
    W = [[0] * rank for _ in range(p * n)] # 16 x rank
    
    col = 0
    
    # Helper to add a rank-1 term
    def add_term(u_idx, v_idx, w_idx, val=1):
        nonlocal col
        U[u_idx][col] = val
        V[v_idx][col] = val
        W[w_idx][col] = val
        col += 1

    # Part 1: A1 B1
    # A1 indices: (0,0), (1,0) in A (row, col). A is 2x3.
    # A[i, 0] for i=0,1.
    # B[0, k] for k=0..7. B is 3x8.
    # C[i, k] = A[i,0] * B[0,k].
    # U index: i*m + 0 = i*3.
    # V index: 0*p + k = k.
    # W index: k*n + i = k*2 + i.
    for i in range(n):
        for k in range(p):
            add_term(i * m + 0, 0 * p + k, k * n + i)
            
    # Part 2: A2 B2
    # A2 indices: A[i, 1], A[i, 2].
    # B2 indices: B[1, k], B[2, k].
    # Split B2 into B2a (k=0..3) and B2b (k=4..7).
    
    # Strassen for 2x2 * 2x4
    # A2 = [[a11, a12], [a21, a22]] corresponding to A[0,1], A[0,2], A[1,1], A[1,2].
    # B_block = [[b11, b12, b13, b14], [b21, b22, b23, b24]]
    # B_block rows correspond to B[1, k] and B[2, k].
    
    def strassen_2x2x4(offset_k):
        # offset_k is 0 for B2a, 4 for B2b.
        # A entries:
        a11 = 0 * m + 1 # A[0,1]
        a12 = 0 * m + 2 # A[0,2]
        a21 = 1 * m + 1 # A[1,1]
        a22 = 1 * m + 2 # A[1,2]
        
        # B entries for k in offset_k .. offset_k+3
        # B[1, k] and B[2, k].
        # V indices: 1*p + k and 2*p + k.
        
        # C entries: C[i, k]. W indices: k*n + i.
        
        # Standard Strassen for 2x2 * 2x2 gives 2x2 result.
        # Here result is 2x4.
        # We can run Strassen on B[1, k: k+2] and B[2, k: k+2]?
        # No, Strassen splits the 2x2 matrix B into 2x2 blocks.
        # Here B is 2x4. We can split it into two 2x2 blocks: B_L and B_R.
        # B_L is B[1, k:k+2], B[2, k:k+2].
        # B_R is B[1, k+2:k+4], B[2, k+2:k+4].
        
        # So we run Strassen for B_L and Strassen for B_R.
        # Each Strassen run uses 7 mults.
        # Total 14 mults.
        
        for block_idx in range(2):
            k_start = offset_k + block_idx * 2
            # B_L/R entries:
            # b11 = B[1, k_start]
            # b12 = B[1, k_start+1]
            # b21 = B[2, k_start]
            # b22 = B[2, k_start+1]
            
            # V indices:
            v11 = 1 * p + k_start
            v12 = 1 * p + k_start + 1
            v21 = 2 * p + k_start
            v22 = 2 * p + k_start + 1
            
            # C entries:
            # C[0, k_start] = C11
            # C[0, k_start+1] = C12
            # C[1, k_start] = C21
            # C[1, k_start+1] = C22
            
            w11 = k_start * n + 0
            w12 = (k_start + 1) * n + 0
            w21 = k_start * n + 1
            w22 = (k_start + 1) * n + 1
            
            # Strassen 7 mults:
            # M1 = (a11 + a22) * (b11 + b22)
            # M2 = (a21 + a22) * b11
            # M3 = a11 * (b22 - b11)
            # M4 = a22 * (b22 - b11)
            # M5 = (a11 + a12) * b22
            # M6 = (a21 - a11) * (b11 + b12)
            # M7 = (a12 - a22) * (b21 + b22)
            
            # We need to express these as linear combinations.
            # U, V, W matrices store coefficients.
            
            # M1
            U[a11][col] += 1
            U[a22][col] += 1
            V[v11][col] += 1
            V[v22][col] += 1
            W[w11][col] += 1
            W[w22][col] += 1
            W[w12][col] -= 1
            W[w21][col] -= 1
            col += 1
            
            # M2
            U[a21][col] += 1
            U[a22][col] += 1
            V[v11][col] += 1
            W[w21][col] += 1
            W[w22][col] += 1
            col += 1
            
            # M3
            U[a11][col] += 1
            V[v22][col] += 1
            V[v11][col] -= 1
            W[w11][col] -= 1
            W[w12][col] += 1
            col += 1
            
            # M4
            U[a22][col] += 1
            V[v22][col] += 1
            V[v11][col] -= 1
            W[w21][col] -= 1
            W[w22][col] += 1
            col += 1
            
            # M5
            U[a11][col] += 1
            U[a12][col] += 1
            V[v22][col] += 1
            W[w12][col] += 1
            col += 1
            
            # M6
            U[a21][col] += 1
            U[a11][col] -= 1
            V[v11][col] += 1
            V[v12][col] += 1
            W[w22][col] -= 1
            col += 1
            
            # M7
            U[a12][col] += 1
            U[a22][col] -= 1
            V[v21][col] += 1
            V[v22][col] += 1
            W[w11][col] += 1
            col += 1

    strassen_2x2x4(0)
    strassen_2x2x4(4)
    
    return {"U": U, "V": V, "W": W}
