"""Rank-55 tensor decomposition for 2x3 @ 3x10 matrix multiplication."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    # Dimensions: A is 2x3, B is 3x10, C is 2x10
    n, m, p = 2, 3, 10
    
    # Strategy: Split middle dimension into blocks of size 2
    # Block 1: j=0,1 -> 2x2 subproblem, use Strassen (rank 7 per block of 2 output columns)
    # Block 2: j=2 -> 1x1 subproblem, schoolbook (rank 1 per output column)
    
    # For each pair of output columns (k0, k1) where k0=2l, k1=2l+1:
    #   Strassen gives 7 multiplications for A[0:2, 0:2] @ B[0:2, k0:k1+1]
    #   Schoolbook gives 2 multiplications for A[0:2, 2] @ B[2, k0:k1+1]
    # Total per pair: 9, and 5 pairs = 45
    # Plus for the last pair (k=8,9) same structure
    # Wait, 10 columns -> 5 pairs, each pair needs 7 (Strassen) + 2 (schoolbook for j=2) = 9
    # Total rank = 5 * 9 = 45
    
    # Let me recount:
    # For columns k0, k1:
    #   Strassen on 2x2 matrices: rank 7
    #   Schoolbook for j=2: A[i,2]*B[2,k0] and A[i,2]*B[2,k1] for i=0,1 -> rank 4
    #   Total per pair: 7 + 4 = 11
    #   5 pairs: 55
    
    rank = 55
    U_mat = [[0] * rank for _ in range(n * m)]  # 6 x 55
    V_mat = [[0] * rank for _ in range(m * p)]  # 30 x 55
    W_mat = [[0] * rank for _ in range(p * n)]  # 20 x 55
    
    idx = 0
    
    for l in range(5):
        k0 = 2 * l
        k1 = 2 * l + 1
        
        # Index helpers
        # A indices: A[i,j] -> i*m + j = i*3 + j
        a00 = 0  # A[0,0]
        a01 = 1  # A[0,1]
        a10 = 3  # A[1,0]
        a11 = 4  # A[1,1]
        
        # B indices: B[j,k] -> j*p + k = j*10 + k
        b0k0 = 0 * 10 + k0  # B[0,k0]
        b0k1 = 0 * 10 + k1  # B[0,k1]
        b1k0 = 1 * 10 + k0  # B[1,k0]
        b1k1 = 1 * 10 + k1  # B[1,k1]
        
        # C indices: C[i,k] -> k*n + i = k*2 + i
        c0k0 = k0 * 2 + 0  # C[0,k0]
        c0k1 = k1 * 2 + 0  # C[0,k1]
        c1k0 = k0 * 2 + 1  # C[1,k0]
        c1k1 = k1 * 2 + 1  # C[1,k1]
        
        # Strassen 2x2 multiplication (7 multiplications)
        # M1 = (A[0,0] + A[1,1]) * (B[0,k0] + B[1,k1])
        U_mat[a00][idx] += 1
        U_mat[a11][idx] += 1
        V_mat[b0k0][idx] += 1
        V_mat[b1k1][idx] += 1
        W_mat[c0k0][idx] += 1
        W_mat[c1k1][idx] += 1
        idx += 1
        
        # M2 = (A[1,0] + A[1,1]) * B[0,k0]
        U_mat[a10][idx] += 1
        U_mat[a11][idx] += 1
        V_mat[b0k0][idx] += 1
        W_mat[c1k0][idx] += 1
        W_mat[c1k1][idx] -= 1
        idx += 1
        
        # M3 = A[0,0] * (B[0,k1] - B[1,k1])
        U_mat[a00][idx] += 1
        V_mat[b0k1][idx] += 1
        V_mat[b1k1][idx] -= 1
        W_mat[c0k1][idx] += 1
        W_mat[c1k1][idx] += 1
        idx += 1
        
        # M4 = A[1,1] * (B[1,k0] - B[0,k0])
        U_mat[a11][idx] += 1
        V_mat[b1k0][idx] += 1
        V_mat[b0k0][idx] -= 1
        W_mat[c0k0][idx] += 1
        W_mat[c1k0][idx] += 1
        idx += 1
        
        # M5 = (A[0,0] + A[0,1]) * B[1,k1]
        U_mat[a00][idx] += 1
        U_mat[a01][idx] += 1
        V_mat[b1k1][idx] += 1
        W_mat[c0k0][idx] -= 1
        W_mat[c0k1][idx] += 1
        idx += 1
        
        # M6 = (A[1,0] - A[0,0]) * (B[0,k0] + B[0,k1])
        U_mat[a10][idx] += 1
        U_mat[a00][idx] -= 1
        V_mat[b0k0][idx] += 1
        V_mat[b0k1][idx] += 1
        W_mat[c1k1][idx] += 1
        idx += 1
        
        # M7 = (A[0,1] - A[1,1]) * (B[1,k0] + B[1,k1])
        U_mat[a01][idx] += 1
        U_mat[a11][idx] -= 1
        V_mat[b1k0][idx] += 1
        V_mat[b1k1][idx] += 1
        W_mat[c0k0][idx] += 1
        idx += 1
        
        # Schoolbook for j=2: 4 multiplications
        # C[i,k0] += A[i,2] * B[2,k0]
        # C[i,k1] += A[i,2] * B[2,k1]
        for i in range(2):
            for k in (k0, k1):
                u_idx = i * 3 + 2
                v_idx = 2 * 10 + k
                w_idx = k * 2 + i
                U_mat[u_idx][idx] += 1
                V_mat[v_idx][idx] += 1
                W_mat[w_idx][idx] += 1
                idx += 1
    
    return {'U': U_mat, 'V': V_mat, 'W': W_mat}
# EVOLVE-BLOCK-END
