"""Rewritten program exploiting transposed problem M_{10,3,2} with block-diagonal decomposition."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 10
    rank = 55
    
    # Build U, V, W for original problem:
    # U: (n*m, rank) = (6, 55)  -- A[i,j] indexed by i*m+j
    # V: (m*p, rank) = (30, 55) -- B[j,k] indexed by j*p+k
    # W: (p*n, rank) = (20, 55) -- C[i,k] indexed by k*n+i (trace convention)
    
    U_mat = [[0] * rank for _ in range(n * m)]
    V_mat = [[0] * rank for _ in range(m * p)]
    W_mat = [[0] * rank for _ in range(p * n)]
    
    idx = 0
    
    # Block 1: 2x2 sub-problem using Strassen (columns 0,1 of A; rows 0,1 of B)
    # A_sub is 2x2: A[i, 0:2] for i in {0,1}
    # B_sub is 2x10: B[0:2, :] 
    # We split B_sub into 5 blocks of 2 columns each
    
    for l in range(5):
        k0 = 2 * l
        k1 = 2 * l + 1
        
        # A indices in U: i * m + j
        a00 = 0 * m + 0  # 0
        a01 = 0 * m + 1  # 1
        a10 = 1 * m + 0  # 3
        a11 = 1 * m + 1  # 4
        
        # B indices in V: j * p + k
        b0k0 = 0 * p + k0
        b0k1 = 0 * p + k1
        b1k0 = 1 * p + k0
        b1k1 = 1 * p + k1
        
        # W indices: k * n + i
        c0k0 = k0 * n + 0
        c0k1 = k1 * n + 0
        c1k0 = k0 * n + 1
        c1k1 = k1 * n + 1
        
        # Strassen 7 multiplications for 2x2 * 2x2 block
        # M1 = (a00 + a11) * (b0k0 + b1k1)
        U_mat[a00][idx] += 1
        U_mat[a11][idx] += 1
        V_mat[b0k0][idx] += 1
        V_mat[b1k1][idx] += 1
        W_mat[c0k0][idx] += 1
        W_mat[c1k1][idx] += 1
        idx += 1
        
        # M2 = (a10 + a11) * b0k0
        U_mat[a10][idx] += 1
        U_mat[a11][idx] += 1
        V_mat[b0k0][idx] += 1
        W_mat[c1k0][idx] += 1
        W_mat[c1k1][idx] -= 1
        idx += 1
        
        # M3 = a00 * (b0k1 - b1k1)
        U_mat[a00][idx] += 1
        V_mat[b0k1][idx] += 1
        V_mat[b1k1][idx] -= 1
        W_mat[c0k1][idx] += 1
        W_mat[c1k1][idx] += 1
        idx += 1
        
        # M4 = a11 * (b1k0 - b0k0)
        U_mat[a11][idx] += 1
        V_mat[b1k0][idx] += 1
        V_mat[b0k0][idx] -= 1
        W_mat[c0k0][idx] += 1
        W_mat[c1k0][idx] += 1
        idx += 1
        
        # M5 = (a00 + a01) * b1k1
        U_mat[a00][idx] += 1
        U_mat[a01][idx] += 1
        V_mat[b1k1][idx] += 1
        W_mat[c0k0][idx] -= 1
        W_mat[c0k1][idx] += 1
        idx += 1
        
        # M6 = (a10 - a00) * (b0k0 + b0k1)
        U_mat[a10][idx] += 1
        U_mat[a00][idx] -= 1
        V_mat[b0k0][idx] += 1
        V_mat[b0k1][idx] += 1
        W_mat[c1k1][idx] += 1
        idx += 1
        
        # M7 = (a01 - a11) * (b1k0 + b1k1)
        U_mat[a01][idx] += 1
        U_mat[a11][idx] -= 1
        V_mat[b1k0][idx] += 1
        V_mat[b1k1][idx] += 1
        W_mat[c0k0][idx] += 1
        idx += 1
    
    # Block 2: Schoolbook for column 2 of A and row 2 of B
    # C[i,k] += A[i,2] * B[2,k]
    for i in range(n):
        for k in range(p):
            u_idx = i * m + 2
            v_idx = 2 * p + k
            w_idx = k * n + i
            
            U_mat[u_idx][idx] += 1
            V_mat[v_idx][idx] += 1
            W_mat[w_idx][idx] += 1
            idx += 1
    
    return {'U': U_mat, 'V': V_mat, 'W': W_mat}
# EVOLVE-BLOCK-END
