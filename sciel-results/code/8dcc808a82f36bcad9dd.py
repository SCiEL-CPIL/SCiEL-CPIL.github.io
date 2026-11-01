"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def _strassen_2x2():
    """
    Returns U, V, W for 2x2 matrix multiplication.
    U: shape (4, 7) corresponding to A indices (i, j) -> i*2+j
    V: shape (4, 7) corresponding to B indices (j, k) -> j*2+k
    W: shape (4, 7) corresponding to C indices (k, i) -> k*2+i  (Note: Task says W[k*n+i, r]. For n=2, W[k*2+i, r])
    
    Wait, the task says:
    U: (n*m, rank) -> (4, rank)
    V: (m*p, rank) -> (4, rank) for 2x2x2? No, for 2x2x2, m=2, p=2. So (4, rank).
    W: (p*n, rank) -> (4, rank). Index W[k*n+i, r].
    
    For 2x2x2:
    n=2, m=2, p=2.
    U indices: i in 0..1, j in 0..1. idx = i*m + j = i*2 + j.
    V indices: j in 0..1, k in 0..1. idx = j*p + k = j*2 + k.
    W indices: k in 0..1, i in 0..1. idx = k*n + i = k*2 + i.
    
    Strassen terms:
    M1 = (A00+A11)(B00+B11) -> C00 += 1, C11 += 1
    M2 = (A10+A11)B00 -> C10 += 1, C11 -= 1
    M3 = A00(B01-B11) -> C01 += 1, C11 += 1
    M4 = A11(B10-B00) -> C00 += 1, C10 += 1
    M5 = (A00+A01)B11 -> C00 -= 1, C01 += 1
    M6 = (A10-A00)(B00+B01) -> C11 += 1
    M7 = (A01-A11)(B10+B11) -> C00 += 1
    
    Let's verify C indices in W.
    W[k*n+i].
    C00 -> k=0, i=0 -> W[0]
    C01 -> k=1, i=0 -> W[2]
    C10 -> k=0, i=1 -> W[1]
    C11 -> k=1, i=1 -> W[3]
    
    So:
    M1: W[0]+=1, W[3]+=1
    M2: W[1]+=1, W[3]-=1
    M3: W[2]+=1, W[3]+=1
    M4: W[0]+=1, W[1]+=1
    M5: W[0]-=1, W[2]+=1
    M6: W[3]+=1
    M7: W[0]+=1
    """
    rank = 7
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    
    # u_terms[r][i*2+j]
    # v_terms[r][j*2+k]
    # w_terms[r][k*2+i]
    
    u = [
        [1, 0, 0, 1], # M1: A00, A11
        [0, 0, 1, 1], # M2: A10, A11
        [1, 0, 0, 0], # M3: A00
        [0, 0, 0, 1], # M4: A11
        [1, 1, 0, 0], # M5: A00, A01
        [-1, 0, 1, 0],# M6: -A00, A10
        [0, 1, 0, -1] # M7: A01, -A11
    ]
    
    v = [
        [1, 0, 0, 1], # M1: B00, B11
        [1, 0, 0, 0], # M2: B00
        [0, 1, 0, -1],# M3: B01, -B11
        [0, -1, 1, 0],# M4: -B00, B10
        [0, 0, 0, 1], # M5: B11
        [1, 1, 0, 0], # M6: B00, B01
        [0, 0, 1, 1]  # M7: B10, B11
    ]
    
    w = [
        [1, 0, 0, 1], # M1: C00, C11
        [0, 1, 0, -1],# M2: C10, -C11
        [0, 0, 1, 1], # M3: C01, C11
        [1, 1, 0, 0], # M4: C00, C10
        [-1, 0, 1, 0],# M5: -C00, C01
        [0, 0, 0, 1], # M6: C11
        [1, 0, 0, 0]  # M7: C00
    ]
    
    for r in range(rank):
        for idx in range(4):
            U[idx][r] = u[r][idx]
            V[idx][r] = v[r][idx]
            W[idx][r] = w[r][idx]
            
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=6
    
    # Get base 2x2 decomposition
    U_base, V_base, W_base = _strassen_2x2()
    rank_base = len(U_base[0]) # 7
    
    # Total rank
    rank = 3 * rank_base
    total_U_rows = n * m # 4
    total_V_rows = m * p # 12
    total_W_rows = p * n # 12
    
    U = [[0] * rank for _ in range(total_U_rows)]
    V = [[0] * rank for _ in range(total_V_rows)]
    W = [[0] * rank for _ in range(total_W_rows)]
    
    # We have 3 blocks of B, each 2x2.
    # B is 2x6. Columns 0-1 are block 0, 2-3 block 1, 4-5 block 2.
    # C is 2x6. Columns 0-1 are block 0, etc.
    
    for b in range(3):
        for r_local in range(rank_base):
            r_global = b * rank_base + r_local
            
            # Fill U
            # U is same for all blocks because A is shared?
            # Wait. The decomposition is for C = A * B.
            # If we split B into B1, B2, B3, then C = A*B1 + A*B2 + A*B3? No.
            # C = [A*B1, A*B2, A*B3].
            # The tensor is T_{i,j,k} = sum A_{i,j} B_{j,k}.
            # This is NOT a sum of independent tensors in the sense that A is the same.
            # However, the bilinear forms are independent for each k.
            # T_{i,j,k} depends on A_{i,j} and B_{j,k}.
            # For fixed k, it's a 2x2 matrix multiplication tensor.
            # Since k ranges over 6 values, and they don't overlap in B indices,
            # the tensor is the direct sum of 3 2x2x2 tensors.
            # Therefore, the U matrix for block b should be the SAME as U_base,
            # but placed in the same rows of U?
            # No, U has shape (n*m, rank).
            # The rank is the number of terms.
            # Term r_global uses A_{i,j} with coefficient U_base[i,j, r_local].
            # So U[i*m+j][r_global] = U_base[i*m+j][r_local].
            
            for i in range(n):
                for j in range(m):
                    idx_u = i * m + j
                    U[idx_u][r_global] = U_base[idx_u][r_local]
            
            # Fill V
            # V has shape (m*p, rank).
            # V index is j*p + k.
            # For block b, k is in range [2*b, 2*b+2).
            # k_local = k - 2*b.
            # V_base index is j*2 + k_local.
            for j in range(m):
                for k_local in range(2):
                    k_global = 2 * b + k_local
                    idx_v = j * p + k_global
                    idx_v_base = j * 2 + k_local
                    V[idx_v][r_global] = V_base[idx_v_base][r_local]
            
            # Fill W
            # W has shape (p*n, rank).
            # W index is k*n + i.
            # For block b, k is in range [2*b, 2*b+2).
            # k_local = k - 2*b.
            # W_base index is k_local*2 + i. (Since n=2 in base)
            for i in range(n):
                for k_local in range(2):
                    k_global = 2 * b + k_local
                    idx_w = k_global * n + i
                    idx_w_base = k_local * 2 + i
                    W[idx_w][r_global] = W_base[idx_w_base][r_local]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
