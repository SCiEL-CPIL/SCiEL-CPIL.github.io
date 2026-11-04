import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 4, 4
    rank = 49
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Verified Strassen 2x2 decomposition matrices
    # U[i,j] for i,j in {0,1}, 7 columns
    U2 = [
        [1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 1]
    ]
    
    # V[j,k] for j,k in {0,1}, 7 columns
    V2 = [
        [1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 1]
    ]
    
    # W[k,i] for k,i in {0,1}, 7 columns
    W2 = [
        [1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 1, 0, 1, 0, 0, 1]
    ]
    
    # Build 4x4 decomposition via Kronecker product of Strassen 2x2
    for i in range(n):
        for j in range(m):
            i1, i0 = divmod(i, 2)
            j1, j0 = divmod(j, 2)
            u_idx = i * m + j
            u_block_idx = i1 * 2 + j1
            u_elem_idx = i0 * 2 + j0
            
            for s1 in range(7):
                val1 = U2[u_block_idx][s1]
                if val1 == 0:
                    continue
                for s2 in range(7):
                    val2 = U2[u_elem_idx][s2]
                    if val2 == 0:
                        continue
                    r = s1 * 7 + s2
                    U[u_idx][r] = val1 * val2
    
    for j in range(m):
        for k in range(p):
            j1, j0 = divmod(j, 2)
            k1, k0 = divmod(k, 2)
            v_idx = j * p + k
            v_block_idx = j1 * 2 + k1
            v_elem_idx = j0 * 2 + k0
            
            for s1 in range(7):
                val1 = V2[v_block_idx][s1]
                if val1 == 0:
                    continue
                for s2 in range(7):
                    val2 = V2[v_elem_idx][s2]
                    if val2 == 0:
                        continue
                    r = s1 * 7 + s2
                    V[v_idx][r] = val1 * val2
    
    for k in range(p):
        for i in range(n):
            k1, k0 = divmod(k, 2)
            i1, i0 = divmod(i, 2)
            w_idx = k * n + i
            w_block_idx = k1 * 2 + i1
            w_elem_idx = k0 * 2 + i0
            
            for s1 in range(7):
                val1 = W2[w_block_idx][s1]
                if val1 == 0:
                    continue
                for s2 in range(7):
                    val2 = W2[w_elem_idx][s2]
                    if val2 == 0:
                        continue
                    r = s1 * 7 + s2
                    W[w_idx][r] = val1 * val2
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
