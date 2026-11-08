import time
import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    
    # Factorization: (3, 2, 4) and (1, 2, 2)
    n1, m1, p1 = 3, 2, 4
    n2, m2, p2 = 1, 2, 2
    
    # Check if factorization matches
    if n1 * n2 != n or m1 * m2 != m or p1 * p2 != p:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    # Build schoolbook for sub 1
    r1 = n1 * m1 * p1
    U1 = [[0] * r1 for _ in range(n1 * m1)]
    V1 = [[0] * r1 for _ in range(m1 * p1)]
    W1 = [[0] * r1 for _ in range(p1 * n1)]
    
    col = 0
    for i1 in range(n1):
        for j1 in range(m1):
            for k1 in range(p1):
                U1[i1 * m1 + j1][col] = 1
                V1[j1 * p1 + k1][col] = 1
                W1[k1 * n1 + i1][col] = 1
                col += 1

    # Build schoolbook for sub 2
    r2 = n2 * m2 * p2
    U2 = [[0] * r2 for _ in range(n2 * m2)]
    V2 = [[0] * r2 for _ in range(m2 * p2)]
    W2 = [[0] * r2 for _ in range(p2 * n2)]
    
    col = 0
    for i2 in range(n2):
        for j2 in range(m2):
            for k2 in range(p2):
                U2[i2 * m2 + j2][col] = 1
                V2[j2 * p2 + k2][col] = 1
                W2[k2 * n2 + i2][col] = 1
                col += 1

    # Kronecker product
    rank = r1 * r2
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Fill Kronecker product
    # U row index: u1 * (n2*m2) + u2
    # V row index: v1 * (m2*p2) + v2
    # W row index: w1 * (p2*n2) + w2
    
    for u1 in range(n1 * m1):
        for v1 in range(m1 * p1):
            for w1 in range(p1 * n1):
                for c1 in range(r1):
                    if U1[u1][c1] == 1:
                        for u2 in range(n2 * m2):
                            for v2 in range(m2 * p2):
                                for w2 in range(p2 * n2):
                                    if U2[u2][0] == 1 and V2[v2][0] == 1 and W2[w2][0] == 1:
                                        # This is inefficient. Better to use numpy or precomputed lists.
                                        pass
                                        
    # Let's use numpy for Kronecker and then convert
    U1_np = np.array(U1)
    U2_np = np.array(U2)
    V1_np = np.array(V1)
    V2_np = np.array(V2)
    W1_np = np.array(W1)
    W2_np = np.array(W2)
    
    U_kr = np.kron(U1_np, U2_np)
    V_kr = np.kron(V1_np, V2_np)
    W_kr = np.kron(W1_np, W2_np)
    
    # Now permute rows to match global indices
    # U global index: i_g * m + j_g
    # i_g = i1 * n2 + i2 = i1
    # j_g = j1 * m2 + j2
    # u_g = i1 * m + (j1 * m2 + j2) = i1 * 4 + j1 * 2 + j2
    
    # U_kr row index: u1 * (n2*m2) + u2
    # u1 = i1 * m1 + j1 = i1 * 2 + j1
    # u2 = i2 * m2 + j2 = i2 * 2 + j2
    # u_kr = (i1 * 2 + j1) * 4 + (i2 * 2 + j2) = 8 i1 + 4 j1 + 2 i2 + j2
    
    # We need to map u_g to u_kr
    # i1 in [0,3), j1 in [0,2), i2 in [0,1), j2 in [0,2)
    # i2 is always 0.
    # u_g = 4 i1 + 2 j1 + j2
    # u_kr = 8 i1 + 4 j1 + j2
    
    # Permutation for U
    perm_U = [0] * (n * m)
    for i1 in range(n1):
        for j1 in range(m1):
            for i2 in range(n2):
                for j2 in range(m2):
                    u_g = i1 * m + j1 * m2 + j2
                    u_kr = (i1 * m1 + j1) * (n2 * m2) + (i2 * m2 + j2)
                    perm_U[u_g] = u_kr
    
    # V global index: j_g * p + k_g
    # j_g = j1 * m2 + j2
    # k_g = k1 * p2 + k2
    # v_g = (j1 * m2 + j2) * p + (k1 * p2 + k2) = (j1 * 2 + j2) * 8 + k1 * 2 + k2
    
    # V_kr row index: v1 * (m2*p2) + v2
    # v1 = j1 * p1 + k1 = j1 * 4 + k1
    # v2 = j2 * p2 + k2 = j2 * 2 + k2
    # v_kr = (j1 * 4 + k1) * 4 + (j2 * 2 + k2) = 16 j1 + 4 k1 + 2 j2 + k2
    
    # Permutation for V
    perm_V = [0] * (m * p)
    for j1 in range(m1):
        for k1 in range(p1):
            for j2 in range(m2):
                for k2 in range(p2):
                    v_g = (j1 * m2 + j2) * p + k1 * p2 + k2
                    v_kr = (j1 * p1 + k1) * (m2 * p2) + (j2 * p2 + k2)
                    perm_V[v_g] = v_kr
                    
    # W global index: k_g * n + i_g
    # k_g = k1 * p2 + k2
    # i_g = i1 * n2 + i2 = i1
    # w_g = (k1 * p2 + k2) * n + i1 = (k1 * 2 + k2) * 3 + i1
    
    # W_kr row index: w1 * (p2*n2) + w2
    # w1 = k1 * n1 + i1 = k1 * 3 + i1
    # w2 = k2 * n2 + i2 = k2 * 1 + i2 = k2
    # w_kr = (k1 * 3 + i1) * 2 + k2 = 6 k1 + 2 i1 + k2
    
    # Permutation for W
    perm_W = [0] * (p * n)
    for k1 in range(p1):
        for i1 in range(n1):
            for k2 in range(p2):
                for i2 in range(n2):
                    w_g = (k1 * p2 + k2) * n + i1 * n2 + i2
                    w_kr = (k1 * n1 + i1) * (p2 * n2) + (k2 * n2 + i2)
                    perm_W[w_g] = w_kr

    # Apply permutations
    U_final = U_kr[perm_U, :]
    V_final = V_kr[perm_V, :]
    W_final = W_kr[perm_W, :]
    
    # Convert to lists of lists
    U_list = U_final.tolist()
    V_list = V_final.tolist()
    W_list = W_final.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
