import time
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Rank 17 decomposition
    rank = 17
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_rank1_term(U_vec, V_vec, W_vec):
        nonlocal col
        if col >= rank:
            return
        for i in range(n * m):
            if U_vec[i] != 0:
                U[i][col] += U_vec[i]
        for i in range(m * p):
            if V_vec[i] != 0:
                V[i][col] += V_vec[i]
        for i in range(p * n):
            if W_vec[i] != 0:
                W[i][col] += W_vec[i]
        col += 1

    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(i, k): return k * n + i

    # 1. Strassen for A[0:2, 0:2] B[0:2, 0:2] -> C[0:2, 0:2]
    # m1
    U1 = [0]*(n*m); V1 = [0]*(m*p); W1 = [0]*(p*n)
    U1[u_idx(0,0)] = 1; U1[u_idx(1,1)] = 1
    V1[v_idx(0,0)] = 1; V1[v_idx(1,1)] = 1
    W1[w_idx(0,0)] = 1; W1[w_idx(1,1)] = 1
    add_rank1_term(U1, V1, W1)
    
    # m2
    U2 = [0]*(n*m); V2 = [0]*(m*p); W2 = [0]*(p*n)
    U2[u_idx(1,0)] = 1; U2[u_idx(1,1)] = 1
    V2[v_idx(0,0)] = 1
    W2[w_idx(1,0)] = 1; W2[w_idx(1,1)] = -1
    add_rank1_term(U2, V2, W2)
    
    # m3
    U3 = [0]*(n*m); V3 = [0]*(m*p); W3 = [0]*(p*n)
    U3[u_idx(0,0)] = 1
    V3[v_idx(0,1)] = 1; V3[v_idx(1,1)] = -1
    W3[w_idx(0,1)] = 1; W3[w_idx(1,1)] = 1
    add_rank1_term(U3, V3, W3)
    
    # m4
    U4 = [0]*(n*m); V4 = [0]*(m*p); W4 = [0]*(p*n)
    U4[u_idx(1,1)] = 1
    V4[v_idx(1,0)] = 1; V4[v_idx(0,0)] = -1
    W4[w_idx(0,0)] = 1; W4[w_idx(1,0)] = 1
    add_rank1_term(U4, V4, W4)
    
    # m5
    U5 = [0]*(n*m); V5 = [0]*(m*p); W5 = [0]*(p*n)
    U5[u_idx(0,0)] = 1; U5[u_idx(0,1)] = 1
    V5[v_idx(1,1)] = 1
    W5[w_idx(0,0)] = -1; W5[w_idx(0,1)] = 1
    add_rank1_term(U5, V5, W5)
    
    # m6
    U6 = [0]*(n*m); V6 = [0]*(m*p); W6 = [0]*(p*n)
    U6[u_idx(1,0)] = 1; U6[u_idx(0,0)] = -1
    V6[v_idx(0,0)] = 1; V6[v_idx(0,1)] = 1
    W6[w_idx(1,1)] = 1
    add_rank1_term(U6, V6, W6)
    
    # m7
    U7 = [0]*(n*m); V7 = [0]*(m*p); W7 = [0]*(p*n)
    U7[u_idx(0,1)] = 1; U7[u_idx(1,1)] = -1
    V7[v_idx(1,0)] = 1; V7[v_idx(1,1)] = 1
    W7[w_idx(0,0)] = 1
    add_rank1_term(U7, V7, W7)

    # 2. Strassen for A[0:2, 0:2] B[0:2, 2:4] -> C[0:2, 2:4]
    # m1
    U1 = [0]*(n*m); V1 = [0]*(m*p); W1 = [0]*(p*n)
    U1[u_idx(0,0)] = 1; U1[u_idx(1,1)] = 1
    V1[v_idx(0,2)] = 1; V1[v_idx(1,3)] = 1
    W1[w_idx(0,2)] = 1; W1[w_idx(1,3)] = 1
    add_rank1_term(U1, V1, W1)
    
    # m2
    U2 = [0]*(n*m); V2 = [0]*(m*p); W2 = [0]*(p*n)
    U2[u_idx(1,0)] = 1; U2[u_idx(1,1)] = 1
    V2[v_idx(0,2)] = 1
    W2[w_idx(1,2)] = 1; W2[w_idx(1,3)] = -1
    add_rank1_term(U2, V2, W2)
    
    # m3
    U3 = [0]*(n*m); V3 = [0]*(m*p); W3 = [0]*(p*n)
    U3[u_idx(0,0)] = 1
    V3[v_idx(0,3)] = 1; V3[v_idx(1,3)] = -1
    W3[w_idx(0,3)] = 1; W3[w_idx(1,3)] = 1
    add_rank1_term(U3, V3, W3)
    
    # m4
    U4 = [0]*(n*m); V4 = [0]*(m*p); W4 = [0]*(p*n)
    U4[u_idx(1,1)] = 1
    V4[v_idx(1,2)] = 1; V4[v_idx(0,2)] = -1
    W4[w_idx(0,2)] = 1; W4[w_idx(1,2)] = 1
    add_rank1_term(U4, V4, W4)
    
    # m5
    U5 = [0]*(n*m); V5 = [0]*(m*p); W5 = [0]*(p*n)
    U5[u_idx(0,0)] = 1; U5[u_idx(0,1)] = 1
    V5[v_idx(1,3)] = 1
    W5[w_idx(0,2)] = -1; W5[w_idx(0,3)] = 1
    add_rank1_term(U5, V5, W5)
    
    # m6
    U6 = [0]*(n*m); V6 = [0]*(m*p); W6 = [0]*(p*n)
    U6[u_idx(1,0)] = 1; U6[u_idx(0,0)] = -1
    V6[v_idx(0,2)] = 1; V6[v_idx(0,3)] = 1
    W6[w_idx(1,3)] = 1
    add_rank1_term(U6, V6, W6)
    
    # m7
    U7 = [0]*(n*m); V7 = [0]*(m*p); W7 = [0]*(p*n)
    U7[u_idx(0,1)] = 1; U7[u_idx(1,1)] = -1
    V7[v_idx(1,2)] = 1; V7[v_idx(1,3)] = 1
    W7[w_idx(0,2)] = 1
    add_rank1_term(U7, V7, W7)

    # 3. Merged Linear Term 1: A[0:2,0:2]B[0:2,4:7] (j=0) + A[0:2,2]B[2,0:7] (i=0)
    # U: U[0,0]=1, U[1,0]=1, U[0,2]=1
    # V: V[0,4..6]=1, V[2,0..6]=1
    # W: W[0..6,0]=1
    U_L1 = [0]*(n*m); V_L1 = [0]*(m*p); W_L1 = [0]*(p*n)
    U_L1[u_idx(0,0)] = 1
    U_L1[u_idx(1,0)] = 1
    U_L1[u_idx(0,2)] = 1
    
    for k in range(4, 7):
        V_L1[v_idx(0,k)] = 1
    for k in range(7):
        V_L1[v_idx(2,k)] = 1
        
    for k in range(7):
        W_L1[w_idx(0,k)] = 1
        
    add_rank1_term(U_L1, V_L1, W_L1)

    # 4. Merged Linear Term 2: A[0:2,0:2]B[0:2,4:7] (j=1) + A[0:2,2]B[2,0:7] (i=1)
    # U: U[0,1]=1, U[1,1]=1, U[1,2]=1
    # V: V[1,4..6]=1, V[2,0..6]=1
    # W: W[0..6,1]=1
    U_L2 = [0]*(n*m); V_L2 = [0]*(m*p); W_L2 = [0]*(p*n)
    U_L2[u_idx(0,1)] = 1
    U_L2[u_idx(1,1)] = 1
    U_L2[u_idx(1,2)] = 1
    
    for k in range(4, 7):
        V_L2[v_idx(1,k)] = 1
    for k in range(7):
        V_L2[v_idx(2,k)] = 1
        
    for k in range(7):
        W_L2[w_idx(1,k)] = 1
        
    add_rank1_term(U_L2, V_L2, W_L2)

    # 5. Merged Linear Term 3: A[2,0:3]B[0:3,0:7]
    # U: U[2,0]=1, U[2,1]=1, U[2,2]=1
    # V: V[0..2,0..6]=1
    # W: W[0..6,2]=1
    U_L3 = [0]*(n*m); V_L3 = [0]*(m*p); W_L3 = [0]*(p*n)
    U_L3[u_idx(2,0)] = 1
    U_L3[u_idx(2,1)] = 1
    U_L3[u_idx(2,2)] = 1
    
    for j in range(3):
        for k in range(7):
            V_L3[v_idx(j,k)] = 1
            
    for k in range(7):
        W_L3[w_idx(2,k)] = 1
        
    add_rank1_term(U_L3, V_L3, W_L3)

    # Verification
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_true = A_test @ B_test
    C_comp = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            val = 0.0
            for j in range(m):
                for r in range(rank):
                    val += U_arr[i*m + j, r] * A_test[i, j] * V_arr[j*p + k, r] * B_test[j, k] * W_arr[k*n + i, r]
            C_comp[i, k] = val
            
    if not np.allclose(C_comp, C_true, atol=1e-9):
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
