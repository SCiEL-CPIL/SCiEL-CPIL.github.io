import numpy as np

def construct():
    n, m, p = 2, 3, 5
    R = 18
    U = np.zeros((n*m, R))
    V = np.zeros((m*p, R))
    W = np.zeros((p*n, R))
    
    col = 0
    
    def add_term(u_vals, v_vals, w_vals):
        nonlocal col
        for i, val in enumerate(u_vals):
            U[i, col] += val
        for j, val in enumerate(v_vals):
            V[j, col] += val
        for k, val in enumerate(w_vals):
            W[k, col] += val
        col += 1

    # Block 1: i in 0..1, j in 0..1, k in 0..1
    # Strassen
    # U indices: i*m + j -> i*3 + j
    # V indices: j*p + k -> j*5 + k
    # W indices: k*n + i -> k*2 + i
    
    # M1
    # A: (1,0) and (1,1) -> U idx 0, 1
    # B: (0,0) and (1,1) -> V idx 0, 6
    # C: (0,0), (0,1), (1,0), (1,1) -> W idx 0, 2, 1, 3
    
    # Strassen coefficients
    # M1 = (A11+A22)(B11+B22)
    u1 = [1, 1, 0, 0] # A11, A22, A12, A21 (order in U)
    # Wait, U index is i*m+j.
    # i=0, j=0 -> 0
    # i=0, j=1 -> 1
    # i=1, j=0 -> 3
    # i=1, j=1 -> 4
    # So A11 is U[0], A12 is U[1], A21 is U[3], A22 is U[4].
    
    # Let's define a helper to get U index
    def u_idx(i, j): return i*m + j
    def v_idx(j, k): return j*p + k
    def w_idx(k, i): return k*n + i
    
    # Strassen for Block 1 (local i,j,k 0..1)
    # M1
    u_m1 = [0]*6
    u_m1[u_idx(0,0)] = 1
    u_m1[u_idx(1,1)] = 1
    v_m1 = [0]*15
    v_m1[v_idx(0,0)] = 1
    v_m1[v_idx(1,1)] = 1
    w_m1 = [0]*10
    w_m1[w_idx(0,0)] = 1
    w_m1[w_idx(0,1)] = 1
    w_m1[w_idx(1,0)] = 1
    w_m1[w_idx(1,1)] = 1
    add_term(u_m1, v_m1, w_m1)
    
    # M2 = (A21+A22)B11
    u_m2 = [0]*6
    u_m2[u_idx(1,0)] = 1
    u_m2[u_idx(1,1)] = 1
    v_m2 = [0]*15
    v_m2[v_idx(0,0)] = 1
    w_m2 = [0]*10
    w_m2[w_idx(0,1)] = 1
    w_m2[w_idx(1,1)] = 1
    add_term(u_m2, v_m2, w_m2)
    
    # M3 = A11(B12-B22)
    u_m3 = [0]*6
    u_m3[u_idx(0,0)] = 1
    v_m3 = [0]*15
    v_m3[v_idx(0,1)] = 1
    v_m3[v_idx(1,1)] = -1
    w_m3 = [0]*10
    w_m3[w_idx(0,0)] = 1
    w_m3[w_idx(1,0)] = 1
    add_term(u_m3, v_m3, w_m3)
    
    # M4 = A22(B21-B11)
    u_m4 = [0]*6
    u_m4[u_idx(1,1)] = 1
    v_m4 = [0]*15
    v_m4[v_idx(1,0)] = 1
    v_m4[v_idx(0,0)] = -1
    w_m4 = [0]*10
    w_m4[w_idx(0,0)] = 1
    w_m4[w_idx(1,0)] = 1
    add_term(u_m4, v_m4, w_m4)
    
    # M5 = (A11+A12)B22
    u_m5 = [0]*6
    u_m5[u_idx(0,0)] = 1
    u_m5[u_idx(0,1)] = 1
    v_m5 = [0]*15
    v_m5[v_idx(1,1)] = 1
    w_m5 = [0]*10
    w_m5[w_idx(0,0)] = 1
    w_m5[w_idx(1,0)] = 1
    add_term(u_m5, v_m5, w_m5)
    
    # M6 = (A21-A11)(B11+B12)
    u_m6 = [0]*6
    u_m6[u_idx(1,0)] = 1
    u_m6[u_idx(0,0)] = -1
    v_m6 = [0]*15
    v_m6[v_idx(0,0)] = 1
    v_m6[v_idx(0,1)] = 1
    w_m6 = [0]*10
    w_m6[w_idx(0,1)] = 1
    w_m6[w_idx(1,1)] = 1
    add_term(u_m6, v_m6, w_m6)
    
    # M7 = (A12-A22)(B21+B22)
    u_m7 = [0]*6
    u_m7[u_idx(0,1)] = 1
    u_m7[u_idx(1,1)] = -1
    v_m7 = [0]*15
    v_m7[v_idx(1,0)] = 1
    v_m7[v_idx(1,1)] = 1
    w_m7 = [0]*10
    w_m7[w_idx(0,0)] = 1
    w_m7[w_idx(1,0)] = 1
    add_term(u_m7, v_m7, w_m7)
    
    # Block 2: i in 0..1, j in 0..1, k in 2..4
    # Sub-block 2a: k in 2..3 (2x2x2)
    # Same as Block 1 but k offset by 2
    # M1a
    u_m1a = [0]*6
    u_m1a[u_idx(0,0)] = 1
    u_m1a[u_idx(1,1)] = 1
    v_m1a = [0]*15
    v_m1a[v_idx(0,2)] = 1
    v_m1a[v_idx(1,3)] = 1
    w_m1a = [0]*10
    w_m1a[w_idx(2,0)] = 1
    w_m1a[w_idx(2,1)] = 1
    w_m1a[w_idx(3,0)] = 1
    w_m1a[w_idx(3,1)] = 1
    add_term(u_m1a, v_m1a, w_m1a)
    
    # M2a
    u_m2a = [0]*6
    u_m2a[u_idx(1,0)] = 1
    u_m2a[u_idx(1,1)] = 1
    v_m2a = [0]*15
    v_m2a[v_idx(0,2)] = 1
    w_m2a = [0]*10
    w_m2a[w_idx(2,1)] = 1
    w_m2a[w_idx(3,1)] = 1
    add_term(u_m2a, v_m2a, w_m2a)
    
    # M3a
    u_m3a = [0]*6
    u_m3a[u_idx(0,0)] = 1
    v_m3a = [0]*15
    v_m3a[v_idx(0,3)] = 1
    v_m3a[v_idx(1,3)] = -1
    w_m3a = [0]*10
    w_m3a[w_idx(2,0)] = 1
    w_m3a[w_idx(3,0)] = 1
    add_term(u_m3a, v_m3a, w_m3a)
    
    # M4a
    u_m4a = [0]*6
    u_m4a[u_idx(1,1)] = 1
    v_m4a = [0]*15
    v_m4a[v_idx(1,2)] = 1
    v_m4a[v_idx(0,2)] = -1
    w_m4a = [0]*10
    w_m4a[w_idx(2,0)] = 1
    w_m4a[w_idx(3,0)] = 1
    add_term(u_m4a, v_m4a, w_m4a)
    
    # M5a
    u_m5a = [0]*6
    u_m5a[u_idx(0,0)] = 1
    u_m5a[u_idx(0,1)] = 1
    v_m5a = [0]*15
    v_m5a[v_idx(1,3)] = 1
    w_m5a = [0]*10
    w_m5a[w_idx(2,0)] = 1
    w_m5a[w_idx(3,0)] = 1
    add_term(u_m5a, v_m5a, w_m5a)
    
    # M6a
    u_m6a = [0]*6
    u_m6a[u_idx(1,0)] = 1
    u_m6a[u_idx(0,0)] = -1
    v_m6a = [0]*15
    v_m6a[v_idx(0,2)] = 1
    v_m6a[v_idx(0,3)] = 1
    w_m6a = [0]*10
    w_m6a[w_idx(2,1)] = 1
    w_m6a[w_idx(3,1)] = 1
    add_term(u_m6a, v_m6a, w_m6a)
    
    # M7a
    u_m7a = [0]*6
    u_m7a[u_idx(0,1)] = 1
    u_m7a[u_idx(1,1)] = -1
    v_m7a = [0]*15
    v_m7a[v_idx(1,2)] = 1
    v_m7a[v_idx(1,3)] = 1
    w_m7a = [0]*10
    w_m7a[w_idx(2,0)] = 1
    w_m7a[w_idx(3,0)] = 1
    add_term(u_m7a, v_m7a, w_m7a)
    
    # Sub-block 2b: k=4 (2x2x1)
    # Rank 2
    # T[i, j, 4] = delta_{j, 4?} No, B index is j, k.
    # B is 3x5. j in 0..2, k in 0..4.
    # Here j in 0..1, k=4.
    # C[i, 4] = sum_j A[i, j] B[j, 4].
    # Tensor T[i, j, 4] = delta_{j, ?}
    # We need to compute sum_{j=0}^1 A[i, j] B[j, 4].
    # This is rank 2.
    # Term 1: j=0. U[i, 0]=1, V[0, 4]=1, W[4, i]=1.
    u_m8 = [0]*6
    u_m8[u_idx(0,0)] = 1
    u_m8[u_idx(1,0)] = 1
    v_m8 = [0]*15
    v_m8[v_idx(0,4)] = 1
    w_m8 = [0]*10
    w_m8[w_idx(4,0)] = 1
    w_m8[w_idx(4,1)] = 1
    add_term(u_m8, v_m8, w_m8)
    
    # Term 2: j=1.
    u_m9 = [0]*6
    u_m9[u_idx(0,1)] = 1
    u_m9[u_idx(1,1)] = 1
    v_m9 = [0]*15
    v_m9[v_idx(1,4)] = 1
    w_m9 = [0]*10
    w_m9[w_idx(4,0)] = 1
    w_m9[w_idx(4,1)] = 1
    add_term(u_m9, v_m9, w_m9)
    
    # Block 3: i in 0..1, j=2, k in 0..1
    # Rank 1
    # C[i, k] = A[i, 2] B[2, k].
    u_m10 = [0]*6
    u_m10[u_idx(0,2)] = 1
    u_m10[u_idx(1,2)] = 1
    v_m10 = [0]*15
    v_m10[v_idx(2,0)] = 1
    v_m10[v_idx(2,1)] = 1
    w_m10 = [0]*10
    w_m10[w_idx(0,0)] = 1
    w_m10[w_idx(0,1)] = 1
    w_m10[w_idx(1,0)] = 1
    w_m10[w_idx(1,1)] = 1
    add_term(u_m10, v_m10, w_m10)
    
    # Block 4: i in 0..1, j=2, k in 2..4
    # Rank 1
    u_m11 = [0]*6
    u_m11[u_idx(0,2)] = 1
    u_m11[u_idx(1,2)] = 1
    v_m11 = [0]*15
    v_m11[v_idx(2,2)] = 1
    v_m11[v_idx(2,3)] = 1
    v_m11[v_idx(2,4)] = 1
    w_m11 = [0]*10
    w_m11[w_idx(2,0)] = 1
    w_m11[w_idx(2,1)] = 1
    w_m11[w_idx(3,0)] = 1
    w_m11[w_idx(3,1)] = 1
    w_m11[w_idx(4,0)] = 1
    w_m11[w_idx(4,1)] = 1
    add_term(u_m11, v_m11, w_m11)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
