import numpy as np
import time

DIMENSIONS = [3, 4, 5]

def get_strassen_2x2x2():
    """
    Rank-7 decomposition for 2x2x2 matrix multiplication.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indices for U, V, W correspond to:
    U: [A00, A01, A10, A11]
    V: [B00, B01, B10, B11]
    W: [C00, C01, C10, C11]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A00 + A11)(B00 + B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10 + A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3 = A00(B01 - B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4 = A11(B10 - B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5 = (A00 + A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6 = (A10 - A00)(B00 + B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A01 - A11)(B10 + B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def get_schoolbook_2x2x1():
    """
    Rank-4 decomposition for 2x2x1 matrix multiplication.
    U: (4, 4), V: (2, 4), W: (2, 4)
    Indices:
    U: [A00, A01, A10, A11]
    V: [B0, B1]
    W: [C0, C1]
    """
    U = np.zeros((4, 4))
    V = np.zeros((2, 4))
    W = np.zeros((2, 4))
    
    # C0 = A00*B0 + A01*B1
    U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
    U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
    
    # C1 = A10*B0 + A11*B1
    U[2, 2] = 1; V[0, 2] = 1; W[1, 2] = 1
    U[3, 3] = 1; V[1, 3] = 1; W[1, 3] = 1
    
    return U, V, W

def get_schoolbook_2x1x2():
    """
    Rank-4 decomposition for 2x1x2 matrix multiplication.
    U: (2, 4), V: (2, 4), W: (4, 4)
    Indices:
    U: [A0, A1]
    V: [B0, B1]
    W: [C00, C01, C10, C11]
    """
    U = np.zeros((2, 4))
    V = np.zeros((2, 4))
    W = np.zeros((4, 4))
    
    # C00 = A0*B0
    U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
    # C01 = A0*B1
    U[0, 1] = 1; V[1, 1] = 1; W[1, 1] = 1
    # C10 = A1*B0
    U[1, 2] = 1; V[0, 2] = 1; W[2, 2] = 1
    # C11 = A1*B1
    U[1, 3] = 1; V[1, 3] = 1; W[3, 3] = 1
    
    return U, V, W

def get_schoolbook_1x2x2():
    """
    Rank-4 decomposition for 1x2x2 matrix multiplication.
    U: (2, 4), V: (4, 4), W: (2, 4)
    Indices:
    U: [A0, A1]
    V: [B00, B01, B10, B11]
    W: [C0, C1]
    """
    U = np.zeros((2, 4))
    V = np.zeros((4, 4))
    W = np.zeros((2, 4))
    
    # C0 = A0*B00 + A1*B10
    U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
    U[1, 1] = 1; V[2, 1] = 1; W[0, 1] = 1
    
    # C1 = A0*B01 + A1*B11
    U[0, 2] = 1; V[1, 2] = 1; W[1, 2] = 1
    U[1, 3] = 1; V[3, 3] = 1; W[1, 3] = 1
    
    return U, V, W

def get_schoolbook_1x2x1():
    """
    Rank-2 decomposition for 1x2x1 matrix multiplication.
    U: (2, 2), V: (2, 2), W: (1, 2)
    Indices:
    U: [A0, A1]
    V: [B0, B1]
    W: [C0]
    """
    U = np.zeros((2, 2))
    V = np.zeros((2, 2))
    W = np.zeros((1, 2))
    
    # C0 = A0*B0 + A1*B1
    U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
    U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    
    start_time = time.time()
    DEADLINE = 140.0 # 10s safety margin
    
    U_parts = []
    V_parts = []
    W_parts = []
    
    def add_block(u_indices, v_indices, w_indices, U_block, V_block, W_block):
        rank = U_block.shape[1]
        for r in range(rank):
            U_vec = np.zeros(n * m)
            V_vec = np.zeros(m * p)
            W_vec = np.zeros(p * n)
            
            for i, idx in enumerate(u_indices):
                U_vec[idx] = U_block[i, r]
            for i, idx in enumerate(v_indices):
                V_vec[idx] = V_block[i, r]
            for i, idx in enumerate(w_indices):
                W_vec[idx] = W_block[i, r]
                
            U_parts.append(U_vec)
            V_parts.append(V_vec)
            W_parts.append(W_vec)

    # --- Helper to check time ---
    def check_time():
        if time.time() - start_time > DEADLINE:
            return True
        return False

    # --- Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2] (Strassen 2x2x2) ---
    # U indices: A[0,0]=0, A[0,1]=1, A[1,0]=4, A[1,1]=5
    # V indices: B[0,0]=0, B[0,1]=1, B[1,0]=5, B[1,1]=6
    # W indices: C[0,0]=0, C[0,1]=3, C[1,0]=1, C[1,1]=4 (Note: W index is k*n+i. n=3. C[0,0]->0*3+0=0. C[0,1]->1*3+0=3. C[1,0]->0*3+1=1. C[1,1]->1*3+1=4)
    # Wait, let's re-verify W index convention: W[k*n+i, r].
    # C is 3x5. C[i, k]. Index in W is k*n + i.
    # C[0,0] -> 0*3+0 = 0
    # C[0,1] -> 1*3+0 = 3
    # C[1,0] -> 0*3+1 = 1
    # C[1,1] -> 1*3+1 = 4
    # C[2,0] -> 0*3+2 = 2
    # C[2,1] -> 1*3+2 = 5
    # C[0,2] -> 2*3+0 = 6
    # C[0,3] -> 3*3+0 = 9
    # C[0,4] -> 4*3+0 = 12
    # C[1,2] -> 2*3+1 = 7
    # C[1,3] -> 3*3+1 = 10
    # C[1,4] -> 4*3+1 = 13
    # C[2,2] -> 2*3+2 = 8
    # C[2,3] -> 3*3+2 = 11
    # C[2,4] -> 4*3+2 = 14
    
    # U indices for A[0:2, 0:2]:
    # A[0,0]=0, A[0,1]=1
    # A[1,0]=4, A[1,1]=5
    u_idx_1 = [0, 1, 4, 5]
    
    # V indices for B[0:2, 0:2]:
    # B[0,0]=0, B[0,1]=1
    # B[1,0]=5, B[1,1]=6
    v_idx_1 = [0, 1, 5, 6]
    
    # W indices for C[0:2, 0:2]:
    # C[0,0]=0, C[0,1]=3
    # C[1,0]=1, C[1,1]=4
    w_idx_1 = [0, 3, 1, 4]
    
    add_block(u_idx_1, v_idx_1, w_idx_1, *get_strassen_2x2x2())
    if check_time(): return {'U': np.array(U_parts).T.tolist(), 'V': np.array(V_parts).T.tolist(), 'W': np.array(W_parts).T.tolist()}

    # --- Block 2: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2] (Strassen 2x2x2) ---
    # U indices for A[0:2, 2:4]:
    # A[0,2]=2, A[0,3]=3
    # A[1,2]=6, A[1,3]=7
    u_idx_2 = [2, 3, 6, 7]
    
    # V indices for B[2:4, 0:2]:
    # B[2,0]=10, B[2,1]=11
    # B[3,0]=15, B[3,1]=16
    v_idx_2 = [10, 11, 15, 16]
    
    # W indices for C[0:2, 0:2]:
    w_idx_2 = [0, 3, 1, 4]
    
    add_block(u_idx_2, v_idx_2, w_idx_2, *get_strassen_2x2x2())
    if check_time(): return {'U': np.array(U_parts).T.tolist(), 'V': np.array(V_parts).T.tolist(), 'W': np.array(W_parts).T.tolist()}

    # --- Block 3: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4] (Strassen 2x2x2) ---
    u_idx_3 = [0, 1, 4, 5]
    
    # V indices for B[0:2, 2:4]:
    # B[0,2]=2, B[0,3]=3
    # B[1,2]=7, B[1,3]=8
    v_idx_3 = [2, 3, 7, 8]
    
    # W indices for C[0:2, 2:4]:
    # C[0,2]=6, C[0,3]=9
    # C[1,2]=7, C[1,3]=10
    w_idx_3 = [6, 9, 7, 10]
    
    add_block(u_idx_3, v_idx_3, w_idx_3, *get_strassen_2x2x2())
    if check_time(): return {'U': np.array(U_parts).T.tolist(), 'V': np.array(V_parts).T.tolist(), 'W': np.array(W_parts).T.tolist()}

    # --- Block 4: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4] (Strassen 2x2x2) ---
    u_idx_4 = [2, 3, 6, 7]
    
    # V indices for B[2:4, 2:4]:
    # B[2,2]=12, B[2,3]=13
    # B[3,2]=17, B[3,3]=18
    v_idx_4 = [12, 13, 17, 18]
    
    # W indices for C[0:2, 2:4]:
    w_idx_4 = [6, 9, 7, 10]
    
    add_block(u_idx_4, v_idx_4, w_idx_4, *get_strassen_2x2x2())
    if check_time(): return {'U': np.array(U_parts).T.tolist(), 'V': np.array(V_parts).T.tolist(), 'W': np.array(W_parts).T.tolist()}

    # --- Block 5: A[0:2, 0:2] * B[0:2, 4:5] -> C[0:2, 4:5] (Schoolbook 2x2x1) ---
    # A is 2x2, B is 2x1, C is 2x1
    u_idx_5 = [0, 1, 4, 5]
    
    # V indices for B[0:2, 4:5]:
    # B[0,4]=4, B[1,4]=9
    v_idx_5 = [4, 9]
    
    # W indices for C[0:2, 4:5]:
    # C[0,4]=12, C[1,4]=13
    w_idx_5 = [12, 13]
    
    add_block(u_idx_5, v_idx_5, w_idx_5, *get_schoolbook_2x2x1())
    if check_time(): return {'U': np.array(U_parts).T.tolist(), 'V': np.array(V_parts).T.tolist(), 'W': np.array(W_parts).T.tolist()}

    # --- Block 6: A[0:2, 2:4] * B[2:4, 4:5] -> C[0:2, 4:5] (Schoolbook 2x2x1) ---
    u_idx_6 = [2, 3, 6, 7]
    
    # V indices for B[2:4, 4:5]:
    # B[2,4]=14, B[3,4]=19
    v_idx_6 = [14, 19]
    
    # W indices for C[0:2, 4:5]:
    w_idx_6 = [12, 13]
    
    add_block(u_idx_6, v_idx_6, w_idx_6, *get_schoolbook_2x2x1())
    if check_time(): return {'U': np.array(U_parts).T.tolist(), 'V': np.array(V_parts).T.tolist(), 'W': np.array(W_parts).T.tolist()}

    # --- Block 7: A[2:3, 0:2] * B[0:2, 0:2] -> C[2:3, 0:2] (Schoolbook 1x2x2) ---
    # A is 1x2, B is 2x2, C is 1x2
    # U indices for A[2:3, 0:2]:
    # A[2,0]=8, A[2,1]=9
    u_idx_7 = [8, 9]
    
    # V indices for B[0:2, 0:2]:
    # B[0,0]=0, B[0,1]=1
    # B[1,0]=5, B[1,1]=6
    v_idx_7 = [0, 1, 5, 6]
    
    # W indices for C[2:3, 0:2]:
    # C[2,0]=2, C[2,1]=5
    w_idx_7 = [2, 5]
    
    add_block(u_idx_7, v_idx_7, w_idx_7, *get_schoolbook_1x2x2())
    if check_time(): return {'U': np.array(U_parts).T.tolist(), 'V': np.array(V_parts).T.tolist(), 'W': np.array(W_parts).T.tolist()}

    # --- Block 8: A[2:3, 2:4] * B[2:4, 0:2] -> C[2:3, 0:2] (Schoolbook 1x2x2) ---
    # U indices for A[2:3, 2:4]:
    # A[2,2]=10, A[2,3]=11
    u_idx_8 = [10, 11]
    
    # V indices for B[2:4, 0:2]:
    # B[2,0]=10, B[2,1]=11
    # B[3,0]=15, B[3,1]=16
    v_idx_8 = [10, 11, 15, 16]
    
    # W indices for C[2:3, 0:2]:
    w_idx_8 = [2, 5]
    
    add_block(u_idx_8, v_idx_8, w_idx_8, *get_schoolbook_1x2x2())
    if check_time(): return {'U': np.array(U_parts).T.tolist(), 'V': np.array(V_parts).T.tolist(), 'W': np.array(W_parts).T.tolist()}

    # --- Block 9: A[2:3, 0:2] * B[0:2, 2:4] -> C[2:3, 2:4] (Schoolbook 1x2x2) ---
    u_idx_9 = [8, 9]
    
    # V indices for B[0:2, 2:4]:
    # B[0,2]=2, B[0,3]=3
    # B[1,2]=7, B[1,3]=8
    v_idx_9 = [2, 3, 7, 8]
    
    # W indices for C[2:3, 2:4]:
    # C[2,2]=8, C[2,3]=11
    w_idx_9 = [8, 11]
    
    add_block(u_idx_9, v_idx_9, w_idx_9, *get_schoolbook_1x2x2())
    if check_time(): return {'U': np.array(U_parts).T.tolist(), 'V': np.array(V_parts).T.tolist(), 'W': np.array(W_parts).T.tolist()}

    # --- Block 10: A[2:3, 2:4] * B[2:4, 2:4] -> C[2:3, 2:4] (Schoolbook 1x2x2) ---
    u_idx_10 = [10, 11]
    
    # V indices for B[2:4, 2:4]:
    # B[2,2]=12, B[2,3]=13
    # B[3,2]=17, B[3,3]=18
    v_idx_10 = [12, 13, 17, 18]
    
    # W indices for C[2:3, 2:4]:
    w_idx_10 = [8, 11]
    
    add_block(u_idx_10, v_idx_10, w_idx_10, *get_schoolbook_1x2x2())
    if check_time(): return {'U': np.array(U_parts).T.tolist(), 'V': np.array(V_parts).T.tolist(), 'W': np.array(W_parts).T.tolist()}

    # --- Block 11: A[2:3, 0:2] * B[0:2, 4:5] -> C[2:3, 4:5] (Schoolbook 1x2x1) ---
    # A is 1x2, B is 2x1, C is 1x1
    u_idx_11 = [8, 9]
    
    # V indices for B[0:2, 4:5]:
    # B[0,4]=4, B[1,4]=9
    v_idx_11 = [4, 9]
    
    # W indices for C[2:3, 4:5]:
    # C[2,4]=14
    w_idx_11 = [14]
    
    add_block(u_idx_11, v_idx_11, w_idx_11, *get_schoolbook_1x2x1())
    if check_time(): return {'U': np.array(U_parts).T.tolist(), 'V': np.array(V_parts).T.tolist(), 'W': np.array(W_parts).T.tolist()}

    # --- Block 12: A[2:3, 2:4] * B[2:4, 4:5] -> C[2:3, 4:5] (Schoolbook 1x2x1) ---
    u_idx_12 = [10, 11]
    
    # V indices for B[2:4, 4:5]:
    # B[2,4]=14, B[3,4]=19
    v_idx_12 = [14, 19]
    
    # W indices for C[2:3, 4:5]:
    w_idx_12 = [14]
    
    add_block(u_idx_12, v_idx_12, w_idx_12, *get_schoolbook_1x2x1())
    if check_time(): return {'U': np.array(U_parts).T.tolist(), 'V': np.array(V_parts).T.tolist(), 'W': np.array(W_parts).T.tolist()}

    U = np.array(U_parts).T
    V = np.array(V_parts).T
    W = np.array(W_parts).T
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
