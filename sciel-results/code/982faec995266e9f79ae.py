"""Tensor decomposition for 2x2 * 2x13 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130
    
    n, m, p = 2, 2, 13
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                U_school[i * m + k, col] = 1.0
                V_school[k * p + j, col] = 1.0
                W_school[j * n + i, col] = 1.0
                col += 1
                
    best_rank = rank_school
    best_U, best_V, best_W = U_school, V_school, W_school

    try:
        # Strassen-based decomposition
        rank_strassen = 6 * 7 + 4 # 6 blocks of 2x2 (rank 7 each) + 1 block of 2x1 (rank 4)
        U = np.zeros((n * m, rank_strassen))
        V = np.zeros((m * p, rank_strassen))
        W = np.zeros((p * n, rank_strassen))
        
        # Define Strassen vectors for 2x2 block
        # Indices for A (4): 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
        # Indices for B (4): 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
        # Indices for C (4): 0=(0,0), 1=(1,0), 2=(0,1), 3=(1,1)  <-- Note: W index is j*n+i
        
        # u_t, v_t, w_t are length 4
        strassen_terms = [
            # 1
            np.array([1, 0, 0, 1]), np.array([1, 0, 0, 1]), np.array([1, 0, 0, 1]),
            # 2
            np.array([0, 0, 1, 1]), np.array([1, 0, 0, 0]), np.array([0, 1, 0, -1]),
            # 3
            np.array([1, 0, 0, 0]), np.array([0, 1, 0, -1]), np.array([0, 0, 1, 1]),
            # 4
            np.array([0, 0, 0, 1]), np.array([-1, 0, 1, 0]), np.array([1, 1, 0, 0]),
            # 5
            np.array([1, 1, 0, 0]), np.array([0, 0, 0, 1]), np.array([-1, 0, 1, 0]),
            # 6
            np.array([-1, 0, 1, 0]), np.array([1, 1, 0, 0]), np.array([0, 0, 0, 1]),
            # 7
            np.array([0, 1, 0, -1]), np.array([0, 0, 1, 1]), np.array([1, 0, 0, 0]),
        ]
        
        col = 0
        for block in range(6):
            k0 = block * 2
            k1 = block * 2 + 1
            
            for t in range(7):
                u_local, v_local, w_local = strassen_terms[t]
                
                u_global = np.zeros(n * m)
                v_global = np.zeros(m * p)
                w_global = np.zeros(p * n)
                
                # Map local A indices to global
                # Local 0->(i=0,k=0), 1->(i=0,k=1), 2->(i=1,k=0), 3->(i=1,k=1)
                # Global index = i*m + k
                u_global[0] = u_local[0] # i=0, k=0
                u_global[1] = u_local[1] # i=0, k=1
                u_global[2] = u_local[2] # i=1, k=0
                u_global[3] = u_local[3] # i=1, k=1
                
                # Map local B indices to global
                # Local 0->(k=0,j=0), 1->(k=0,j=1), 2->(k=1,j=0), 3->(k=1,j=1)
                # Global index = k*p + j
                # Block covers j=k0, k1 for k=0,1
                # Wait, B is 2x13. Rows are k=0,1. Cols are j=0..12.
                # Block covers cols k0, k1.
                # Local B index 0 is (k=0, j_local=0). Global j = k0.
                # Local B index 1 is (k=0, j_local=1). Global j = k1.
                # Local B index 2 is (k=1, j_local=0). Global j = k0.
                # Local B index 3 is (k=1, j_local=1). Global j = k1.
                
                v_global[0 * p + k0] = v_local[0]
                v_global[0 * p + k1] = v_local[1]
                v_global[1 * p + k0] = v_local[2]
                v_global[1 * p + k1] = v_local[3]
                
                # Map local C indices to global
                # Local 0->(j=0,i=0), 1->(j=0,i=1), 2->(j=1,i=0), 3->(j=1,i=1)
                # Global index = j*n + i
                # Block covers j=k0, k1.
                # Local C index 0 is (j_local=0, i=0). Global j = k0.
                # Local C index 1 is (j_local=0, i=1). Global j = k0.
                # Local C index 2 is (j_local=1, i=0). Global j = k1.
                # Local C index 3 is (j_local=1, i=1). Global j = k1.
                
                w_global[k0 * n + 0] = w_local[0]
                w_global[k0 * n + 1] = w_local[1]
                w_global[k1 * n + 0] = w_local[2]
                w_global[k1 * n + 1] = w_local[3]
                
                U[:, col] = u_global
                V[:, col] = v_global
                W[:, col] = w_global
                col += 1
        
        # Last block: column 12 (2x1)
        k_last = 12
        for i in range(n):
            for kk in range(m):
                U[i * m + kk, col] = 1.0
                V[kk * p + k_last, col] = 1.0
                W[k_last * n + i, col] = 1.0
                col += 1
        
        # Verify
        T_target = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    T_target[i * m + k, k * p + j, j * n + i] = 1.0
                    
        T_approx = np.einsum('it,jt,kt->ijk', U, V, W)
        error = np.max(np.abs(T_approx - T_target))
        
        if error < 1e-8:
            best_rank = rank_strassen
            best_U, best_V, best_W = U, V, W
    
    except Exception:
        pass
    
    if time.time() > deadline:
        pass
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
