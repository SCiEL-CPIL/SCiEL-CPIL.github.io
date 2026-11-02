"""Tensor decomposition for 2x2 * 2x13 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill
    
    n, m, p = 2, 2, 13
    
    # Schoolbook fallback: rank = n*m*p = 52
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
        # Split 13 columns into 6 blocks of 2 + 1 block of 1
        # Each 2x2 * 2x2 block uses rank-7 Strassen
        # The 2x2 * 2x1 block uses rank-4 schoolbook
        # Total rank = 6*7 + 4 = 46
        
        rank_total = 6 * 7 + 4
        U = np.zeros((n * m, rank_total))
        V = np.zeros((m * p, rank_total))
        W = np.zeros((p * n, rank_total))
        
        # Strassen terms for 2x2x2x2:
        # A = [[a0,a1],[a2,a3]], B = [[b0,b1],[b2,b3]], C = A*B
        # C indices in W: j*n+i, so for a 2-col block starting at j0:
        #   W[j0*n+0] = c00, W[j0*n+1] = c01
        #   W[j1*n+0] = c10, W[j1*n+1] = c11
        # where c00=a0*b0+a1*b2, c01=a0*b1+a1*b3, c10=a2*b0+a3*b2, c11=a2*b1+a3*b3
        
        # Standard Strassen decomposition (7 terms):
        # M1 = (a0+a3)(b0+b3) -> contributes to c00, c11
        # M2 = (a2+a3)b0       -> contributes to c10, c11
        # M3 = a0(b1-b3)       -> contributes to c00, c01
        # M4 = a3(b2-b0)       -> contributes to c10, c11
        # M5 = (a0+a1)b2       -> contributes to c00, c10
        # M6 = (a2-a0)(b0+b1)  -> contributes to c10, c01
        # M7 = (a1-a3)(b3+b2)  -> contributes to c01, c11
        
        strassen_terms = [
            # (u, v, w) where u has 4 entries [a0,a1,a2,a3], v has [b0,b1,b2,b3],
            # w has 4 entries [c00,c01,c10,c11]
            (np.array([1, 0, 0, 1]), np.array([1, 0, 0, 1]), np.array([1, 0, 0, 1])),   # M1
            (np.array([0, 0, 1, 1]), np.array([1, 0, 0, 0]), np.array([0, 0, 1, 1])),   # M2
            (np.array([1, 0, 0, 0]), np.array([0, 1, 0, -1]), np.array([1, 1, 0, 0])),  # M3
            (np.array([0, 0, 0, 1]), np.array([-1, 0, 1, 0]), np.array([0, 0, 1, 1])),  # M4
            (np.array([1, 1, 0, 0]), np.array([0, 0, 1, 0]), np.array([1, 0, 1, 0])),   # M5
            (np.array([-1, 0, 1, 0]), np.array([1, 1, 0, 0]), np.array([0, 1, 1, 0])),  # M6
            (np.array([0, 1, 0, -1]), np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1])),  # M7
        ]
        
        col = 0
        for block in range(6):
            j0 = block * 2
            j1 = block * 2 + 1
            
            for t in range(7):
                u_local, v_local, w_local = strassen_terms[t]
                
                # A: global index = i*m + k
                U[0, col] = u_local[0]  # a0
                U[1, col] = u_local[1]  # a1
                U[2, col] = u_local[2]  # a2
                U[3, col] = u_local[3]  # a3
                
                # B: global index = k*p + j
                V[0 * p + j0, col] = v_local[0]  # b0
                V[0 * p + j1, col] = v_local[1]  # b1
                V[1 * p + j0, col] = v_local[2]  # b2
                V[1 * p + j1, col] = v_local[3]  # b3
                
                # C: global index = j*n + i
                W[j0 * n + 0, col] = w_local[0]  # c00
                W[j0 * n + 1, col] = w_local[1]  # c01
                W[j1 * n + 0, col] = w_local[2]  # c10
                W[j1 * n + 1, col] = w_local[3]  # c11
                
                col += 1
        
        # Last column (j=12): schoolbook 2x2 x 2x1, rank 4
        j_last = 12
        for i in range(n):
            for k in range(m):
                U[i * m + k, col] = 1.0
                V[k * p + j_last, col] = 1.0
                W[j_last * n + i, col] = 1.0
                col += 1
        
        # Verify correctness
        T_target = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    T_target[i * m + k, k * p + j, j * n + i] = 1.0
                    
        T_approx = np.einsum('it,jt,kt->ijk', U, V, W)
        error = np.max(np.abs(T_approx - T_target))
        
        if error < 1e-10:
            best_rank = rank_total
            best_U, best_V, best_W = U, V, W
    
    except Exception:
        pass
    
    if time.time() > deadline:
        pass
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
