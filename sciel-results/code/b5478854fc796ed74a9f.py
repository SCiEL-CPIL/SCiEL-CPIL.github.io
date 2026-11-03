import time
import numpy as np

DIMENSIONS = [3, 3, 8]
DEADLINE_S = 110  # safety margin from 120s limit

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    def time_left():
        return DEADLINE_S - (time.time() - start_time)
    
    def schoolbook():
        rank = n*m*p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1.0
                    V[j*p + k, col] = 1.0
                    W[k*n + i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    def validate(U, V, W, tol=1e-10):
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        rank = U.shape[1]
        if V.shape[1] != rank or W.shape[1] != rank:
            return False
        T_expected = np.ones((n, m, p))
        T_computed = np.zeros((n, m, p))
        for c in range(rank):
            u_mat = U[:, c].reshape(n, m)
            v_mat = V[:, c].reshape(m, p)
            w_mat = W[:, c].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    if abs(u_mat[i,j]) < 1e-15:
                        continue
                    for k in range(p):
                        T_computed[i,j,k] += u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
        return np.max(np.abs(T_computed - T_expected)) < tol
    
    def strassen_2x2x2_block():
        """
        Returns U, V, W for a 2x2x2 Strassen decomposition (rank 7).
        U: (4, 7) - indices (i,j) with i,j in {0,1}
        V: (4, 7) - indices (j,k) with j,k in {0,1}
        W: (4, 7) - indices (k,i) with k,i in {0,1}
        """
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # Strassen's 7 multiplication terms for 2x2x2
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B00)
        # M4 = A11(B01-B11)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A11)(B01+B11)
        
        # U coefficients (A matrix)
        # M1: A00 + A11 -> (0,0)=1, (1,1)=1
        U[0*2+0, 0] = 1; U[1*2+1, 0] = 1
        # M2: A10 + A11 -> (1,0)=1, (1,1)=1
        U[1*2+0, 1] = 1; U[1*2+1, 1] = 1
        # M3: A00 -> (0,0)=1
        U[0*2+0, 2] = 1
        # M4: A11 -> (1,1)=1
        U[1*2+1, 3] = 1
        # M5: A00 + A01 -> (0,0)=1, (0,1)=1
        U[0*2+0, 4] = 1; U[0*2+1, 4] = 1
        # M6: A10 - A00 -> (1,0)=1, (0,0)=-1
        U[1*2+0, 5] = 1; U[0*2+0, 5] = -1
        # M7: A01 - A11 -> (0,1)=1, (1,1)=-1
        U[0*2+1, 6] = 1; U[1*2+1, 6] = -1
        
        # V coefficients (B matrix)
        # M1: B00 + B11 -> (0,0)=1, (1,1)=1
        V[0*2+0, 0] = 1; V[1*2+1, 0] = 1
        # M2: B00 -> (0,0)=1
        V[0*2+0, 1] = 1
        # M3: B10 - B00 -> (1,0)=1, (0,0)=-1
        V[1*2+0, 2] = 1; V[0*2+0, 2] = -1
        # M4: B01 - B11 -> (0,1)=1, (1,1)=-1
        V[0*2+1, 3] = 1; V[1*2+1, 3] = -1
        # M5: B11 -> (1,1)=1
        V[1*2+1, 4] = 1
        # M6: B00 + B10 -> (0,0)=1, (1,0)=1
        V[0*2+0, 5] = 1; V[1*2+0, 5] = 1
        # M7: B01 + B11 -> (0,1)=1, (1,1)=1
        V[0*2+1, 6] = 1; V[1*2+1, 6] = 1
        
        # W coefficients (C matrix reconstruction)
        # C00 = M1 - M4 + M5 + M6
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M3 + M6 - M7
        W[0*2+0, 0] = 1; W[1*2+1, 0] = -1; W[0*2+1, 0] = 1; W[1*2+0, 0] = 1
        W[0*2+1, 1] = 1; W[0*2+1, 2] = 1
        W[1*2+0, 3] = 1; W[1*2+1, 4] = 1
        W[1*2+1, 5] = 1; W[1*2+0, 6] = -1; W[0*2+1, 6] = 1; W[0*2+0, 6] = -1
        
        return U, V, W
    
    def _get_17_rank_block():
        """
        Returns U, V, W for a 3x3x2 block with rank 17.
        U: (9, 17), V: (6, 17), W: (6, 17)
        Uses Strassen for the 2x2x2 sub-block (i,j in {0,1}, k in {0,1})
        and schoolbook for the remaining 14 terms.
        """
        rank = 17
        U = np.zeros((9, rank))
        V = np.zeros((6, rank))
        W = np.zeros((6, rank))
        
        U_s, V_s, W_s = strassen_2x2x2_block()
        
        col = 0
        # Place Strassen terms (7 terms) for (i,j) in {0,1} x {0,1}, k in {0,1}
        for l in range(7):
            for i in range(2):
                for j in range(2):
                    U[i*3 + j, col] = U_s[i*2 + j, l]
            for j in range(2):
                for k in range(2):
                    V[j*2 + k, col] = V_s[j*2 + k, l]
            for k in range(2):
                for i in range(2):
                    W[k*2 + i, col] = W_s[k*2 + i, l]
            col += 1
        
        # Schoolbook part for remaining (i,j,k) in 3x3x2 not in Strassen region
        for i in range(3):
            for j in range(3):
                for k in range(2):
                    if i in (0,1) and j in (0,1):
                        continue
                    U[i*3 + j, col] = 1.0
                    V[j*2 + k, col] = 1.0
                    W[k*2 + i, col] = 1.0
                    col += 1
        
        return U, V, W
    
    def build_4block_strassen():
        """
        4 independent 3x3x2 blocks, each with rank 17.
        Total rank: 68.
        """
        U_block, V_block, W_block = _get_17_rank_block()
        rank_block = 17
        rank_total = 4 * rank_block
        U = np.zeros((9, rank_total))
        V = np.zeros((24, rank_total))
        W = np.zeros((24, rank_total))
        
        for g in range(4):
            k0 = 2 * g
            col_start = g * rank_block
            col_end = (g + 1) * rank_block
            
            U[:, col_start:col_end] = U_block
            
            for c in range(rank_block):
                for j in range(3):
                    for k_local in range(2):
                        k_global = k0 + k_local
                        V[j*8 + k_global, col_start + c] = V_block[j*2 + k_local, c]
            
            for c in range(rank_block):
                for k_local in range(2):
                    k_global = k0 + k_local
                    for i in range(3):
                        W[k_global*3 + i, col_start + c] = W_block[k_local*2 + i, c]
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Use the 4-block Strassen (rank 68)
    result = build_4block_strassen()
    
    if validate(result["U"], result["V"], result["W"]):
        return result
    else:
        return schoolbook()
