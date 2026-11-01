import time
import random
import numpy as np

DIMENSIONS = [2, 4, 8]

def _verify(U, V, W, n=2, m=4, p=8, trials=20, tol=1e-9):
    """Verify that the decomposition correctly computes matrix multiplication."""
    rank = len(U[0]) if U else 0
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    
    for _ in range(trials):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        C_computed = np.zeros((n, p))
        
        for c in range(rank):
            # factor_A[i] = sum_j A[i,j] * U[i*m+j, c]
            factor_A = np.zeros(n)
            for i in range(n):
                factor_A[i] = np.dot(A[i, :], U_np[i*m:(i+1)*m, c])
            
            # factor_B[k] = sum_j V[j*p+k, c] * B[j,k]
            factor_B = np.zeros(p)
            for k in range(p):
                # V indices for fixed k are j*p + k for j in 0..m-1
                # V_np is (m*p, rank). We need V_np[j*p+k, c]
                # This is not a contiguous slice.
                v_col = V_np[:, c]
                # v_col[j*p + k]
                factor_B[k] = np.dot(v_col[::p][0:m], B[:, k]) # Incorrect slicing
                # Let's just loop
                s = 0
                for j in range(m):
                    s += V_np[j*p + k, c] * B[j, k]
                factor_B[k] = s
            
            # C[i,k] += factor_A[i] * factor_B[k] * W[k*n+i, c]
            for i in range(n):
                for k in range(p):
                    C_computed[i, k] += factor_A[i] * factor_B[k] * W_np[k*n + i, c]
        
        if np.max(np.abs(C_computed - C_expected)) > tol:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Build schoolbook decomposition as fallback
    U_school = [[0] * schoolbook_rank for _ in range(n * m)]
    V_school = [[0] * schoolbook_rank for _ in range(m * p)]
    W_school = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    # Try Strassen rank-56 decomposition
    try:
        U = np.zeros((n*m, 56))
        V = np.zeros((m*p, 56))
        W = np.zeros((p*n, 56))
        
        u_strassen = [
            [1, 0, 0, 1],
            [0, 0, 1, 1],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [1, 0, 1, 0],
            [-1, 0, 1, 0],
            [0, 1, 0, -1]
        ]
        v_strassen = [
            [1, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 0, 1, -1],
            [-1, 1, 0, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 1]
        ]
        w_strassen = [
            [1, 0, 0, 1],
            [0, 0, 1, -1],
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [-1, 1, 0, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0]
        ]
        
        col_offset = 0
        for k_pair in range(4):
            # Part 1: A_L * B_T_part
            for s in range(7):
                for i in range(2):
                    for j in range(2):
                        U[i*2 + j, col_offset] = u_strassen[s][i*2 + j]
                for j in range(2):
                    for k in range(2):
                        V[j*8 + k_pair*2 + k, col_offset] = v_strassen[s][j*2 + k]
                for i in range(2):
                    for k in range(2):
                        W[(k_pair*2 + k)*2 + i, col_offset] = w_strassen[s][i*2 + k]
                col_offset += 1
            
            # Part 2: A_R * B_B_part
            for s in range(7):
                for i in range(2):
                    for j in range(2):
                        U[i*4 + (2 + j), col_offset] = u_strassen[s][i*2 + j]
                for j in range(2):
                    for k in range(2):
                        V[(2 + j)*8 + k_pair*2 + k, col_offset] = v_strassen[s][j*2 + k]
                for i in range(2):
                    for k in range(2):
                        W[(k_pair*2 + k)*2 + i, col_offset] = w_strassen[s][i*2 + k]
                col_offset += 1
        
        if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p, trials=10, tol=1e-9):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        else:
            # Fallback to schoolbook
            return {"U": U_school, "V": V_school, "W": W_school}
            
    except Exception:
        return {"U": U_school, "V": V_school, "W": W_school}

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Build schoolbook decomposition as fallback
    U_school = [[0] * schoolbook_rank for _ in range(n * m)]
    V_school = [[0] * schoolbook_rank for _ in range(m * p)]
    W_school = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    # Try Strassen rank-56 decomposition
    try:
        U = np.zeros((n*m, 56))
        V = np.zeros((m*p, 56))
        W = np.zeros((p*n, 56))
        
        u_strassen = [
            [1, 0, 0, 1],
            [0, 0, 1, 1],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [1, 0, 1, 0],
            [-1, 0, 1, 0],
            [0, 1, 0, -1]
        ]
        v_strassen = [
            [1, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 0, 1, -1],
            [-1, 1, 0, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 1]
        ]
        w_strassen = [
            [1, 0, 0, 1],
            [0, 0, 1, -1],
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [-1, 1, 0, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0]
        ]
        
        col_offset = 0
        for k_pair in range(4):
            # Part 1: A_L * B_T_part
            for s in range(7):
                for i in range(2):
                    for j in range(2):
                        U[i*2 + j, col_offset] = u_strassen[s][i*2 + j]
                for j in range(2):
                    for k in range(2):
                        V[j*8 + k_pair*2 + k, col_offset] = v_strassen[s][j*2 + k]
                for i in range(2):
                    for k in range(2):
                        W[(k_pair*2 + k)*2 + i, col_offset] = w_strassen[s][i*2 + k]
                col_offset += 1
            
            # Part 2: A_R * B_B_part
            for s in range(7):
                for i in range(2):
                    for j in range(2):
                        U[i*4 + (2 + j), col_offset] = u_strassen[s][i*2 + j]
                for j in range(2):
                    for k in range(2):
                        V[(2 + j)*8 + k_pair*2 + k, col_offset] = v_strassen[s][j*2 + k]
                for i in range(2):
                    for k in range(2):
                        W[(k_pair*2 + k)*2 + i, col_offset] = w_strassen[s][i*2 + k]
                col_offset += 1
        
        if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p, trials=10, tol=1e-9):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        else:
            # Fallback to schoolbook
            return {"U": U_school, "V": V_school, "W": W_school}
            
    except Exception:
        return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
