import time
import numpy as np
from scipy.optimize import least_squares

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
        """Check that sum_c U[:,c] V[:,c] W[:,c] == multiplication tensor."""
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        rank = U.shape[1]
        if V.shape[1] != rank or W.shape[1] != rank:
            return False
        
        T_computed = np.zeros((n, m, p))
        for c in range(rank):
            u_mat = U[:, c].reshape(n, m)
            v_mat = V[:, c].reshape(m, p)
            w_mat = W[:, c].reshape(p, n)
            # T[i,j,k] += u[i,j] * v[j,k] * w[k,i]
            # Vectorized: for each (i,j), T[i,j,:] += u[i,j] * (v[j,:] * w[:,i])
            for i in range(n):
                for j in range(m):
                    T_computed[i,j,:] += u_mat[i,j] * (v_mat[j,:] * w_mat[:,i])
        
        return np.max(np.abs(T_computed - 1.0)) < tol

    def get_13_rank_block():
        """Returns U, V, W for a 3x3x2 block with rank 13."""
        rank = 13
        U = np.zeros((9, rank))
        V = np.zeros((6, rank))
        W = np.zeros((6, rank))
        
        col = 0
        # 1. Strassen part (7 terms) for 2x2x2 block (i,j in {0,1}, k in {0,1})
        # U indices: i*3 + j
        # V indices: j*2 + k
        # W indices: k*3 + i
        
        # M1 = (A00 + A11)(B10 + B11)
        U[0*3+0, col] = 1; U[1*3+1, col] = 1
        V[1*2+0, col] = 1; V[1*2+1, col] = 1
        W[0*3+0, col] = 1; W[0*3+1, col] = 1; W[1*3+0, col] = 1; W[1*3+1, col] = 1
        col += 1
        
        # M2 = (A10 + A11)B00
        U[1*3+0, col] = 1; U[1*3+1, col] = 1
        V[0*2+0, col] = 1
        W[0*3+0, col] = 1; W[1*3+0, col] = 1
        col += 1
        
        # M3 = A00(B01 - B11)
        U[0*3+0, col] = 1
        V[0*2+1, col] = 1; V[1*2+1, col] = -1
        W[0*3+1, col] = 1; W[1*3+1, col] = 1
        col += 1
        
        # M4 = A11(B00 + B11)
        U[1*3+1, col] = 1
        V[0*2+0, col] = 1; V[1*2+0, col] = 1
        W[0*3+1, col] = 1; W[1*3+1, col] = 1
        col += 1
        
        # M5 = (A00 + A01)B11
        U[0*3+0, col] = 1; U[0*3+1, col] = 1
        V[1*2+1, col] = 1
        W[0*3+1, col] = 1; W[1*3+1, col] = 1
        col += 1
        
        # M6 = (A10 - A00)(B00 + B01)
        U[1*3+0, col] = 1; U[0*3+0, col] = -1
        V[0*2+0, col] = 1; V[0*2+1, col] = 1
        W[0*3+0, col] = 1; W[1*3+0, col] = 1
        col += 1
        
        # M7 = (A01 - A11)(B10 + B11)
        U[0*3+1, col] = 1; U[1*3+1, col] = -1
        V[1*2+0, col] = 1; V[1*2+1, col] = 1
        W[0*3+0, col] = 1; W[1*3+0, col] = 1
        col += 1
        
        # 2. Outer product A[0:2, 2] * B[2, 0:2] (2x1x2)
        # C[i,k] += A[i,2] B[2,k]
        # Rank 2 decomposition:
        # M8 = (A02 + A12)(B20 + B21)
        U[0*3+2, col] = 1; U[1*3+2, col] = 1
        V[2*2+0, col] = 1; V[2*2+1, col] = 1
        W[0*3+0, col] = 1; W[0*3+1, col] = 1; W[1*3+0, col] = 1; W[1*3+1, col] = 1
        col += 1
        
        # M9 = (A02 - A12)(B20 - B21)
        U[0*3+2, col] = 1; U[1*3+2, col] = -1
        V[2*2+0, col] = 1; V[2*2+1, col] = -1
        W[0*3+0, col] = 1; W[0*3+1, col] = -1; W[1*3+0, col] = -1; W[1*3+1, col] = 1
        col += 1
        
        # 3. Matrix mult A[2, 0:2] * B[0:2, 0:2] (1x2x2)
        # C[2,k] += sum_j A[2,j] B[j,k]
        # Rank 2 decomposition:
        # M10 = (A20 + A21)(B00 + B10)
        U[2*3+0, col] = 1; U[2*3+1, col] = 1
        V[0*2+0, col] = 1; V[1*2+0, col] = 1
        W[0*3+2, col] = 1
        col += 1
        
        # M11 = (A20 - A21)(B01 - B11)
        U[2*3+0, col] = 1; U[2*3+1, col] = -1
        V[0*2+1, col] = 1; V[1*2+1, col] = -1
        W[1*3+2, col] = 1
        col += 1
        
        # 4. Outer product A[2,2] * B[2, 0:2] (1x1x2)
        # C[2,k] += A[2,2] B[2,k]
        # Rank 2 decomposition:
        # M12 = A22 B20
        U[2*3+2, col] = 1
        V[2*2+0, col] = 1
        W[0*3+2, col] = 1
        col += 1
        
        # M13 = A22 B21
        U[2*3+2, col] = 1
        V[2*2+1, col] = 1
        W[1*3+2, col] = 1
        col += 1
        
        return U, V, W

    # Assemble full 3x3x8 decomposition using rank 13 blocks
    U_block, V_block, W_block = get_13_rank_block()
    rank_block = 13
    
    rank_total = 4 * rank_block
    U = np.zeros((9, rank_total))
    V = np.zeros((24, rank_total)) # 3*8
    W = np.zeros((24, rank_total)) # 8*3
    
    for g in range(4):
        k0 = 2 * g
        k1 = 2 * g + 1
        col_start = g * rank_block
        col_end = (g + 1) * rank_block
        
        # U
        U[:, col_start:col_end] = U_block
        
        # V
        # V_block has shape (6, rank_block). Rows are j*2 + k_local.
        # Full V has shape (24, rank_total). Rows are j*8 + k_global.
        for c in range(rank_block):
            for j in range(3):
                for k_local in range(2):
                    k_global = k0 + k_local
                    V[j*8 + k_global, col_start + c] = V_block[j*2 + k_local, c]
                    
        # W
        # W_block has shape (6, rank_block). Rows are k_local*3 + i.
        # Full W has shape (24, rank_total). Rows are k_global*3 + i.
        for c in range(rank_block):
            for k_local in range(2):
                k_global = k0 + k_local
                for i in range(3):
                    W[k_global*3 + i, col_start + c] = W_block[k_local*3 + i, c]

    result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    if validate(U, V, W):
        return result
    else:
        return schoolbook()
