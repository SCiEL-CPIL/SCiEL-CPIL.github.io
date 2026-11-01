import time
import random
import numpy as np

DIMENSIONS = [2, 4, 8]

def _verify_fast(U, V, W, n=2, m=4, p=8, trials=100, tol=1e-9):
    """Fast verification using vectorized numpy operations."""
    rank = len(U[0]) if U else 0
    U_mat = np.array(U)  # shape (n*m, rank)
    V_mat = np.array(V)  # shape (m*p, rank)
    W_mat = np.array(W)  # shape (p*n, rank)
    
    for _ in range(trials):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        # Reshape A and B for matrix multiplication with factors
        A_flat = A.flatten()  # shape (n*m,)
        B_flat = B.flatten()  # shape (m*p,)
        
        # Compute factor_A: for each rank component, sum over j
        # factor_A[i, c] = sum_j A[i,j] * U[i*m+j, c]
        # This is: reshape A to (n, m), then for each c, compute A @ U_col_c reshaped
        
        # More efficiently: 
        # U_mat is (n*m, rank), reshape to (n, m, rank)
        U_reshaped = U_mat.reshape(n, m, rank)  # (n, m, rank)
        # factor_A[i, c] = sum_j A[i,j] * U_reshaped[i,j,c]
        factor_A = np.einsum('ij,ijc->ic', A, U_reshaped)  # (n, rank)
        
        # V_mat is (m*p, rank), reshape to (m, p, rank)
        V_reshaped = V_mat.reshape(m, p, rank)  # (m, p, rank)
        # factor_B[k, c] = sum_j B[j,k] * V_reshaped[j,k,c]
        factor_B = np.einsum('jk,jkc->kc', B, V_reshaped)  # (p, rank)
        
        # W_mat is (p*n, rank), reshape to (p, n, rank)
        W_reshaped = W_mat.reshape(p, n, rank)  # (p, n, rank)
        
        # C_computed[i,k] = sum_c factor_A[i,c] * factor_B[k,c] * W_reshaped[k,i,c]
        C_computed = np.einsum('ic,kc,kic->ik', factor_A, factor_B, W_reshaped)
        
        if np.max(np.abs(C_computed - C_expected)) > tol:
            return False
    return True

def _try_rank_reduction(U, V, W, n=2, m=4, p=8, max_rank=56, time_limit=30):
    """Try to find a lower rank decomposition by removing redundant columns."""
    start_time = time.time()
    rank = len(U[0]) if U else 0
    
    # Try removing columns one by one
    U_cur = [row[:] for row in U]
    V_cur = [row[:] for row in V]
    W_cur = [row[:] for row in W]
    
    for target_rank in range(rank - 1, 0, -1):
        if time.time() - start_time > time_limit:
            break
        
        # Try to find a subset of columns of size target_rank that still works
        # This is expensive, so use a heuristic: try random subsets
        for attempt in range(10):
            if time.time() - start_time > time_limit:
                break
            
            if target_rank >= rank:
                cols = list(range(rank))
            else:
                cols = random.sample(range(rank), target_rank)
            
            U_test = [row[c] for row in U_cur for c in cols]
            # Reshape back
            U_test = [[U_cur[i*m+j][c] for c in cols] for i in range(n) for j in range(m)]
            V_test = [[V_cur[j*p+k][c] for c in cols] for j in range(m) for k in range(p)]
            W_test = [[W_cur[k*n+i][c] for c in cols] for k in range(p) for i in range(n)]
            
            if _verify_fast(U_test, V_test, W_test, n, m, p, trials=20, tol=1e-9):
                return U_test, V_test, W_test, target_rank
    
    return None, None, None, rank

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    n, m, p = DIMENSIONS
    rank = 56
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen factors for 2x2x2
    U_strassen = [
        [1, 0, 0, 1], # P1
        [0, 0, 1, 1], # P2
        [1, 0, 0, 0], # P3
        [0, 0, 0, 1], # P4
        [1, 1, 0, 0], # P5
        [0, 0, 1, -1],# P6
        [0, 1, 0, 1]  # P7
    ]
    
    V_strassen = [
        [1, 0, 0, 1], # P1
        [0, 0, 0, 1], # P2
        [1, 0, -1, 0],# P3
        [0, -1, 0, 1],# P4
        [0, 0, 1, 0], # P5
        [1, 1, 0, 0], # P6
        [0, 0, 1, 1]  # P7
    ]
    
    W_strassen = [
        [1, 0, 0, 1, -1, 0, 1], # C00
        [0, 0, 1, 0, 1, 0, 0],  # C01
        [0, 1, 0, 0, 0, 1, 0],  # C10
        [1, -1, 1, 0, 0, 1, 0]  # C11
    ]
    
    # Define blocks
    A1_idx = [0, 1, 4, 5]
    A2_idx = [2, 3, 6, 7]
    
    B_top_0 = [0*8+0, 0*8+1, 1*8+0, 1*8+1]
    B_top_1 = [0*8+2, 0*8+3, 1*8+2, 1*8+3]
    B_top_2 = [0*8+4, 0*8+5, 1*8+4, 1*8+5]
    B_top_3 = [0*8+6, 0*8+7, 1*8+6, 1*8+7]
    
    B_bot_0 = [2*8+0, 2*8+1, 3*8+0, 3*8+1]
    B_bot_1 = [2*8+2, 2*8+3, 3*8+2, 3*8+3]
    B_bot_2 = [2*8+4, 2*8+5, 3*8+4, 3*8+5]
    B_bot_3 = [2*8+6, 2*8+7, 3*8+6, 3*8+7]
    
    C_01 = [0*2+0, 0*2+1, 1*2+0, 1*2+1]
    C_23 = [2*2+0, 2*2+1, 3*2+0, 3*2+1]
    C_45 = [4*2+0, 4*2+1, 5*2+0, 5*2+1]
    C_67 = [6*2+0, 6*2+1, 7*2+0, 7*2+1]
    
    blocks = []
    for i in range(4):
        blocks.append((A1_idx, [B_top_0, B_top_1, B_top_2, B_top_3][i], [C_01, C_23, C_45, C_67][i]))
        blocks.append((A2_idx, [B_bot_0, B_bot_1, B_bot_2, B_bot_3][i], [C_01, C_23, C_45, C_67][i]))
    
    col_offset = 0
    for a_idx, b_idx, c_idx in blocks:
        for f in range(7):
            col = col_offset + f
            for r in range(4):
                U[a_idx[r]][col] = U_strassen[f][r]
                V[b_idx[r]][col] = V_strassen[f][r]
                W[c_idx[r]][col] = W_strassen[r][f]
        col_offset += 7
    
    # Fast verification
    if _verify_fast(U, V, W, n, m, p, trials=50, tol=1e-9):
        # Try to reduce rank if time allows
        if time.time() - start_time < deadline - 30:
            U_red, V_red, W_red, new_rank = _try_rank_reduction(U, V, W, n, m, p, rank, time_limit=25)
            if U_red is not None:
                return {"U": U_red, "V": V_red, "W": W_red}
        return {"U": U, "V": W, "W": W}
    else:
        # Fallback to schoolbook
        schoolbook_rank = n * m * p
        U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
        V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
        W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j][col] = 1
                    V_sb[j * p + k][col] = 1
                    W_sb[k * n + i][col] = 1
                    col += 1
        return {"U": U_sb, "V": V_sb, "W": W_sb}
