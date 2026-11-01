import numpy as np
import time
import random

def _verify_fast(U, V, W, n=2, m=4, p=8, trials=50, tol=1e-9):
    """
    Fast vectorized verification.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    """
    rank = U.shape[1]
    rng = np.random.RandomState(42)
    
    U_reshaped = U.reshape(n, m, rank)
    V_reshaped = V.reshape(m, p, rank)
    W_reshaped = W.reshape(p, n, rank)
    
    for _ in range(trials):
        A = rng.randn(n, m)
        B = rng.randn(m, p)
        C_expected = A @ B
        
        factor_A = np.einsum('ij,ijc->ic', A, U_reshaped)
        factor_B = np.einsum('lk,lkc->kc', B, V_reshaped)
        
        C_computed = np.einsum('kic,ic,kc->ik', W_reshaped, factor_A, factor_B)
        
        if np.max(np.abs(C_computed - C_expected)) > tol:
            return False
    return True

def _get_strassen_2x2x2():
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _construct_rank_56():
    n, m, p = 2, 4, 8
    U_total = np.zeros((n*m, 56))
    V_total = np.zeros((m*p, 56))
    W_total = np.zeros((p*n, 56))
    
    U_str, V_str, W_str = _get_strassen_2x2x2()
    
    for t in range(4):
        # Block t handles B columns 2t, 2t+1
        # Splits A into left (cols 0,1) and right (cols 2,3)
        # Splits B block into top (rows 0,1) and bottom (rows 2,3)
        
        # Part 1: A_left * B_top
        base1 = t * 14
        for c in range(7):
            col_idx = base1 + c
            
            # U: A_left (cols 0,1)
            for i in range(2):
                for j in range(2):
                    val = U_str[i*2+j, c]
                    if val != 0:
                        U_total[i*4+j, col_idx] += val
                        
            # V: B_top (rows 0,1, cols 2t, 2t+1)
            for l in range(2):
                for k in range(2):
                    val = V_str[l*2+k, c]
                    if val != 0:
                        V_total[l*8 + (2*t + k), col_idx] += val
                        
            # W: C_sub (rows 0,1, cols 2t, 2t+1)
            for k in range(2):
                for i in range(2):
                    val = W_str[k*2+i, c]
                    if val != 0:
                        W_total[(2*t + k)*2 + i, col_idx] += val
                        
        # Part 2: A_right * B_bot
        base2 = t * 14 + 7
        for c in range(7):
            col_idx = base2 + c
            
            # U: A_right (cols 2,3)
            for i in range(2):
                for j in range(2):
                    val = U_str[i*2+j, c]
                    if val != 0:
                        U_total[i*4+(2+j), col_idx] += val
                        
            # V: B_bot (rows 2,3, cols 2t, 2t+1)
            for l in range(2):
                for k in range(2):
                    val = V_str[l*2+k, c]
                    if val != 0:
                        V_total[(2+l)*8 + (2*t + k), col_idx] += val
                        
            # W: C_sub (rows 0,1, cols 2t, 2t+1)
            for k in range(2):
                for i in range(2):
                    val = W_str[k*2+i, c]
                    if val != 0:
                        W_total[(2*t + k)*2 + i, col_idx] += val
                        
    return U_total, V_total, W_total

def construct():
    start_time = time.time()
    deadline = start_time + 130
    
    n, m, p = 2, 4, 8
    
    # Get baseline rank 56
    U, V, W = _construct_rank_56()
    
    if not _verify_fast(U, V, W, n, m, p, trials=20):
        # Fallback to schoolbook if construction is broken
        rank = n*m*p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Attempt local search to reduce rank
    # Try to see if any column is redundant?
    # This is hard. 
    # Instead, just return the verified 56.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
