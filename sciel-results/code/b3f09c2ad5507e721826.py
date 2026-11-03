import numpy as np
import time

DIMENSIONS = [4, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    n, m, p = DIMENSIONS
    nm = n * m  # 16
    mp = m * p  # 32
    pn = p * n  # 32
    
    # Schoolbook decomposition as fallback
    rank_school = n * m * p  # 128
    U_school = np.zeros((nm, rank_school))
    V_school = np.zeros((mp, rank_school))
    W_school = np.zeros((pn, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1
                V_school[j*p + k, col] = 1
                W_school[k*n + i, col] = 1
                col += 1
    
    # Strassen-based decomposition for 4x4 * 4x8
    # Split 4x4 into 2x2 blocks, use Strassen's 7 multiplications
    # For each 2x2 block pair, we need 7 multiplications
    # 4x4 has 4 block positions, so 4*7 = 28 rank terms
    # But we need to handle the 4x8 output properly
    
    # Let's build a Strassen decomposition for 4x4 * 4x4 first,
    # then extend to 4x8 by doubling the output columns
    
    # For 4x4 * 4x4, split into 2x2 blocks:
    # A = [[A00, A01], [A10, A11]] where each block is 2x2
    # B = [[B00, B01], [B10, B11]]
    # C = A*B = [[C00, C01], [C10, C11]]
    
    # Strassen for 2x2 blocks:
    # m1 = (A00 + A11) * (B00 + B11)
    # m2 = (A10 + A11) * B00
    # m3 = A00 * (B01 - B11)
    # m4 = A11 * (B10 - B00)
    # m5 = (A00 + A10) * B11
    # m6 = (A01 - A11) * (B10 + B11)
    # m7 = (A10 - A00) * (B00 + B01)
    
    # C00 = m1 - m4 + m5 - m7
    # C01 = m3 + m5
    # C10 = m2 + m4
    # C11 = m1 - m2 + m3 + m6
    
    # For 4x4 * 4x8, we have:
    # A is 4x4, B is 4x8
    # Split B into two 4x4 halves: B = [B_left, B_right]
    # C = A*B = [A*B_left, A*B_right]
    # Each A*B_half is 4x4, so we can use Strassen for each
    
    # This gives us 2 * 7 = 14 rank terms? No, that's not right.
    # Actually, we need to be more careful.
    
    # Let me try a different approach: use the fact that 4x4*4x8
    # can be seen as 4x4*4x4 twice, giving 2*28 = 56 rank terms?
    # Or can we do better?
    
    # Actually, let's just use the schoolbook for now and try to optimize
    # using alternating minimization
    
    best_rank = rank_school
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Try alternating minimization to reduce rank
    for target_rank in [98, 90, 80, 70]:
        if time.time() > deadline - 20:
            break
        
        np.random.seed(42)
        U = np.random.randn(nm, target_rank) * 0.1
        V = np.random.randn(mp, target_rank) * 0.1
        W = np.random.randn(pn, target_rank) * 0.1
        
        # Alternating minimization
        for iteration in range(200):
            if time.time() > deadline - 10:
                break
            
            # Update U
            for i in range(n):
                for j in range(m):
                    # For fixed i,j, we need:
                    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all k
                    # This is a linear system in U[i*m+j, :]
                    A_mat = np.zeros((p, target_rank))
                    for k in range(p):
                        A_mat[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    
                    try:
                        U[i*m+j, :], _, _, _ = np.linalg.lstsq(A_mat.T, np.ones(p), rcond=None)
                    except:
                        pass
            
            # Update V
            for j in range(m):
                for k in range(p):
                    # For fixed j,k, we need:
                    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i
                    A_mat = np.zeros((n, target_rank))
                    for i in range(n):
                        A_mat[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    
                    try:
                        V[j*p+k, :], _, _, _ = np.linalg.lstsq(A_mat.T, np.ones(n), rcond=None)
                    except:
                        pass
            
            # Update W
            for k in range(p):
                for i in range(n):
                    # For fixed k,i, we need:
                    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all j
                    A_mat = np.zeros((m, target_rank))
                    for j in range(m):
                        A_mat[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    
                    try:
                        W[k*n+i, :], _, _, _ = np.linalg.lstsq(A_mat.T, np.ones(m), rcond=None)
                    except:
                        pass
        
        # Check validity
        max_err = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    max_err = max(max_err, abs(val - 1))
        
        if max_err < 1e-6:
            best_rank = target_rank
            best_U, best_V, best_W = U, V, W
            break
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
