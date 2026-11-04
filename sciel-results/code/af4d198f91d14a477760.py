import numpy as np
import time

def construct():
    n, m, p = 2, 3, 9
    nm, mp, pn = n * m, m * p, p * n
    
    # Build the target tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
                
    def to_list(mat):
        return [[float(x) for x in row] for row in mat]

    # Schoolbook solution as fallback
    rank_schoolbook = n * m * p
    U_sb = np.zeros((nm, rank_schoolbook))
    V_sb = np.zeros((mp, rank_schoolbook))
    W_sb = np.zeros((pn, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
                
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try to find a rank 16 decomposition for 2x3x3
    # If found, use it to build a rank 48 decomposition for 2x3x9
    # T_block is 6x9x6
    T_block = np.zeros((6, 9, 6))
    for i in range(2):
        for j in range(3):
            for l in range(3):
                T_block[i*3+j, j*3+l, l*2+i] = 1.0
                
    def is_half_integer(mat, tol=1e-6):
        scaled = 2.0 * mat
        rounded = np.round(scaled)
        if np.all(np.abs(scaled - rounded) < tol):
            return True
        return False

    def verify_solution(U, V, W, T, tol=1e-9):
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_approx, T, atol=tol)

    found_block = False
    U_block = None
    V_block = None
    W_block = None
    
    # Quick search for rank 16 in 2x3x3
    # Time limit 10s
    deadline = time.time() + 10
    for r in range(16, 19): # Try 16, 17, 18
        if time.time() > deadline:
            break
        for trial in range(50):
            if time.time() > deadline:
                break
            np.random.seed(42 + r * 100 + trial)
            V = np.random.randint(-1, 2, size=(9, r))
            W = np.random.randint(-1, 2, size=(6, r))
            M = (V[:, None, :] * W[None, :, :]).reshape(54, r)
            if np.linalg.matrix_rank(M) < r:
                continue
            U = np.zeros((6, r))
            valid = True
            for a in range(6):
                target = T_block[a, :, :].flatten()
                try:
                    sol, residuals, rank, sv = np.linalg.lstsq(M, target, rcond=None)
                    U[a, :] = sol
                except:
                    valid = False
                    break
            if not valid:
                continue
            if is_half_integer(U):
                U_snapped = np.round(2.0 * U) / 2.0
                if verify_solution(U_snapped, V, W, T_block):
                    U_block, V_block, W_block = U_snapped, V, W
                    best_rank = 3 * r
                    found_block = True
                    break
        if found_block:
            break
            
    if found_block:
        # Construct full 2x3x9 decomposition from 2x3x3 blocks
        rank_full = 3 * len(U_block[0])
        U_full = np.zeros((6, rank_full))
        V_full = np.zeros((27, rank_full))
        W_full = np.zeros((18, rank_full))
        
        for l in range(3): # Block index
            for r_idx in range(len(U_block[0])):
                R = l * len(U_block[0]) + r_idx
                # U is same for all blocks
                U_full[:, R] = U_block[:, r_idx]
                
                # V: j in 0..2, k in 0..8. k = 3*l + s (s in 0..2)
                # V_block index: j*3 + s
                for j in range(3):
                    for s in range(3):
                        k = 3*l + s
                        V_full[j*9 + k, R] = V_block[j*3 + s, r_idx]
                        
                # W: k in 0..8, i in 0..1. k = 3*l + s
                # W_block index: s*2 + i
                for s in range(3):
                    for i in range(2):
                        k = 3*l + s
                        W_full[k*2 + i, R] = W_block[s*2 + i, r_idx]
                        
        best_U, best_V, best_W = U_full, V_full, W_full
    else:
        best_U, best_V, best_W = U_sb, V_sb, W_sb
        best_rank = 54

    U_list = to_list(best_U)
    V_list = to_list(best_V)
    W_list = to_list(best_W)
    
    return {"U": U_list, "V": V_list, "W": W_list}
