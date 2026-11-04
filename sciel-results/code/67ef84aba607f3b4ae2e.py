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

    # Schoolbook solution
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
                
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_schoolbook

    # Try to find rank-5 decomposition for 2x3x3 block
    # T_block size 6x9x6
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
    
    # Search for rank 5 in 2x3x3
    deadline = time.time() + 10
    for r in range(5, 19):
        if time.time() > deadline:
            break
        for trial in range(100):
            if time.time() > deadline:
                break
            np.random.seed(42 + r * 100 + trial)
            # Random initialization
            V = np.random.randn(9, r)
            W = np.random.randn(6, r)
            # Normalize to avoid scaling issues
            V /= np.linalg.norm(V, axis=0, keepdims=True)
            W /= np.linalg.norm(W, axis=0, keepdims=True)
            
            # ALS iterations
            U = np.zeros((6, r))
            for it in range(20):
                # Solve for U
                M = (V[:, None, :] * W[None, :, :]).reshape(54, r)
                # M shape (54, r). Target T_block reshaped to (6, 54)
                # T_block[a, b, c] -> M[b, r] * U[a, r]
                # We want sum_r U[a,r] M[b,r] = T_block[a, b, c]
                # This is not a simple linear system in U if M is fixed?
                # T_block[a, :, :] = sum_r U[a,r] (V[:,r] * W[:,r]).reshape(9,6)
                # Let M_r = (V[:,r] * W[:,r]).reshape(54)
                # Then T_block[a, :, :].flatten() = sum_r U[a,r] M_r
                # This is a linear system: U * M^T = T_block_flat
                # U (6, r) * M^T (r, 54) = T_block_flat (6, 54)
                try:
                    U, _, _, _ = np.linalg.lstsq(M.T, T_block.reshape(6, 54).T, rcond=None)
                    U = U.T
                except:
                    break
                
                # Solve for V
                # T_block[a, b, c] = sum_r U[a,r] V[b,r] W[c,r]
                # Fix U, W. Solve for V.
                # For each b, T_block[:, b, :].flatten() = sum_r V[b,r] (U[:,r] * W[:,r]).flatten()
                # Let N_r = (U[:,r] * W[:,r]).flatten() (size 36)
                # T_block[:, b, :].flatten() (size 36) = sum_r V[b,r] N_r
                # V (9, r) * N^T (r, 36) = T_block_reshaped (9, 36)
                N = (U[:, None, :] * W[None, :, :]).reshape(36, r)
                try:
                    V, _, _, _ = np.linalg.lstsq(N.T, T_block.reshape(6, 9, 6).transpose(1, 0, 2).reshape(9, 36).T, rcond=None)
                    V = V.T
                except:
                    break
                    
                # Solve for W
                # T_block[a, b, c] = sum_r U[a,r] V[b,r] W[c,r]
                # For each c, T_block[:, :, c].flatten() = sum_r W[c,r] (U[:,r] * V[:,r]).flatten()
                # Let P_r = (U[:,r] * V[:,r]).flatten() (size 54)
                # T_block[:, :, c].flatten() (size 54) = sum_r W[c,r] P_r
                # W (6, r) * P^T (r, 54) = T_block_reshaped (6, 54)
                P = (U[:, None, :] * V[None, :, :]).reshape(54, r)
                try:
                    W, _, _, _ = np.linalg.lstsq(P.T, T_block.reshape(6, 9, 6).transpose(2, 0, 1).reshape(6, 54).T, rcond=None)
                    W = W.T
                except:
                    break
                
            # Check solution
            if verify_solution(U, V, W, T_block):
                if is_half_integer(U) and is_half_integer(V) and is_half_integer(W):
                    U_block = np.round(2.0 * U) / 2.0
                    V_block = np.round(2.0 * V) / 2.0
                    W_block = np.round(2.0 * W) / 2.0
                    if verify_solution(U_block, V_block, W_block, T_block):
                        found_block = True
                        break
        if found_block:
            break

    if found_block:
        # Construct full 2x3x9 decomposition
        rank_full = 3 * 5
        U_full = np.zeros((6, rank_full))
        V_full = np.zeros((27, rank_full))
        W_full = np.zeros((18, rank_full))
        
        for l in range(3):
            for r_idx in range(5):
                R = l * 5 + r_idx
                U_full[:, R] = U_block[:, r_idx]
                
                for j in range(3):
                    for s in range(3):
                        k = 3*l + s
                        V_full[j*9 + k, R] = V_block[j*3 + s, r_idx]
                        
                for s in range(3):
                    for i in range(2):
                        k = 3*l + s
                        W_full[k*2 + i, R] = W_block[s*2 + i, r_idx]
                        
        best_U, best_V, best_W = U_full, V_full, W_full
        best_rank = 15
    else:
        best_U, best_V, best_W = U_sb, V_sb, W_sb
        best_rank = 54

    U_list = to_list(best_U)
    V_list = to_list(best_V)
    W_list = to_list(best_W)
    
    return {"U": U_list, "V": V_list, "W": W_list}
