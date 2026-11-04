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

    # 2x3x3 block tensor
    T_block = np.zeros((6, 9, 6))
    for i in range(2):
        for j in range(3):
            for l in range(3):
                T_block[i*3+j, j*3+l, l*2+i] = 1.0
                
    def verify_solution(U, V, W, T, tol=1e-9):
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_approx, T, atol=tol)

    def verify_block(U, V, W, tol=1e-8):
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_approx, T_block, atol=tol)

    found_block = False
    U_block = None
    V_block = None
    W_block = None
    block_rank = 0

    # Use F_2^3 field multiplication structure to guide initialization
    # F_2^3 = F_2[x]/(x^3+x+1)
    # Multiplication tensor M (3,3,3)
    M = np.zeros((3,3,3))
    M[0,0,0] = 1; M[1,2,0] = 1; M[2,1,0] = 1
    M[0,1,1] = 1; M[1,0,1] = 1; M[1,2,1] = 1; M[2,1,1] = 1; M[2,2,1] = 1
    M[0,2,2] = 1; M[1,1,2] = 1; M[2,0,2] = 1; M[2,2,2] = 1

    # Find low-rank decomposition of field mult tensor
    deadline_field = time.time() + 15
    field_rank = 6
    field_U = field_V = field_W = None
    
    for r in range(3, 8):
        if time.time() > deadline_field:
            break
        for trial in range(150):
            if time.time() > deadline_field:
                break
            np.random.seed(5000 + r * 1000 + trial)
            V = np.random.randn(3, r)
            W = np.random.randn(3, r)
            V /= np.linalg.norm(V, axis=0, keepdims=True)
            W /= np.linalg.norm(W, axis=0, keepdims=True)
            U = np.zeros((3, r))
            for it in range(30):
                Mv = (V[:, None, :] * W[None, :, :]).reshape(9, r)
                try:
                    U_sol, _, _, _ = np.linalg.lstsq(Mv.T, M.reshape(3, 9).T, rcond=None)
                    U = U_sol.T
                except:
                    break
                Nv = (U[:, None, :] * W[None, :, :]).reshape(9, r)
                try:
                    V_sol, _, _, _ = np.linalg.lstsq(Nv.T, M.transpose(1,0,2).reshape(3,9).T, rcond=None)
                    V = V_sol.T
                except:
                    break
                Pv = (U[:, None, :] * V[None, :, :]).reshape(9, r)
                try:
                    W_sol, _, _, _ = np.linalg.lstsq(Pv.T, M.transpose(2,0,1).reshape(3,9).T, rcond=None)
                    W = W_sol.T
                except:
                    break
            M_approx = np.einsum('ar,br,cr->abc', U, V, W)
            if np.allclose(M_approx, M, atol=1e-8):
                field_rank = r
                field_U = U.copy()
                field_V = V.copy()
                field_W = W.copy()
                break
        if field_rank == r:
            break

    # Search for block decomposition, using field structure for initialization
    deadline_block = time.time() + 90
    for r in range(5, 20):
        if time.time() > deadline_block:
            break
        for trial in range(400):
            if time.time() > deadline_block:
                break
            
            if field_U is not None and r >= 2 * field_rank:
                # Initialize using field structure
                # Block tensor is 2 copies of field mult (for i=0,1)
                # U_block[i*3+j, r_idx] = field_U[j, r_idx % field_rank] for appropriate mapping
                np.random.seed(9000 + r * 1000 + trial)
                U = np.zeros((6, r))
                V = np.zeros((9, r))
                W = np.zeros((6, r))
                
                # For i=0: use first field_rank columns
                # For i=1: use next field_rank columns
                # But we need r >= 2*field_rank
                if r >= 2 * field_rank:
                    for j in range(3):
                        for rr in range(field_rank):
                            U[j, rr] = field_U[j, rr]
                            U[3+j, field_rank+rr] = field_U[j, rr]
                    for j in range(3):
                        for s in range(3):
                            for rr in range(field_rank):
                                V[j*3+s, rr] = field_V[j, rr]
                                V[j*3+s, field_rank+rr] = field_V[j, rr]
                    for s in range(3):
                        for i in range(2):
                            for rr in range(field_rank):
                                W[s*2+i, rr] = field_W[s, rr] if i == 0 else 0
                                W[s*2+i, field_rank+rr] = 0 if i == 0 else field_W[s, rr]
                    
                    # Add random perturbation for remaining columns if r > 2*field_rank
                    if r > 2 * field_rank:
                        U[:, 2*field_rank:] = np.random.randn(6, r - 2*field_rank)
                        V[:, 2*field_rank:] = np.random.randn(9, r - 2*field_rank)
                        W[:, 2*field_rank:] = np.random.randn(6, r - 2*field_rank)
                else:
                    break
            else:
                np.random.seed(42 + r * 1000 + trial)
                V = np.random.randn(9, r)
                W = np.random.randn(6, r)
                V /= np.linalg.norm(V, axis=0, keepdims=True)
                W /= np.linalg.norm(W, axis=0, keepdims=True)
                U = np.zeros((6, r))
            
            # ALS iterations
            for it in range(30):
                M_mat = (V[:, None, :] * W[None, :, :]).reshape(54, r)
                try:
                    U_sol, _, _, _ = np.linalg.lstsq(M_mat.T, T_block.reshape(6, 54).T, rcond=None)
                    U = U_sol.T
                except:
                    break
                
                N_mat = (U[:, None, :] * W[None, :, :]).reshape(36, r)
                try:
                    V_sol, _, _, _ = np.linalg.lstsq(N_mat.T, T_block.transpose(1, 0, 2).reshape(9, 36).T, rcond=None)
                    V = V_sol.T
                except:
                    break
                    
                P_mat = (U[:, None, :] * V[None, :, :]).reshape(54, r)
                try:
                    W_sol, _, _, _ = np.linalg.lstsq(P_mat.T, T_block.transpose(2, 0, 1).reshape(6, 54).T, rcond=None)
                    W = W_sol.T
                except:
                    break
            
            if verify_block(U, V, W):
                U_block = U.copy()
                V_block = V.copy()
                W_block = W.copy()
                block_rank = r
                found_block = True
                break
        if found_block:
            break

    if found_block:
        rank_full = 3 * block_rank
        U_full = np.zeros((6, rank_full))
        V_full = np.zeros((27, rank_full))
        W_full = np.zeros((18, rank_full))
        
        for l in range(3):
            for r_idx in range(block_rank):
                R = l * block_rank + r_idx
                U_full[:, R] = U_block[:, r_idx]
                
                for j in range(3):
                    for s in range(3):
                        k = 3*l + s
                        V_full[j*9 + k, R] = V_block[j*3 + s, r_idx]
                        
                for s in range(3):
                    for i in range(2):
                        k = 3*l + s
                        W_full[k*2 + i, R] = W_block[s*2 + i, r_idx]
                        
        if verify_solution(U_full, V_full, W_full, T):
            best_U, best_V, best_W = U_full, V_full, W_full
            best_rank = rank_full
        else:
            best_U, best_V, best_W = U_sb, V_sb, W_sb
            best_rank = 54
    else:
        best_U, best_V, best_W = U_sb, V_sb, W_sb
        best_rank = 54

    U_list = to_list(best_U)
    V_list = to_list(best_V)
    W_list = to_list(best_W)
    
    return {"U": U_list, "V": V_list, "W": W_list}
