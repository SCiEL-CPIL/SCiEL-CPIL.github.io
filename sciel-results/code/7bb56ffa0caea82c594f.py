import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    deadline = time.time() + 110  # 10s safety margin
    
    # Build the target tensor T
    # T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k, else 0
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Reshape for ALS
    T_mat = T.reshape(nm*mp, pn)  # (108, 12)
    
    best_rank = nm*mp*p  # schoolbook
    best_result = None
    
    # Schoolbook solution as fallback
    rank_school = nm*mp*p
    U_s = np.zeros((nm, rank_school))
    V_s = np.zeros((mp, rank_school))
    W_s = np.zeros((pn, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_s[i*m+j, col] = 1
                V_s[j*p+k, col] = 1
                W_s[k*n+i, col] = 1
                col += 1
    best_result = (U_s, V_s, W_s)
    best_rank = rank_school
    
    # Try ranks from 24 upward
    for R in range(24, 37):
        if time.time() > deadline - 10:
            break
        found = False
        for trial in range(50):
            if time.time() > deadline - 5:
                break
            # Random initialization
            np.random.seed(trial * 1000 + R)
            U = np.random.randn(nm, R) * 0.5
            V = np.random.randn(mp, R) * 0.5
            W = np.random.randn(pn, R) * 0.5
            
            # ALS iterations
            for it in range(200):
                # Solve for W: A_mat @ W^T = T_mat
                A_mat = np.zeros((nm*mp, R))
                for a in range(nm):
                    for b in range(mp):
                        A_mat[a*mp+b, :] = U[a, :] * V[b, :]
                # W^T = A_mat \ T_mat
                W_T, residuals, rank_a, sv = np.linalg.lstsq(A_mat, T_mat, rcond=None)
                W = W_T.T
                
                # Solve for U: for each (b,c), sum_a U[a,r] * (V[b,r]*W[c,r]) = T[a,b,c]
                # B_mat[(b,c), r] = V[b,r]*W[c,r]
                # U^T @ B_mat^T = T reshaped... 
                # Actually: for each (b,c): sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c] for all a
                # So: for each (b,c), the vector (T[:,b,c]) = U^T @ (V[b,:]*W[c,:])
                # Stack: B_mat[(b,c), r] = V[b,r]*W[c,r], shape (mp*pn, R)
                # T_reshaped: (mp*pn, nm) where T_reshaped[(b,c), a] = T[a,b,c]
                B_mat = np.zeros((mp*pn, R))
                for b in range(mp):
                    for c in range(pn):
                        B_mat[b*pn+c, :] = V[b, :] * W[c, :]
                T_bc = T.transpose(1,2,0).reshape(mp*pn, nm)  # T_bc[(b,c), a] = T[a,b,c]
                U_T, _, _, _ = np.linalg.lstsq(B_mat, T_bc, rcond=None)
                U = U_T.T
                
                # Solve for V: for each (a,c), sum_r V[b,r] * (U[a,r]*W[c,r]) = T[a,b,c] for all b
                C_mat = np.zeros((nm*pn, R))
                for a in range(nm):
                    for c in range(pn):
                        C_mat[a*pn+c, :] = U[a, :] * W[c, :]
                T_ac = T.transpose(0,2,1).reshape(nm*pn, mp)  # T_ac[(a,c), b] = T[a,b,c]
                V_T, _, _, _ = np.linalg.lstsq(C_mat, T_ac, rcond=None)
                V = V_T.T
            
            # Check if solution is exact
            error = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(R))
                        error += abs(val - 1.0)
            # Also check off-diagonal entries are ~0
            for a in range(nm):
                for b in range(mp):
                    for c in range(pn):
                        if T[a,b,c] == 0:
                            val = sum(U[a, r] * V[b, r] * W[c, r] for r in range(R))
                            error += abs(val)
            
            if error < 1e-6:
                # Try to snap to integers/half-integers
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                err_snap = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = sum(U_snap[i*m+j, r] * V_snap[j*p+k, r] * W_snap[k*n+i, r] for r in range(R))
                            err_snap += abs(val - 1.0)
                for a in range(nm):
                    for b in range(mp):
                        for c in range(pn):
                            if T[a,b,c] == 0:
                                val = sum(U_snap[a, r] * V_snap[b, r] * W_snap[c, r] for r in range(R))
                                err_snap += abs(val)
                if err_snap < 1e-6:
                    best_rank = R
                    best_result = (U_snap, V_snap, W_snap)
                    found = True
                    break
                else:
                    # Use the non-snapped version
                    best_rank = R
                    best_result = (U, V, W)
                    found = True
                    break
        
        if found:
            break
    
    U, V, W = best_result
    # Convert to lists
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    return {"U": U_list, "V": V_list, "W": W_list}
