"""Matrix multiplication 3,4,8 - rank-minimal exact tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    # Build the multiplication tensor T with shape (n*m, m*p, p*n)
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Schoolbook as fallback
    rank_school = n * m * p
    U_sb = np.zeros((n * m, rank_school))
    V_sb = np.zeros((m * p, rank_school))
    W_sb = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1.0
                V_sb[j * p + k, col] = 1.0
                W_sb[k * n + i, col] = 1.0
                col += 1
    best_result = (U_sb, V_sb, W_sb)
    best_rank = rank_school
    
    np.random.seed(42)
    
    # Try ranks from lower bound upward
    for rank in range(32, best_rank + 1):
        if time.time() > deadline - 30.0:
            break
        
        found = False
        for trial in range(3):
            if time.time() > deadline - 10.0:
                break
            
            U = np.random.randn(n * m, rank) * 0.5
            V = np.random.randn(m * p, rank) * 0.5
            W = np.random.randn(p * n, rank) * 0.5
            
            for iteration in range(100):
                if time.time() > deadline - 5.0:
                    break
                
                # Solve for U: T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
                # M_U[r, idx] where idx = b*(p*n) + c, M_U[r, idx] = V[b,r]*W[c,r]
                M_U = np.zeros((rank, m * p * p * n))
                for r in range(rank):
                    M_U[r, :] = (V[:, r].reshape(m * p, 1) * W[:, r].reshape(1, p * n)).flatten()
                
                T_flat = T.reshape(n * m, -1)  # shape (n*m, m*p*p*n)
                # U @ M_U = T_flat => M_U.T @ U.T = T_flat.T
                try:
                    U_new = np.linalg.lstsq(M_U.T, T_flat.T, rcond=None)[0].T
                    U = U_new
                except:
                    break
                
                # Solve for V: T[a,b,c] = sum_r V[b,r] * U[a,r] * W[c,r]
                # For each b: T[:, b, :].flatten() = V[b,:] @ N where N[r, idx] = U[a,r]*W[c,r], idx = a*(p*n)+c
                N = np.zeros((rank, n * m * p * n))
                for r in range(rank):
                    N[r, :] = (U[:, r].reshape(n * m, 1) * W[:, r].reshape(1, p * n)).flatten()
                
                # T[:, b, :] has shape (n*m, p*n), flatten to (n*m*p*n,)
                T_for_V = T.transpose(1, 0, 2).reshape(m * p, -1)  # shape (m*p, n*m*p*n)
                try:
                    V_new = np.linalg.lstsq(N.T, T_for_V.T, rcond=None)[0].T
                    V = V_new
                except:
                    break
                
                # Solve for W: T[a,b,c] = sum_r W[c,r] * U[a,r] * V[b,r]
                # For each c: T[:, :, c].flatten() = W[c,:] @ P where P[r, idx] = U[a,r]*V[b,r], idx = a*(m*p)+b
                P = np.zeros((rank, n * m * m * p))
                for r in range(rank):
                    P[r, :] = (U[:, r].reshape(n * m, 1) * V[:, r].reshape(1, m * p)).flatten()
                
                T_for_W = T.transpose(2, 0, 1).reshape(p * n, -1)  # shape (p*n, n*m*m*p)
                try:
                    W_new = np.linalg.lstsq(P.T, T_for_W.T, rcond=None)[0].T
                    W = W_new
                except:
                    break
                
                # Check error
                # Compute max absolute error
                err = 0.0
                for a in range(n * m):
                    for b in range(m * p):
                        for c in range(p * n):
                            val = np.dot(U[a, :], V[b, :] * W[c, :])
                            err = max(err, abs(val - T[a, b, c]))
                
                if err < 1e-10:
                    found = True
                    break
            
            if found:
                # Try to snap to integers or half-integers
                for scale in [1, 2]:
                    U_snap = np.round(U * scale) / scale
                    V_snap = np.round(V * scale) / scale
                    W_snap = np.round(W * scale) / scale
                    
                    if (np.abs(U_snap - U) < 1e-6).all() and \
                       (np.abs(V_snap - V) < 1e-6).all() and \
                       (np.abs(W_snap - W) < 1e-6).all():
                        # Verify
                        valid = True
                        for a in range(n * m):
                            for b in range(m * p):
                                for c in range(p * n):
                                    val = np.dot(U_snap[a, :], V_snap[b, :] * W_snap[c, :])
                                    if abs(val - T[a, b, c]) > 1e-9:
                                        valid = False
                                        break
                                if not valid:
                                    break
                            if not valid:
                                break
                        if valid:
                            best_result = (U_snap, V_snap, W_snap)
                            best_rank = rank
                            break
                if best_rank <= rank:
                    break
        
        if best_rank <= rank:
            break
    
    U_best, V_best, W_best = best_result
    U_list = [[float(x) for x in row] for row in U_best]
    V_list = [[float(x) for x in row] for row in V_best]
    W_list = [[float(x) for x in row] for row in W_best]
    
    return {"U": U_list, "V": V_list, "W": W_list}
