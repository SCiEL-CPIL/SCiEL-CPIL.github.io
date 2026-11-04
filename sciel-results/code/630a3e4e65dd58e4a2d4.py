import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    
    # Schoolbook as fallback
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    best = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    best_rank = rank_school

    # Precompute the flattened multiplication tensor M
    # M indices: (i, j, j', k, k', i') -> delta_{i,i'} delta_{j,j'} delta_{k,k'}
    # We need M reshaped into matrices for the linear systems.
    
    # M_mat_W: rows correspond to (i, j, j', k), cols to (k', i')
    # Size: (n*m*m*p) x (p*n)
    M_mat_W = np.zeros((n*m*m*p, p*n))
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for kp in range(p):
                    idx_row = i*m*m*p + j*m*p + jp*p + kp
                    idx_col = kp*n + i
                    M_mat_W[idx_row, idx_col] = 1.0
    M_vec_W = M_mat_W.flatten()

    # M_mat_V: rows correspond to (i, j, k', i'), cols to (j', k)
    # Size: (n*m*p*n) x (m*p)
    M_mat_V = np.zeros((n*m*p*n, m*p))
    for i in range(n):
        for j in range(m):
            for kp in range(p):
                for ip in range(n):
                    idx_row = i*m*p*n + j*p*n + kp*n + ip
                    idx_col = j*p + kp
                    M_mat_V[idx_row, idx_col] = 1.0
    M_vec_V = M_mat_V.flatten()

    # M_mat_U: rows correspond to (j', k, k', i'), cols to (i, j)
    # Size: (m*p*p*n) x (n*m)
    M_mat_U = np.zeros((m*p*p*n, n*m))
    for jp in range(m):
        for kp in range(p):
            for ip in range(n):
                for i in range(n):
                    idx_row = jp*p*p*n + kp*p*n + ip*n + i
                    idx_col = i*m + jp
                    M_mat_U[idx_row, idx_col] = 1.0
    M_vec_U = M_mat_U.flatten()

    def check_decomposition(U, V, W, tol=1e-6):
        n, m, p = DIMENSIONS
        rank = U.shape[1]
        # Reconstruct tensor and check
        # T[i,j,jp,kp,kp_out,ip_out] = sum_r U[i,j,r] V[jp,kp,r] W[kp_out,ip_out,r]
        # We want T[i,j,jp,kp,kp,ip] = 1 and 0 elsewhere.
        # Efficient check:
        # For each r, U_r, V_r, W_r.
        # Let's just do the full einsum for small sizes.
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        
        # Sum over r
        # Result shape: (n, m, m, p, p, n)
        # T[i, j, jp, kp, kp_out, ip_out]
        T = np.einsum('ijr,jkr,krno->ijjknno', U3, V3, W3)
        # Wait, indices:
        # U: i, j, r
        # V: jp, kp, r  (Note: V is m x p, so indices are jp, kp)
        # W: kp_out, ip_out, r (Note: W is p x n, so indices are kp_out, ip_out)
        
        # Let's correct the einsum string.
        # U3: i, j, r
        # V3: jp, kp, r
        # W3: kp_out, ip_out, r
        # Target: i, j, jp, kp, kp_out, ip_out
        T = np.einsum('ijr,jpr,kqr->ijjpkq', U3, V3, W3) # p is m, k is p, q is n? No.
        
        # Let's stick to explicit loops for check to avoid index confusion, it's small.
        for ip in range(n):
            for j in range(m):
                for jp in range(m):
                    for kp in range(p):
                        s = np.zeros((p, n))
                        for r in range(rank):
                            s += W3[:, :, r] * U3[ip, j, r] * V3[jp, kp, r]
                        expected = np.zeros((p, n))
                        expected[kp, ip] = 1.0
                        if np.max(np.abs(s - expected)) > tol:
                            return False
        return True

    def alternating_minimization(rank, max_iter=500, seed=0):
        """Vectorized Alternating Minimization."""
        n, m, p = DIMENSIONS
        np.random.seed(seed)
        
        U = np.random.randn(n*m, rank)
        V = np.random.randn(m*p, rank)
        W = np.random.randn(p*n, rank)
        
        prev_err = float('inf')
        
        for it in range(max_iter):
            # 1. Solve for W
            # A_W: rows (i,j,jp,kp), cols r. A_W[idx, r] = U[i,j,r] * V[jp,kp,r]
            U_reshaped = U.reshape(n, m, rank)
            V_reshaped = V.reshape(m, p, rank)
            
            # Compute A_W
            # U_reshaped: (n, m, R)
            # V_reshaped: (m, p, R)
            # We need A_W[i, j, jp, kp, r] = U[i, j, r] * V[jp, kp, r]
            # Broadcast: U[i,j,r] * V[jp,kp,r]
            # Result shape: (n, m, m, p, R)
            A_W_5d = U_reshaped[:, :, np.newaxis, np.newaxis, :] * V_reshaped[np.newaxis, np.newaxis, :, :, :]
            A_W = A_W_5d.reshape(n*m*m*p, rank)
            
            K_W = np.kron(np.eye(p*n), A_W)
            W_vec, _, _, _ = np.linalg.lstsq(K_W, M_vec_W, rcond=None)
            W = W_vec.reshape(p*n, rank)
            
            # 2. Solve for V
            # B_V: rows (i,j,kp,ip), cols r. B_V[idx, r] = U[i,j,r] * W[kp,ip,r]
            W_reshaped = W.reshape(p, n, rank)
            
            # B_V[i, j, kp, ip, r] = U[i, j, r] * W[kp, ip, r]
            B_V_5d = U_reshaped[:, :, np.newaxis, np.newaxis, :] * W_reshaped[np.newaxis, np.newaxis, :, :, :]
            B_V = B_V_5d.reshape(n*m*p*n, rank)
            
            K_V = np.kron(np.eye(m*p), B_V)
            V_vec, _, _, _ = np.linalg.lstsq(K_V, M_vec_V, rcond=None)
            V = V_vec.reshape(m*p, rank)
            
            # 3. Solve for U
            # C_U: rows (jp,kp,kp_out,ip_out), cols r. C_U[idx, r] = V[jp,kp,r] * W[kp_out,ip_out,r]
            # Note: V indices are jp, kp. W indices are kp_out, ip_out.
            # C_U[jp, kp, kp_out, ip_out, r] = V[jp, kp, r] * W[kp_out, ip_out, r]
            C_U_5d = V_reshaped[:, :, np.newaxis, np.newaxis, :] * W_reshaped[np.newaxis, np.newaxis, :, :, :]
            C_U = C_U_5d.reshape(m*p*p*n, rank)
            
            K_U = np.kron(np.eye(n*m), C_U)
            U_vec, _, _, _ = np.linalg.lstsq(K_U, M_vec_U, rcond=None)
            U = U_vec.reshape(n*m, rank)
            
            # Calculate error
            # err = || T - M ||^2
            # T = sum_r U_r \otimes V_r \otimes W_r
            # We can compute the Frobenius norm of the difference.
            # T_flat is M_vec_W if we use the same flattening.
            # Let's compute T_mat_W and compare with M_mat_W.
            U_reshaped = U.reshape(n, m, rank)
            V_reshaped = V.reshape(m, p, rank)
            W_reshaped = W.reshape(p, n, rank)
            
            A_W_5d = U_reshaped[:, :, np.newaxis, np.newaxis, :] * V_reshaped[np.newaxis, np.newaxis, :, :, :]
            # T_mat_W[row, col] = sum_r A_W[row, r] * W[col, r]
            # T_mat_W = A_W @ W.T
            T_mat_W = A_W @ W.T
            
            err = np.sum((T_mat_W - M_mat_W)**2)
            
            if abs(prev_err - err) < 1e-10:
                break
            prev_err = err
            
            if err < 1e-8:
                break
        
        return U, V, W, err

    # Try ranks
    # Lower bound is roughly max(n*m, m*p, p*n) / something?
    # Start from 15 (half of schoolbook) up to 29
    for rank in range(15, 30):
        if time.time() > deadline - 20:
            break
        
        for trial in range(20):
            if time.time() > deadline - 15:
                break
            
            seed = int(time.time() * 1000) % 10000 + trial
            U, V, W, err = alternating_minimization(rank, max_iter=200, seed=seed)
            
            if err < 1e-6 and check_decomposition(U, V, W, tol=1e-5):
                best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = rank
                break
        
        if best_rank == rank:
            break
    
    return best
