import numpy as np
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Leave 15s safety margin
    
    n, m, p = 3, 3, 6
    
    # Build the target tensor T of shape (n*m, m*p, p*n)
    # T[im, jp, kn] = 1 if im = i*m+j, jp = j'*p+k, kn = k'*n+i
    # and j == j' (the matching index)
    # Actually, T[i*m+j, j'*p+k, k'*n+i] = 1 if j == j' else 0
    
    target = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m + j, j*p + k, k*n + i] = 1.0
    
    # Try to find rank-r decomposition
    # Start with schoolbook rank and try to reduce
    best_U, best_V, best_W = None, None, None
    best_rank = n * m * p  # 54
    
    # Schoolbook as fallback
    U_sb = np.zeros((n*m, best_rank))
    V_sb = np.zeros((m*p, best_rank))
    W_sb = np.zeros((p*n, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1
                V_sb[j*p + k, col] = 1
                W_sb[k*n + i, col] = 1
                col += 1
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()
    
    # Try various ranks with optimization
    # For each candidate rank, initialize and iterate
    
    ranks_to_try = [54, 50, 45, 40, 36, 32, 30, 27, 24]
    
    for target_rank in ranks_to_try:
        if time.time() > deadline:
            break
        
        # Try multiple initializations
        n_trials = 30
        found = False
        
        for trial in range(n_trials):
            if time.time() > deadline:
                break
            
            rng = np.random.RandomState(trial * 42 + target_rank)
            
            # Random initialization
            U = rng.randn(n*m, target_rank)
            V = rng.randn(m*p, target_rank)
            W = rng.randn(p*n, target_rank)
            
            # Alternating least squares
            # At each step, fix two matrices and solve for the third
            converged = False
            for iteration in range(200):
                if time.time() > deadline:
                    break
                
                # Compute the "residual tensor" contributions
                # T = sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]
                # To solve for U with V,W fixed:
                # For each row of U (index im), the equation is:
                # target[im, :, :] = sum_r U[im,r] * V[:,r] ⊗ W[:,r]
                
                # Reshape V and W into matrices for linear system
                # V is (m*p, rank), W is (p*n, rank)
                # V ⊗ W outer product per component: (m*p, p*n)
                # Stack these into a design matrix
                
                # Design matrix for U: shape (rank, m*p * p*n)
                # Actually, for each im, we have:
                # target[im].reshape(-1) = U[im] @ (stack of V[:,r] * W[:,r] for r)
                
                # Build design matrix D_U: (rank, m*p*p*n)
                # D_U[r, :].reshape(m*p, p*n) = V[:, r].reshape(m*p, 1) @ W[:, r].reshape(1, p*n)
                # = np.outer(V[:, r], W[:, r])
                
                D_U = np.zeros((target_rank, m*p * p*n))
                for r in range(target_rank):
                    D_U[r, :] = np.outer(V[:, r], W[:, r]).flatten()
                
                # Solve: target.reshape(n*m, m*p*p*n) = U @ D_U^T
                # So U = target.reshape(n*m, m*p*p*n) @ D_U^T @ (D_U @ D_U^T)^{-1}
                # Or use least squares: U[im] = argmin ||target[im] - D_U^T @ U[im]||
                
                tgt_reshaped = target.reshape(n*m, m*p * p*n)
                
                # Solve linear system: D_U.T @ U.T = tgt_reshaped.T
                # i.e., for each im: D_U.T @ U[im] = tgt_reshaped[im]
                
                try:
                    # Use pseudoinverse or lstsq
                    U_new, _, _, _ = np.linalg.lstsq(D_U.T, tgt_reshaped.T, rcond=None)
                    U = U_new.T
                except np.linalg.LinAlgError:
                    break
                
                # Solve for V with U,W fixed
                D_V = np.zeros((target_rank, n*m * p*n))
                for r in range(target_rank):
                    D_V[r, :] = np.outer(U[:, r], W[:, r]).flatten()
                
                # Reshape target for V: for each jp index
                # target[:, jp, :] reshaped to (n*m, p*n)
                # target_v[jp] = sum_r V[jp, r] * (U[:,r] ⊗ W[:,r])
                # So target_v[jp].reshape(-1) = V[jp] @ D_V^T
                # where target_v[jp] = target[:, jp, :].reshape(-1)
                
                tgt_v = np.zeros((m*p, n*m * p*n))
                for jp in range(m*p):
                    tgt_v[jp, :] = target[:, jp, :].flatten()
                
                try:
                    V_new, _, _, _ = np.linalg.lstsq(D_V.T, tgt_v, rcond=None)
                    V = V_new
                except np.linalg.LinAlgError:
                    break
                
                # Solve for W with U,V fixed
                D_W = np.zeros((target_rank, n*m * m*p))
                for r in range(target_rank):
                    D_W[r, :] = np.outer(U[:, r], V[:, r]).flatten()
                
                # For each kn index:
                # target[:, :, kn] reshaped to (n*m, m*p)
                # target_w[kn] = sum_r W[kn, r] * (U[:,r] ⊗ V[:,r])
                # target_w[kn].reshape(-1) = W[kn] @ D_W^T
                
                tgt_w = np.zeros((p*n, n*m * m*p))
                for kn in range(p*n):
                    tgt_w[kn, :] = target[:, :, kn].flatten()
                
                try:
                    W_new, _, _, _ = np.linalg.lstsq(D_W.T, tgt_w, rcond=None)
                    W = W_new
                except np.linalg.LinAlgError:
                    break
                
                # Check convergence
                # Compute current reconstruction error
                rec = np.zeros_like(target)
                for r in range(target_rank):
                    rec += np.outer(U[:, r], V[:, r]).reshape(n*m, 1, m*p) * W[:, r].reshape(1, p*n, 1)
                    # Actually: rec[i, j, k] += U[i,r] * V[j,r] * W[k,r]
                    # So rec = sum_r np.einsum('ir,jr,kr->ijk', U, V, W)
                    # But that's expensive, let's just check periodically
                
                if iteration % 20 == 0:
                    rec = np.zeros_like(target)
                    for r in range(target_rank):
                        rec += np.einsum('ir,jr,kr->ijk', U, V, W)
                    err = np.max(np.abs(rec - target))
                    if err < 1e-8:
                        converged = True
                        break
            else:
                if time.time() > deadline:
                    break
            
            if time.time() > deadline:
                break
            
            # Final error check
            rec = np.zeros_like(target)
            for r in range(target_rank):
                rec += np.einsum('ir,jr,kr->ijk', U, V, W)
            err = np.max(np.abs(rec - target))
            
            if err < 1e-6:
                # Round to integers or half-integers if possible
                U_rounded = np.round(U * 2) / 2.0
                V_rounded = np.round(V * 2) / 2.0
                W_rounded = np.round(W * 2) / 2.0
                
                # Check if rounded version still works
                rec_r = np.zeros_like(target)
                for r in range(target_rank):
                    rec_r += np.einsum('ir,jr,kr->ijk', U_rounded, V_rounded, W_rounded)
                err_r = np.max(np.abs(rec_r - target))
                
                if err_r < 1e-6:
                    U, V, W = U_rounded, V_rounded, W_rounded
                
                best_U, best_V, best_W = U, V, W
                best_rank = target_rank
                found = True
                break
        
        if found:
            # Don't break immediately; try to go lower
            continue
    
    if time.time() > deadline:
        break
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
