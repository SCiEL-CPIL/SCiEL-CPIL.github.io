"""Matrix multiplication 3,4,8 using rank-88 decomposition with efficient verification and rank reduction."""
import time
import numpy as np

DIMENSIONS = [3, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit (evaluator kills at 150s)
    
    n, m, p = DIMENSIONS
    
    # Build target tensor
    target = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Initial decomposition: rank-88
    rank = 88
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    # Part 1: i=0, schoolbook for 1x4x8 (rank 32)
    for j in range(4):
        for k in range(8):
            U[0 * m + j, col] = 1
            V[j * p + k, col] = 1
            W[k * n + 0, col] = 1
            col += 1
    
    # Part 2: i=1,2, Strassen on 2x2 blocks (rank 56)
    s_U = [
        [1, 0, 0, 1], 
        [0, 0, 1, 1], 
        [1, 0, 0, 0], 
        [0, 0, 0, 1], 
        [1, 1, 0, 0], 
        [-1, 0, 1, 0], 
        [0, 1, 0, -1]  
    ]
    s_V = [
        [1, 0, 0, 1], 
        [1, 0, 0, 0], 
        [0, 1, 0, -1], 
        [-1, 0, 1, 0], 
        [0, 0, 0, 1], 
        [1, 1, 0, 0], 
        [0, 0, 1, 1]  
    ]
    s_W = [
        [1, 0, 0, 1], 
        [0, 0, 1, -1], 
        [0, 1, 0, 1], 
        [1, 0, 1, 0], 
        [-1, 1, 0, 0], 
        [0, 0, 0, 1], 
        [1, 0, 0, 0]  
    ]
    
    for j_block in range(2):
        j_start = j_block * 2
        for k_block in range(4):
            k_start = k_block * 2
            for t in range(7):
                for l in range(4):
                    if s_U[t][l] != 0:
                        li = l // 2
                        lj = l % 2
                        gi = 1 + li
                        gj = j_start + lj
                        U[gi * m + gj, col] += s_U[t][l]
                    
                    if s_V[t][l] != 0:
                        lj = l // 2
                        lk = l % 2
                        gj = j_start + lj
                        gk = k_start + lk
                        V[gj * p + gk, col] += s_V[t][l]
                    
                    if s_W[t][l] != 0:
                        li = l // 2
                        lk = l % 2
                        gi = 1 + li
                        gk = k_start + lk
                        W[gk * n + gi, col] += s_W[t][l]
                col += 1
    
    # Efficient verification using matrix multiplication
    def verify_fast(U_mat, V_mat, W_mat):
        """Check if U, V, W decompose the target tensor using efficient contraction."""
        # T_recon[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # = sum_r (U[:,r] * W[c,r]) * V[b,r]
        # For fixed r: outer(U[:,r], V[:,r]) gives (a,b) matrix, then multiply by W[c,r]
        # More efficient: for each pair (a,b), T_recon[a,b,:] = sum_r U[a,r]*V[b,r]*W[:,r]
        # = (U[a,:] * V[b,:]) @ W.T  -- no, that's not right either
        
        # Best approach: use einsum which is optimized
        recon = np.einsum('ar,br,cr->abc', U_mat, V_mat, W_mat)
        return np.allclose(recon, target, atol=1e-10)
    
    # Verify initial decomposition
    if not verify_fast(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to reduce rank by finding linear dependencies
    # Strategy: for each column r, try to express it as a combination of others
    # by solving a least-squares problem on the "effective" contribution
    
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Precompute the "slice" contributions for faster checking
    # For each rank term r, its contribution to the tensor is:
    # contrib_r[a,b,c] = U[a,r] * V[b,r] * W[c,r]
    
    # Try removing each column and checking if the remaining columns can still span
    # We'll use a greedy approach: try to remove columns that are "redundant"
    
    for r_remove in range(rank):
        if time.time() > deadline - 5:
            break
        
        # Skip if already removed
        if r_remove >= U.shape[1]:
            continue
        
        new_rank = U.shape[1] - 1
        if new_rank < best_rank:
            # Check if we can remove this column
            # The contribution of column r_remove is:
            # c[a,b,c] = U[a,r_remove] * V[b,r_remove] * W[c,r_remove]
            # We need: sum_{s != r_remove} U[a,s]*V[b,s]*W[c,s] + alpha * U[a,r_remove]*V[b,r_remove]*W[c,r_remove] == target
            # This is equivalent to: can we find alpha such that the residual is zero?
            # Actually, we need the remaining terms to exactly equal target.
            
            U_new = np.delete(U, r_remove, axis=1)
            V_new = np.delete(V, r_remove, axis=1)
            W_new = np.delete(W, r_remove, axis=1)
            
            # Quick check: if the removed column is all zeros, it's definitely removable
            if np.allclose(U[:, r_remove], 0) and np.allclose(V[:, r_remove], 0) and np.allclose(W[:, r_remove], 0):
                if verify_fast(U_new, V_new, W_new):
                    best_rank = new_rank
                    best_U, best_V, best_W = U_new, V_new, W_new
                    U, V, W = U_new, V_new, W_new
                    # Restart loop with new rank
                    break
            else:
                # Try to see if the contribution can be absorbed
                # This is harder - skip for now, just check zero columns
                pass
    
    if best_rank < 88:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
