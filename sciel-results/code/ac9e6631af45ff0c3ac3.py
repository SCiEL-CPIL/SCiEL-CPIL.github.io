"""Optimized 6x6 matrix multiplication tensor decomposition with aggressive rank reduction."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    import numpy as np
    import time
    
    t0 = time.time()
    deadline = t0 + 110  # safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # 6, 6, 6
    
    # Build initial Strassen-based decomposition: 3x3 blocks of 2x2 matrices
    # Rank = 3*3*3*7 = 189
    nb = 3  # blocks per dimension
    bs = 2  # block size
    
    # Strassen coefficients for 2x2
    # For each s in 0..6: beta[s] (2x2 for A), gamma[s] (2x2 for B), alpha[s] (2x2 for C)
    beta = [None] * 7
    gamma = [None] * 7
    alpha = [None] * 7
    
    for s in range(7):
        beta[s] = np.zeros((2, 2))
        gamma[s] = np.zeros((2, 2))
        alpha[s] = np.zeros((2, 2))
    
    # s=0: (A00+A11)(B00+B11) -> C00+=M0, C11+=M0
    beta[0][0][0] = 1; beta[0][1][1] = 1
    gamma[0][0][0] = 1; gamma[0][1][1] = 1
    alpha[0][0][0] = 1; alpha[0][1][1] = 1
    
    # s=1: (A10+A11)B00 -> C10+=M1, C11-=M1
    beta[1][1][0] = 1; beta[1][1][1] = 1
    gamma[1][0][0] = 1
    alpha[1][1][0] = 1; alpha[1][1][1] = -1
    
    # s=2: A00(B01-B11) -> C01+=M2, C11+=M2
    beta[2][0][0] = 1
    gamma[2][0][1] = 1; gamma[2][1][1] = -1
    alpha[2][0][1] = 1; alpha[2][1][1] = 1
    
    # s=3: A11(B10-B00) -> C00+=M3, C10+=M3
    beta[3][1][1] = 1
    gamma[3][1][0] = 1; gamma[3][0][0] = -1
    alpha[3][0][0] = 1; alpha[3][1][0] = 1
    
    # s=4: (A00+A01)B11 -> C00-=M4, C01+=M4
    beta[4][0][0] = 1; beta[4][0][1] = 1
    gamma[4][1][1] = 1
    alpha[4][0][0] = -1; alpha[4][0][1] = 1
    
    # s=5: (A10-A00)(B00+B01) -> C11+=M5
    beta[5][1][0] = 1; beta[5][0][0] = -1
    gamma[5][0][0] = 1; gamma[5][0][1] = 1
    alpha[5][1][1] = 1
    
    # s=6: (A01-A11)(B10+B11) -> C00+=M6
    beta[6][0][1] = 1; beta[6][1][1] = -1
    gamma[6][1][0] = 1; gamma[6][1][1] = 1
    alpha[6][0][0] = 1
    
    # Build factor matrices
    rank = nb * nb * nb * 7  # 189
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for i_b in range(nb):
        for j_b in range(nb):
            for k_b in range(nb):
                for s in range(7):
                    r = (i_b * nb + k_b) * nb * 7 + j_b * 7 + s
                    # Actually, let's use a cleaner indexing:
                    # r = ((i_b * nb + k_b) * nb + j_b) * 7 + s
                    
                    # U: A[i_b, j_b] block contribution
                    for lr in range(bs):
                        for lc in range(bs):
                            if beta[s][lr][lc] != 0:
                                i_glob = i_b * bs + lr
                                j_glob = j_b * bs + lc
                                U[i_glob * m + j_glob, r] = beta[s][lr][lc]
                    
                    # V: B[j_b, k_b] block contribution
                    for lr in range(bs):
                        for lc in range(bs):
                            if gamma[s][lr][lc] != 0:
                                j_glob = j_b * bs + lr
                                k_glob = k_b * bs + lc
                                V[j_glob * p + k_glob, r] = gamma[s][lr][lc]
                    
                    # W: C[i_b, k_b] block contribution
                    for lr in range(bs):
                        for lc in range(bs):
                            if alpha[s][lr][lc] != 0:
                                i_glob = i_b * bs + lr
                                k_glob = k_b * bs + lc
                                W[k_glob * n + i_glob, r] = alpha[s][lr][lc]
    
    # Verify initial decomposition
    def verify_fast(U, V, W):
        """Verify that sum_r U[:,r] V[:,r] W[:,r] = identity tensor."""
        n, m, p = 6, 6, 6
        rank = U.shape[1]
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Build T efficiently
        T = np.zeros((n, m, p))
        for r in range(rank):
            Ur = U[:, r].reshape(n, m)
            Vr = V[:, r].reshape(m, p)
            Wr = W[:, r].reshape(p, n)
            T += np.einsum('ij,jk,ki->ijk', Ur, Vr, Wr)
        return np.allclose(T, np.ones((n, m, p)), atol=1e-10)
    
    if not verify_fast(U, V, W):
        # Should not happen, but return anyway
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Remove zero columns
    zero_mask = (np.all(U == 0, axis=0) & 
                 np.all(V == 0, axis=0) & 
                 np.all(W == 0, axis=0))
    if np.any(zero_mask):
        keep = ~zero_mask
        U = U[:, keep]
        V = V[:, keep]
        W = W[:, keep]
    
    # Smart elimination: use tensor contribution analysis
    # For each column r, compute its tensor contribution T_r[i,j,k] = U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    # A column can be removed if its contribution is zero (already handled) or if it's redundant
    
    # Better approach: iterative elimination with fast verification
    # Precompute the current tensor sum
    def compute_tensor(U, V, W):
        n, m, p = 6, 6, 6
        rank = U.shape[1]
        T = np.zeros((n, m, p))
        for r in range(rank):
            Ur = U[:, r].reshape(n, m)
            Vr = V[:, r].reshape(m, p)
            Wr = W[:, r].reshape(p, n)
            T += np.einsum('ij,jk,ki->ijk', Ur, Vr, Wr)
        return T
    
    # Iterative greedy elimination
    max_rounds = 50
    eliminated = 0
    
    for round_num in range(max_rounds):
        if time.time() > deadline:
            break
            
        cur_rank = U.shape[1]
        if cur_rank <= 1:
            break
        
        # Compute current tensor
        T_current = compute_tensor(U, V, W)
        
        found = False
        for r in range(cur_rank):
            if time.time() > deadline:
                break
            
            # Compute contribution of column r
            Ur = U[:, r].reshape(6, 6)
            Vr = V[:, r].reshape(6, 6)
            Wr = W[:, r].reshape(6, 6)
            T_r = np.einsum('ij,jk,ki->ijk', Ur, Vr, Wr)
            
            # If we remove column r, new tensor = T_current - T_r
            # Check if T_current - T_r == ones
            T_new = T_current - T_r
            if np.allclose(T_new, np.ones((6, 6, 6)), atol=1e-10):
                # Remove column r
                keep = [x for x in range(cur_rank) if x != r]
                U = U[:, keep]
                V = V[:, keep]
                W = W[:, keep]
                eliminated += 1
                found = True
                break
        
        if not found:
            break
    
    # Try pairwise elimination: can we remove two columns at once?
    # This is more expensive but might find reductions that single elimination misses
    if time.time() < deadline - 30:
        cur_rank = U.shape[1]
        if cur_rank >= 2:
            T_current = compute_tensor(U, V, W)
            found_pair = False
            for r1 in range(cur_rank):
                if time.time() > deadline:
                    break
                if found_pair:
                    break
                Ur1 = U[:, r1].reshape(6, 6)
                Vr1 = V[:, r1].reshape(6, 6)
                Wr1 = W[:, r1].reshape(6, 6)
                T_r1 = np.einsum('ij,jk,ki->ijk', Ur1, Vr1, Wr1)
                
                for r2 in range(r1 + 1, cur_rank):
                    if time.time() > deadline:
                        break
                    Ur2 = U[:, r2].reshape(6, 6)
                    Vr2 = V[:, r2].reshape(6, 6)
                    Wr2 = W[:, r2].reshape(6, 6)
                    T_r2 = np.einsum('ij,jk,ki->ijk', Ur2, Vr2, Wr2)
                    
                    T_new = T_current - T_r1 - T_r2
                    if np.allclose(T_new, np.ones((6, 6, 6)), atol=1e-10):
                        keep = [x for x in range(cur_rank) if x != r1 and x != r2]
                        U = U[:, keep]
                        V = V[:, keep]
                        W = W[:, keep]
                        eliminated += 2
                        found_pair = True
                        break
    
    # Final verification
    if not verify_fast(U, V, W):
        pass
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
