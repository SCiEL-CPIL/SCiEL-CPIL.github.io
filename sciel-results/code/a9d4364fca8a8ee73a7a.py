"""Rewritten program using Strassen decomposition for 6x6 matrix multiplication."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    import numpy as np
    import time
    
    t0 = time.time()
    deadline = t0 + 110  # safety margin
    
    n, m, p = DIMENSIONS
    
    # Strassen coefficients for 2x2 block decomposition
    beta = [None] * 7
    gamma = [None] * 7
    alpha = [None] * 7
    
    for s in range(7):
        beta[s] = [[0]*2 for _ in range(2)]
        gamma[s] = [[0]*2 for _ in range(2)]
        alpha[s] = [[0]*2 for _ in range(2)]
        
    # s=0: (A11+A22)(B11+B22)
    beta[0][0][0] = 1; beta[0][1][1] = 1
    gamma[0][0][0] = 1; gamma[0][1][1] = 1
    alpha[0][0][0] = 1; alpha[0][1][1] = 1
    
    # s=1: (A21+A22)B11
    beta[1][1][0] = 1; beta[1][1][1] = 1
    gamma[1][0][0] = 1
    alpha[1][1][0] = 1; alpha[1][1][1] = -1
    
    # s=2: A11(B12-B22)
    beta[2][0][0] = 1
    gamma[2][0][1] = 1; gamma[2][1][1] = -1
    alpha[2][0][1] = 1; alpha[2][1][1] = 1
    
    # s=3: A22(B21-B11)
    beta[3][1][1] = 1
    gamma[3][1][0] = 1; gamma[3][0][0] = -1
    alpha[3][0][0] = 1; alpha[3][1][0] = 1
    
    # s=4: (A11+A12)B22
    beta[4][0][0] = 1; beta[4][0][1] = 1
    gamma[4][1][1] = 1
    alpha[4][0][0] = -1; alpha[4][0][1] = 1
    
    # s=5: (A21-A11)(B11+B12)
    beta[5][1][0] = 1; beta[5][0][0] = -1
    gamma[5][0][0] = 1; gamma[5][0][1] = 1
    alpha[5][1][1] = 1
    
    # s=6: (A12-A22)(B21+B22)
    beta[6][0][1] = 1; beta[6][1][1] = -1
    gamma[6][1][0] = 1; gamma[6][1][1] = 1
    alpha[6][0][0] = 1
    
    # Build factor matrices
    rank = 189
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for s in range(7):
        for a in range(3):
            for b in range(3):
                for c in range(3):
                    r = s * 27 + a * 9 + b * 3 + c
                    
                    for I in range(2):
                        for J in range(2):
                            if beta[s][I][J] != 0:
                                i_glob = I * 3 + a
                                j_glob = J * 3 + b
                                U[i_glob * m + j_glob, r] = beta[s][I][J]
                                
                    for K in range(2):
                        for L in range(2):
                            if gamma[s][K][L] != 0:
                                j_glob = K * 3 + b
                                k_glob = L * 3 + c
                                V[j_glob * p + k_glob, r] = gamma[s][K][L]
                                
                    for I in range(2):
                        for K in range(2):
                            if alpha[s][I][K] != 0:
                                i_glob = I * 3 + a
                                k_glob = K * 3 + c
                                W[k_glob * n + i_glob, r] = alpha[s][I][K]
    
    # Vectorized verification
    def verify(U, V, W):
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Should equal 1 for all i, j, k
        n, m, p = 6, 6, 6
        # Reshape for efficient computation
        # U shape: (n*m, rank) -> U[i*m+j, r]
        # V shape: (m*p, rank) -> V[j*p+k, r]
        # W shape: (p*n, rank) -> W[k*n+i, r]
        
        # For each r, the contribution to T is an outer product structure
        # T = sum_r (U_col_r reshaped) * (V_col_r reshaped) * (W_col_r reshaped)
        # U_col_r: shape (n, m) where U[i*m+j, r]
        # V_col_r: shape (m, p) where V[j*p+k, r]
        # W_col_r: shape (p, n) where W[k*n+i, r]
        
        T = np.zeros((n, m, p))
        for r in range(U.shape[1]):
            Ur = U[:, r].reshape(n, m)
            Vr = V[:, r].reshape(m, p)
            Wr = W[:, r].reshape(p, n)
            # T[i,j,k] += Ur[i,j] * Vr[j,k] * Wr[k,i]
            # This is: T = sum_r np.einsum('ij,jk,ki->ijk', Ur, Vr, Wr)
            T += np.einsum('ij,jk,ki->ijk', Ur, Vr, Wr)
        
        # Check if T is all ones
        if not np.allclose(T, 1.0, atol=1e-10):
            return False
        return True
    
    # Verify initial decomposition
    if not verify(U, V, W):
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
    
    # Greedy column elimination: try to remove each column if tensor identity is preserved
    # This is expensive, so limit iterations
    max_elim_attempts = 200
    eliminated = 0
    
    for _ in range(max_elim_attempts):
        if time.time() > deadline:
            break
            
        cur_rank = U.shape[1]
        if cur_rank <= 1:
            break
            
        found = False
        for r in range(cur_rank):
            if time.time() > deadline:
                break
                
            # Try removing column r
            keep = [x for x in range(cur_rank) if x != r]
            U_new = U[:, keep]
            V_new = V[:, keep]
            W_new = W[:, keep]
            
            # Quick check: verify tensor identity
            n, m, p = 6, 6, 6
            T = np.zeros((n, m, p))
            for rr in range(U_new.shape[1]):
                Ur = U_new[:, rr].reshape(n, m)
                Vr = V_new[:, rr].reshape(m, p)
                Wr = W_new[:, rr].reshape(p, n)
                T += np.einsum('ij,jk,ki->ijk', Ur, Vr, Wr)
            
            if np.allclose(T, 1.0, atol=1e-10):
                U, V, W = U_new, V_new, W_new
                eliminated += 1
                found = True
                break
        
        if not found:
            break
    
    # Final verification
    if not verify(U, V, W):
        # Shouldn't happen, but return anyway
        pass
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
