import numpy as np
import time

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # The multiplication tensor T has shape (n, m, p) with T[i,j,k] = 1 (it's the identity tensor for MM)
    # Actually, the tensor for matrix multiplication: T[i,j,k] represents the coefficient
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor is T[i,j,k] = 1 for all i,j,k in the "multiplication" sense
    # We need U, V, W such that:
    # U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] summed over r gives the product
    
    # Let's try to find a low-rank decomposition
    # The tensor T of shape (n*m, m*p, p*n) where T[(i,j),(j,k),(k,i)] = 1
    
    # Build the tensor
    nm = n * m
    mp = m * p
    pn = p * n
    
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1
    
    # Try rank decomposition
    # For a tensor, rank-r decomposition means T = sum_r u_r ⊗ v_r ⊗ w_r
    # i.e., T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    
    # Let's try various ranks
    # Start with a guess based on known results
    # For 2x4x5, let's try rank 20 first, then go lower
    
    deadline = time.time() + 140  # Leave 10s safety margin
    
    best_rank = nm * mp * pn // pn  # schoolbook is n*m*p = 40
    best_U = None
    best_V = None
    best_W = None
    
    # Try ranks from low to high
    # Known: for 2xm x mxp, there might be structure
    # Let's try rank 20, 25, 30, 35, 40
    
    for rank in range(20, 41):
        if time.time() > deadline:
            break
        
        # Try to find a rank-r decomposition
        # Use a simple approach: random initialization + gradient descent
        # Or use SVD-based approximation
        
        # Method: initialize U, V, W randomly and optimize
        # T[a,b,c] ≈ sum_r U[a,r]*V[b,r]*W[c,r]
        
        # Use ALS (Alternating Least Squares)
        np.random.seed(rank)
        
        U = np.random.randn(nm, rank) * 0.1
        V = np.random.randn(mp, rank) * 0.1
        W = np.random.randn(pn, rank) * 0.1
        
        # ALS iterations
        for it in range(200):
            if time.time() > deadline:
                break
            
            # Update U: for each a, U[a,:] should satisfy
            # T[a,:,:] ≈ sum_r U[a,r] * (V[:,r] ⊗ W[:,r])
            # Reshape: for fixed a, T[a] is (mp, pn), and we want
            # T[a] ≈ sum_r U[a,r] * np.outer(V[:,r], W[:,r])
            # So U[a,:] = (V⊗W)^T * T[a] flattened... 
            # Actually: T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
            # For fixed a: T[a,:,:] = sum_r U[a,r] * np.outer(V[:,r], W[:,r])
            # Let M_r = np.outer(V[:,r], W[:,r]) reshaped to (mp*pn,)
            # Then T[a].flatten() = sum_r U[a,r] * M_r
            # So U[a,:] = solve(M, T[a].flatten()) where M[r, s] = M_r[s]
            
            # Build the matrix for U update
            # M has shape (rank, mp*pn)
            M = np.zeros((rank, mp * pn))
            for r in range(rank):
                M[r, :] = np.outer(V[:, r], W[:, r]).flatten()
            
            # U[a, r] such that M @ U[a, :] = T[a, :].flatten()
            # U[a, :] = (M^T M)^{-1} M^T T[a].flatten()
            # Use least squares
            try:
                U = np.linalg.lstsq(M, T.reshape(nm, mp * pn).T, rcond=None)[0].T
            except:
                pass
            
            # Update V
            Mv = np.zeros((rank, nm * pn))
            for r in range(rank):
                Mv[r, :] = np.outer(U[:, r], W[:, r]).flatten()
            try:
                V = np.linalg.lstsq(Mv, T.transpose(1, 0, 2).reshape(mp, nm * pn).T, rcond=None)[0].T
            except:
                pass
            
            # Update W
            Mw = np.zeros((rank, nm * mp))
            for r in range(rank):
                Mw[r, :] = np.outer(U[:, r], V[:, r]).flatten()
            try:
                W = np.linalg.lstsq(Mw, T.transpose(2, 0, 1).reshape(pn, nm * mp).T, rcond=None)[0].T
            except:
                pass
        
        # Check if we found an exact decomposition
        # Reconstruct and check
        T_recon = np.zeros((nm, mp, pn))
        for r in range(rank):
            T_recon += np.outer(U[:, r], np.outer(V[:, r], W[:, r]).reshape(mp, pn)) \
                        .reshape(nm, mp, pn)
        
        # Check exactness (with tolerance for floating point)
        if np.allclose(T_recon, T, atol=1e-9):
            # Found a valid decomposition!
            best_rank = rank
            best_U = U
            best_V = V
            best_W = W
            break
    
    if best_U is None:
        # Fall back to schoolbook
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Round to nearest half-integer for cleanliness
    U_rounded = np.round(best_U * 2) / 2
    V_rounded = np.round(best_V * 2) / 2
    W_rounded = np.round(best_W * 2) / 2
    
    # Verify rounded version
    T_recon2 = np.zeros((nm, mp, pn))
    for r in range(best_rank):
        T_recon2 += np.outer(U_rounded[:, r], np.outer(V_rounded[:, r], W_rounded[:, r]).reshape(mp, pn)) \
                     .reshape(nm, mp, pn)
    
    if np.allclose(T_recon2, T, atol=1e-9):
        return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
    else:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
