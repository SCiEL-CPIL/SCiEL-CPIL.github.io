import numpy as np

def construct():
    n, m, p = 3, 5, 5
    
    # Target tensor is all 1s
    # We want U (nm x R), V (mp x R), W (pn x R)
    # such that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    
    def check_decomposition(U, V, W, n, m, p):
        # U shape (n*m, R)
        # V shape (m*p, R)
        # W shape (p*n, R)
        # Check for all i,j,k
        # Reshape U to (n, m, R), V to (m, p, R), W to (p, n, R)
        U_t = U.reshape(n, m, -1)
        V_t = V.reshape(m, p, -1)
        W_t = W.reshape(p, n, -1)
        
        # Compute T[i,j,k] = sum_r U_t[i,j,r] * V_t[j,k,r] * W_t[k,i,r]
        # Use einsum
        T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
        # Check if T is all 1s
        if not np.allclose(T, np.ones((n,m,p)), atol=1e-9):
            return False
        return True

    def round_to_half(x):
        return np.round(x * 2) / 2.0

    def als_decompose(n, m, p, rank, iters=50, seed=0):
        np.random.seed(seed)
        # Initialize with random small half-integers
        # Values in {-1, -0.5, 0.5, 1}
        vals = np.array([-1, -0.5, 0.5, 1])
        U = np.random.choice(vals, size=(n*m, rank))
        V = np.random.choice(vals, size=(m*p, rank))
        W = np.random.choice(vals, size=(p*n, rank))
        
        for it in range(iters):
            # Update U
            # For each i, j: solve U[i,j,:] @ (V[j,:,k] * W[k,i,:])^T = 1
            # Let's vectorize
            # M_U[i, j, k, r] = V[j, k, r] * W[k, i, r]
            # We want sum_r U[i, j, r] * M_U[i, j, k, r] = 1 for all k
            
            # Reshape V to (m, p, rank), W to (p, n, rank)
            V_t = V.reshape(m, p, rank)
            W_t = W.reshape(p, n, rank)
            
            # Compute M_U: shape (n, m, p, rank)
            # M_U[i, j, k, r] = V_t[j, k, r] * W_t[k, i, r]
            # This is a batched multiplication.
            # V_t[j, k, :] * W_t[k, i, :]
            # We can use einsum
            # M_U[i, j, k, r] = sum? No, just element-wise product of V and W parts.
            # V_t[j, k, r] and W_t[k, i, r].
            # We need to align indices.
            # V_t is (m, p, R). W_t is (p, n, R).
            # We want to compute for each i, j:
            #   M[i, j, k, r] = V_t[j, k, r] * W_t[k, i, r]
            #   Then solve U[i, j, :] M[i, j, :, :]^T = 1_p
            
            # This is expensive to do in pure Python loops for n=3, m=5, p=5.
            # 15 pairs (i,j). For each, a 5xR system.
            # 15 * (5xR) is small.
            
            U_new = np.zeros((n, m, rank))
            for i in range(n):
                for j in range(m):
                    # Construct M: (p, rank)
                    # M[k, r] = V_t[j, k, r] * W_t[k, i, r]
                    M = V_t[j, :, :] * W_t[:, i, :] # (p, rank)
                    # Solve U_new[i, j, :] M^T = 1
                    # U_new[i, j, :] is (1, rank)
                    # M^T is (rank, p)
                    # We want U M^T = [1, 1, ..., 1]
                    # This is an underdetermined system if rank > p.
                    # Use least squares: U = 1_p M (M^T M)^{-1}
                    # Or use np.linalg.lstsq
                    # A = M.T (rank x p), b = ones(p)
                    # We want x such that x A = b -> A.T x.T = b.T
                    # So solve A.T x.T = b.T
                    A = M.T # (rank, p)
                    b = np.ones(p)
                    # Solve A.T x.T = b.T
                    # x.T is (rank, 1)
                    # Use lstsq on A.T
                    # Actually, np.linalg.lstsq(A.T, b.T) gives x.T
                    x, residuals, rank_s, s = np.linalg.lstsq(A.T, b.T, rcond=None)
                    U_new[i, j, :] = x.flatten()
            
            U = U_new.flatten()
            
            # Update V
            V_new = np.zeros((m, p, rank))
            U_t = U.reshape(n, m, rank)
            W_t = W.reshape(p, n, rank)
            for j in range(m):
                for k in range(p):
                    # M[i, r] = U_t[i, j, r] * W_t[k, i, r]
                    # We want sum_r V[j, k, r] M[i, r] = 1 for all i
                    M = U_t[:, j, :] * W_t[k, :, :] # (n, rank)
                    A = M.T # (rank, n)
                    b = np.ones(n)
                    x, _, _, _ = np.linalg.lstsq(A.T, b.T, rcond=None)
                    V_new[j, k, :] = x.flatten()
            V = V_new.flatten()
            
            # Update W
            W_new = np.zeros((p, n, rank))
            U_t = U.reshape(n, m, rank)
            V_t = V.reshape(m, p, rank)
            for k in range(p):
                for i in range(n):
                    # M[j, r] = U_t[i, j, r] * V_t[j, k, r]
                    # We want sum_r W[k, i, r] M[j, r] = 1 for all j
                    M = U_t[i, :, :] * V_t[:, k, :] # (m, rank)
                    A = M.T # (rank, m)
                    b = np.ones(m)
                    x, _, _, _ = np.linalg.lstsq(A.T, b.T, rcond=None)
                    W_new[k, i, :] = x.flatten()
            W = W_new.flatten()
            
            # Optional: check convergence
            # If error is small, break
            
        # Round to half-integers
        U_r = round_to_half(U)
        V_r = round_to_half(V)
        W_r = round_to_half(W)
        
        # Check
        if check_decomposition(U_r, V_r, W_r, n, m, p):
            return U_r, V_r, W_r
        return None

    # Try different ranks
    # Start from a low rank and go up
    # Theoretical lower bound is roughly max(nm, mp, pn) / 2? No.
    # For 2x2x2, 7.
    # For 3x5x5, let's try 50 to 70.
    
    best_rank = 1000
    best_factors = None
    
    # We can also try to use the current program's result as a fallback or upper bound
    # But we must return a valid dict.
    
    for rank in range(50, 72):
        # Try a few seeds
        for seed in range(5):
            res = als_decompose(n, m, p, rank, iters=20, seed=seed)
            if res is not None:
                U, V, W = res
                if rank < best_rank:
                    best_rank = rank
                    best_factors = (U, V, W)
                break # Found valid for this rank, go to next rank? No, we want min rank.
        if best_rank <= rank:
            break # Found a valid decomposition for this rank, which is the lowest so far.
            
    if best_factors is None:
        # Fallback to a known valid decomposition (e.g. schoolbook rank 75)
        # Or use the recursive one? The directive says "design a new program... do not reuse".
        # But I need to return *something*.
        # Let's construct a simple schoolbook decomposition.
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, idx] = 1
                    V[j*p+k, idx] = 1
                    W[k*n+i, idx] = 1
                    idx += 1
        best_factors = (U, V, W)
        best_rank = rank

    U, V, W = best_factors
    return {"U": U, "V": V, "W": W}
