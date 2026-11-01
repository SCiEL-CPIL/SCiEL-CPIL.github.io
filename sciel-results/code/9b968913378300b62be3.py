"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    # For 2x2 * 2x4, the rank-minimal decomposition is known to be 8 (same as schoolbook)
    # but we can try to find a lower rank via numerical search.
    # However, for small matrices, the rank is often equal to the schoolbook rank or slightly less.
    # Let's first check if we can find a rank < 16 solution.
    
    # The tensor T has dimensions (n*m) x (m*p) x (p*n) = 4 x 8 x 8.
    # We want to find rank r such that T[i,j,k] = sum_c U[i,c] * V[j,c] * W[k,c]
    # where i in [0,n*m), j in [0,m*p), k in [0,p*n).
    # The matrix multiplication tensor for A (n x m) * B (m x p) = C (n x p)
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # In our indexing:
    #   U index: i*m + j (row of A, col of A) -> wait, U is (n*m, rank)
    #   V index: j*p + k (row of B, col of B)
    #   W index: k*n + i (col of C, row of C)
    # T[i*m+j, j*p+k, k*n+i] = 1 if the indices are consistent, else 0.
    # Actually, the tensor entry T[a,b,c] is 1 if a = i*m+j, b = j'*p+k, c = k'*n+i' 
    # and i=i', j=j', k=k'. Otherwise 0.
    
    # Let's define the tensor T explicitly.
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # We will attempt to find a decomposition of rank r for r from some lower bound up to 16.
    # The lower bound for matrix multiplication rank is at least max(n*m, m*p, p*n) / something?
    # Actually, the rank is at least the matrix dimension of the largest factor.
    # For 2x2 * 2x4, the factors have sizes 4, 8, 8.
    # A trivial lower bound is max(n, m, p) but that's not tight.
    # The rank is at least the dimension of the output space, which is n*p = 8? No, that's not right either.
    # Let's just try ranks from 8 to 16.
    
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    # We'll use a randomized approach: fix U and V randomly, then solve for W?
    # Or use a more systematic approach.
    # Since the problem is small, we can try to use a rank-revealing decomposition or iterative methods.
    
    # Let's try rank 8 first, then 9, etc.
    # If we find a valid rank, return it. Otherwise return the schoolbook rank 16.
    
    def try_rank(r):
        """Try to find a decomposition of rank r."""
        np.random.seed(42)
        
        # Precompute transposes of T for solving each factor
        # T shape: (n*m, m*p, p*n)
        # Indices: a (n*m), b (m*p), c (p*n)
        
        # To solve for W (shape p*n, r):
        # T[a,b,c] = sum_d U[a,d] V[b,d] W[c,d]
        # Fix a,b. T[a,b,:] is a vector of length p*n.
        # Coeff for W[c,d] is U[a,d] V[b,d].
        # We want Y_w[ (a,b), c ] = T[a,b,c].
        # A_w[ (a,b), d ] = U[a,d] V[b,d].
        # Solve A_w @ W.T = Y_w.T ? Or A_w @ X = Y_w where X is (r, p*n)?
        # A_w is (n*m*m*p, r). Y_w is (n*m*m*p, p*n).
        # X will be (r, p*n). W is (p*n, r). So W = X.T.
        
        T_w = T.reshape(n*m * m*p, p*n)
        
        # To solve for V (shape m*p, r):
        # T[a,b,c] = sum_d U[a,d] V[b,d] W[c,d]
        # Fix a,c. T[a,:,c] is a vector of length m*p.
        # We need T permuted to (a, c, b).
        T_perm_v = np.transpose(T, (0, 2, 1)) # (n*m, p*n, m*p)
        T_v = T_perm_v.reshape(n*m * p*n, m*p)
        
        # To solve for U (shape n*m, r):
        # T[a,b,c] = sum_d U[a,d] V[b,d] W[c,d]
        # Fix b,c. T[:,b,c] is a vector of length n*m.
        # We need T permuted to (b, c, a).
        T_perm_u = np.transpose(T, (1, 2, 0)) # (m*p, p*n, n*m)
        T_u = T_perm_u.reshape(m*p * p*n, n*m)

        for attempt in range(10):
            if time.time() > deadline:
                return None
            
            U = np.random.randn(n*m, r)
            V = np.random.randn(m*p, r)
            W = np.random.randn(p*n, r)
            
            for it in range(50):
                if time.time() > deadline:
                    return None
                
                # Solve for W
                # A_w[ (a,b), d ] = U[a,d] V[b,d]
                # U: (n*m, r), V: (m*p, r)
                # Use einsum: 'ad,bd->abd' then reshape
                A_w = np.einsum('ad,bd->abd', U, V).reshape(n*m * m*p, r)
                W_sol, _, _, _ = np.linalg.lstsq(A_w, T_w, rcond=None)
                W = W_sol.T # W_sol is (r, p*n), W is (p*n, r)
                
                # Solve for V
                # A_v[ (a,c), d ] = U[a,d] W[c,d]
                A_v = np.einsum('ad,cd->acd', U, W).reshape(n*m * p*n, r)
                V_sol, _, _, _ = np.linalg.lstsq(A_v, T_v, rcond=None)
                V = V_sol.T # V_sol is (r, m*p), V is (m*p, r)
                
                # Solve for U
                # A_u[ (b,c), d ] = V[b,d] W[c,d]
                A_u = np.einsum('bd,cd->bcd', V, W).reshape(m*p * p*n, r)
                U_sol, _, _, _ = np.linalg.lstsq(A_u, T_u, rcond=None)
                U = U_sol.T # U_sol is (r, n*m), U is (n*m, r)
            
            # Check error
            # T_recon[a,b,c] = sum_d U[a,d] V[b,d] W[c,d]
            T_recon = np.einsum('ad,bd,cd->abc', U, V, W)
            error = np.linalg.norm(T - T_recon)
            
            if error < 1e-6:
                U_list = [[float(x) for x in row] for row in U]
                V_list = [[float(x) for x in row] for row in V]
                W_list = [[float(x) for x in row] for row in W]
                return U_list, V_list, W_list
        
        return None
    
    # Try ranks from 8 to 15
    for r in range(8, 16):
        if time.time() > deadline:
            break
        result = try_rank(r)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Fall back to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
