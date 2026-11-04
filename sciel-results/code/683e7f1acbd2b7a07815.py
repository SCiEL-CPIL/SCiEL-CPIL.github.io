import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 8
    
    # Target tensor T[i, j, k] = 1 for all valid i, j, k
    # We represent the tensor as a flat vector of length n*m*p
    # Index mapping:
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    def tensor_to_flat(T):
        return T.flatten()
        
    # The tensor T is all ones of shape (n, m, p)
    T = np.ones((n, m, p))
    T_flat = T.flatten()
    
    def error_function(params, rank):
        """
        Unpack parameters into U, V, W and compute the error.
        params is a flat vector of length rank * (n*m + m*p + p*n)
        """
        size_U = n * m
        size_V = m * p
        size_W = p * n
        
        U_flat = params[:size_U * rank]
        V_flat = params[size_U * rank : size_U * rank + size_V * rank]
        W_flat = params[size_U * rank + size_V * rank : size_U * rank + size_V * rank + size_W * rank]
        
        U = U_flat.reshape(size_U, rank)
        V = V_flat.reshape(size_V, rank)
        W = W_flat.reshape(size_W, rank)
        
        # Compute the reconstructed tensor
        # Reconstructed[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # We can compute this efficiently using einsum or loops
        # Rec[i, j, k] = sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
        # Note: U is indexed by (i, j), V by (j, k), W by (k, i)
        
        Rec = np.zeros((n, m, p))
        for r in range(rank):
            # u_r is vector of length n*m, indexed by i*m+j
            # v_r is vector of length m*p, indexed by j*p+k
            # w_r is vector of length p*n, indexed by k*n+i
            
            # Outer product logic:
            # For each i, j, k:
            # Rec[i, j, k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            
            # We can reshape U[:, r] to (n, m), V[:, r] to (m, p), W[:, r] to (p, n)
            u_r = U[:, r].reshape(n, m)
            v_r = V[:, r].reshape(m, p)
            w_r = W[:, r].reshape(p, n)
            
            # The term is u_r[i, j] * v_r[j, k] * w_r[k, i]
            # This is a 3-way interaction.
            # We can compute this as:
            # Temp[i, j, k] = u_r[i, j] * v_r[j, k]
            # Then sum over k? No, w_r depends on k and i.
            
            # Let's use einsum: 'ij,jk,ki->ijk'
            # u_r is (i, j), v_r is (j, k), w_r is (k, i)
            # Result is (i, j, k)
            term = np.einsum('ij,jk,ki->ijk', u_r, v_r, w_r)
            Rec += term
            
        Rec_flat = Rec.flatten()
        diff = Rec_flat - T_flat
        return np.sum(diff * diff)
        
    def get_factors_from_params(params, rank):
        size_U = n * m
        size_V = m * p
        size_W = p * n
        
        U_flat = params[:size_U * rank]
        V_flat = params[size_U * rank : size_U * rank + size_V * rank]
        W_flat = params[size_U * rank + size_V * rank : size_U * rank + size_V * rank + size_W * rank]
        
        U = U_flat.reshape(size_U, rank)
        V = V_flat.reshape(size_V, rank)
        W = W_flat.reshape(size_W, rank)
        
        return U, V, W

    def round_to_half_integer(arr):
        return np.round(arr * 2) / 2.0

    def verify_factors(U, V, W, rank):
        Rec = np.zeros((n, m, p))
        for r in range(rank):
            u_r = U[:, r].reshape(n, m)
            v_r = V[:, r].reshape(m, p)
            w_r = W[:, r].reshape(p, n)
            term = np.einsum('ij,jk,ki->ijk', u_r, v_r, w_r)
            Rec += term
            
        # Check if Rec is exactly T (all ones)
        # Due to floating point, we check closeness
        if np.allclose(Rec, T, atol=1e-9):
            return True
        return False

    # Try ranks from lower to higher
    # Lower bound for rank of matrix multiplication tensor M_{n,m,p} is known to be at least max(n*m, m*p, p*n)
    # For 2x3x8: max(6, 24, 16) = 24.
    # Schoolbook is 48.
    # Let's try from 24 to 48.
    
    # To speed up, we can try a few random initializations for each rank.
    
    for rank in range(24, 49):
        # Try a few random starts
        for attempt in range(10):
            np.random.seed(42 + attempt) # Reproducibility
            
            size_U = n * m
            size_V = m * p
            size_W = p * n
            total_params = rank * (size_U + size_V + size_W)
            
            # Initialize with small random values
            x0 = np.random.randn(total_params) * 0.1
            
            res = minimize(error_function, x0, args=(rank,), method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6})
            
            if res.success or res.fun < 1e-4:
                U, V, W = get_factors_from_params(res.x, rank)
                
                # Round to half-integers
                U_rounded = round_to_half_integer(U)
                V_rounded = round_to_half_integer(V)
                W_rounded = round_to_half_integer(W)
                
                if verify_factors(U_rounded, V_rounded, W_rounded, rank):
                    # Convert to list of lists for output
                    return {
                        "U": U_rounded.tolist(),
                        "V": V_rounded.tolist(),
                        "W": W_rounded.tolist()
                    }
                    
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1.0
                V[j * p + k][column] = 1.0
                W[k * n + i][column] = 1.0
                column += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
