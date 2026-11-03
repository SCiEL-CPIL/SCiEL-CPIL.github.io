"""
Rewritten program using a Greedy Rank-1 Subtraction approach.
This is distinct from the seed which uses a constructive Kronecker product.
The goal is to find a decomposition with rank < 96.
If no improvement is found within the time limit, it falls back to the seed.
"""
import numpy as np
import time

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Build the target tensor T
    # T[alpha, beta, gamma] = 1 if indices are consistent, else 0
    # alpha = i*m + j
    # beta = j*p + k
    # gamma = k*n + i
    T = np.zeros((dim_u, dim_v, dim_w), dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha = i * m + j
                beta = j * p + k
                gamma = k * n + i
                T[alpha, beta, gamma] = 1.0
                
    T_rem = T.copy()
    
    # Lists to store the factors
    U_factors = []
    V_factors = []
    W_factors = []
    
    start_time = time.time()
    time_limit = 100.0 # 100 seconds to leave buffer for evaluation overhead
    
    # Greedy search
    # We try to subtract rank-1 tensors u v^T w^T from T_rem
    # We restrict u, v to be sparse vectors with entries in {-1, 1}
    # and solve for w optimally, then round to half-integers.
    
    max_iter = 20000
    for _ in range(max_iter):
        if time.time() - start_time > time_limit:
            break
            
        if not np.any(T_rem):
            break
            
        best_reduction = 0
        best_u = None
        best_v = None
        best_w = None
        
        # Sample a few random sparse candidates for u and v
        # To speed up, we can pick a random non-zero entry and build u, v around it
        nonzero_indices = np.argwhere(T_rem != 0)
        if len(nonzero_indices) == 0:
            break
            
        # Pick 5 random candidates
        for _ in range(5):
            alpha, beta, gamma = nonzero_indices[np.random.randint(len(nonzero_indices))]
            
            # Construct u and v with 1s at alpha, beta
            # Try different sparsities
            u = np.zeros(dim_u)
            v = np.zeros(dim_v)
            
            # Base case: just the one index
            u[alpha] = 1.0
            v[beta] = 1.0
            
            # Calculate optimal w (real)
            # Minimize || T_rem - u v^T w^T ||^2
            # This is equivalent to minimizing sum_{a,b} (T_rem[a,b,:] - u[a]v[b]w)^2
            # Gradient wrt w: sum_{a,b} -2 (T_rem[a,b,:] - u[a]v[b]w) u[a]v[b] = 0
            # sum_{a,b} T_rem[a,b,:] u[a]v[b] = w sum_{a,b} (u[a]v[b])^2
            # w = (sum_{a,b} T_rem[a,b,:] u[a]v[b]) / (sum_{a,b} (u[a]v[b])^2)
            
            numerator = np.tensordot(T_rem, np.outer(u, v), axes=([0, 1], [0, 1]))
            denominator = np.sum((u * v.reshape(dim_u, 1, 1))**2) # Wait, u is (dim_u,), v is (dim_v,)
            # u[a] v[b]
            uv_prod = np.outer(u, v)
            denominator = np.sum(uv_prod**2)
            
            if denominator == 0:
                continue
                
            w_real = numerator / denominator
            
            # Round w_real to nearest half-integer
            w_half = np.round(w_real * 2) / 2
            
            # Calculate reduction
            # Error before: ||T_rem||^2
            # Error after: ||T_rem - u v^T w_half^T||^2
            # Reduction = 2 * sum(T_rem * u v^T w_half) - sum((u v^T w_half)^2)
            
            term = np.einsum('ijk,i,j,k->', T_rem, u, v, w_half)
            term_sq = np.sum((np.einsum('i,j,k->ijk', u, v, w_half))**2)
            
            reduction = 2 * term - term_sq
            
            if reduction > best_reduction:
                best_reduction = reduction
                best_u = u
                best_v = v
                best_w = w_half
        
        if best_reduction > 1e-9:
            # Accept
            T_rem -= np.einsum('i,j,k->ijk', best_u, best_v, best_w)
            U_factors.append(best_u)
            V_factors.append(best_v)
            W_factors.append(best_w)
        else:
            # Stuck, break to avoid infinite loop
            break
            
    # If we found a decomposition with rank < 96, use it
    if len(U_factors) > 0 and len(U_factors) < n * m * p:
        # Convert to lists of lists
        U = [list(u) for u in U_factors]
        V = [list(v) for v in V_factors]
        W = [list(w) for w in W_factors]
        
        # Note: The return format expects U, V, W to be matrices of shape (dim, rank)
        # The seed returns U as (n*m, rank)
        # My U_factors is a list of rank vectors, each of length dim_u.
        # So I need to transpose.
        
        rank = len(U_factors)
        U_mat = np.zeros((dim_u, rank))
        V_mat = np.zeros((dim_v, rank))
        W_mat = np.zeros((dim_w, rank))
        
        for r in range(rank):
            U_mat[:, r] = U_factors[r]
            V_mat[:, r] = V_factors[r]
            W_mat[:, r] = W_factors[r]
            
        # Convert to list of lists
        return {
            "U": U_mat.tolist(),
            "V": V_mat.tolist(),
            "W": W_mat.tolist()
        }
    else:
        # Fallback to seed
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    c = i * (m * p) + j * p + k
                    U[i * m + j][c] = 1
                    V[j * p + k][c] = 1
                    W[k * n + i][c] = 1
        return {"U": U, "V": V, "W": W}
