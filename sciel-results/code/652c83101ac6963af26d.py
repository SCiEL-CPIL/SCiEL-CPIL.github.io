import numpy as np
from scipy.optimize import least_squares
import time

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target tensor T is defined by T[(i,j), (j,k), (i,k)] = 1
    # We flatten the indices.
    # U: (n*m, R)
    # V: (m*p, R)
    # W: (p*n, R) -> Note: prompt says W[k*n+i, r]. So index is k*n+i.
    
    # We try to find a decomposition of rank R=20.
    R = 20
    
    # Initialize U, V, W
    # Use a random initialization with small values
    np.random.seed(42)
    U = np.random.randn(n*m, R) * 0.1
    V = np.random.randn(m*p, R) * 0.1
    W = np.random.randn(p*n, R) * 0.1
    
    # Define the residual function
    # We need to check all (i,j, j',k', i'',k'')
    # But most are 0.
    # Non-zero constraints:
    # 1. Sum_r U[i,j,r] V[j,k,r] W[i,k,r] = 1  (for all i,j,k)
    # 2. Sum_r U[i,j,r] V[j',k',r] W[i'',k'',r] = 0 (for others)
    
    # It is more efficient to construct the full tensor residual.
    # Total size of tensor: (n*m) * (m*p) * (p*n) = 9 * 12 * 12 = 1296.
    # This is small. We can compute the full tensor and compare.
    
    def residual(params):
        # params: flattened U, V, W
        U_flat = params[:R*n*m]
        V_flat = params[R*n*m:2*R*n*m + R*m*p] # Wait, sizes are different
        # U: n*m * R
        # V: m*p * R
        # W: p*n * R
        
        size_U = n*m * R
        size_V = m*p * R
        size_W = p*n * R
        
        U_flat = params[:size_U]
        V_flat = params[size_U:size_U+size_V]
        W_flat = params[size_U+size_V:size_U+size_V+size_W]
        
        U_mat = U_flat.reshape(n*m, R)
        V_mat = V_flat.reshape(m*p, R)
        W_mat = W_flat.reshape(p*n, R)
        
        # Compute tensor T_decomp
        # T_decomp[i*m+j, j'*p+k', k''*n+i''] = sum_r U[i*m+j, r] V[j'*p+k', r] W[k''*n+i'', r]
        # We can use einsum
        # U: (i,j, r)
        # V: (j',k', r)
        # W: (k'',i'', r)
        # Result: (i,j, j',k', k'',i'')
        
        # Reshape to 3D
        U_3d = U_mat.reshape(n, m, R)
        V_3d = V_mat.reshape(m, p, R)
        W_3d = W_mat.reshape(p, n, R)
        
        # Compute sum_r U[i,j,r] V[j',k',r] W[k'',i'',r]
        # This is a 6D tensor.
        # T_decomp = np.einsum('ijr,jkr,klr->ijkkl', U_3d, V_3d, W_3d) # Indices mismatch
        
        # Let's use explicit loops or matmul for speed if needed, but 9x12x12 is small.
        # T_decomp[i, j, j_prime, k_prime, k_double_prime, i_double_prime]
        
        # Actually, we can just check the non-zero constraints.
        # Constraints 1: i=i'', j=j', k=k'' -> sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1
        # Constraints 2: others -> 0
        
        res = []
        # Constraint 1: n*m*p terms
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # U index: i*m + j
                    # V index: j*p + k
                    # W index: k*n + i
                    val = np.dot(U_mat[i*m+j], V_mat[j*p+k]) # No, sum over r
                    # val = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    # This is not a simple dot product of U and V.
                    # It is a trilinear form.
                    # sum_r U_a,r V_b,r W_c,r
                    # Can be computed as:
                    # Let M_r = U[:,r] outer V[:,r] outer W[:,r]? No.
                    # Just loop r.
                    s = 0
                    for r in range(R):
                        s += U_mat[i*m+j, r] * V_mat[j*p+k, r] * W_mat[k*n+i, r]
                    res.append(s - 1)
                    
        # Constraint 2: Others
        # Total terms (n*m) * (m*p) * (p*n) = 1296.
        # We already covered n*m*p = 36 terms.
        # Remaining 1296 - 36 = 1260 terms.
        # We can iterate over all and skip the ones handled above.
        
        for idx_u in range(n*m):
            i = idx_u // m
            j = idx_u % m
            for idx_v in range(m*p):
                j_prime = idx_v // p
                k_prime = idx_v % p
                for idx_w in range(p*n):
                    k_double_prime = idx_w // n
                    i_double_prime = idx_w % n
                    
                    # Check if this is a "matching" term
                    if i == i_double_prime and j == j_prime and k_prime == k_double_prime:
                        continue # Already handled
                    
                    s = 0
                    for r in range(R):
                        s += U_mat[idx_u, r] * V_mat[idx_v, r] * W_mat[idx_w, r]
                    res.append(s)
                    
        return np.array(res)
        
    # Initial guess
    x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    # Optimize
    start_time = time.time()
    try:
        # Use least_squares
        # This might be slow if not vectorized.
        # Vectorizing the residual:
        # We can form the tensor T_decomp and subtract T_target.
        
        def residual_vectorized(params):
            size_U = n*m * R
            size_V = m*p * R
            size_W = p*n * R
            
            U_flat = params[:size_U]
            V_flat = params[size_U:size_U+size_V]
            W_flat = params[size_U+size_V:size_U+size_V+size_W]
            
            U_mat = U_flat.reshape(n*m, R)
            V_mat = V_flat.reshape(m*p, R)
            W_mat = W_flat.reshape(p*n, R)
            
            # T_decomp shape: (n*m, m*p, p*n)
            # T_decomp[a, b, c] = sum_r U[a,r] V[b,r] W[c,r]
            
            # We can compute this using matmul?
            # T_decomp = (U * W) ? No.
            # T_decomp[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            
            # Let's use einsum
            # U: (A, R), V: (B, R), W: (C, R)
            # T: (A, B, C)
            # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            # This is not a standard einsum unless we expand.
            # T = np.einsum('ar,br,cr->abc', U_mat, V_mat, W_mat)
            
            T_decomp = np.einsum('ar,br,cr->abc', U_mat, V_mat, W_mat)
            
            # Target Tensor T_target
            # T_target[a,b,c] = 1 if a=(i,j), b=(j,k), c=(k,i) else 0
            T_target = np.zeros((n*m, m*p, p*n))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i*m + j
                        b = j*p + k
                        c = k*n + i
                        T_target[a,b,c] = 1
            
            return (T_decomp - T_target).flatten()
        
        # Run optimization
        # We need to be careful with time.
        # 120s limit.
        # We will run a few iterations.
        
        # Use L-BFGS-B or similar
        result = least_squares(residual_vectorized, x0, method='lm', max_nfev=100)
        
        if time.time() - start_time > 100:
            return _schoolbook()
            
        if result.cost < 1e-6:
            # Extract solution
            size_U = n*m * R
            size_V = m*p * R
            size_W = p*n * R
            
            U_sol = result.x[:size_U].reshape(n*m, R)
            V_sol = result.x[size_U:size_U+size_V].reshape(m*p, R)
            W_sol = result.x[size_U+size_V:size_U+size_V+size_W].reshape(p*n, R)
            
            # Round to nearest 0.5
            U_sol = np.round(U_sol * 2) / 2
            V_sol = np.round(V_sol * 2) / 2
            W_sol = np.round(W_sol * 2) / 2
            
            # Verify
            T_decomp = np.einsum('ar,br,cr->abc', U_sol, V_sol, W_sol)
            T_target = np.zeros((n*m, m*p, p*n))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i*m + j
                        b = j*p + k
                        c = k*n + i
                        T_target[a,b,c] = 1
            
            if np.allclose(T_decomp, T_target, atol=1e-6):
                # Convert to list of lists
                return {
                    "U": U_sol.tolist(),
                    "V": V_sol.tolist(),
                    "W": W_sol.tolist()
                }
                
    except Exception:
        pass
        
    return _schoolbook()

def _schoolbook():
    n, m, p = DIMENSIONS
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
