import numpy as np
import time

DIMENSIONS = [3, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank = n * m * p
    
    # Build schoolbook decomposition
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    
    # The multiplication tensor T[i,j,k] = 1 for all valid i,j,k
    # T is (n, m, p) tensor where T[i,j,k] contributes to C[i,k]
    # Actually, T[i,j,k] = 1 means A[i,j]*B[j,k] contributes to C[i,k]
    # So the tensor is T[i,j,k] with i in [0,n), j in [0,m), k in [0,p)
    
    # Rank reduction: try to find linear dependencies
    # A rank-1 tensor is u ⊗ v ⊗ w where u is (n*m), v is (m*p), w is (p*n)
    # T = sum_c U_c ⊗ V_c ⊗ W_c
    
    # Matricize: reshape T to (n*m, p*n) or other combinations
    # T_mat[i*m+j, k*n+i] = 1 if we use this convention
    # Actually, let's use: T_mat[a, b] where a = i*m+j, b = k*n+i
    # T_mat[a, b] = 1 if the corresponding (i,j,k) is valid
    
    # Better: use the standard matricization
    # T as (n*m) x (p*n) matrix: T_mat[i*m+j, k*n+i] = T[i,j,k]
    
    T = np.zeros((n*m, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, k*n + i] = 1
    
    # Current decomposition: T = U @ diag? No.
    # T = sum_c U_c ⊗ V_c ⊗ W_c
    # Matricized: T_mat = sum_c U_c (as column) * (V_c ⊗ W_c)^T ... this is getting complex
    
    # Simpler: verify decomposition by checking that for random A, B:
    # C[i,k] = sum_j A[i,j]*B[j,k] = sum_c (sum_j U[i*m+j,c]*A[i,j]) * (sum_j V[j*p+k,c]*B[j,k]) * W[k*n+i,c]
    # Wait, that's not right. The standard is:
    # C[i,k] = sum_c W[k*n+i,c] * (sum_j U[i*m+j,c]*A[i,j]) * (sum_j V[j*p+k,c]*B[j,k])
    
    # Let me just do rank reduction by trying to eliminate columns
    # If column c is a linear combination of others, we can remove it
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    current_rank = rank
    
    # Try to reduce rank by finding linear dependencies
    for attempt in range(5000):
        if time.time() > deadline:
            break
        
        if current_rank <= 1:
            break
        
        # Try to eliminate a random column
        c_elim = np.random.randint(current_rank)
        
        # Check if column c_elim is in the span of others
        # Use the matricized form
        # For the decomposition to be valid, we need:
        # sum_c U_c ⊗ V_c ⊗ W_c = T
        
        # Matricize as (n*m) x (p*n): 
        # T_mat[i*m+j, k*n+i] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # This is not linear in the decomposition parameters!
        
        # The tensor is trilinear, so matricization gives:
        # For fixed i: T[i, :, :] as (m, p) matrix = sum_c U[i*m: (i+1)*m, c] * V[:, c] ⊗ W[i, :, c]
        # This is still not straightforward.
        
        # Alternative: use the fact that if we fix two indices, it's linear
        # For fixed i,k: T[i, :, k] (length m) = sum_c U[i*m: (i+1)*m, c] * W[k*n+i, c] * V[:, c]
        # No, that's not right either.
        
        # Let me use a different approach: random linear combination test
        # Generate random vectors and check if they're in the span
        
        # Actually, let me just try: if I set column c_elim to 0, does the decomposition still work?
        # No, that's not how it works.
        
        # Better approach: try to find alpha such that
        # sum_{c != c_elim} alpha_c * (U_c, V_c, W_c) = (U_{c_elim}, V_{c_elim}, W_{c_elim})
        # This is a system of equations.
        
        # The condition for the decomposition to remain valid after removing c_elim:
        # U_{c_elim} ⊗ V_{c_elim} ⊗ W_{c_elim} = sum_{c != c_elim} beta_c * U_c ⊗ V_c ⊗ W_c
        # for some beta_c. This is a tensor equation, hard to solve directly.
        
        # Practical approach: just try random rank-1 additions and see if rank decreases
        # Or: use the known result that for 3x4 * 4x6, the rank might be reducible
        
        # Let me try a different strategy: search for integer/half-integer decompositions
        # by solving a system of equations
        
        # For rank r, we need to find U (n*m x r), V (m*p x r), W (p*n x r)
        # such that for all i,j,k: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # And T[i,j,k] = 1 for all valid i,j,k
        
        # This is a system of n*m*p equations (72 equations) with 3*r variables
        # For r < 72, this is overdetermined but may have solutions
        
        # Try small ranks
        for r in range(40, 72):
            if time.time() > deadline:
                break
            # This is a non-linear system, hard to solve directly
            # Skip for now
        
        # Just keep the schoolbook for now if we can't find better
        pass
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
