import numpy as np

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    dim_total = dim_U * dim_V * dim_W
    
    # Generate schoolbook tensors
    tensors = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = np.zeros(dim_U)
                v = np.zeros(dim_V)
                w = np.zeros(dim_W)
                
                u[i * m + j] = 1
                v[j * p + k] = 1
                w[k * n + i] = 1
                
                tensors.append((u, v, w))
                
    # Check linear dependence
    M = np.zeros((dim_total, len(tensors)))
    for idx, (u, v, w) in enumerate(tensors):
        M[:, idx] = np.kron(np.kron(u, v), w)
        
    # Compute rank
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    tol = 1e-9
    rank = np.sum(s > tol)
    
    # If rank < 42, we can potentially reduce
    # For now, if rank < 42, we just return the first 'rank' tensors? 
    # This is incorrect as they might not sum to T.
    # But if they are dependent, T is in the span.
    # We can solve M x = T for x.
    
    T = np.sum(M, axis=1)
    
    if rank < len(tensors):
        # Find a sparse solution x
        # We can use the null space to reduce the number of non-zeros
        # For simplicity, let's just return the schoolbook if we can't easily solve
        pass
        
    # Return schoolbook
    U_out = []
    V_out = []
    W_out = []
    for u, v, w in tensors:
        U_out.append(u.tolist())
        V_out.append(v.tolist())
        W_out.append(w.tolist())
        
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
