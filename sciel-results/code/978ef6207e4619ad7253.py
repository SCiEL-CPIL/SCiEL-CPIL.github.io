import numpy as np

def construct():
    n, m, p = 2, 2, 16
    
    # Strassen factors for 2x2x2
    # u, v, w are 2-vectors
    # The tensor decomposition is sum_r u_r (v_r . B) w_r^T
    # In the flattened tensor form:
    # U[i, r] = u_r[i]
    # V[j, r] = v_r[j]
    # W[k, r] = w_r[k]
    # And the sum U[i,r] V[j,r] W[k,r] should equal T[i,j,k]
    
    # Correct Strassen factors for T[i,j,k] = 1 (all ones) is impossible (rank 2).
    # The tensor T[i,j,k] for matrix mult is:
    # T[i,j,k] = 1 if the term A[i,j] B[j,k] contributes to C[i,k].
    # This is always 1.
    # BUT, the Strassen decomposition works on the *map* (A,B) -> C.
    # The tensor is the structure tensor.
    # The factors are:
    
    factors = [
        ([1, 1], [1, 1], [1, 1]),
        ([0, 1], [1, 0], [0, 1]),
        ([1, 0], [0, 1], [1, 0]),
        ([0, 1], [1, -1], [1, 0]),
        ([1, 1], [0, 1], [0, 1]),
        ([1, -1], [1, 1], [0, 1]),
        ([1, -1], [0, 1], [1, 0]),
    ]
    
    rank = 7 * (p // 2)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for t in range(p // 2):
        k0 = 2 * t
        k1 = 2 * t + 1
        for u, v, w in factors:
            # u is 2-vector for i (rows of A/C)
            # v is 2-vector for j (cols of A / rows of B)
            # w is 2-vector for k (cols of B/C)
            
            # Map to global indices
            # U row: i * m + j. Here i is from u, j is from v?
            # No, U depends on i and j?
            # In the tensor U[i*m+j, r], the index is i*m+j.
            # The factor u_r is a vector of length n (2).
            # The factor v_r is a vector of length m (2).
            # The factor w_r is a vector of length p (2 for block).
            
            # The term is u_r[i] * v_r[j] * w_r[k].
            # So U[i*m+j, col] = u_r[i] * v_r[j]
            # V[j*p+k, col] = v_r[j] * w_r[k]
            # W[k*n+i, col] = w_r[k] * u_r[i]
            
            for i in range(2):
                for j in range(2):
                    U[i * m + j][col] = u[i] * v[j]
            
            for j in range(2):
                # k0 corresponds to local k=0, k1 to local k=1
                V[j * p + k0][col] = v[j] * w[0]
                V[j * p + k1][col] = v[j] * w[1]
            
            for k_local in range(2):
                k_global = k0 + k_local
                for i in range(2):
                    W[k_global * n + i][col] = w[k_local] * u[i]
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
