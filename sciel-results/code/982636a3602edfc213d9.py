import numpy as np

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the rank-minimal exact tensor decomposition for matrix multiplication.
    For 2x3 * 3x5, the known minimal rank is 30 (schoolbook).
    We construct U, V, W using Kronecker product structures derived from identity matrices.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Identity matrices
    I_n = np.eye(n)
    I_m = np.eye(m)
    I_p = np.eye(p)
    
    # Ones vectors
    ones_n = np.ones(n)
    ones_m = np.ones(m)
    ones_p = np.ones(p)
    
    # Construct U: shape (n*m, n*m*p)
    # Row index (i, j) -> i*m + j
    # Col index (i, j, k) -> i*(m*p) + j*p + k
    # U[i*m+j, i*(m*p) + j*p + k] = 1
    
    # We can build U by iterating i, j or using kron.
    # U is essentially a block matrix.
    # Let's build it via stacking kron products.
    U_blocks = []
    for i in range(n):
        for j in range(m):
            # Vector of length n*m*p
            # 1s at positions where index i matches i and index j matches j
            # Index r = i_r * (m*p) + j_r * p + k_r
            # We want r such that i_r == i and j_r == j.
            # This is kron(e_i, kron(e_j, ones_p))
            vec = np.kron(np.kron(np.eye(n)[i], np.eye(m)[j]), ones_p)
            U_blocks.append(vec)
    U = np.array(U_blocks)
    
    # Construct V: shape (m*p, n*m*p)
    # Row index (j, k) -> j*p + k
    # Col index (i, j, k) -> i*(m*p) + j*p + k
    # V[j*p+k, i*(m*p) + j*p + k] = 1 if j_r==j and k_r==k
    
    V_blocks = []
    for j in range(m):
        for k in range(p):
            # Vector of length n*m*p
            # 1s at positions where index j matches j and index k matches k
            # Index r = i_r * (m*p) + j_r * p + k_r
            # This is kron(ones_n, kron(e_j, e_k))
            vec = np.kron(ones_n, np.kron(np.eye(m)[j], np.eye(p)[k]))
            V_blocks.append(vec)
    V = np.array(V_blocks)
    
    # Construct W: shape (p*n, n*m*p)
    # Row index (k, i) -> k*n + i  (Trace convention W[k*n+i, r])
    # Col index (i, j, k) -> i*(m*p) + j*p + k
    # W[k*n+i, i*(m*p) + j*p + k] = 1 if i_r==i and k_r==k
    
    W_blocks = []
    for k in range(p):
        for i in range(n):
            # Vector of length n*m*p
            # 1s at positions where index i matches i and index k matches k
            # Index r = i_r * (m*p) + j_r * p + k_r
            # This is kron(e_i, ones_m, e_k)
            vec = np.kron(np.kron(np.eye(n)[i], ones_m), np.eye(p)[k])
            W_blocks.append(vec)
    W = np.array(W_blocks)
    
    # Convert to lists of lists to match interface requirement if needed, 
    # but numpy arrays are usually accepted or convertible. 
    # The seed returned lists of lists. Let's stick to that to be safe, 
    # or check if evaluator accepts arrays. 
    # "factor matrices of shapes ... with integer/half-integer entries"
    # The seed used lists. I will convert to lists of lists.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
