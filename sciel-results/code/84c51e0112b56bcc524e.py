import numpy as np

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the rank-minimal exact tensor decomposition for matrix multiplication.
    Uses Kronecker product structure for U, V and a permuted Kronecker product for W.
    """
    n, m, p = DIMENSIONS
    
    # Identity matrices
    I_n = np.eye(n)
    I_m = np.eye(m)
    I_p = np.eye(p)
    
    # Row vectors of ones
    J_n = np.ones((1, n))
    J_m = np.ones((1, m))
    J_p = np.ones((1, p))
    
    # U: (n*m) x (n*m*p)
    # Structure: I_n (n x n) o I_m (m x m) o J_p (1 x p)
    # Result shape: (n*m*1) x (n*m*p)
    U = np.kron(np.kron(I_n, I_m), J_p)
    
    # V: (m*p) x (n*m*p)
    # Structure: J_n (1 x n) o I_m (m x m) o I_p (p x p)
    # Result shape: (1*m*p) x (n*m*p)
    V = np.kron(np.kron(J_n, I_m), I_p)
    
    # W: (p*n) x (n*m*p)
    # Base structure M: J_m (1 x m) o I_p (p x p) o I_n (n x n)
    # Result shape: (1*p*n) x (m*p*n)
    # Row index corresponds to (k, i) with linear index k*n + i.
    # Column index corresponds to (j, k, i) with linear index j*(p*n) + k*n + i.
    M_W = np.kron(np.kron(J_m, I_p), I_n)
    
    # We need to permute the columns of M_W to match the order (i, j, k) 
    # with linear index i*(m*p) + j*p + k.
    # Current column order in M_W: (j, k, i) -> idx_old = j*(p*n) + k*n + i
    # Target column order: (i, j, k) -> idx_new = i*(m*p) + j*p + k
    
    rank = n * m * p
    P = np.zeros((rank, rank))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_old = j * (p * n) + k * n + i
                idx_new = i * (m * p) + j * p + k
                P[idx_old, idx_new] = 1
                
    W = M_W @ P
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
