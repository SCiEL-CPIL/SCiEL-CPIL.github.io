import numpy as np

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    # We construct a rank-54 decomposition (schoolbook rank) for the 2x3 * 3x9 multiplication.
    # The decomposition is structured as a direct sum over the output columns k=0..8.
    # For each column k, we need to compute C[:,k] = A * B[:,k], which is a (2,3,1) tensor.
    # The (2,3,1) tensor has rank 6, so total rank = 9 * 6 = 54.
    
    rank = n * m * p  # 54
    # Initialize factor matrices as lists of lists (will be converted to numpy arrays)
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for k in range(p):
        for i in range(n):
            for j in range(m):
                # U index: i*m + j (row i, col j of A)
                U[i * m + j, col] = 1
                # V index: j*p + k (row j, col k of B)
                V[j * p + k, col] = 1
                # W index: k*n + i (trace convention: row k*n + i)
                W[k * n + i, col] = 1
                col += 1
    
    # Convert to lists of lists as required by the interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    # Ensure integer/half-integer entries (they are already 0 or 1)
    return {"U": U_list, "V": V_list, "W": W_list}
