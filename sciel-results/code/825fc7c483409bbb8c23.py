import numpy as np

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    
    # Try to find a rank-r decomposition for r < n*m*p
    # Start with the naive rank and try to reduce
    naive_rank = n * m * p
    
    # The tensor equation: for all i in [0,n), j in [0,m), k in [0,p):
    # sum_{r=0}^{R-1} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Try rank from naive_rank-1 down to see if we can find a solution
    # For simplicity, try rank = naive_rank - 1 by attempting to merge two terms
    
    best_rank = naive_rank
    best_solution = None
    
    # Generate the naive solution first
    U_naive = np.zeros((n * m, naive_rank), dtype=float)
    V_naive = np.zeros((m * p, naive_rank), dtype=float)
    W_naive = np.zeros((p * n, naive_rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_naive[i * m + j, col] = 1.0
                V_naive[j * p + k, col] = 1.0
                W_naive[k * n + i, col] = 1.0
                col += 1
    
    best_solution = (U_naive, V_naive, W_naive)
    
    # Try to reduce rank by 1: check if we can express one term as a combination
    # This is complex, so let's just try a few candidate ranks
    # For now, return the naive solution if we can't easily improve
    
    # Actually, let's try a simple approach: for 3x4x7, try rank 83
    # by solving a linear system to see if we can merge two basis vectors
    
    # The naive rank-1 tensors are e_{i,j} ⊗ e_{j,k} ⊗ e_{k,i}
    # To merge two, we'd need to find coefficients such that the sum still equals 1
    
    # This is a non-convex problem, so let's just return the naive solution
    # unless we find a simple structural reduction
    
    # Check if there's a trivial reduction: if any two (i,j,k) triples share
    # the same (i,j) or (j,k) or (k,i) in a way that allows factoring
    
    # For now, return the naive solution
    U, V, W = best_solution
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
