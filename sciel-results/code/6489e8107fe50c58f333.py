"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_sb = n * m * p  # 144
    
    # Build schoolbook decomposition
    U = np.zeros((n*m, rank_sb), dtype=float)
    V = np.zeros((m*p, rank_sb), dtype=float)
    W = np.zeros((p*n, rank_sb), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1.0
                V[j*p + k, col] = 1.0
                W[k*n + i, col] = 1.0
                col += 1
    
    # The tensor T[a,b,d] = sum_c U[a,c]*V[b,c]*W[d,c]
    # Verify schoolbook is correct
    # T should equal delta_{j,j'}*delta_{k,k'}*delta_{i,i'} where
    # a=i*m+j, b=j'*p+k, d=k'*n+i'
    
    # Try to reduce rank by finding linear dependencies
    # The columns of [U; V; W] stacked... actually the tensor is a 3-way array.
    
    # Flatten the tensor: T_flat[a,b,d] 
    # We have T = sum_c U[:,c] V[:,c]^T ... no, it's a 3-way tensor.
    
    # Alternative: try random perturbations and rank reduction
    # For now, return schoolbook as baseline
    # Convert to lists
    best_U = U.tolist()
    best_V = V.tolist()
    best_W = W.tolist()
    best_rank = rank_sb
    
    # Try simple optimization: check if any columns are linearly dependent
    # in the combined tensor representation
    
    # Represent the tensor as a matrix: reshape to (n*m*m*p, p*n)
    # T[a,b,d] -> matrix M[a, (b,d)]
    # T = sum_c U[:,c] * (V[:,c] outer W[:,c])
    # So M = U @ (V * W elementwise per column)
    
    # Let's try: for each pair of columns, check if one is a linear combo
    # This is expensive for 144 columns.
    
    # Instead, try a greedy rank reduction:
    # Pick two columns c1, c2 and try to merge them
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 10:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    # Try to reduce rank by 1: find if any column is redundant
    # A column c is redundant if U[:,c], V[:,c], W[:,c] can be expressed
    # as linear combinations of other columns in a way that preserves the tensor.
    
    # This is hard to check directly. Instead, try random rank-reduction attempts.
    
    # For now, return schoolbook
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
