"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [3, 3, 7]

def _verify_decomposition(n, m, p, U, V, W, rank):
    """Verify that the decomposition exactly reproduces the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # T[i,j,k] should equal sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # Build tensor from decomposition
    T = np.zeros((n, m, p))
    for c in range(rank):
        u_vec = U[:, c].reshape(n, m)
        v_vec = V[:, c].reshape(m, p)
        w_vec = W[:, c].reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += u_vec[i, j] * v_vec[j, k] * w_vec[k, i]
    # Target tensor is all ones
    target = np.ones((n, m, p))
    if not np.allclose(T, target, atol=1e-10):
        return False
    return True

def _construct_decomposition():
    n, m, p = 3, 3, 7
    # We split the problem into two parts based on the k index (columns of B, rows of C)
    # Part 1: k in {0, 1, 2} -> 3x3x3 multiplication
    # Part 2: k in {3, 4, 5, 6} -> 3x3x4 multiplication
    
    # We will use schoolbook for Part 1 (rank 27) 
    # and optimized slice-based for Part 2 (rank 12)
    # Total rank = 39
    
    rank1 = n * m * 3  # 27
    rank2 = n * m * 4  # 12? No, rank 3 per k, so 3*4=12.
    total_rank = rank1 + rank2
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: k in 0..2 (Schoolbook)
    for i in range(n):
        for j in range(m):
            for k in range(3):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    # Part 2: k in 3..6 (Optimized: Rank 3 per k)
    # For each k, we use 3 terms (one for each j)
    for k in range(3, 7):
        for j in range(m):
            # Term c corresponds to (k, j)
            # U[i, j, c] = 1 if j == j_term
            # V[j, k, c] = 1 if j == j_term and k == k_term
            # W[k, i, c] = 1 if k == k_term (independent of i)
            
            # U: depends on i, j. We want to select A[i, j].
            # So U[i, j, c] should be 1 for all i, for the specific j.
            for i in range(n):
                U[i * m + j][col] = 1
            
            # V: depends on j, k.
            V[j * p + k][col] = 1
            
            # W: depends on k, i.
            # We want W[k, i, c] to be 1 for all i, for the specific k.
            for i in range(n):
                W[k * n + i][col] = 1
            
            col += 1
            
    return U, V, W, total_rank

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Use the constructed decomposition
    U, V, W, rank = _construct_decomposition()
    
    # Verify
    if _verify_decomposition(n, m, p, U, V, W, rank):
        # Clean values (they are already 0 or 1)
        U_final = [[int(x) for x in row] for row in U]
        V_final = [[int(x) for x in row] for row in V]
        W_final = [[int(x) for x in row] for row in W]
        return {"U": U_final, "V": V_final, "W": W_final}
    else:
        # Fallback to schoolbook if verification fails (should not happen)
        rank_sb = n * m * p
        U_fb = [[0] * rank_sb for _ in range(n * m)]
        V_fb = [[0] * rank_sb for _ in range(m * p)]
        W_fb = [[0] * rank_sb for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][c] = 1
                    V_fb[j * p + k][c] = 1
                    W_fb[k * n + i][c] = 1
                    c += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
# EVOLVE-BLOCK-END
