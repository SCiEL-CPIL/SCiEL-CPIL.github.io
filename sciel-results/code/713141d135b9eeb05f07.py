"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 3, 3
    # Use Naive Decomposition (Rank 27) to ensure correctness and avoid index errors
    rank = n * m * p
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    c = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                # A[i, k] * B[k, j] -> C[i, j]
                u_idx = i * m + k
                v_idx = k * p + j
                # W index convention: W[k*n+i] in prompt text? 
                # Prompt: "W[k*n+i,r]". 
                # If W is p*n, and index is k*n+i, then k in [0,p), i in [0,n).
                # This maps to C[j, i] if we assume standard row-major for C (n x p) is C[i, j].
                # Let's assume the prompt's "k*n+i" refers to the linear index of the flattened W matrix.
                # If W represents C^T (p x n), then W[j, i] is at j*n + i.
                # So w_idx = j * n + i.
                w_idx = j * n + i
                
                U[u_idx][c] = 1
                V[v_idx][c] = 1
                W[w_idx][c] = 1
                c += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
