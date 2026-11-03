"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 7
    start_time = time.time()
    deadline = start_time + 120  # Safety margin
    
    U_rows = []
    V_rows = []
    W_rows = []
    
    def add_term(u_vec, v_vec, w_vec):
        """Add a rank-1 term."""
        U_rows.append(list(u_vec))
        V_rows.append(list(v_vec))
        W_rows.append(list(w_vec))
    
    def zeros(length):
        return [0] * length

    # Rank 12 decomposition
    # For each i in n and j in m, we add a term that handles the entire row i of A 
    # and column j of A, summing over all k in B.
    # Term (i,j):
    # U[i*m+j, r] = 1
    # V[j*p+k, r] = 1 for all k
    # W[k*n+i, r] = 1 for all k
    
    rank = 0
    for i in range(n):
        for j in range(m):
            u = zeros(n * m)
            v = zeros(m * p)
            w = zeros(p * n)
            
            u[i * m + j] = 1
            
            for k in range(p):
                v[j * p + k] = 1
                w[k * n + i] = 1
                
            add_term(u, v, w)
            rank += 1
            
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    # Convert to proper matrices: U_rows is (rank, n*m), we want (n*m, rank)
    if not U_rows:
        # Fallback to schoolbook if something went wrong
        rank_schoolbook = n * m * p
        U = [[0] * rank_schoolbook for _ in range(n * m)]
        V = [[0] * rank_schoolbook for _ in range(m * p)]
        W = [[0] * rank_schoolbook for _ in range(p * n)]
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    U = [list(col) for col in zip(*U_rows)]
    V = [list(col) for col in zip(*V_rows)]
    W = [list(col) for col in zip(*W_rows)]
    
    # Verify correctness
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Reconstruct the tensor
    T_reconstructed = np.zeros((n * m, m * p, p * n))
    for r in range(rank):
        T_reconstructed += np.einsum('a,b,c->abc', U_np[:, r], V_np[:, r], W_np[:, r])
    
    # Check against the true multiplication tensor
    T_true = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_true[i * m + j, j * p + k, k * n + i] = 1
                
    if not np.allclose(T_reconstructed, T_true, atol=1e-9):
        # Fallback to schoolbook approach
        rank_schoolbook = n * m * p
        U = [[0] * rank_schoolbook for _ in range(n * m)]
        V = [[0] * rank_schoolbook for _ in range(m * p)]
        W = [[0] * rank_schoolbook for _ in range(p * n)]
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
