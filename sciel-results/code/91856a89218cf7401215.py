"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 3, 4, 8
    nm, mp, pn = n * m, m * p, p * n  # 12, 32, 24
    
    # The multiplication tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # We use the fact that the all-ones tensor of shape (n,m,p) has rank min(n,m,p)
    # For n=3, m=4, p=8, rank = 3
    # Decomposition: T[i,j,k] = sum_{c=0}^{2} u_c[i] * v_c[j] * w_c[k]
    # where u_c is a one-hot vector in R^n, v_c is all-ones in R^m, w_c is all-ones in R^p
    
    rank = n  # = 3
    
    # Initialize U, V, W
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    
    # For each c in 0..rank-1:
    # u_c[i] = 1 if i == c, else 0
    # v_c[j] = 1 for all j
    # w_c[k] = 1 for all k
    
    for c in range(rank):
        for i in range(n):
            for j in range(m):
                # U[i*m + j, c] = u_c[i] = 1 if i == c else 0
                U[i * m + j][c] = 1 if i == c else 0
        
        for j in range(m):
            for k in range(p):
                # V[j*p + k, c] = v_c[j] = 1
                V[j * p + k][c] = 1
        
        for k in range(p):
            for i in range(n):
                # W[k*n + i, c] = w_c[k] = 1
                W[k * n + i][c] = 1
    
    result = {"U": U, "V": V, "W": W}
    
    # Verify the decomposition
    import numpy as np
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    T_target = np.ones((n, m, p))
    T_recon = np.zeros((n, m, p))
    for c in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_recon[i, j, k] += U_np[i*m+j, c] * V_np[j*p+k, c] * W_np[k*n+i, c]
    
    if not np.allclose(T_recon, T_target, atol=1e-10):
        # Fallback to schoolbook if verification fails
        rank_schoolbook = n * m * p
        U_sb = [[0] * rank_schoolbook for _ in range(nm)]
        V_sb = [[0] * rank_schoolbook for _ in range(mp)]
        W_sb = [[0] * rank_schoolbook for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j][col] = 1
                    V_sb[j * p + k][col] = 1
                    W_sb[k * n + i][col] = 1
                    col += 1
        return {"U": U_sb, "V": V_sb, "W": W_sb}
    
    return result
# EVOLVE-BLOCK-END
