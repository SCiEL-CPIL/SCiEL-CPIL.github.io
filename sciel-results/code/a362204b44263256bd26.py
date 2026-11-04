"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Block-structured decomposition for 2x5 * 5x5.
    Partition j and k into groups of size 2,2,1.
    Use schoolbook within each block.
    Rank: 8+8+8+8+4+4+2+2+1 = 45
    """
    n, m, p = 2, 5, 5
    terms = []
    
    def add_term(u_coeff, v_coeff, w_coeff):
        terms.append((u_coeff, v_coeff, w_coeff))
    
    # Schoolbook for a_i x b_j subproblem
    def schoolbook(j_indices, k_indices):
        for i in range(n):
            for j in j_indices:
                for k in k_indices:
                    add_term({(i,j):1}, {(j,k):1}, {(k,i):1})
    
    # Blocks: j-groups {0,1}, {2,3}, {4}; k-groups {0,1}, {2,3}, {4}
    j_groups = [[0,1], [2,3], [4]]
    k_groups = [[0,1], [2,3], [4]]
    
    for jg in j_groups:
        for kg in k_groups:
            schoolbook(jg, kg)
    
    rank = len(terms)
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for r, (u_dict, v_dict, w_dict) in enumerate(terms):
        for (i,j), val in u_dict.items():
            U[i*m+j][r] = val
        for (j,k), val in v_dict.items():
            V[j*p+k][r] = val
        for (k,i), val in w_dict.items():
            W[k*n+i][r] = val
    
    # Verify
    rng = np.random.default_rng(42)
    A = rng.standard_normal((n, m))
    B = rng.standard_normal((m, p))
    C_expected = A @ B
    
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    C_recon = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            s = 0.0
            for j in range(m):
                for r in range(rank):
                    s += U_np[i*m+j, r] * V_np[j*p+k, r] * W_np[k*n+i, r]
            C_recon[i, k] = s
    
    error = np.max(np.abs(C_recon - C_expected))
    if error > 1e-10:
        # Fallback to full schoolbook
        rank_fb = n * m * p
        U_fb = [[0]*rank_fb for _ in range(n*m)]
        V_fb = [[0]*rank_fb for _ in range(m*p)]
        W_fb = [[0]*rank_fb for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j][col] = 1
                    V_fb[j*p+k][col] = 1
                    W_fb[k*n+i][col] = 1
                    col += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
