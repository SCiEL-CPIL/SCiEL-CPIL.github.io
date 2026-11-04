import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank decomposition for 5x5 matrix multiplication using schoolbook multiplication.
    Rank: 125.
    """
    start_time = time.time()
    deadline = 120.0
    safety_margin = 5.0
    
    n, m, p = DIMENSIONS
    nm = n * m  # 25
    mp = m * p  # 25
    pn = p * n  # 25
    
    terms = []
    
    def add_term(u_idx, v_idx, w_idx, coeff=1.0):
        if coeff != 0:
            terms.append((u_idx, v_idx, w_idx, coeff))

    # Index mapping:
    # A: row i, col j -> i*m + j
    # B: row j, col k -> j*p + k
    # C: row i, col k -> k*n + i (Trace convention W[k*n+i])
    
    def get_A_idx(i, j):
        return i * m + j
        
    def get_B_idx(j, k):
        return j * p + k
        
    def get_C_idx(i, k):
        return k * n + i

    # Schoolbook for the entire 5x5x5
    for i in range(n):
        for j in range(m):
            for k in range(p):
                add_term(
                    get_A_idx(i, j),
                    get_B_idx(j, k),
                    get_C_idx(i, k),
                    1
                )

    rank = len(terms)
    
    U = np.zeros((nm, rank), dtype=np.float64)
    V = np.zeros((mp, rank), dtype=np.float64)
    W = np.zeros((pn, rank), dtype=np.float64)
    
    for r, (u_idx, v_idx, w_idx, coeff) in enumerate(terms):
        U[u_idx, r] = coeff
        V[v_idx, r] = coeff
        W[w_idx, r] = coeff
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
