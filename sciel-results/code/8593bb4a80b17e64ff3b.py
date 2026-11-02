import numpy as np
import time

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x2 by 2x4 matrix multiplication.
    U: (n*m, rank) = (4, rank)
    V: (m*p, rank) = (8, rank)
    W: (p*n, rank) = (8, rank)
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n, m, p = 2, 2, 4
    rank_14 = 14
    
    # First, build the rank-14 Strassen decomposition as fallback
    def build_strassen_14():
        rank = 14
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        
        strassen_terms = [
            ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
            ([(2, 1), (3, 1)], [(0, 1)], [(1, 1), (3, -1)]),
            ([(0, 1)], [(1, 1), (3, -1)], [(2, 1), (3, 1)]),
            ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (1, 1)]),
            ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (2, 1)]),
            ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
            ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
        ]
        
        for block in range(2):
            for r_local in range(7):
                r_global = block * 7 + r_local
                u_terms, v_terms, w_terms = strassen_terms[r_local]
                
                for idx, val in u_terms:
                    U[idx, r_global] += val
                    
                for idx, val in v_terms:
                    j = idx // 2
                    k = idx % 2
                    k_global = k + block * 2
                    global_v_idx = j * p + k_global
                    V[global_v_idx, r_global] += val
                    
                for idx, val in w_terms:
                    k = idx // 2
                    i = idx % 2
                    k_global = k + block * 2
                    global_w_idx = k_global * n + i
                    W[global_w_idx, r_global] += val
        
        return U, V, W
    
    U14, V14, W14 = build_strassen_14()
    
    # Verify the solution
    def build_tensor():
        T = np.zeros((n*m, m*p, p*n), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m+j, j*p+k, k*n+i] = 1.0
        return T
    
    T = build_tensor()
    T_rec = np.zeros((4, 8, 8), dtype=float)
    for r in range(14):
        T_rec += np.einsum('i,j,k->ijk', U14[:,r], V14[:,r], W14[:,r])
    
    if not np.allclose(T_rec, T, atol=1e-10):
        # This should not happen, but fallback to naive if it does
        # Naive rank 16
        U = np.zeros((4, 16), dtype=float)
        V = np.zeros((8, 16), dtype=float)
        W = np.zeros((8, 16), dtype=float)
        idx = 0
        for i in range(2):
            for j in range(2):
                for k in range(4):
                    U[i*2+j, idx] = 1
                    V[j*4+k, idx] = 1
                    W[k*2+i, idx] = 1
                    idx += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    return {
        "U": U14.tolist(),
        "V": V14.tolist(),
        "W": W14.tolist()
    }
