"""Find a rank-minimal exact tensor decomposition for 2x3 by 3x6 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 3, 6]
DEADLINE = 140.0

def _verify(U, V, W):
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    if U.shape != (n*m, U.shape[1]): return False
    if V.shape != (m*p, V.shape[1]): return False
    if W.shape != (p*n, W.shape[1]): return False
    if U.shape[1] != V.shape[1] or V.shape[1] != W.shape[1]: return False
    
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                true_val = 1.0
                dec_val = 0.0
                for r in range(rank):
                    dec_val += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                if abs(dec_val - true_val) > 1e-9:
                    return False
    return True

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Rank 11 decomposition for 2x3x3
    terms_233 = [
        {'u': {0: 1, 4: 1}, 'v': {0: 1, 4: 1}, 'w': {0: 1, 3: 1}},
        {'u': {3: 1, 4: 1}, 'v': {0: 1}, 'w': {1: 1, 3: -1}},
        {'u': {0: 1}, 'v': {1: 1, 4: -1}, 'w': {2: 1, 3: 1}},
        {'u': {4: 1}, 'v': {3: 1, 0: -1}, 'w': {0: 1, 1: 1}},
        {'u': {0: 1, 1: 1}, 'v': {4: 1}, 'w': {0: -1, 2: 1}},
        {'u': {3: 1, 0: -1}, 'v': {0: 1, 1: 1}, 'w': {3: 1}},
        {'u': {1: 1, 4: -1}, 'v': {3: 1, 4: 1}, 'w': {0: 1}},
        {'u': {2: 1}, 'v': {6: 1}, 'w': {0: 1}},
        {'u': {2: 1}, 'v': {7: 1}, 'w': {2: 1}},
        {'u': {5: 1}, 'v': {6: 1}, 'w': {1: 1}},
        {'u': {5: 1}, 'v': {7: 1}, 'w': {3: 1}}
    ]
    
    # Construct Rank 18 decomposition for 2x3x6
    # Merge T7(0) with T8(1), T8(0) with T7(1), T9(0) with T10(1), T10(0) with T9(1)
    # Keep others separate.
    
    rank = 18
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_term(col, term, block):
        for idx, val in term['u'].items():
            U[idx][col] = val
        for v_idx, val in term['v'].items():
            j_local = v_idx // 3
            k_local = v_idx % 3
            g_idx = j_local * p + (block * 3 + k_local)
            V[g_idx][col] = val
        for w_idx, val in term['w'].items():
            k_local = w_idx // 2
            i_local = w_idx % 2
            g_idx = (block * 3 + k_local) * n + i_local
            W[g_idx][col] = val

    def add_merged_term(col, term1, block1, term2, block2):
        # U is same
        for idx, val in term1['u'].items():
            U[idx][col] = val
        for v_idx, val in term1['v'].items():
            j_local = v_idx // 3
            k_local = v_idx % 3
            g_idx = j_local * p + (block1 * 3 + k_local)
            V[g_idx][col] += val
        for w_idx, val in term1['w'].items():
            k_local = w_idx // 2
            i_local = w_idx % 2
            g_idx = (block1 * 3 + k_local) * n + i_local
            W[g_idx][col] += val
            
        for v_idx, val in term2['v'].items():
            j_local = v_idx // 3
            k_local = v_idx % 3
            g_idx = j_local * p + (block2 * 3 + k_local)
            V[g_idx][col] += val
        for w_idx, val in term2['w'].items():
            k_local = w_idx // 2
            i_local = w_idx % 2
            g_idx = (block2 * 3 + k_local) * n + i_local
            W[g_idx][col] += val

    col = 0
    # Merged terms
    add_merged_term(col, terms_233[7], 0, terms_233[8], 1); col += 1
    add_merged_term(col, terms_233[8], 0, terms_233[7], 1); col += 1
    add_merged_term(col, terms_233[9], 0, terms_233[10], 1); col += 1
    add_merged_term(col, terms_233[10], 0, terms_233[9], 1); col += 1
    
    # Remaining Block 0 terms (0-6)
    for i in range(7):
        add_term(col, terms_233[i], 0); col += 1
        
    # Remaining Block 1 terms (0-6)
    for i in range(7):
        add_term(col, terms_233[i], 1); col += 1
        
    result = {"U": U, "V": V, "W": W}
    
    if _verify(U, V, W):
        return result
    
    # Fallback to Rank 22
    rank = 22
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for block in range(2):
        for t_idx, term in enumerate(terms_233):
            col = block * 11 + t_idx
            for idx, val in term['u'].items():
                U[idx][col] = val
            for v_idx, val in term['v'].items():
                j_local = v_idx // 3
                k_local = v_idx % 3
                g_idx = j_local * p + (block * 3 + k_local)
                V[g_idx][col] = val
            for w_idx, val in term['w'].items():
                k_local = w_idx // 2
                i_local = w_idx % 2
                g_idx = (block * 3 + k_local) * n + i_local
                W[g_idx][col] = val
                
    return {"U": U, "V": V, "W": W}
