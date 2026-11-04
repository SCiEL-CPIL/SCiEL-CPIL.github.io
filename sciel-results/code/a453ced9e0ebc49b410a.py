import time
import numpy as np

_START_TIME = time.time()
TIME_LIMIT = 110.0

def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)

def _strassen_2x2x2():
    """Return the 7 rank-1 terms for 2x2x2 Strassen."""
    return [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]

def _strassen_4x4x4():
    """
    Return rank-1 terms for 4x4x4 using recursive Strassen.
    Each term is (U_dict, V_dict, W_dict) where keys are (row, col) in 4x4, 4x4, 4x4.
    Rank = 7 * 7 = 49? No, recursive Strassen on 4x4x4:
    Split into 8 blocks of 2x2x2. Each has rank 7. Total rank = 8*7 = 56.
    But actually, Strassen on 4x4: we split 4x4 into four 2x2 blocks.
    The 4x4x4 tensor splits into 8 sub-tensors of 2x2x2, each rank 7.
    So rank = 8*7 = 56. That's the same as before.
    
    Wait, that's not right. Let me think again.
    For 4x4x4: T_{i,j,k} with i,j,k in 0..3.
    Split i = 2*i1 + i2, j = 2*j1 + j2, k = 2*k1 + k2 where i1,j1,k1 in 0..1, i2,j2,k2 in 0..1.
    Then T_{i,j,k} = T_{i1,i2,j1,j2,k1,k2}.
    The tensor is a sum over (i1,j1,k1) of 2x2x2 blocks.
    Each 2x2x2 block has rank 7. There are 8 blocks. Total rank = 8*7 = 56.
    
    Hmm, that's the same as the current approach for the 4x4x4 part.
    But the current approach does 4x4x6 as 7*8 = 56 by splitting k into 3+3.
    
    Let me try a different split for 4x4x4. Actually, there's no better known decomposition for 4x4x4 than 56 via this method. The true rank of 4x4x4 is known to be at most 49 (Landsberg) but constructing it is complex.
    
    Let me just return the 56-rank decomposition for 4x4x4.
    """
    S2 = _strassen_2x2x2()
    terms = []
    for u1, v1, w1 in S2:
        for u2, v2, w2 in S2:
            U_dict = {}
            V_dict = {}
            W_dict = {}
            for (i1, j1), c1 in u1.items():
                for (i2, j2), c2 in u2.items():
                    i = i1 * 2 + i2
                    j = j1 * 2 + j2
                    U_dict[(i, j)] = U_dict.get((i, j), 0) + c1 * c2
            for (j1, k1), c1 in v1.items():
                for (j2, k2), c2 in v2.items():
                    j = j1 * 2 + j2
                    k = k1 * 2 + k2
                    V_dict[(j, k)] = V_dict.get((j, k), 0) + c1 * c2
            for (k1, i1), c1 in w1.items():
                for (k2, i2), c2 in w2.items():
                    k = k1 * 2 + k2
                    i = i1 * 2 + i2
                    W_dict[(k, i)] = W_dict.get((k, i), 0) + c1 * c2
            terms.append((U_dict, V_dict, W_dict))
    return terms

def _rank1_4x4x2():
    """
    Return rank-1 terms for 4x4x2 tensor.
    Split i into 2+2, j into 2+2, giving 4 blocks of 2x2x2, each rank 7.
    Total rank = 4*7 = 28.
    """
    S2 = _strassen_2x2x2()
    terms = []
    for i1 in range(2):
        for j1 in range(2):
            for u2, v2, w2 in S2:
                U_dict = {}
                V_dict = {}
                W_dict = {}
                for (i2, j2), c in u2.items():
                    i = i1 * 2 + i2
                    j = j1 * 2 + j2
                    U_dict[(i, j)] = c
                for (j2, k2), c in v2.items():
                    j = j1 * 2 + j2
                    k = k2  # k is 0 or 1
                    V_dict[(j, k)] = c
                for (k2, i2), c in w2.items():
                    k = k2
                    i = i1 * 2 + i2
                    W_dict[(k, i)] = c
                terms.append((U_dict, V_dict, W_dict))
    return terms

def construct():
    n, m, p = 4, 4, 6
    
    # Split k (6) into k4 (4) + k2 (2)
    # Part 1: 4x4x4 (k=0..3), rank 56 via recursive Strassen
    # Part 2: 4x4x2 (k=4..5), rank 28 via block Strassen
    
    terms_4x4x4 = _strassen_4x4x4()  # 56 terms
    terms_4x4x2 = _rank1_4x4x2()     # 28 terms
    
    rank = len(terms_4x4x4) + len(terms_4x4x2)
    
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Fill part 1: 4x4x4 (k offset = 0)
    for idx, (U_dict, V_dict, W_dict) in enumerate(terms_4x4x4):
        for (i, j), c in U_dict.items():
            U[i * m + j][idx] += c
        for (j, k), c in V_dict.items():
            V[j * p + k][idx] += c
        for (k, i), c in W_dict.items():
            W[k * n + i][idx] += c
    
    # Fill part 2: 4x4x2 (k offset = 4)
    for idx, (U_dict, V_dict, W_dict) in enumerate(terms_4x4x2):
        idx2 = len(terms_4x4x4) + idx
        for (i, j), c in U_dict.items():
            U[i * m + j][idx2] += c
        for (j, k), c in V_dict.items():
            V[j * p + (k + 4)][idx2] += c
        for (k, i), c in W_dict.items():
            W[(k + 4) * n + i][idx2] += c
    
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback: naive rank-96 decomposition
    rank_fb = n * m * p
    U_fb = [[0] * rank_fb for _ in range(n * m)]
    V_fb = [[0] * rank_fb for _ in range(m * p)]
    W_fb = [[0] * rank_fb for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][column] = 1
                V_fb[j * p + k][column] = 1
                W_fb[k * n + i][column] = 1
                column += 1
    return {"U": U_fb, "V": V_fb, "W": W_fb}

def _verify_decomposition(U, V, W, n, m, p):
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.int64)
    V_arr = np.array(V, dtype=np.int64)
    W_arr = np.array(W, dtype=np.int64)
    
    i_vals, j_vals, k_vals = np.meshgrid(
        np.arange(n), np.arange(m), np.arange(p), indexing='ij'
    )
    u_indices = (i_vals * m + j_vals).ravel()
    v_indices = (j_vals * p + k_vals).ravel()
    w_indices = (k_vals * n + i_vals).ravel()
    
    U_sel = U_arr[u_indices, :]
    V_sel = V_arr[v_indices, :]
    W_sel = W_arr[w_indices, :]
    
    totals = np.sum(U_sel * V_sel * W_sel, axis=1)
    
    return bool(np.all(totals == 1))
