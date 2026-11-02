import numpy as np

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    rank = 88
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    def add_rank1(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        nonlocal col
        if col >= rank:
            return
        U[u_idx, col] = u_val
        V[v_idx, col] = v_val
        W[w_idx, col] = w_val
        col += 1

    # Strassen 2x2x2
    # Terms t=0..6
    # U indices (i', j') and values
    # V indices (j', k') and values
    # W indices (k', i') and values
    
    strassen_data = [
        # M1
        {
            'U': [((0,0), 1), ((1,1), 1)],
            'V': [((0,0), 1), ((1,1), 1)],
            'W': [((0,0), 1), ((1,1), 1)]
        },
        # M2
        {
            'U': [((1,0), 1), ((1,1), 1)],
            'V': [((0,0), 1)],
            'W': [((0,1), 1), ((1,1), -1)]
        },
        # M3
        {
            'U': [((0,0), 1)],
            'V': [((1,0), 1), ((1,1), -1)],
            'W': [((1,0), 1), ((1,1), 1)]
        },
        # M4
        {
            'U': [((1,1), 1)],
            'V': [((0,1), 1), ((1,1), 1)],
            'W': [((0,0), 1), ((0,1), 1)]
        },
        # M5
        {
            'U': [((0,0), 1), ((0,1), 1)],
            'V': [((1,1), 1)],
            'W': [((0,0), -1), ((1,0), 1)]
        },
        # M6
        {
            'U': [((1,0), 1), ((0,0), -1)],
            'V': [((0,0), 1), ((0,1), 1)],
            'W': [] # Not used in reconstruction?
        },
        # M7
        {
            'U': [((0,1), 1), ((1,1), -1)],
            'V': [((1,0), 1), ((1,1), 1)],
            'W': [((0,0), -1), ((1,1), 1)]
        }
    ]

    for b in range(4):
        k_off = 2 * b
        for s in range(2):
            j_off = 2 * s
            
            # 7 Strassen terms
            for t in range(7):
                data = strassen_data[t]
                
                # We need to combine multiple U entries into one column?
                # No, a rank-1 term has a SINGLE value in U, V, W for each index?
                # No, U is a vector. It can have multiple non-zero entries.
                # But it's a single column in the matrix U.
                # So for a given rank-1 term, U[:, col] is a vector with multiple non-zeros.
                
                # So I should initialize U[:, col], V[:, col], W[:, col] to 0
                # and then fill them.
                
                if col >= rank:
                    continue
                
                # U
                for (li, lj), val in data['U']:
                    i_g = li
                    j_g = j_off + lj
                    U[i_g * m + j_g, col] = val
                
                # V
                for (lj, lk), val in data['V']:
                    j_g = j_off + lj
                    k_g = k_off + lk
                    V[j_g * p + k_g, col] = val
                
                # W
                for (lk, li), val in data['W']:
                    k_g = k_off + lk
                    i_g = li
                    W[k_g * n + i_g, col] = val
                
                col += 1
                
            # 4 Schoolbook terms for i=2
            for lj in range(2):
                for lk in range(2):
                    if col >= rank:
                        continue
                    i_g = 2
                    j_g = j_off + lj
                    k_g = k_off + lk
                    
                    U[i_g * m + j_g, col] = 1
                    V[j_g * p + k_g, col] = 1
                    W[k_g * n + i_g, col] = 1
                    
                    col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
