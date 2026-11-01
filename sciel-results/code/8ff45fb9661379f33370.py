import time
import numpy as np
import random

DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Precompute minimal ranks for p up to 13 using Strassen-like recursion
    # R(1) = 4, R(2) = 7
    # R(p) = min(R(k) + R(p-k))
    
    def compute_ranks(max_p):
        R = [0] * (max_p + 1)
        R[1] = 4
        R[2] = 7
        for i in range(3, max_p + 1):
            min_r = float('inf')
            split = 1
            for k in range(1, i):
                if R[k] + R[i-k] < min_r:
                    min_r = R[k] + R[i-k]
                    split = k
            R[i] = min_r
            R[i] = min_r # Store split? No, just store rank. We need split for construction.
            # Let's store splits in a separate dict
        return R

    ranks = [0] * 14
    splits = [0] * 14
    ranks[1] = 4
    ranks[2] = 7
    for i in range(3, 14):
        min_r = float('inf')
        best_split = 1
        for k in range(1, i):
            r_val = ranks[k] + ranks[i-k]
            if r_val < min_r:
                min_r = r_val
                best_split = k
        ranks[i] = min_r
        splits[i] = best_split

    def get_base_decomp(p):
        """
        Returns U, V, W for M_{2,2,p} for p=1 or p=2.
        U: (4, r)
        V: (2p, r)
        W: (2p, r)
        """
        if p == 1:
            r = 4
            U = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ], dtype=float)
            V = np.array([
                [1, 0, 1, 0],
                [0, 1, 0, 1]
            ], dtype=float)
            W = np.array([
                [1, 1, 0, 0],
                [0, 0, 1, 1]
            ], dtype=float)
            return U, V, W
        elif p == 2:
            r = 7
            # Strassen decomposition for 2x2 * 2x2
            # A = [[a, b], [c, d]]
            # B = [[e, f], [g, h]]
            # C = [[i, j], [k, l]]
            # 
            # M1 = (a+d)(e+h)
            # M2 = (c+d)e
            # M3 = a(f-h)
            # M4 = d(g-e)
            # M5 = (a+b)h
            # M6 = (c-a)(e+f)
            # M7 = (b-d)(g+h)
            #
            # i = M1 + M4 - M5 + M7
            # j = M3 + M5
            # k = M2 + M4
            # l = M1 - M2 + M3 - M6
            
            # U columns correspond to a, b, c, d
            # V columns correspond to e, f, g, h
            # W columns correspond to i, j, k, l
            
            U = np.zeros((4, r), dtype=float)
            V = np.zeros((4, r), dtype=float) # 2*p = 4 rows
            W = np.zeros((4, r), dtype=float) # 2*p = 4 rows
            
            # M1: (a+d)(e+h)
            U[0, 0] = 1; U[3, 0] = 1
            V[0, 0] = 1; V[3, 0] = 1
            W[0, 0] = 1; W[3, 0] = -1 # i += M1, l -= M1? No, l = M1 - M2...
            # Let's be careful with signs.
            # i = M1 + M4 - M5 + M7
            # j = M3 + M5
            # k = M2 + M4
            # l = M1 - M2 + M3 - M6
            
            # Col 0: M1
            U[0, 0] = 1; U[3, 0] = 1
            V[0, 0] = 1; V[3, 0] = 1
            W[0, 0] = 1 # i
            W[3, 0] = 1 # l
            
            # Col 1: M2
            U[2, 1] = 1; U[3, 1] = 1
            V[0, 1] = 1
            W[2, 1] = 1 # k
            W[3, 1] = -1 # l
            
            # Col 2: M3
            U[0, 2] = 1
            V[1, 2] = 1; V[3, 2] = -1
            W[1, 2] = 1 # j
            W[3, 2] = 1 # l
            
            # Col 3: M4
            U[3, 3] = 1
            V[2, 3] = 1; V[0, 3] = -1
            W[0, 3] = 1 # i
            W[2, 3] = 1 # k
            
            # Col 4: M5
            U[0, 4] = 1; U[1, 4] = 1
            V[3, 4] = 1
            W[0, 4] = -1 # i
            W[1, 4] = 1 # j
            
            # Col 5: M6
            U[2, 5] = 1; U[0, 5] = -1
            V[0, 5] = 1; V[1, 5] = 1
            W[3, 5] = -1 # l
            
            # Col 6: M7
            U[1, 6] = 1; U[3, 6] = -1
            V[2, 6] = 1; V[3, 6] = 1
            W[0, 6] = 1 # i
            
            return U, V, W

    def combine_decomps(U1, V1, W1, p1, U2, V2, W2, p2):
        """
        Combine two decompositions for p1 and p2 into one for p1+p2.
        """
        r1 = U1.shape[1]
        r2 = U2.shape[1]
        r = r1 + r2
        p = p1 + p2
        
        U = np.hstack([U1, U2])
        
        # V has 2p rows.
        # V1 has 2p1 rows. Pad with 2p2 zeros at the bottom.
        # V2 has 2p2 rows. Pad with 2p1 zeros at the top.
        V1_padded = np.vstack([V1, np.zeros((2*p2, r1))])
        V2_padded = np.vstack([np.zeros((2*p1, r2)), V2])
        V = np.hstack([V1_padded, V2_padded])
        
        # W has 2p rows.
        # W1 has 2p1 rows. Pad with 2p2 zeros at the bottom.
        # W2 has 2p2 rows. Pad with 2p1 zeros at the top.
        W1_padded = np.vstack([W1, np.zeros((2*p2, r1))])
        W2_padded = np.vstack([np.zeros((2*p1, r2)), W2])
        W = np.hstack([W1_padded, W2_padded])
        
        return U, V, W

    def build_decomp(p):
        if p == 1:
            return get_base_decomp(1)
        if p == 2:
            return get_base_decomp(2)
        
        p1 = splits[p]
        p2 = p - p1
        
        U1, V1, W1 = build_decomp(p1)
        U2, V2, W2 = build_decomp(p2)
        
        return combine_decomps(U1, V1, W1, p1, U2, V2, W2, p2)

    U, V, W = build_decomp(p)
    
    # Verify validity
    # Target indices: (i*m+j, j*p+k, k*n+i)
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target_indices.append((i * m + j, j * p + k, k * n + i))
                
    target_arr = np.array(target_indices)
    a_idx = target_arr[:, 0]
    b_idx = target_arr[:, 1]
    c_idx = target_arr[:, 2]
    
    U_t = U[a_idx, :]
    V_t = V[b_idx, :]
    W_t = W[c_idx, :]
    
    prod = U_t * V_t * W_t
    sums = np.sum(prod, axis=1)
    
    if not np.all(np.abs(sums - 1.0) < 1e-9):
        # Fallback to schoolbook if construction fails
        R0 = n * m * p
        U = np.zeros((nm, R0), dtype=float)
        V = np.zeros((mp, R0), dtype=float)
        W = np.zeros((pn, R0), dtype=float)
        col_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col_idx] = 1.0
                    V[j * p + k, col_idx] = 1.0
                    W[k * n + i, col_idx] = 1.0
                    col_idx += 1
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
