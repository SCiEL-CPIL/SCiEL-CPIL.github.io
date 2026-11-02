"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Block sizes
    s_i = [1, 2]
    s_j = [1, 2, 2]
    s_k = [2, 2, 2]
    
    # Offsets for each block
    # i offsets
    off_i = []
    cur = 0
    for size in s_i:
        off_i.append(cur)
        cur += size
    # j offsets
    off_j = []
    cur = 0
    for size in s_j:
        off_j.append(cur)
        cur += size
    # k offsets
    off_k = []
    cur = 0
    for size in s_k:
        off_k.append(cur)
        cur += size

    # Precompute local Strassen terms
    # Each term is a dict mapping (local_u, local_v) to coeff for U, (local_v, local_w) to coeff for V, (local_w, local_u) to coeff for W?
    # No, U is indexed by (u, v). V by (v, w). W by (w, u).
    # Wait, W index is k*n + i. So W is indexed by (w, u) where w is from B col, u is from A row.
    # Yes.
    
    # Strassen terms for 2x2x2
    # U coeffs: (u, v) -> val
    # V coeffs: (v, w) -> val
    # W coeffs: (w, u) -> val
    strassen_terms = [
        # M1
        {
            'U': { (0,0): 1, (1,1): 1 },
            'V': { (0,0): 1, (1,1): 1 },
            'W': { (0,0): 1, (1,1): 1 }
        },
        # M2
        {
            'U': { (1,0): 1, (1,1): 1 },
            'V': { (0,0): 1 },
            'W': { (0,1): 1 } # C10 = u=1, v=0? No, C index is (w, u). C10 means w=0, u=1.
        },
        # M3
        {
            'U': { (0,0): 1 },
            'V': { (1,0): 1, (1,1): -1 }, # B10-B11. B index (v,w). v=1. w=0,1.
            'W': { (1,0): 1, (1,1): 1 } # C01 (w=1,u=0) and C11 (w=1,u=1).
        },
        # M4
        {
            'U': { (1,1): 1 },
            'V': { (1,1): 1 },
            'W': { (0,0): 1, (0,1): 1 } # C00 (w=0,u=0) and C10 (w=0,u=1).
        },
        # M5
        {
            'U': { (0,0): 1, (0,1): 1 }, # A00+A01. v=0,1. u=0.
            'V': { (1,1): 1 },
            'W': { (0,0): -1, (1,0): 1 } # C00 -M5, C01 +M5.
        },
        # M6
        {
            'U': { (1,0): 1, (0,0): -1 }, # A10-A00.
            'V': { (0,0): 1, (0,1): 1 }, # B00+B01.
            'W': { (1,1): 1 } # C11 +M6.
        },
        # M7
        {
            'U': { (0,1): 1, (1,1): -1 }, # A01-A11.
            'V': { (1,0): 1, (1,1): 1 }, # B10+B11.
            'W': { (0,0): 1 } # C00 +M7.
        }
    ]
    
    # Schoolbook terms generator
    def get_schoolbook_terms(si, sj, sk):
        terms = []
        for u in range(si):
            for v in range(sj):
                for w in range(sk):
                    terms.append({
                        'U': { (u, v): 1 },
                        'V': { (v, w): 1 },
                        'W': { (w, u): 1 }
                    })
        return terms

    rank = 0
    U_cols = []
    V_cols = []
    W_cols = []

    for i in range(len(s_i)):
        for j in range(len(s_j)):
            for k in range(len(s_k)):
                si = s_i[i]
                sj = s_j[j]
                sk = s_k[k]
                
                if si == 2 and sj == 2 and sk == 2:
                    terms = strassen_terms
                else:
                    terms = get_schoolbook_terms(si, sj, sk)
                
                for term in terms:
                    # Create vectors for this column
                    u_vec = [0] * (n * m)
                    v_vec = [0] * (m * p)
                    w_vec = [0] * (p * n)
                    
                    for (lu, lv), coeff in term['U'].items():
                        gu = off_i[i] + lu
                        gv = off_j[j] + lv
                        u_vec[gu * m + gv] += coeff
                    
                    for (lv, lw), coeff in term['V'].items():
                        gv = off_j[j] + lv
                        gw = off_k[k] + lw
                        v_vec[gv * p + gw] += coeff
                    
                    for (lw, lu), coeff in term['W'].items():
                        gw = off_k[k] + lw
                        gu = off_i[i] + lu
                        w_vec[gw * n + gu] += coeff
                    
                    U_cols.append(u_vec)
                    V_cols.append(v_vec)
                    W_cols.append(w_vec)
                    rank += 1

    U = [list(map(int, col)) for col in U_cols] # Note: U_cols are rows of the matrix? No, they are columns.
    # The return format expects U, V, W as lists of rows?
    # "factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    # Usually matrices are row-major lists.
    # So U[i][c] is the element at row i, col c.
    # My U_cols[c][r] is row r, col c.
    # I need to transpose them.
    
    U_mat = [[0]*rank for _ in range(n*m)]
    V_mat = [[0]*rank for _ in range(m*p)]
    W_mat = [[0]*rank for _ in range(p*n)]
    
    for c in range(rank):
        for r in range(n*m):
            U_mat[r][c] = U_cols[c][r]
        for r in range(m*p):
            V_mat[r][c] = V_cols[c][r]
        for r in range(p*n):
            W_mat[r][c] = W_cols[c][r]
            
    return {"U": U_mat, "V": V_mat, "W": W_mat}

# EVOLVE-BLOCK-END
