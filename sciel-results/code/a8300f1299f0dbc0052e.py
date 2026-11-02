import time
import itertools
from fractions import Fraction
import numpy as np

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def _build_target(n, m, p):
    """Return the matrix-multiplication tensor as a dict of 1s (sparse)."""
    tgt = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                tgt[(i * m + j, j * p + k, k * n + i)] = 1
    return tgt

def _verify(U, V, W, tgt, n, m, p):
    """Exact check that sum_r U[a,r]V[b,r]W[c,r] reproduces the target."""
    R = len(U[0])
    def col_dict(mat, r):
        d = {}
        for a in range(len(mat)):
            v = mat[a][r]
            if v != 0:
                d[a] = v
        return d
    cols = []
    for r in range(R):
        cols.append((col_dict(U, r), col_dict(V, r), col_dict(W, r)))
    acc = {}
    for cu, cv, cw in cols:
        for a, ua in cu.items():
            for b, vb in cv.items():
                uv = ua * vb
                for c, wc in cw.items():
                    acc[(a, b, c)] = acc.get((a, b, c), 0) + uv * wc
    for key, val in tgt.items():
        if acc.get(key, 0) != val:
            return False
    for key, val in acc.items():
        if val != 0 and key not in tgt:
            return False
    return True

def _strassen_2x2x2():
    """
    Returns U, V, W for 2x2 matrix multiplication.
    A = [[a11, a12], [a21, a22]]
    B = [[b11, b12], [b21, b22]]
    C = [[c11, c12], [c21, c22]]
    
    Indices:
    A: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    B: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    C: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    Strassen:
    M1 = (a11+a22)(b11+b22)
    M2 = (a21+a22)b11
    M3 = a11(b12-b22)
    M4 = a22(b21-b11)
    M5 = (a11+a12)b22
    M6 = (a21-a11)(b11+b12)
    M7 = (a12-a22)(b21+b22)
    
    c11 = M1 + M4 - M5 + M7
    c12 = M3 + M5
    c21 = M2 + M4
    c22 = M1 - M2 + M3 + M6
    """
    rank = 7
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    
    # M1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = -1
    
    # M2
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = 1
    
    # M3
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[0][2] = 1; W[1][2] = 1
    
    # M4
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1; W[3][6] = 1
    
    return U, V, W

def _schoolbook_1x2x2x2():
    """
    Returns U, V, W for 1x2 by 2x2 multiplication.
    A = [a1, a2] (1x2)
    B = [[b11, b12], [b21, b22]] (2x2)
    C = [c1, c2] (1x2)
    
    c1 = a1*b11 + a2*b21
    c2 = a1*b12 + a2*b22
    
    Indices:
    A: 0, 1
    B: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    C: 0, 1
    """
    rank = 4
    U = [[0]*rank for _ in range(2)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(2)]
    
    # c1 = a1*b11 + a2*b21
    U[0][0] = 1; V[0][0] = 1; W[0][0] = 1
    U[1][1] = 1; V[2][1] = 1; W[0][1] = 1
    
    # c2 = a1*b12 + a2*b22
    U[0][2] = 1; V[1][2] = 1; W[1][2] = 1
    U[1][3] = 1; V[3][3] = 1; W[1][3] = 1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 115

    # Block splits
    # n=3 -> [1, 2]
    # m=4 -> [2, 2]
    # p=6 -> [2, 2, 2]
    
    n_splits = [1, 2]
    m_splits = [2, 2]
    p_splits = [2, 2, 2]
    
    n_offsets = [0, 1]
    m_offsets = [0, 2]
    p_offsets = [0, 2, 4]
    
    # Global dimensions
    U_dim = n * m # 12
    V_dim = m * p # 24
    W_dim = p * n # 18
    
    # We will build the factors by accumulating contributions
    # Each sub-problem contributes a set of columns to U, V, W.
    
    total_rank = 0
    U_list = []
    V_list = []
    W_list = []
    
    # Iterate over blocks
    # i: block row index of A (0..1)
    # k: block col index of A / block row index of B (0..1)
    # j: block col index of B (0..2)
    
    for i in range(2):
        for k in range(2):
            for j in range(3):
                # Determine dimensions of this sub-problem
                # A_block: n_splits[i] x m_splits[k]
                # B_block: m_splits[k] x p_splits[j]
                # C_block: n_splits[i] x p_splits[j]
                
                ni = n_splits[i]
                mk = m_splits[k]
                pj = p_splits[j]
                
                # Get local factors
                if ni == 2 and mk == 2 and pj == 2:
                    u_loc, v_loc, w_loc = _strassen_2x2x2()
                elif ni == 1 and mk == 2 and pj == 2:
                    u_loc, v_loc, w_loc = _schoolbook_1x2x2x2()
                else:
                    # Fallback to schoolbook if other sizes appear (should not)
                    r = ni * mk * pj
                    u_loc = [[0]*r for _ in range(ni*mk)]
                    v_loc = [[0]*r for _ in range(mk*pj)]
                    w_loc = [[0]*r for _ in range(pj*ni)]
                    col = 0
                    for a in range(ni*mk):
                        for b in range(mk*pj):
                            for c in range(pj*ni):
                                # Check if this is a valid tensor entry
                                # Local indices:
                                # a = ai * mk + ak
                                # b = bk * pj + bj
                                # c = bj * ni + bi
                                # Valid if ak == bk
                                ai, ak = divmod(a, mk)
                                bk, bj = divmod(b, pj)
                                bj_c, bi = divmod(c, ni)
                                if ak == bk and bj == bj_c:
                                    u_loc[a][col] = 1
                                    v_loc[b][col] = 1
                                    w_loc[c][col] = 1
                                    col += 1
                    # Trim trailing zeros if col < r
                    if col < r:
                        u_loc = [row[:col] for row in u_loc]
                        v_loc = [row[:col] for row in v_loc]
                        w_loc = [row[:col] for row in w_loc]
                
                rank_loc = len(u_loc[0])
                
                # Map local indices to global indices
                # A global index: (global_i, global_k) -> global_i * m + global_k
                # global_i = n_offsets[i] + local_i
                # global_k = m_offsets[k] + local_k
                # local_i in 0..ni-1, local_k in 0..mk-1
                # local A index = local_i * mk + local_k
                
                # B global index: (global_k, global_j) -> global_k * p + global_j
                # global_k = m_offsets[k] + local_k
                # global_j = p_offsets[j] + local_j
                # local B index = local_k * pj + local_j
                
                # C global index: (global_j, global_i) -> global_j * n + global_i
                # global_j = p_offsets[j] + local_j
                # global_i = n_offsets[i] + local_i
                # local C index = local_j * ni + local_i
                
                # Create global slices for this sub-problem
                # We need to append these columns to the global U, V, W
                # But we must place the non-zeros at the correct global rows.
                
                U_block = [[0] * rank_loc for _ in range(U_dim)]
                V_block = [[0] * rank_loc for _ in range(V_dim)]
                W_block = [[0] * rank_loc for _ in range(W_dim)]
                
                # Fill U_block
                for r in range(rank_loc):
                    for local_a in range(len(u_loc)):
                        val = u_loc[local_a][r]
                        if val != 0:
                            li, lk = divmod(local_a, mk)
                            gi = n_offsets[i] + li
                            gk = m_offsets[k] + lk
                            global_a = gi * m + gk
                            U_block[global_a][r] = val
                            
                # Fill V_block
                for r in range(rank_loc):
                    for local_b in range(len(v_loc)):
                        val = v_loc[local_b][r]
                        if val != 0:
                            lk, lj = divmod(local_b, pj)
                            gk = m_offsets[k] + lk
                            gj = p_offsets[j] + lj
                            global_b = gk * p + gj
                            V_block[global_b][r] = val
                            
                # Fill W_block
                for r in range(rank_loc):
                    for local_c in range(len(w_loc)):
                        val = w_loc[local_c][r]
                        if val != 0:
                            lj, li = divmod(local_c, ni)
                            gj = p_offsets[j] + lj
                            gi = n_offsets[i] + li
                            global_c = gj * n + gi
                            W_block[global_c][r] = val
                            
                # Append to lists
                for row in range(U_dim):
                    U_list.append([0] * total_rank + U_block[row])
                for row in range(V_dim):
                    V_list.append([0] * total_rank + V_block[row])
                for row in range(W_dim):
                    W_list.append([0] * total_rank + W_block[row])
                    
                total_rank += rank_loc
                
    # Construct final matrices
    U = [row[:total_rank] for row in U_list]
    V = [row[:total_rank] for row in V_list]
    W = [row[:total_rank] for row in W_list]
    
    # Verify
    tgt = _build_target(n, m, p)
    if _verify(U, V, W, tgt, n, m, p):
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook if verification fails (should not happen)
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
