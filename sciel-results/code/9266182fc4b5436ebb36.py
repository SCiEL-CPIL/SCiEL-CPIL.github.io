"""Rank-reduced exact decomposition for 3x4 times 4x4 matrix multiplication.

Uses Strassen-7 for 2x2x2 blocks and schoolbook for the remaining row.
Total rank: 4*7 + 4*4 = 44. Verified correct.
"""
import time
import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Exact decomposition for C = A(3x4) * B(4x4).
    
    Uses Strassen-7 for four 2x2x2 sub-blocks covering the top 2 rows,
    and schoolbook (rank 16) for the bottom row.
    Total rank: 4*7 + 16 = 44.
    """
    t0 = time.time()
    deadline = t0 + 100  # safety margin before 120s limit
    
    result = _construct_rank44()
    
    # Verify
    U, V, W = result["U"], result["V"], result["W"]
    n, m, p = 3, 4, 4
    rank = len(U[0])
    
    def verify(U_mat, V_mat, W_mat, rank_val):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank_val):
                        val += U_mat[i * m + j][r] * V_mat[j * p + k][r] * W_mat[k * n + i][r]
                    if abs(val - 1.0) > 1e-10:
                        return False, (i, j, k, val)
        return True, None
    
    is_valid, err = verify(U, V, W, rank)
    
    if not is_valid:
        # Should not happen, but return schoolbook rank-48 as last resort
        return _construct_rank48()
    
    return result


def _construct_rank44():
    """Rank-44 decomposition using Strassen-7 for 2x2x2 blocks and schoolbook for row 2."""
    n, m, p = 3, 4, 4
    
    cols = []
    
    def add_term(u_entries, v_entries, w_entries):
        cols.append((u_entries, v_entries, w_entries))
    
    def u(i, j): return i * m + j
    def v(j, k): return j * p + k
    def w(k, i): return k * n + i
    
    def add_strassen_2x2x2(a11, a12, a21, a22, b11, b12, b21, b22, c11, c12, c21, c22):
        # Strassen 7-term decomposition for 2x2x2 matrix multiplication
        # c11 = a11*b11 + a21*b21  ... standard Strassen
        M1 = (a11,1),(a22,1)  # (a11+a22)
        M2 = (a21,1),(a22,1)  # (a21+a22)
        M3 = (a11,1)          # a11
        M4 = (a22,1)          # a22
        M5 = (a11,1),(a12,1)  # (a11+a12)
        M6 = (a21,1),(a11,-1) # (a21-a11)
        M7 = (a12,1),(a22,-1) # (a12-a22)
        
        # M1*(b11+b22) -> c11+=, c22+=
        add_term([(a11,1),(a22,1)], [(b11,1),(b22,1)], [(c11,1),(c22,1)])
        # M2*b11 -> c21+=, c22-=
        add_term([(a21,1),(a22,1)], [(b11,1)], [(c21,1),(c22,-1)])
        # M3*(b12-b22) -> c12+=, c22+=
        add_term([(a11,1)], [(b12,1),(b22,-1)], [(c12,1),(c22,1)])
        # M4*(b21-b11) -> c11+=, c21+=
        add_term([(a22,1)], [(b21,1),(b11,-1)], [(c11,1),(c21,1)])
        # M5*b22 -> c12+=, c11-=
        add_term([(a11,1),(a12,1)], [(b22,1)], [(c12,1),(c11,-1)])
        # M6*(b11+b12) -> c22+=
        add_term([(a21,1),(a11,-1)], [(b11,1),(b12,1)], [(c22,1)])
        # M7*(b21+b22) -> c11+=
        add_term([(a12,1),(a22,-1)], [(b21,1),(b22,1)], [(c11,1)])
    
    # Block 1: C[0:2,0:2] += A[0:2,0:2]*B[0:2,0:2]
    add_strassen_2x2x2(
        u(0,0), u(0,1), u(1,0), u(1,1),
        v(0,0), v(0,1), v(1,0), v(1,1),
        w(0,0), w(1,0), w(0,1), w(1,1)
    )
    
    # Block 2: C[0:2,0:2] += A[0:2,2:4]*B[2:4,0:2]
    add_strassen_2x2x2(
        u(0,2), u(0,3), u(1,2), u(1,3),
        v(2,0), v(2,1), v(3,0), v(3,1),
        w(0,0), w(1,0), w(0,1), w(1,1)
    )
    
    # Block 3: C[0:2,2:4] += A[0:2,0:2]*B[0:2,2:4]
    add_strassen_2x2x2(
        u(0,0), u(0,1), u(1,0), u(1,1),
        v(0,2), v(0,3), v(1,2), v(1,3),
        w(2,0), w(3,0), w(2,1), w(3,1)
    )
    
    # Block 4: C[0:2,2:4] += A[0:2,2:4]*B[2:4,2:4]
    add_strassen_2x2x2(
        u(0,2), u(0,3), u(1,2), u(1,3),
        v(2,2), v(2,3), v(3,2), v(3,3),
        w(2,0), w(3,0), w(2,1), w(3,1)
    )
    
    # Schoolbook for row 2: C[2,k] = sum_j A[2,j]*B[j,k]
    for j in range(4):
        for k in range(4):
            add_term([(u(2,j),1)], [(v(j,k),1)], [(w(k,2),1)])
    
    rank = len(cols)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, (u_entries, v_entries, w_entries) in enumerate(cols):
        for idx, val in u_entries:
            U[idx][r] = val
        for idx, val in v_entries:
            V[idx][r] = val
        for idx, val in w_entries:
            W[idx][r] = val
    
    return {"U": U, "V": V, "W": W}


def _construct_rank48():
    """Last-resort schoolbook rank-48 decomposition."""
    n, m, p = 3, 4, 4
    cols = []
    
    def u(i, j): return i * m + j
    def v(j, k): return j * p + k
    def w(k, i): return k * n + i
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                cols.append(([(u(i,j),1)], [(v(j,k),1)], [(w(k,i),1)]))
    
    rank = len(cols)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, (u_entries, v_entries, w_entries) in enumerate(cols):
        for idx, val in u_entries:
            U[idx][r] = val
        for idx, val in v_entries:
            V[idx][r] = val
        for idx, val in w_entries:
            W[idx][r] = val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
