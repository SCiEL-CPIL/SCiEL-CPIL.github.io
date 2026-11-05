import time
import numpy as np

DIMENSIONS = [3, 4, 4]

def construct():
    """
    Exact decomposition for C = A(3x4) * B(4x4) with rank 44.
    Uses Strassen's algorithm for 2x2x2 sub-blocks and schoolbook for 1x2x2 sub-blocks.
    """
    n, m, p = 3, 4, 4
    
    cols = []
    
    def add_term(u_entries, v_entries, w_entries):
        cols.append((u_entries, v_entries, w_entries))
    
    def u(i, j): return i * m + j
    def v(j, k): return j * p + k
    def w(k, i): return k * n + i
    
    def add_strassen_2x2x2(a11, a12, a21, a22, b11, b12, b21, b22, c11, c12, c21, c22):
        # Strassen's 7 multiplications for 2x2x2
        add_term([(a11,1),(a22,1)], [(b11,1),(b22,1)], [(c11,1),(c22,1)])
        add_term([(a21,1),(a22,1)], [(b11,1)], [(c21,1),(c22,-1)])
        add_term([(a11,1)], [(b12,1),(b22,-1)], [(c12,1),(c22,1)])
        add_term([(a22,1)], [(b21,1),(b11,-1)], [(c11,1),(c21,1)])
        add_term([(a11,1),(a12,1)], [(b22,1)], [(c12,1),(c11,-1)])
        add_term([(a21,1),(a11,-1)], [(b11,1),(b12,1)], [(c22,1)])
        add_term([(a12,1),(a22,-1)], [(b21,1),(b22,1)], [(c11,1)])

    # Block 1: C[0:2,0:2] from A[0:2,0:2]*B[0:2,0:2]
    add_strassen_2x2x2(
        u(0,0), u(0,1), u(1,0), u(1,1),
        v(0,0), v(0,1), v(1,0), v(1,1),
        w(0,0), w(1,0), w(0,1), w(1,1)
    )
    
    # Block 2: C[0:2,0:2] from A[0:2,2:4]*B[2:4,0:2]
    add_strassen_2x2x2(
        u(0,2), u(0,3), u(1,2), u(1,3),
        v(2,0), v(2,1), v(3,0), v(3,1),
        w(0,0), w(1,0), w(0,1), w(1,1)
    )
    
    # Block 3: C[0:2,2:4] from A[0:2,0:2]*B[0:2,2:4]
    add_strassen_2x2x2(
        u(0,0), u(0,1), u(1,0), u(1,1),
        v(0,2), v(0,3), v(1,2), v(1,3),
        w(2,0), w(3,0), w(2,1), w(3,1)
    )
    
    # Block 4: C[0:2,2:4] from A[0:2,2:4]*B[2:4,2:4]
    add_strassen_2x2x2(
        u(0,2), u(0,3), u(1,2), u(1,3),
        v(2,2), v(2,3), v(3,2), v(3,3),
        w(2,0), w(3,0), w(2,1), w(3,1)
    )
    
    # Blocks 5-8: schoolbook for row 2
    # C[2,k] = sum_j A[2,j] B[j,k]
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
