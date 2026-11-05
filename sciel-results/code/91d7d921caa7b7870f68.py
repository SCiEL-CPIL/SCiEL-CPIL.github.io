"""Rank-reduced exact decomposition for 3x4 times 4x4 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Exact decomposition for C = A(3x4) * B(4x4).
    
    Strategy:
    - Use Strassen's 7-term algorithm for the top-left 2x2x2 block (A[0:2,0:2]*B[0:2,0:2] -> C[0:2,0:2])
    - Use schoolbook for all remaining terms
    - This gives rank 7 + (48-8) = 47... wait, let's count properly.
    
    Full schoolbook is 3*4*4 = 48 terms.
    The 2x2x2 block A[0:2,0:2]*B[0:2,0:2] contributes to C[0:2,0:2].
    In schoolbook, this is 2*2*2 = 8 terms.
    Strassen replaces these 8 terms with 7 terms.
    So total rank = 48 - 8 + 7 = 47.
    
    But wait, I need to be careful. The Strassen terms for A[0:2,0:2]*B[0:2,0:2] 
    only contribute to C[0:2,0:2]. They don't interfere with other C entries.
    So yes, rank 47.
    
    Can I do better? What if I use Strassen for multiple 2x2x2 blocks?
    The 4x4 B matrix has four 2x2 blocks: B[0:2,0:2], B[0:2,2:4], B[2:4,0:2], B[2:4,2:4].
    The 3x4 A matrix has: A[0:2,0:2], A[0:2,2:4], A[2,0:2], A[2,2:4].
    
    For C[0:2,0:2] = A[0:2,0:2]*B[0:2,0:2] + A[0:2,2:4]*B[2:4,0:2]
    The first term is 2x2x2, the second is 2x2x2 (A[0:2,2:4] is 2x2, B[2:4,0:2] is 2x2).
    So C[0:2,0:2] involves two 2x2x2 multiplications.
    Schoolbook: 8+8=16 terms.
    Strassen: 7+7=14 terms.
    Saving: 2.
    
    For C[0:2,2:4] = A[0:2,0:2]*B[0:2,2:4] + A[0:2,2:4]*B[2:4,2:4]
    Similarly: 16 schoolbook -> 14 Strassen. Saving: 2.
    
    For C[2,0:2] = A[2,0:2]*B[0:2,0:2] + A[2,2:4]*B[2:4,0:2]
    A[2,0:2] is 1x2, B[0:2,0:2] is 2x2. This is 1x2x2, rank 4 schoolbook.
    A[2,2:4] is 1x2, B[2:4,0:2] is 2x2. This is 1x2x2, rank 4 schoolbook.
    Total: 8 terms. Can we do better? 1x2x2 has rank 4 (it's just 4 scalar multiplications).
    No Strassen benefit here since one dimension is 1.
    
    For C[2,2:4] = A[2,0:2]*B[0:2,2:4] + A[2,2:4]*B[2:4,2:4]
    Similarly: 8 terms, no improvement.
    
    Total with Strassen on the 2x2x2 blocks:
    C[0:2,0:2]: 14
    C[0:2,2:4]: 14
    C[2,0:2]: 8
    C[2,2:4]: 8
    Total: 44.
    
    That's rank 44! Better than 47.
    """
    n, m, p = 3, 4, 4
    
    start_time = time.time()
    deadline = start_time + 110
    
    cols = []
    
    def add_term(u_entries, v_entries, w_entries):
        cols.append((u_entries, v_entries, w_entries))
    
    def u(i, j): return i * m + j
    def v(j, k): return j * p + k
    def w(k, i): return k * n + i
    
    def add_strassen_2x2x2(a11, a12, a21, a22, b11, b12, b21, b22, c11, c12, c21, c22):
        """
        Strassen's 7-term algorithm for 2x2x2 matrix multiplication.
        A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]], C = [[c11, c12], [c21, c22]]
        """
        # M1 = (a11+a22)(b11+b22)
        add_term([(a11,1),(a22,1)], [(b11,1),(b22,1)], [(c11,1),(c22,1)])
        
        # M2 = (a21+a22)b11
        # C21 += M2, C22 -= M2
        add_term([(a21,1),(a22,1)], [(b11,1)], [(c21,1),(c22,-1)])
        
        # M3 = a11(b12-b22)
        # C12 += M3, C22 += M3
        add_term([(a11,1)], [(b12,1),(b22,-1)], [(c12,1),(c22,1)])
        
        # M4 = a22(b21-b11)
        # C11 += M4, C21 += M4
        add_term([(a22,1)], [(b21,1),(b11,-1)], [(c11,1),(c21,1)])
        
        # M5 = (a11+a12)b22
        # C12 += M5, C11 -= M5
        add_term([(a11,1),(a12,1)], [(b22,1)], [(c12,1),(c11,-1)])
        
        # M6 = (a21-a11)(b11+b12)
        # C22 += M6
        add_term([(a21,1),(a11,-1)], [(b11,1),(b12,1)], [(c22,1)])
        
        # M7 = (a12-a22)(b21+b22)
        # C11 += M7
        add_term([(a12,1),(a22,-1)], [(b21,1),(b22,1)], [(c11,1)])
    
    # C[0:2,0:2] = A[0:2,0:2]*B[0:2,0:2] + A[0:2,2:4]*B[2:4,0:2]
    # First part: Strassen for A[0:2,0:2]*B[0:2,0:2]
    add_strassen_2x2x2(
        u(0,0), u(0,1), u(1,0), u(1,1),  # A[0:2,0:2]
        v(0,0), v(0,1), v(1,0), v(1,1),  # B[0:2,0:2]
        w(0,0), w(1,0), w(0,1), w(1,1)   # C[0:2,0:2]
    )
    # Second part: Strassen for A[0:2,2:4]*B[2:4,0:2]
    add_strassen_2x2x2(
        u(0,2), u(0,3), u(1,2), u(1,3),  # A[0:2,2:4]
        v(2,0), v(2,1), v(3,0), v(3,1),  # B[2:4,0:2]
        w(0,0), w(1,0), w(0,1), w(1,1)   # C[0:2,0:2]
    )
    
    # C[0:2,2:4] = A[0:2,0:2]*B[0:2,2:4] + A[0:2,2:4]*B[2:4,2:4]
    # First part: Strassen for A[0:2,0:2]*B[0:2,2:4]
    add_strassen_2x2x2(
        u(0,0), u(0,1), u(1,0), u(1,1),  # A[0:2,0:2]
        v(0,2), v(0,3), v(1,2), v(1,3),  # B[0:2,2:4]
        w(2,0), w(3,0), w(2,1), w(3,1)   # C[0:2,2:4]
    )
    # Second part: Strassen for A[0:2,2:4]*B[2:4,2:4]
    add_strassen_2x2x2(
        u(0,2), u(0,3), u(1,2), u(1,3),  # A[0:2,2:4]
        v(2,2), v(2,3), v(3,2), v(3,3),  # B[2:4,2:4]
        w(2,0), w(3,0), w(2,1), w(3,1)   # C[0:2,2:4]
    )
    
    # C[2,0:2] = A[2,0:2]*B[0:2,0:2] + A[2,2:4]*B[2:4,0:2]
    # Schoolbook for 1x2x2 blocks
    for j in range(2):
        for k in range(2):
            add_term([(u(2,j),1)], [(v(j,k),1)], [(w(k,2),1)])
    for j in range(2):
        for k in range(2):
            add_term([(u(2,2+j),1)], [(v(2+j,k),1)], [(w(k,2),1)])
    
    # C[2,2:4] = A[2,0:2]*B[0:2,2:4] + A[2,2:4]*B[2:4,2:4]
    # Schoolbook for 1x2x2 blocks
    for j in range(2):
        for k in range(2):
            add_term([(u(2,j),1)], [(v(j,2+k),1)], [(w(2+k,2),1)])
    for j in range(2):
        for k in range(2):
            add_term([(u(2,2+j),1)], [(v(2+j,2+k),1)], [(w(2+k,2),1)])
    
    rank = len(cols)
    
    # Build U, V, W matrices
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
    
    # Verify
    def verify(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(val - 1.0) > 1e-10:
                        return False, (i, j, k, val)
        return True, None
    
    is_valid, err = verify(U, V, W, rank)
    
    if not is_valid:
        # Fallback to schoolbook rank-48
        R = n * m * p
        U = [[0] * R for _ in range(n * m)]
        V = [[0] * R for _ in range(m * p)]
        W = [[0] * R for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
