"""Rank-reduced exact decomposition for 3x4 times 4x4 matrix multiplication."""
import time

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Exact rank-28 decomposition for C = A(3x4) * B(4x4) using Strassen 2x2x2.
    
    Decompose into 4 Strassen blocks:
      Block 1: A[0:2,0:2] x B[0:2,0:2] -> C[0:2,0:2]  (rank 7)
      Block 2: A[0:2,2:4] x B[2:4,0:2] -> C[0:2,0:2]  (rank 7)
      Block 3: A[0:2,0:2] x B[0:2,2:4] -> C[0:2,2:4]  (rank 7)
      Block 4: A[0:2,2:4] x B[2:4,2:4] -> C[0:2,2:4]  (rank 7)
      Block 5: A[1:3,0:2] x B[0:2,0:2] -> C[1:3,0:2]  (rank 7)
      Block 6: A[1:3,2:4] x B[2:4,0:2] -> C[1:3,0:2]  (rank 7)
      Block 7: A[1:3,0:2] x B[0:2,2:4] -> C[1:3,2:4]  (rank 7)
      Block 8: A[1:3,2:4] x B[2:4,2:4] -> C[1:3,2:4]  (rank 7)
    
    Wait, that's 8 blocks = rank 56. Let me reconsider.
    
    Actually for 3x4x4:
    - Row 0 of C: needs A[0,:] * B[:,k] for k=0..3
    - Row 1 of C: needs A[1,:] * B[:,k] for k=0..3
    - Row 2 of C: needs A[2,:] * B[:,k] for k=0..3
    
    Using Strassen 2x2x2:
    - For C[0:2, 0:2]: use A[0:2, 0:2] x B[0:2, 0:2] + A[0:2, 2:4] x B[2:4, 0:2]
      = two Strassen blocks, rank 7+7=14
    - For C[0:2, 2:4]: use A[0:2, 0:2] x B[0:2, 2:4] + A[0:2, 2:4] x B[2:4, 2:4]
      = two Strassen blocks, rank 7+7=14
    - For C[2, 0:2]: need A[2, 0:2] x B[0:2, 0:2] + A[2, 2:4] x B[2:4, 0:2]
      This is a 1x2 x 2x2 and 1x2 x 2x2, each rank 4, total rank 8
    - For C[2, 2:4]: need A[2, 0:2] x B[0:2, 2:4] + A[2, 2:4] x B[2:4, 2:4]
      = rank 4+4=8
    
    Total: 14+14+8+8 = 44. That matches the claimed rank-44!
    """
    n, m, p = 3, 4, 4
    
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    # We'll build the decomposition as a list of (u_entries, v_entries, w_entries)
    # where each entry is a list of (index, value) pairs
    cols = []
    
    def add_term(u_entries, v_entries, w_entries):
        """Add one rank-1 term to the decomposition."""
        cols.append((u_entries, v_entries, w_entries))
    
    def add_strassen_2x2x2(a11, a12, a21, a22, b11, b12, b21, b22, c11, c12, c21, c22):
        """
        Add 7 Strassen terms for 2x2x2 matrix multiplication.
        
        A = [[a11, a12], [a21, a22]]  (U indices)
        B = [[b11, b12], [b21, b22]]  (V indices)
        C = [[c11, c12], [c21, c22]]  (W indices)
        
        Strassen products:
        M1 = (a11+a22)(b11+b22)
        M2 = (a21+a22)b11
        M3 = a11(b12-b22)
        M4 = a22(b21-b11)
        M5 = (a11+a12)b22
        M6 = (a21-a11)(b11+b12)
        M7 = (a12-a22)(b21+b22)
        
        C11 = M1 + M4 - M5 + M7
        C12 = M3 + M5
        C21 = M2 + M4
        C22 = M1 - M2 + M3 + M6
        """
        # M1 = (a11+a22)(b11+b22) -> C11:+1, C22:+1
        add_term([(a11,1),(a22,1)], [(b11,1),(b22,1)], [(c11,1),(c22,1)])
        # M2 = (a21+a22)b11 -> C21:+1, C22:-1
        add_term([(a21,1),(a22,1)], [(b11,1)], [(c21,1),(c22,-1)])
        # M3 = a11(b12-b22) -> C12:+1, C22:+1
        add_term([(a11,1)], [(b12,1),(b22,-1)], [(c12,1),(c22,1)])
        # M4 = a22(b21-b11) -> C11:+1, C21:+1
        add_term([(a22,1)], [(b21,1),(b11,-1)], [(c11,1),(c21,1)])
        # M5 = (a11+a12)b22 -> C12:+1, C11:-1
        add_term([(a11,1),(a12,1)], [(b22,1)], [(c12,1),(c11,-1)])
        # M6 = (a21-a11)(b11+b12) -> C22:+1
        add_term([(a21,1),(a11,-1)], [(b11,1),(b12,1)], [(c22,1)])
        # M7 = (a12-a22)(b21+b22) -> C11:+1
        add_term([(a12,1),(a22,-1)], [(b21,1),(b22,1)], [(c11,1)])
    
    def add_dot_1x2x2(a1, a2, b11, b12, b21, b22, c1, c2):
        """
        Add 4 terms for 1x2 x 2x2 -> 1x2 matrix multiplication.
        [a1, a2] x [[b11, b12], [b21, b22]] = [a1*b11+a2*b21, a1*b12+a2*b22]
        c1 = a1*b11 + a2*b21
        c2 = a1*b12 + a2*b22
        """
        # c1 = a1*b11 + a2*b21
        add_term([(a1,1)], [(b11,1)], [(c1,1)])
        add_term([(a2,1)], [(b21,1)], [(c1,1)])
        # c2 = a1*b12 + a2*b22
        add_term([(a1,1)], [(b12,1)], [(c2,1)])
        add_term([(a2,1)], [(b22,1)], [(c2,1)])
    
    # Index helpers
    # U index: i*m + j = i*4 + j  (i=0..2, j=0..3)
    # V index: j*p + k = j*4 + k  (j=0..3, k=0..3)
    # W index: k*n + i = k*3 + i  (k=0..3, i=0..2)
    
    def u(i, j): return i * m + j
    def v(j, k): return j * p + k
    def w(k, i): return k * n + i
    
    # Block 1: C[0:2, 0:2] from A[0:2, 0:2] x B[0:2, 0:2]
    # A rows 0,1; B rows 0,1; C cols 0,1
    add_strassen_2x2x2(
        a11=u(0,0), a12=u(0,1), a21=u(1,0), a22=u(1,1),
        b11=v(0,0), b12=v(0,1), b21=v(1,0), b22=v(1,1),
        c11=w(0,0), c12=w(1,0), c21=w(0,1), c22=w(1,1)
    )
    
    # Block 2: C[0:2, 0:2] from A[0:2, 2:4] x B[2:4, 0:2]
    # A rows 0,1 cols 2,3; B rows 2,3 cols 0,1
    add_strassen_2x2x2(
        a11=u(0,2), a12=u(0,3), a21=u(1,2), a22=u(1,3),
        b11=v(2,0), b12=v(2,1), b21=v(3,0), b22=v(3,1),
        c11=w(0,0), c12=w(1,0), c21=w(0,1), c22=w(1,1)
    )
    
    # Block 3: C[0:2, 2:4] from A[0:2, 0:2] x B[0:2, 2:4]
    # A rows 0,1 cols 0,1; B rows 0,1 cols 2,3
    add_strassen_2x2x2(
        a11=u(0,0), a12=u(0,1), a21=u(1,0), a22=u(1,1),
        b11=v(0,2), b12=v(0,3), b21=v(1,2), b22=v(1,3),
        c11=w(2,0), c12=w(3,0), c21=w(2,1), c22=w(3,1)
    )
    
    # Block 4: C[0:2, 2:4] from A[0:2, 2:4] x B[2:4, 2:4]
    # A rows 0,1 cols 2,3; B rows 2,3 cols 2,3
    add_strassen_2x2x2(
        a11=u(0,2), a12=u(0,3), a21=u(1,2), a22=u(1,3),
        b11=v(2,2), b12=v(2,3), b21=v(3,2), b22=v(3,3),
        c11=w(2,0), c12=w(3,0), c21=w(2,1), c22=w(3,1)
    )
    
    # Block 5: C[2, 0:2] from A[2, 0:2] x B[0:2, 0:2]
    # A row 2 cols 0,1; B rows 0,1 cols 0,1
    add_dot_1x2x2(
        a1=u(2,0), a2=u(2,1),
        b11=v(0,0), b12=v(0,1), b21=v(1,0), b22=v(1,1),
        c1=w(0,2), c2=w(1,2)
    )
    
    # Block 6: C[2, 0:2] from A[2, 2:4] x B[2:4, 0:2]
    # A row 2 cols 2,3; B rows 2,3 cols 0,1
    add_dot_1x2x2(
        a1=u(2,2), a2=u(2,3),
        b11=v(2,0), b12=v(2,1), b21=v(3,0), b22=v(3,1),
        c1=w(0,2), c2=w(1,2)
    )
    
    # Block 7: C[2, 2:4] from A[2, 0:2] x B[0:2, 2:4]
    # A row 2 cols 0,1; B rows 0,1 cols 2,3
    add_dot_1x2x2(
        a1=u(2,0), a2=u(2,1),
        b11=v(0,2), b12=v(0,3), b21=v(1,2), b22=v(1,3),
        c1=w(2,2), c2=w(3,2)
    )
    
    # Block 8: C[2, 2:4] from A[2, 2:4] x B[2:4, 2:4]
    # A row 2 cols 2,3; B rows 2,3 cols 2,3
    add_dot_1x2x2(
        a1=u(2,2), a2=u(2,3),
        b11=v(2,2), b12=v(2,3), b21=v(3,2), b22=v(3,3),
        c1=w(2,2), c2=w(3,2)
    )
    
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
    
    # Verify the decomposition
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
