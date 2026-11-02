import time

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a rank-1 term
    def add_term(u_entries, v_entries, w_entries):
        nonlocal col
        if col >= rank:
            return
        for idx, val in u_entries.items():
            U[idx][col] += val
        for idx, val in v_entries.items():
            V[idx][col] += val
        for idx, val in w_entries.items():
            W[idx][col] += val
        col += 1

    # Helper to add Strassen decomposition for a 2x2x2 block
    # i0, i1: row indices of A (0..1)
    # j0, j1: col indices of A (0..3)
    # k0, k1: col indices of B (0..4)
    def add_strassen(i0, i1, j0, j1, k0, k1):
        # Map local indices to global indices
        # U index: i*m + j
        # V index: j*p + k
        # W index: k*n + i
        
        # M1
        add_term(
            {i0*m+j0: 1, i1*m+j1: 1},
            {j0*p+k0: 1, j1*p+k1: 1},
            {k0*n+i0: 1, k1*n+i1: 1}
        )
        # M2
        add_term(
            {i1*m+j0: 1, i1*m+j1: 1},
            {j0*p+k0: 1},
            {k0*n+i1: 1, k1*n+i1: -1}
        )
        # M3
        add_term(
            {i0*m+j0: 1},
            {j1*p+k0: 1, j1*p+k1: -1},
            {k0*n+i0: 1}
        )
        # M4
        add_term(
            {i1*m+j1: 1},
            {j0*p+k1: 1, j1*p+k1: 1},
            {k0*n+i0: 1, k0*n+i1: 1}
        )
        # M5
        add_term(
            {i0*m+j0: 1, i0*m+j1: 1},
            {j1*p+k1: 1},
            {k1*n+i0: 1, k0*n+i0: -1}
        )
        # M6
        add_term(
            {i1*m+j0: 1, i0*m+j0: -1},
            {j0*p+k0: 1, j0*p+k1: 1},
            {k1*n+i1: 1}
        )
        # M7
        add_term(
            {i0*m+j1: 1, i1*m+j1: -1},
            {j1*p+k0: 1, j1*p+k1: 1},
            {k0*n+i0: 1}
        )

    # Helper to add Schoolbook decomposition for a 2x2x1 block
    def add_schoolbook_2x2x1(i0, i1, j0, j1, k):
        add_term(
            {i0*m+j0: 1},
            {j0*p+k: 1},
            {k*n+i0: 1}
        )
        add_term(
            {i0*m+j1: 1},
            {j1*p+k: 1},
            {k*n+i0: 1}
        )
        add_term(
            {i1*m+j0: 1},
            {j0*p+k: 1},
            {k*n+i1: 1}
        )
        add_term(
            {i1*m+j1: 1},
            {j1*p+k: 1},
            {k*n+i1: 1}
        )

    # Block 1: i in {0,1}, j in {0,1}, k in {0,1} -> Strassen
    add_strassen(0, 1, 0, 1, 0, 1)
    
    # Block 2: i in {0,1}, j in {0,1}, k in {2,3} -> Strassen
    add_strassen(0, 1, 0, 1, 2, 3)
    
    # Block 3: i in {0,1}, j in {0,1}, k=4 -> Schoolbook
    add_schoolbook_2x2x1(0, 1, 0, 1, 4)
    
    # Block 4: i in {0,1}, j in {2,3}, k in {0,1} -> Strassen
    add_strassen(0, 1, 2, 3, 0, 1)
    
    # Block 5: i in {0,1}, j in {2,3}, k in {2,3} -> Strassen
    add_strassen(0, 1, 2, 3, 2, 3)
    
    # Block 6: i in {0,1}, j in {2,3}, k=4 -> Schoolbook
    add_schoolbook_2x2x1(0, 1, 2, 3, 4)

    return {"U": U, "V": V, "W": W}
