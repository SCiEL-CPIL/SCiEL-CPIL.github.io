"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=7
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    U_rows = []
    V_rows = []
    W_rows = []
    
    def add_term(u, v, w):
        """u: dict mapping (i,j) -> coeff, v: dict mapping (j,k) -> coeff, w: dict mapping (k,i) -> coeff"""
        U_row = [0.0] * 0  # placeholder
        # We'll build full rows at the end
        U_rows.append(u)
        V_rows.append(v)
        W_rows.append(w)
    
    # Part 1: j=0 (A[:,0] * B[0,:])
    # 14 terms: for each i in {0,1}, k in {0..6}
    for i in range(n):
        for k in range(p):
            u = {(i, 0): 1.0}
            v = {(0, k): 1.0}
            w = {(k, i): 1.0}
            add_term(u, v, w)
    
    # Part 2: j={1,2} (A[:,1:3] * B[1:3,:])
    # Split p=7 columns into [0,1], [2,3], [4,5], [6]
    col_groups = [(0,1), (2,3), (4,5), (6,)]
    
    for group in col_groups:
        if len(group) == 2:
            c0, c1 = group
            # Strassen 2x2x2x2
            # A1 indices: (0,1)->A[0,1], (0,2)->A[0,2], (1,1)->A[1,1], (1,2)->A[1,2]
            # B1 indices: (1,c0)->B[1,c0], (1,c1)->B[1,c1], (2,c0)->B[2,c0], (2,c1)->B[2,c1]
            # C indices: (c0,0)->C[0,c0], (c1,0)->C[0,c1], (c0,1)->C[1,c0], (c1,1)->C[1,c1]
            
            # M1 = (A00+A11)(B00+B11)
            add_term(
                {(0,1): 1.0, (1,2): 1.0},
                {(1,c0): 1.0, (2,c1): 1.0},
                {(c0,0): 1.0, (c1,1): 1.0}
            )
            # M2 = (A10+A11)B00
            add_term(
                {(1,1): 1.0, (1,2): 1.0},
                {(1,c0): 1.0},
                {(c0,1): 1.0, (c1,1): -1.0}
            )
            # M3 = A00(B01-B11)
            add_term(
                {(0,1): 1.0},
                {(1,c1): 1.0, (2,c1): -1.0},
                {(c1,0): 1.0, (c1,1): 1.0}
            )
            # M4 = A11(B10-B00)
            add_term(
                {(1,2): 1.0},
                {(2,c0): 1.0, (1,c0): -1.0},
                {(c0,0): 1.0, (c0,1): 1.0}
            )
            # M5 = (A00+A01)B11
            add_term(
                {(0,1): 1.0, (0,2): 1.0},
                {(2,c1): 1.0},
                {(c1,0): 1.0, (c1,1): -1.0}
            )
            # M6 = (A10-A00)(B00+B01)
            add_term(
                {(1,1): 1.0, (0,1): -1.0},
                {(1,c0): 1.0, (1,c1): 1.0},
                {(c1,1): 1.0}
            )
            # M7 = (A01-A11)(B10+B11)
            add_term(
                {(0,2): 1.0, (1,2): -1.0},
                {(2,c0): 1.0, (2,c1): 1.0},
                {(c0,0): 1.0}
            )
        elif len(group) == 1:
            c0 = group[0]
            # Schoolbook 2x2x2x1: 4 terms
            for i in range(n):
                for jj in range(2):  # jj=0,1 corresponds to j=1,2
                    add_term(
                        {(i, 1+jj): 1.0},
                        {(1+jj, c0): 1.0},
                        {(c0, i): 1.0}
                    )
    
    # Convert to matrix form
    rank = len(U_rows)
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    for c in range(rank):
        for (i, j), val in U_rows[c].items():
            U[i * m + j][c] = val
        for (j, k), val in V_rows[c].items():
            V[j * p + k][c] = val
        for (k, i), val in W_rows[c].items():
            W[k * n + i][c] = val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
