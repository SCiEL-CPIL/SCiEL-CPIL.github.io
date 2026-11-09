import itertools

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Total rank calculation:
    # 6 Strassen blocks * 7 = 42
    # 1 Trivial 2x1x6 block * 12 = 12
    # Total = 54
    rank = 54
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Strassen 2x2x2 decomposition matrices
    # U rows: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # V rows: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # W rows: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    S_U = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    S_V = [
        [1, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, -1, 1, 0, 1]
    ]
    S_W = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    
    def add_strassen(i_start, j_start, k_start, col_offset):
        # Map local indices to global
        # Local U row (li, lj) -> Global U row (i_start+li, j_start+lj)
        # Local V row (lj, lk) -> Global V row (j_start+lj, k_start+lk)
        # Local W row (lk, li) -> Global W row (k_start+lk, i_start+li)
        
        for r in range(7):
            for li in range(2):
                for lj in range(2):
                    u_row_local = li * 2 + lj
                    u_row_global = (i_start + li) * m + (j_start + lj)
                    U[u_row_global][col_offset + r] += S_U[u_row_local][r]
                    
            for lj in range(2):
                for lk in range(2):
                    v_row_local = lj * 2 + lk
                    v_row_global = (j_start + lj) * p + (k_start + lk)
                    V[v_row_global][col_offset + r] += S_V[v_row_local][r]
                    
            for lk in range(2):
                for li in range(2):
                    w_row_local = lk * 2 + li
                    w_row_global = (k_start + lk) * n + (i_start + li)
                    W[w_row_global][col_offset + r] += S_W[w_row_local][r]
                    
    def add_trivial_2_1_6(i_start, j_start, k_start, col_offset):
        # n=2, m=1, p=6
        # i in 0..1, j in 0 (local), k in 0..5
        # Global j is j_start (which is 4)
        # Global k is k_start + lk
        
        col = col_offset
        for li in range(2):
            for lk in range(6):
                u_row_global = (i_start + li) * m + j_start
                v_row_global = j_start * p + (k_start + lk)
                w_row_global = (k_start + lk) * n + (i_start + li)
                
                U[u_row_global][col] = 1.0
                V[v_row_global][col] = 1.0
                W[w_row_global][col] = 1.0
                col += 1
        return col - col_offset # returns 12
    
    col = 0
    
    # 1. Sub1a1a: 2x2x2 at i=0, j=0, k=0
    add_strassen(0, 0, 0, col)
    col += 7
    
    # 2. Sub1a1b: 2x2x2 at i=0, j=0, k=2
    add_strassen(0, 0, 2, col)
    col += 7
    
    # 3. Sub1a2a: 2x2x2 at i=0, j=2, k=0
    add_strassen(0, 2, 0, col)
    col += 7
    
    # 4. Sub1a2b: 2x2x2 at i=0, j=2, k=2
    add_strassen(0, 2, 2, col)
    col += 7
    
    # 5. Sub1b1: 2x2x2 at i=0, j=0, k=4
    add_strassen(0, 0, 4, col)
    col += 7
    
    # 6. Sub1b2: 2x2x2 at i=0, j=2, k=4
    add_strassen(0, 2, 4, col)
    col += 7
    
    # 7. Sub2: 2x1x6 at i=0, j=4, k=0
    # Note: i_start is 0 because n=2, i is always 0..1 in global
    # j_start is 4
    # k_start is 0
    add_trivial_2_1_6(0, 4, 0, col)
    col += 12
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
