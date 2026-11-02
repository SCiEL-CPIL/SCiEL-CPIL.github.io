"""Matrix multiplication 3x4x6 tensor decomposition using Strassen on 2x2x2 sub-blocks."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    n_singles = [0]
    n_pairs = [(1, 2)]
    m_pairs = [(0, 1), (2, 3)]
    p_pairs = [(0, 1), (2, 3), (4, 5)]
    
    U_rows = [[] for _ in range(n * m)]
    V_rows = [[] for _ in range(m * p)]
    W_rows = [[] for _ in range(p * n)]
    
    def add_rank1(u_map, v_map, w_map):
        for r in range(n * m):
            U_rows[r].append(u_map.get(r, 0))
        for r in range(m * p):
            V_rows[r].append(v_map.get(r, 0))
        for r in range(p * n):
            W_rows[r].append(w_map.get(r, 0))
    
    # Schoolbook: n_single x m_pair x p_pair blocks (4 terms each)
    for i in n_singles:
        for (j1, j2) in m_pairs:
            for (k1, k2) in p_pairs:
                for j in (j1, j2):
                    for k in (k1, k2):
                        add_rank1({i*m+j: 1}, {j*p+k: 1}, {k*n+i: 1})
    
    # Strassen: n_pair x m_pair x p_pair blocks (7 terms each)
    for (i1, i2) in n_pairs:
        for (j1, j2) in m_pairs:
            for (k1, k2) in p_pairs:
                add_rank1({i1*m+j1: 1, i2*m+j2: 1},
                          {j1*p+k1: 1, j2*p+k2: 1},
                          {k1*n+i1: 1, k2*n+i2: 1})
                add_rank1({i2*m+j1: 1, i2*m+j2: 1},
                          {j1*p+k1: 1},
                          {k1*n+i2: 1, k2*n+i2: -1})
                add_rank1({i1*m+j1: 1},
                          {j1*p+k2: 1, j2*p+k2: -1},
                          {k2*n+i1: 1})
                add_rank1({i2*m+j2: 1},
                          {j2*p+k1: 1, j1*p+k1: -1},
                          {k1*n+i1: 1, k1*n+i2: 1})
                add_rank1({i1*m+j1: 1, i1*m+j2: 1},
                          {j2*p+k2: 1},
                          {k1*n+i1: -1, k2*n+i1: 1})
                add_rank1({i2*m+j1: 1, i1*m+j1: -1},
                          {j1*p+k1: 1, j1*p+k2: 1},
                          {k2*n+i2: 1})
                add_rank1({i1*m+j2: 1, i2*m+j2: -1},
                          {j2*p+k1: 1, j2*p+k2: 1},
                          {k1*n+i1: 1})
    
    return {"U": U_rows, "V": V_rows, "W": W_rows}
# EVOLVE-BLOCK-END
