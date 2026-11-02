import numpy as np

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    total_rank = 104
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    rank_offset = 0
    
    def add_schoolbook(n, m, p, i_off, j_off, k_off, r_off):
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[(i + i_off) * m + (j + j_off), r_off + r] = 1
                    V[(j + j_off) * p + (k + k_off), r_off + r] = 1
                    W[(k + k_off) * n + (i + i_off), r_off + r] = 1
                    r += 1
        return r

    # Part 1: j=1,2
    # 1x2x7 (i=1)
    rank_offset += add_schoolbook(1, 2, 7, 0, 0, 0, rank_offset)
    
    # 2x2x7 (i=2,3)
    # Strassen for k=1,2
    def add_strassen_222(i_off, j_off, k_off, r_off):
        def add_term(u, v, w, r):
            for (i,j), val in u.items(): U[(i+i_off)*m + (j+j_off), r_off+r] = val
            for (j,k), val in v.items(): V[(j+j_off)*p + (k+k_off), r_off+r] = val
            for (k,i), val in w.items(): W[(k+k_off)*n + (i+i_off), r_off+r] = val
        add_term({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, 0)
        add_term({(1,0):1, (1,1):1}, {(0,0):1}, {(0,1):1}, 1)
        add_term({(0,0):1}, {(1,1):1, (0,1):-1}, {(1,0):1}, 2)
        add_term({(1,1):1}, {(0,0):1, (1,0):-1}, {(0,0):1, (0,1):1}, 3)
        add_term({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (1,0):1}, 4)
        add_term({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}, 5)
        add_term({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1, (1,1):1}, 6)
        return 7
    rank_offset += add_strassen_222(1, 0, 0, rank_offset)
    rank_offset += add_schoolbook(2, 2, 5, 1, 0, 2, rank_offset)
    
    # Part 2: j=3,4,5
    # 3x3x7
    rank_offset += add_schoolbook(3, 3, 7, 0, 2, 0, rank_offset)
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
