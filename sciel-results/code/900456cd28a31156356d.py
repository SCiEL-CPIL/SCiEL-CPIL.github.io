"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 4, 6  # A is n x m, B is m x p, C is n x p
    # U: (n*m, rank) = (12, rank)
    # V: (m*p, rank) = (24, rank)
    # W: (p*n, rank) = (18, rank)
    
    U = [[0] * 0 for _ in range(n * m)]
    V = [[0] * 0 for _ in range(m * p)]
    W = [[0] * 0 for _ in range(p * n)]
    
    rank = 0
    
    def add_block(i_start, i_size, j_start, j_size, k_start, k_size, 
                  u_entries, v_entries, w_entries):
        nonlocal rank
        num_cols = len(u_entries)
        
        for c in range(num_cols):
            col = rank + c
            for row in range(n * m):
                U[row].append(0)
            for row in range(m * p):
                V[row].append(0)
            for row in range(p * n):
                W[row].append(0)
            
            for (li, lj, val) in u_entries[c]:
                U[(i_start + li) * m + (j_start + lj)][col] = val
            for (lj, lk, val) in v_entries[c]:
                V[(j_start + lj) * p + (k_start + lk)][col] = val
            for (lk, li, val) in w_entries[c]:
                W[(k_start + lk) * n + (i_start + li)][col] = val
        
        rank += num_cols
    
    def block_1x2x2():
        u = []
        v = []
        w = []
        # (A[0,0], B[0,0]) -> C[0,0]
        u.append([(0, 0, 1)])
        v.append([(0, 0, 1)])
        w.append([(0, 0, 1)])
        # (A[0,0], B[0,1]) -> C[0,1]
        u.append([(0, 0, 1)])
        v.append([(0, 1, 1)])
        w.append([(1, 0, 1)])
        # (A[0,1], B[1,0]) -> C[0,0]
        u.append([(0, 1, 1)])
        v.append([(1, 0, 1)])
        w.append([(0, 0, 1)])
        # (A[0,1], B[1,1]) -> C[0,1]
        u.append([(0, 1, 1)])
        v.append([(1, 1, 1)])
        w.append([(1, 0, 1)])
        return u, v, w
    
    def block_2x2x2():
        u = []
        v = []
        w = []
        
        # M1 = (A00+A11)(B00+B11)
        u.append([(0, 0, 1), (1, 1, 1)])
        v.append([(0, 0, 1), (1, 1, 1)])
        w.append([(0, 0, 1), (1, 1, 1)])
        
        # M2 = (A10+A11)B00
        u.append([(1, 0, 1), (1, 1, 1)])
        v.append([(0, 0, 1)])
        w.append([(1, 0, 1), (1, 1, -1)])
        
        # M3 = A00(B10-B11)
        u.append([(0, 0, 1)])
        v.append([(1, 0, 1), (1, 1, -1)])
        w.append([(0, 1, 1), (1, 1, 1)])
        
        # M4 = A11(B01-B10)
        u.append([(1, 1, 1)])
        v.append([(0, 1, 1), (1, 0, -1)])
        w.append([(0, 0, 1), (1, 0, 1)])
        
        # M5 = (A00+A01)B11
        u.append([(0, 0, 1), (0, 1, 1)])
        v.append([(1, 1, 1)])
        w.append([(0, 1, 1), (0, 0, -1)])
        
        # M6 = (A10-A00)(B00+B10)
        u.append([(1, 0, 1), (0, 0, -1)])
        v.append([(0, 0, 1), (1, 0, 1)])
        w.append([(1, 1, 1)])
        
        # M7 = (A01-A11)(B01+B11)
        u.append([(0, 1, 1), (1, 1, -1)])
        v.append([(0, 1, 1), (1, 1, 1)])
        w.append([(0, 0, 1)])
        
        return u, v, w
    
    # i-blocks: [(start, size)]
    # i: 3 rows -> split into 1 + 2
    # j: 4 cols -> split into 2 + 2
    # k: 6 cols -> split into 2 + 2 + 2
    i_blocks = [(0, 1), (1, 2)]
    j_blocks = [(0, 2), (2, 2)]
    k_blocks = [(0, 2), (2, 2), (4, 2)]
    
    for i_start, i_size in i_blocks:
        for j_start, j_size in j_blocks:
            for k_start, k_size in k_blocks:
                if i_size == 1 and j_size == 2 and k_size == 2:
                    u, v, w = block_1x2x2()
                elif i_size == 2 and j_size == 2 and k_size == 2:
                    u, v, w = block_2x2x2()
                else:
                    u = []
                    v = []
                    w = []
                    for li in range(i_size):
                        for lj in range(j_size):
                            for lk in range(k_size):
                                u.append([(li, lj, 1)])
                                v.append([(lj, lk, 1)])
                                w.append([(lk, li, 1)])
                add_block(i_start, i_size, j_start, j_size, k_start, k_size, u, v, w)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
