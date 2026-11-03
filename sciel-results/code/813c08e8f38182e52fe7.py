import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We will use a block decomposition.
    # Partitions:
    # I (rows of A, size n=3): I1=[0,1] (size 2), I2=[2] (size 1)
    # J (cols of A, size m=5): J1=[0,1] (size 2), J2=[2,3] (size 2), J3=[4] (size 1)
    # K (cols of B, size p=6): K1=[0,1] (size 2), K2=[2,3] (size 2), K3=[4,5] (size 2)
    
    # Calculate total rank required
    # 6 blocks of 2x2x2 -> 6 * 7 = 42
    # 3 blocks of 2x1x2 -> 3 * 4 = 12
    # 6 blocks of 1x2x2 -> 6 * 4 = 24
    # 3 blocks of 1x1x2 -> 3 * 2 = 6
    # Total Rank = 84
    
    RANK = 84
    U = [[0] * RANK for _ in range(n * m)]
    V = [[0] * RANK for _ in range(m * p)]
    W = [[0] * RANK for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(i_local, j_local, k_local, i_off, j_off, k_off, coeff):
        nonlocal col
        if col >= RANK:
            return # Should not happen
        
        i_g = i_off + i_local
        j_g = j_off + j_local
        k_g = k_off + k_local
        
        # U index: i * m + j
        # V index: j * p + k
        # W index: k * n + i
        
        u_idx = i_g * m + j_g
        v_idx = j_g * p + k_g
        w_idx = k_g * n + i_g
        
        U[u_idx][col] = coeff
        V[v_idx][col] = coeff
        W[w_idx][col] = coeff
        col += 1
        
    # Helper for 2x2x2 Strassen block
    def add_strassen_222(i_off, j_off, k_off):
        # M1 = (A00+A11)(B00+B11)
        # Terms: A00B00, A00B11, A11B00, A11B11
        add_term(0, 0, 0, i_off, j_off, k_off, 1)
        add_term(0, 0, 1, i_off, j_off, k_off, 1)
        add_term(1, 1, 0, i_off, j_off, k_off, 1)
        add_term(1, 1, 1, i_off, j_off, k_off, 1)
        
        # M2 = (A10+A11)B00
        # Terms: A10B00, A11B00
        add_term(1, 0, 0, i_off, j_off, k_off, 1)
        add_term(1, 1, 0, i_off, j_off, k_off, 1)
        
        # M3 = A00(B10-B11)
        # Terms: A00B10, A00B11(-1)
        add_term(0, 0, 1, i_off, j_off, k_off, 1)
        add_term(0, 0, 0, i_off, j_off, k_off, -1)
        
        # M4 = A11(B01-B00)
        # Terms: A11B01, A11B00(-1)
        add_term(1, 1, 1, i_off, j_off, k_off, 1)
        add_term(1, 1, 0, i_off, j_off, k_off, -1)
        
        # M5 = (A00+A10)B11
        # Terms: A00B11, A10B11
        add_term(0, 0, 1, i_off, j_off, k_off, 1)
        add_term(1, 0, 1, i_off, j_off, k_off, 1)
        
        # M6 = (A11-A00)(B00+B10)
        # Terms: A11B00, A11B10, A00B00(-1), A00B10(-1)
        add_term(1, 1, 0, i_off, j_off, k_off, 1)
        add_term(1, 1, 1, i_off, j_off, k_off, 1)
        add_term(0, 0, 0, i_off, j_off, k_off, -1)
        add_term(0, 0, 1, i_off, j_off, k_off, -1)
        
        # M7 = (A01-A11)(B11+B01)
        # Terms: A01B11, A01B01, A11B11(-1), A11B01(-1)
        add_term(0, 1, 1, i_off, j_off, k_off, 1)
        add_term(0, 1, 0, i_off, j_off, k_off, 1)
        add_term(1, 1, 1, i_off, j_off, k_off, -1)
        add_term(1, 1, 0, i_off, j_off, k_off, -1)

    # Helper for rectangular blocks (Schoolbook)
    def add_schoolbook(nb, mb, pb, i_off, j_off, k_off):
        for i in range(nb):
            for j in range(mb):
                for k in range(pb):
                    add_term(i, j, k, i_off, j_off, k_off, 1)

    # Define blocks
    # Offsets for I: 0 (size 2), 2 (size 1)
    # Offsets for J: 0 (size 2), 2 (size 2), 4 (size 1)
    # Offsets for K: 0 (size 2), 2 (size 2), 4 (size 2)
    
    i_offsets = [0, 2]
    i_sizes = [2, 1]
    j_offsets = [0, 2, 4]
    j_sizes = [2, 2, 1]
    k_offsets = [0, 2, 4]
    k_sizes = [2, 2, 2]
    
    for ii in range(len(i_offsets)):
        for jj in range(len(j_offsets)):
            for kk in range(len(k_offsets)):
                ib = i_sizes[ii]
                jb = j_sizes[jj]
                kb = k_sizes[kk]
                io = i_offsets[ii]
                jo = j_offsets[jj]
                ko = k_offsets[kk]
                
                if ib == 2 and jb == 2 and kb == 2:
                    add_strassen_222(io, jo, ko)
                else:
                    # Schoolbook for remaining blocks
                    add_schoolbook(ib, jb, kb, io, jo, ko)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
