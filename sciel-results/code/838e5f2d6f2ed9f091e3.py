import time

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 88
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_indices, v_indices, w_indices, val):
        nonlocal col
        for u_idx in u_indices:
            U[u_idx][col] += val
        for v_idx in v_indices:
            V[v_idx][col] += val
        for w_idx in w_indices:
            W[w_idx][col] += val
        col += 1

    # Helper for 2x2x2 Strassen block
    def add_strassen(i0, j0, k0):
        # i0, j0, k0 are global starting indices
        # Local indices: i in 0..1, j in 0..1, k in 0..1
        # U index: (i0+i)*m + (j0+j)
        # V index: (j0+j)*p + (k0+k)
        # W index: (k0+k)*n + (i0+i)
        
        # P1 = (A00 + A11)(B00 + B11)
        # U: (i0, j0), (i0+1, j0+1)
        # V: (j0, k0), (j0+1, k0+1)
        # W: (k0, i0) coeff 1, (k0+1, i0+1) coeff 1
        add_term(
            [(i0)*m + (j0), (i0+1)*m + (j0+1)],
            [(j0)*p + (k0), (j0+1)*p + (k0+1)],
            [(k0)*n + (i0), (k0+1)*n + (i0+1)],
            1
        )
        
        # P2 = (A10 + A11)B00
        # U: (i0+1, j0), (i0+1, j0+1)
        # V: (j0, k0)
        # W: (k0, i0+1) coeff 1
        add_term(
            [(i0+1)*m + (j0), (i0+1)*m + (j0+1)],
            [(j0)*p + (k0)],
            [(k0)*n + (i0+1)],
            1
        )
        
        # P3 = A00(B01 - B11)
        # U: (i0, j0)
        # V: (j0, k0+1) coeff 1, (j0+1, k0+1) coeff -1
        # W: (k0+1, i0) coeff 1
        add_term(
            [(i0)*m + (j0)],
            [(j0)*p + (k0+1), (j0+1)*p + (k0+1)],
            [(k0+1)*n + (i0)],
            1
        )
        # Note: add_term adds val to all. We need different vals for V.
        # My add_term implementation adds the same val to all indices.
        # I need a more flexible add_term or manual filling.
        
    # Let's rewrite add_term to handle lists of (index, value)
    def add_term_flex(u_list, v_list, w_list):
        nonlocal col
        for idx, val in u_list:
            U[idx][col] += val
        for idx, val in v_list:
            V[idx][col] += val
        for idx, val in w_list:
            W[idx][col] += val
        col += 1

    def add_strassen_flex(i0, j0, k0):
        # P1
        add_term_flex(
            [((i0)*m + (j0), 1), ((i0+1)*m + (j0+1), 1)],
            [((j0)*p + (k0), 1), ((j0+1)*p + (k0+1), 1)],
            [((k0)*n + (i0), 1), ((k0+1)*n + (i0+1), 1)]
        )
        # P2
        add_term_flex(
            [((i0+1)*m + (j0), 1), ((i0+1)*m + (j0+1), 1)],
            [((j0)*p + (k0), 1)],
            [((k0)*n + (i0+1), 1)]
        )
        # P3
        add_term_flex(
            [((i0)*m + (j0), 1)],
            [((j0)*p + (k0+1), 1), ((j0+1)*p + (k0+1), -1)],
            [((k0+1)*n + (i0), 1)]
        )
        # P4
        add_term_flex(
            [((i0+1)*m + (j0+1), 1)],
            [((j0+1)*p + (k0), 1), ((j0)*p + (k0), -1)],
            [((k0)*n + (i0+1), 1)]
        )
        # P5
        add_term_flex(
            [((i0)*m + (j0), 1), ((i0)*m + (j0+1), 1)],
            [((j0+1)*p + (k0+1), 1)],
            [((k0+1)*n + (i0), 1)]
        )
        # P6
        add_term_flex(
            [((i0+1)*m + (j0), 1), ((i0)*m + (j0), -1)],
            [((j0)*p + (k0), 1), ((j0)*p + (k0+1), 1)],
            [((k0+1)*n + (i0+1), 1)]
        )
        # P7
        add_term_flex(
            [((i0)*m + (j0+1), 1), ((i0+1)*m + (j0+1), -1)],
            [((j0+1)*p + (k0), 1), ((j0+1)*p + (k0+1), 1)],
            [((k0)*n + (i0), 1)]
        )

    def add_dot_product(i0, j0, k0):
        # 1x2x1: C = A0 B0 + A1 B1
        # A indices: (i0, j0), (i0, j0+1)
        # B indices: (j0, k0), (j0+1, k0)
        # C index: (k0, i0)
        
        # Term 1: A0 B0
        add_term_flex(
            [((i0)*m + (j0), 1)],
            [((j0)*p + (k0), 1)],
            [((k0)*n + (i0), 1)]
        )
        # Term 2: A1 B1
        add_term_flex(
            [((i0)*m + (j0+1), 1)],
            [((j0+1)*p + (k0), 1)],
            [((k0)*n + (i0), 1)]
        )

    # Main loop
    # m is split into 2 blocks of size 2
    for j_block in range(2):
        j0 = j_block * 2
        
        # n is split into 1 block of size 2 and 1 block of size 1
        # Block 1: i in [0, 1]
        for i_block in range(2):
            i0 = i_block # 0 and 1
            
            # p is split into 4 blocks of size 2
            for k_block in range(4):
                k0 = k_block * 2
                add_strassen_flex(i0, j0, k0)
                
        # Block 2: i in [2, 2] (size 1)
        i0 = 2
        # p is split into 8 blocks of size 1
        for k_block in range(8):
            k0 = k_block
            add_dot_product(i0, j0, k0)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
