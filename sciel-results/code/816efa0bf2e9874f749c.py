import numpy as np

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Target rank
    rank = 39
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, coeff=1):
        nonlocal col
        U[u_idx][col] += coeff
        V[v_idx][col] += coeff
        W[w_idx][col] += coeff
        col += 1

    # Helper to add a Strassen block for 2x2x2
    # Assumes A is rows i0, i1, cols j0, j1
    # B is rows j0, j1, cols k0, k1
    # C is rows k0, k1, cols i0, i1
    def add_strassen(i0, j0, k0):
        nonlocal col
        # Indices
        u00 = i0 * m + j0
        u01 = i0 * m + j0 + 1
        u10 = (i0 + 1) * m + j0
        u11 = (i0 + 1) * m + j0 + 1
        
        v00 = j0 * p + k0
        v01 = j0 * p + k0 + 1
        v10 = (j0 + 1) * p + k0
        v11 = (j0 + 1) * p + k0 + 1
        
        w00 = k0 * n + i0
        w01 = k0 * n + i0 + 1
        w10 = (k0 + 1) * n + i0
        w11 = (k0 + 1) * n + i0 + 1
        
        # M1: (A00+A11)(B00+B11) -> C00, C11
        # U: u00, u11; V: v00, v11; W: w00, w11
        # We need to add 1 to U[u00], U[u11], V[v00], V[v11], W[w00], W[w11]
        # But add_term adds 1 to all.
        # We need to handle multiple indices per term.
        # The current add_term only handles single indices.
        # We need a more flexible add_term or multiple calls.
        # Let's modify add_term to take lists.
        
        pass

    # Re-implementing with a more flexible add_term
    def add_term_multi(u_idxs, v_idxs, w_idxs, coeffs=None):
        nonlocal col
        if coeffs is None:
            coeffs = [1] * len(u_idxs)
        for u, v, w, c in zip(u_idxs, v_idxs, w_idxs, coeffs):
            U[u][col] += c
            V[v][col] += c
            W[w][col] += c
        col += 1

    def add_strassen_block(i0, j0, k0):
        # M1
        add_term_multi([i0*m+j0, (i0+1)*m+j0+1], [j0*p+k0, (j0+1)*p+k0+1], [k0*n+i0, (k0+1)*n+i0+1])
        # M2
        add_term_multi([(i0+1)*m+j0, (i0+1)*m+j0+1], [j0*p+k0], [(k0+1)*n+i0, (k0+1)*n+i0+1], [1, 1, 1, -1])
        # M3
        add_term_multi([i0*m+j0], [j0*p+k0+1, (j0+1)*p+k0+1], [k0*n+i0+1, (k0+1)*n+i0+1], [1, 1, 1, 1])
        # M4
        add_term_multi([(i0+1)*m+j0+1], [(j0+1)*p+k0, j0*p+k0], [k0*n+i0, (k0+1)*n+i0], [1, 1, 1, 1])
        # M5
        add_term_multi([i0*m+j0, i0*m+j0+1], [(j0+1)*p+k0+1], [k0*n+i0, k0*n+i0+1], [1, 1, -1, 1])
        # M6
        add_term_multi([(i0+1)*m+j0, i0*m+j0], [j0*p+k0, j0*p+k0+1], [(k0+1)*n+i0+1], [1, -1, 1, 1, 1])
        # M7
        add_term_multi([i0*m+j0+1, (i0+1)*m+j0+1], [(j0+1)*p+k0, (j0+1)*p+k0+1], [k0*n+i0], [1, -1, 1, 1, 1])

    # 3 Strassen blocks for k=0,2,4
    add_strassen_block(0, 0, 0)
    add_strassen_block(0, 0, 2)
    add_strassen_block(0, 0, 4)
    
    # k=6 block (2x2x1)
    # A[0:2, 0:2] x B[0:2, 6] x C[6, 0:2]
    # Terms:
    # A00 B06 C60
    # A01 B16 C60
    # A10 B06 C61
    # A11 B16 C61
    add_term_multi([0*m+0], [0*p+6], [6*n+0])
    add_term_multi([0*m+1], [1*p+6], [6*n+0])
    add_term_multi([1*m+0], [0*p+6], [6*n+1])
    add_term_multi([1*m+1], [1*p+6], [6*n+1])
    
    # j=2 block (2x1x7)
    # A[0:2, 2] x B[2, 0:7] x C[0:7, 0:2]
    for k in range(7):
        # i=0
        add_term_multi([0*m+2], [2*p+k], [k*n+0])
        # i=1
        add_term_multi([1*m+2], [2*p+k], [k*n+1])
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
