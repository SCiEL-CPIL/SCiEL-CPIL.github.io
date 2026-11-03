import time

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Total rank estimate: 28 (from 4 Strassen blocks) + 2 + 2 + 1 = 33
    rank = 33
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_idx = 0
    
    # Helper to add a Strassen block for M_{2,2,2}
    # i0, l0: top-left index of A block (i, l)
    # j0, k0: top-left index of B block (j, k)
    # r0, s0: top-left index of C block (r, s)
    def add_strassen(i0, l0, j0, k0, r0, s0):
        nonlocal col_idx
        # Strassen coefficients for A (n=2, m=2)
        # A local (a,b) -> global (i0+a, l0+b) -> flat (i0+a)*m + (l0+b)
        # B local (c,d) -> global (j0+c, k0+d) -> flat (j0+c)*p + (k0+d)
        # C local (e,f) -> global (r0+e, s0+f) -> flat (s0+f)*n + (r0+e)
        
        # M1 = A00 * (B00 + B11)
        U[(i0+0)*m + (l0+0)][col_idx] += 1
        V[(j0+0)*p + (k0+0)][col_idx] += 1
        V[(j0+1)*p + (k0+1)][col_idx] += 1
        # C00 += M1, C11 += M1
        W[(s0+0)*n + (r0+0)][col_idx] += 1
        W[(s0+1)*n + (r0+1)][col_idx] += 1
        col_idx += 1
        
        # M2 = (A00 + A10) * B11
        U[(i0+0)*m + (l0+0)][col_idx] += 1
        U[(i0+1)*m + (l0+0)][col_idx] += 1
        V[(j0+1)*p + (k0+1)][col_idx] += 1
        # C10 += M2, C11 -= M2
        W[(s0+0)*n + (r0+1)][col_idx] += 1
        W[(s0+1)*n + (r0+1)][col_idx] += -1
        col_idx += 1
        
        # M3 = (A10 + A11) * B00
        U[(i0+1)*m + (l0+0)][col_idx] += 1
        U[(i0+1)*m + (l0+1)][col_idx] += 1
        V[(j0+0)*p + (k0+0)][col_idx] += 1
        # C01 += M3, C11 += M3
        W[(s0+1)*n + (r0+0)][col_idx] += 1
        W[(s0+1)*n + (r0+1)][col_idx] += 1
        col_idx += 1
        
        # M4 = A11 * (B00 - B10)
        U[(i0+1)*m + (l0+1)][col_idx] += 1
        V[(j0+0)*p + (k0+0)][col_idx] += 1
        V[(j0+1)*p + (k0+0)][col_idx] += -1
        # C00 += M4, C10 += M4
        W[(s0+0)*n + (r0+0)][col_idx] += 1
        W[(s0+0)*n + (r0+1)][col_idx] += 1
        col_idx += 1
        
        # M5 = (A00 + A01) * B11
        U[(i0+0)*m + (l0+0)][col_idx] += 1
        U[(i0+0)*m + (l0+1)][col_idx] += 1
        V[(j0+1)*p + (k0+1)][col_idx] += 1
        # C00 -= M5, C01 += M5
        W[(s0+0)*n + (r0+0)][col_idx] += -1
        W[(s0+1)*n + (r0+0)][col_idx] += 1
        col_idx += 1
        
        # M6 = (A10 - A00) * (B00 + B01)
        U[(i0+1)*m + (l0+0)][col_idx] += 1
        U[(i0+0)*m + (l0+0)][col_idx] += -1
        V[(j0+0)*p + (k0+0)][col_idx] += 1
        V[(j0+0)*p + (k0+1)][col_idx] += 1
        # C11 += M6
        W[(s0+1)*n + (r0+1)][col_idx] += 1
        col_idx += 1
        
        # M7 = (A01 - A11) * (B10 + B11)
        U[(i0+0)*m + (l0+1)][col_idx] += 1
        U[(i0+1)*m + (l0+1)][col_idx] += -1
        V[(j0+1)*p + (k0+0)][col_idx] += 1
        V[(j0+1)*p + (k0+1)][col_idx] += 1
        # C00 += M7
        W[(s0+0)*n + (r0+0)][col_idx] += 1
        col_idx += 1

    # Block 1: M_{2,2,8} -> 4 * M_{2,2,2}
    # A: i in {0,1}, l in {0,1}
    # B: j in {0,1}, k in {0..7}
    # C: r in {0,1}, s in {0..7}
    # We split B/C into 4 pairs of columns
    for t in range(4):
        k_offset = t * 2
        s_offset = t * 2
        add_strassen(0, 0, 0, k_offset, 0, s_offset)
        
    # Block 2: M_{2,1,8}
    # A: i in {0,1}, l=2
    # B: j=2, k in {0..7}
    # C: r in {0,1}, s in {0..7}
    # Rank 2
    # r=0 term: A_{0,2} * B_{2,*} -> C_{0,*}
    for k in range(p):
        U[0*m + 2][col_idx] += 1
        V[2*p + k][col_idx] += 1
        for s in range(p):
            W[s*n + 0][col_idx] += 1
    col_idx += 1
    
    # r=1 term: A_{1,2} * B_{2,*} -> C_{1,*}
    for k in range(p):
        U[1*m + 2][col_idx] += 1
        V[2*p + k][col_idx] += 1
        for s in range(p):
            W[s*n + 1][col_idx] += 1
    col_idx += 1
    
    # Block 3: M_{1,2,8}
    # A: i=2, l in {0,1}
    # B: j in {0,1}, k in {0..7}
    # C: r=2, s in {0..7}
    # Rank 2
    # r=0 term: A_{2,0} * B_{0,*} -> C_{2,*}
    for k in range(p):
        U[2*m + 0][col_idx] += 1
        V[0*p + k][col_idx] += 1
        for s in range(p):
            W[s*n + 2][col_idx] += 1
    col_idx += 1
    
    # r=1 term: A_{2,1} * B_{1,*} -> C_{2,*}
    for k in range(p):
        U[2*m + 1][col_idx] += 1
        V[1*p + k][col_idx] += 1
        for s in range(p):
            W[s*n + 2][col_idx] += 1
    col_idx += 1
    
    # Block 4: M_{1,1,8}
    # A: i=2, l=2
    # B: j=2, k in {0..7}
    # C: r=2, s in {0..7}
    # Rank 1
    for k in range(p):
        U[2*m + 2][col_idx] += 1
        V[2*p + k][col_idx] += 1
        for s in range(p):
            W[s*n + 2][col_idx] += 1
    col_idx += 1
    
    if col_idx != rank:
        # This should not happen
        pass

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
