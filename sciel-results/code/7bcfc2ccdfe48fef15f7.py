"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    # Rank 36 decomposition
    # Partition p=5 into p1=2, p2=3
    # Partition m=4 into m1=2, m2=2
    # Block 1: A1 (2x2) * B11 (2x2) -> Rank 7 (Strassen)
    # Block 2: A2 (2x2) * B12 (2x2) -> Rank 7 (Strassen)
    # Block 3: A1 (2x2) * B21 (2x3) -> Rank 11 (Strassen 7 + Schoolbook 4)
    # Block 4: A2 (2x2) * B22 (2x3) -> Rank 11 (Strassen 7 + Schoolbook 4)
    
    n, m, p = DIMENSIONS
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a 2x2x2 Strassen block
    def add_strassen(i_off, j_off, k_off):
        nonlocal col
        # Local indices i in 0..1, j in 0..1, k in 0..1
        # Global indices:
        # I = i_off + i
        # J = j_off + j
        # K = k_off + k
        
        # M1: (a+d)(e+h)
        U[(i_off+0)*m + (j_off+0)][col] = 1
        U[(i_off+1)*m + (j_off+1)][col] = 1
        V[(j_off+0)*p + (k_off+0)][col] = 1
        V[(j_off+1)*p + (k_off+1)][col] = 1
        W[(k_off+0)*n + (i_off+0)][col] = 1
        W[(k_off+1)*n + (i_off+1)][col] = 1
        col += 1
        
        # M2: (c+d)e
        U[(i_off+1)*m + (j_off+0)][col] = 1
        U[(i_off+1)*m + (j_off+1)][col] = 1
        V[(j_off+0)*p + (k_off+0)][col] = 1
        W[(k_off+0)*n + (i_off+1)][col] = 1
        W[(k_off+1)*n + (i_off+1)][col] = -1
        col += 1
        
        # M3: a(f-h)
        U[(i_off+0)*m + (j_off+0)][col] = 1
        V[(j_off+0)*p + (k_off+1)][col] = 1
        V[(j_off+1)*p + (k_off+1)][col] = -1
        W[(k_off+1)*n + (i_off+0)][col] = 1
        W[(k_off+1)*n + (i_off+1)][col] = 1
        col += 1
        
        # M4: d(i-e)
        U[(i_off+1)*m + (j_off+1)][col] = 1
        V[(j_off+1)*p + (k_off+0)][col] = 1
        V[(j_off+0)*p + (k_off+0)][col] = -1
        W[(k_off+0)*n + (i_off+0)][col] = 1
        W[(k_off+0)*n + (i_off+1)][col] = 1
        col += 1
        
        # M5: (a+b)h
        U[(i_off+0)*m + (j_off+0)][col] = 1
        U[(i_off+0)*m + (j_off+1)][col] = 1
        V[(j_off+1)*p + (k_off+1)][col] = 1
        W[(k_off+0)*n + (i_off+0)][col] = -1
        W[(k_off+1)*n + (i_off+0)][col] = 1
        col += 1
        
        # M6: (c-a)(e+f)
        U[(i_off+1)*m + (j_off+0)][col] = 1
        U[(i_off+0)*m + (j_off+0)][col] = -1
        V[(j_off+0)*p + (k_off+0)][col] = 1
        V[(j_off+0)*p + (k_off+1)][col] = 1
        W[(k_off+1)*n + (i_off+1)][col] = 1
        col += 1
        
        # M7: (b-d)(g+h)
        U[(i_off+0)*m + (j_off+1)][col] = 1
        U[(i_off+1)*m + (j_off+1)][col] = -1
        V[(j_off+1)*p + (k_off+0)][col] = 1
        V[(j_off+1)*p + (k_off+1)][col] = 1
        W[(k_off+0)*n + (i_off+0)][col] = 1
        col += 1

    # Helper to add a 2x2x3 block (Rank 11)
    def add_2x2x3(i_off, j_off, k_off):
        nonlocal col
        # First 2 columns: Strassen (7 ranks)
        add_strassen(i_off, j_off, k_off)
        
        # Last 1 column: Schoolbook (4 ranks)
        # k_local = 2
        # c02 = a00*g + a10*j
        # c12 = a01*g + a11*j
        # U: A indices, V: B indices, W: C indices
        
        # Factor 1: a00 * g
        U[(i_off+0)*m + (j_off+0)][col] = 1
        V[(j_off+0)*p + (k_off+2)][col] = 1
        W[(k_off+2)*n + (i_off+0)][col] = 1
        col += 1
        
        # Factor 2: a10 * j
        U[(i_off+1)*m + (j_off+0)][col] = 1
        V[(j_off+1)*p + (k_off+2)][col] = 1
        W[(k_off+2)*n + (i_off+0)][col] = 1
        col += 1
        
        # Factor 3: a01 * g
        U[(i_off+0)*m + (j_off+1)][col] = 1
        V[(j_off+0)*p + (k_off+2)][col] = 1
        W[(k_off+2)*n + (i_off+1)][col] = 1
        col += 1
        
        # Factor 4: a11 * j
        U[(i_off+1)*m + (j_off+1)][col] = 1
        V[(j_off+1)*p + (k_off+2)][col] = 1
        W[(k_off+2)*n + (i_off+1)][col] = 1
        col += 1

    # Block 1: i=0..1, j=0..1, k=0..1
    add_strassen(0, 0, 0)
    
    # Block 2: i=0..1, j=2..3, k=0..1
    add_strassen(0, 2, 0)
    
    # Block 3: i=0..1, j=0..1, k=2..4
    add_2x2x3(0, 0, 2)
    
    # Block 4: i=0..1, j=2..3, k=2..4
    add_2x2x3(0, 2, 2)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
