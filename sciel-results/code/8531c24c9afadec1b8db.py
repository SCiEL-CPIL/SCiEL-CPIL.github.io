"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
_start_time = time.time()
DEADLINE = _start_time + 140.0  # 140s safety margin before 150s kill

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strategy:
    # Split p=8 into two blocks of p=4.
    # For each p=4 block, decompose 2x3x4 multiplication.
    # 2x3x4 is split into 2x2x4 (rank 14) and 2x1x4 (rank 8).
    # Total rank per block = 22.
    # Total rank = 44.
    
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0

    def add_strassen_2x2x2(i_off, j_off, k_off):
        """Adds a 2x2x2 Strassen decomposition."""
        nonlocal col
        # U indices: (i, j) -> i*m + j
        # V indices: (j, k) -> j*p + k
        # W indices: (k, i) -> k*n + i
        
        # 1. (a+d)(e+h)
        U[(i_off+0)*m + (j_off+0)][col] += 1
        U[(i_off+1)*m + (j_off+1)][col] += 1
        V[(j_off+0)*p + (k_off+0)][col] += 1
        V[(j_off+1)*p + (k_off+1)][col] += 1
        W[(k_off+0)*n + (i_off+0)][col] += 1
        W[(k_off+1)*n + (i_off+1)][col] += 1
        col += 1
        
        # 2. (c+d)e
        U[(i_off+1)*m + (j_off+0)][col] += 1
        U[(i_off+1)*m + (j_off+1)][col] += 1
        V[(j_off+0)*p + (k_off+0)][col] += 1
        W[(k_off+0)*n + (i_off+1)][col] += 1
        W[(k_off+1)*n + (i_off+1)][col] += -1
        col += 1
        
        # 3. a(f-h)
        U[(i_off+0)*m + (j_off+0)][col] += 1
        V[(j_off+0)*p + (k_off+1)][col] += 1
        V[(j_off+1)*p + (k_off+1)][col] += -1
        W[(k_off+1)*n + (i_off+0)][col] += 1
        W[(k_off+1)*n + (i_off+1)][col] += 1
        col += 1
        
        # 4. d(g-e)
        U[(i_off+1)*m + (j_off+1)][col] += 1
        V[(j_off+1)*p + (k_off+0)][col] += 1
        V[(j_off+0)*p + (k_off+0)][col] += -1
        W[(k_off+0)*n + (i_off+0)][col] += 1
        W[(k_off+0)*n + (i_off+1)][col] += 1
        col += 1
        
        # 5. (a+b)h
        U[(i_off+0)*m + (j_off+0)][col] += 1
        U[(i_off+0)*m + (j_off+1)][col] += 1
        V[(j_off+1)*p + (k_off+1)][col] += 1
        W[(k_off+0)*n + (i_off+0)][col] += -1
        W[(k_off+1)*n + (i_off+0)][col] += 1
        col += 1
        
        # 6. (c-a)(e+f)
        U[(i_off+1)*m + (j_off+0)][col] += 1
        U[(i_off+0)*m + (j_off+0)][col] += -1
        V[(j_off+0)*p + (k_off+0)][col] += 1
        V[(j_off+0)*p + (k_off+1)][col] += 1
        W[(k_off+1)*n + (i_off+1)][col] += 1
        col += 1
        
        # 7. (b-d)(g+h)
        U[(i_off+0)*m + (j_off+1)][col] += 1
        U[(i_off+1)*m + (j_off+1)][col] += -1
        V[(j_off+1)*p + (k_off+0)][col] += 1
        V[(j_off+1)*p + (k_off+1)][col] += 1
        W[(k_off+0)*n + (i_off+0)][col] += 1
        col += 1

    def add_schoolbook_2x1x4(i_off, j_off, k_off):
        """Adds a 2x1x4 schoolbook decomposition."""
        nonlocal col
        for i in range(2):
            for k in range(4):
                U[(i_off+i)*m + (j_off+0)][col] = 1
                V[(j_off+0)*p + (k_off+k)][col] = 1
                W[(k_off+k)*n + (i_off+i)][col] = 1
                col += 1

    # Process Block 1 (k=0..3)
    # Split m=3 into m=0,1 (2x2 part) and m=2 (2x1 part)
    # Split p=4 into p=0,1 and p=2,3 for the 2x2 part
    
    # 2x2x2 at j=0,1 k=0,1
    add_strassen_2x2x2(0, 0, 0)
    # 2x2x2 at j=0,1 k=2,3
    add_strassen_2x2x2(0, 0, 2)
    # 2x1x4 at j=2 k=0..3
    add_schoolbook_2x1x4(0, 2, 0)

    # Process Block 2 (k=4..7)
    # 2x2x2 at j=0,1 k=4,5
    add_strassen_2x2x2(0, 0, 4)
    # 2x2x2 at j=0,1 k=6,7
    add_strassen_2x2x2(0, 0, 6)
    # 2x1x4 at j=2 k=4..7
    add_schoolbook_2x1x4(0, 2, 4)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
