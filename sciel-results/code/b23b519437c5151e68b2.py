"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 44 decomposition
    rank = 44
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    
    # Helper to add a 2x2x2 Strassen block
    def add_strassen_222(i_off, j_off, k_off):
        nonlocal col
        # Local indices:
        # A: (0,0)->a, (0,1)->b, (1,0)->c, (1,1)->d
        # B: (0,0)->e, (0,1)->f, (1,0)->g, (1,1)->h
        # C: (0,0)->i, (0,1)->j, (1,0)->k, (1,1)->l
        
        # Map local to global
        # U index: (i_off + li) * m + (j_off + lj)
        # V index: (j_off + lj) * p + (k_off + lk)
        # W index: (k_off + lk) * n + (i_off + li)
        
        # Strassen terms:
        # 1. M1 = (a+d)(e+h) -> i=1, l=1
        U[(i_off+0)*m + (j_off+0), col] = 1
        U[(i_off+1)*m + (j_off+1), col] = 1
        V[(j_off+0)*p + (k_off+0), col] = 1
        V[(j_off+1)*p + (k_off+1), col] = 1
        W[(k_off+0)*n + (i_off+0), col] = 1
        W[(k_off+1)*n + (i_off+1), col] = 1
        col += 1
        
        # 2. M2 = (c+d)e -> k=1, l=-1
        U[(i_off+1)*m + (j_off+0), col] = 1
        U[(i_off+1)*m + (j_off+1), col] = 1
        V[(j_off+0)*p + (k_off+0), col] = 1
        W[(k_off+0)*n + (i_off+1), col] = 1
        W[(k_off+1)*n + (i_off+1), col] = -1
        col += 1
        
        # 3. M3 = a(f-h) -> j=1, l=1
        U[(i_off+0)*m + (j_off+0), col] = 1
        V[(j_off+0)*p + (k_off+1), col] = 1
        V[(j_off+1)*p + (k_off+1), col] = -1
        W[(k_off+1)*n + (i_off+0), col] = 1
        W[(k_off+1)*n + (i_off+1), col] = 1
        col += 1
        
        # 4. M4 = d(g-e) -> i=1, k=1
        U[(i_off+1)*m + (j_off+1), col] = 1
        V[(j_off+1)*p + (k_off+0), col] = 1
        V[(j_off+0)*p + (k_off+0), col] = -1
        W[(k_off+0)*n + (i_off+0), col] = 1
        W[(k_off+0)*n + (i_off+1), col] = 1
        col += 1
        
        # 5. M5 = (a+b)h -> i=-1, j=1
        U[(i_off+0)*m + (j_off+0), col] = 1
        U[(i_off+0)*m + (j_off+1), col] = 1
        V[(j_off+1)*p + (k_off+1), col] = 1
        W[(k_off+0)*n + (i_off+0), col] = -1
        W[(k_off+1)*n + (i_off+0), col] = 1
        col += 1
        
        # 6. M6 = (c-a)(e+f) -> l=1
        U[(i_off+1)*m + (j_off+0), col] = 1
        U[(i_off+0)*m + (j_off+0), col] = -1
        V[(j_off+0)*p + (k_off+0), col] = 1
        V[(j_off+0)*p + (k_off+1), col] = 1
        W[(k_off+1)*n + (i_off+1), col] = 1
        col += 1
        
        # 7. M7 = (b-d)(g+h) -> i=1
        U[(i_off+0)*m + (j_off+1), col] = 1
        U[(i_off+1)*m + (j_off+1), col] = -1
        V[(j_off+1)*p + (k_off+0), col] = 1
        V[(j_off+1)*p + (k_off+1), col] = 1
        W[(k_off+0)*n + (i_off+0), col] = 1
        col += 1

    # Helper to add a 1x2x2 block (rank 4)
    def add_122(i_off, j_off, k_off):
        nonlocal col
        # A: (0,0)->a, (0,1)->b
        # B: (0,0)->e, (0,1)->f, (1,0)->g, (1,1)->h
        # C: (0,0)->i, (0,1)->j
        
        # 1. ae -> i
        U[(i_off+0)*m + (j_off+0), col] = 1
        V[(j_off+0)*p + (k_off+0), col] = 1
        W[(k_off+0)*n + (i_off+0), col] = 1
        col += 1
        
        # 2. bg -> i
        U[(i_off+0)*m + (j_off+1), col] = 1
        V[(j_off+1)*p + (k_off+0), col] = 1
        W[(k_off+0)*n + (i_off+0), col] = 1
        col += 1
        
        # 3. af -> j
        U[(i_off+0)*m + (j_off+0), col] = 1
        V[(j_off+0)*p + (k_off+1), col] = 1
        W[(k_off+1)*n + (i_off+0), col] = 1
        col += 1
        
        # 4. bh -> j
        U[(i_off+0)*m + (j_off+1), col] = 1
        V[(j_off+1)*p + (k_off+1), col] = 1
        W[(k_off+1)*n + (i_off+0), col] = 1
        col += 1

    # Top part: A[0:2, :] * B[:, :]
    # B blocks: B[0:2, 0:2] (0,0), B[0:2, 2:4] (0,1), B[2:4, 0:2] (1,0), B[2:4, 2:4] (1,1)
    
    # C_top[0:2, 0:2] = A_top[0:2, 0:2] * B[0:2, 0:2] + A_top[0:2, 2:4] * B[2:4, 0:2]
    # Wait, A_top is 2x4.
    # A_top[:, 0:2] is 2x2. A_top[:, 2:4] is 2x2.
    
    # Block 1: A_top[:, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    add_strassen_222(0, 0, 0)
    
    # Block 2: A_top[:, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    # Note: This adds to the same C indices.
    # U indices for A_top[:, 2:4]: i in 0,1; j in 2,3.
    # V indices for B[2:4, 0:2]: j in 2,3; k in 0,1.
    # W indices for C[0:2, 0:2]: k in 0,1; i in 0,1.
    add_strassen_222(0, 2, 0)
    
    # Block 3: A_top[:, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    add_strassen_222(0, 0, 2)
    
    # Block 4: A_top[:, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    add_strassen_222(0, 2, 2)
    
    # Bottom part: A[2, :] * B[:, :]
    # A_bot is 1x4.
    # A_bot[0:2] is 1x2. A_bot[2:4] is 1x2.
    
    # Block 5: A_bot[0:2] * B[0:2, 0:2] -> C[2, 0:2]
    add_122(2, 0, 0)
    
    # Block 6: A_bot[2:4] * B[2:4, 0:2] -> C[2, 0:2]
    add_122(2, 2, 0)
    
    # Block 7: A_bot[0:2] * B[0:2, 2:4] -> C[2, 2:4]
    add_122(2, 0, 2)
    
    # Block 8: A_bot[2:4] * B[2:4, 2:4] -> C[2, 2:4]
    add_122(2, 2, 2)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
