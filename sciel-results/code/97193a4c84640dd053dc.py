"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    # We use a block decomposition.
    # A is 4x4, split into 2x2 blocks of size 2x2.
    # B is 4x6, split into 2x3 blocks of size 2x2.
    # C is 4x6, split into 2x3 blocks of size 2x2.
    
    # Total rank will be 12 * 7 = 84.
    rank = 84
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen decomposition for a 2x2x2 block multiplication
    # A_block is at (ai, aj) in A grid (2x2 grid of 2x2 blocks)
    # B_block is at (bi, bj) in B grid (2x3 grid of 2x2 blocks)
    # C_block is at (ci, cj) in C grid (2x3 grid of 2x2 blocks)
    # Note: C_block is the result of A_block * B_block?
    # No, C is sum of products.
    # We will iterate over each term in the block matrix multiplication.
    
    # A indices: i in 0..1, j in 0..1 (block indices)
    # B indices: k in 0..1, l in 0..2 (block indices)
    # C indices: i in 0..1, l in 0..2 (block indices)
    # C[i, l] = sum_k A[i, k] * B[k, l]
    
    # For each i, k, l, we have a term A[i, k] * B[k, l] contributing to C[i, l].
    # There are 2 * 2 * 3 = 12 such terms.
    # Each term is a 2x2 * 2x2 multiplication.
    # We apply Strassen to each.
    
    # Strassen formulas for C = A * B (2x2)
    # M1 = (A00+A11)(B01+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B00-B01)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We need to map local 2x2 indices to global indices.
    # A block (bi, bj) covers A[2*bi:2*bi+2, 2*bj:2*bj+2]
    # B block (bk, bl) covers B[2*bk:2*bk+2, 2*bl:2*bl+2]
    # C block (ci, cl) covers C[2*ci:2*ci+2, 2*cl:2*cl+2]
    
    # Local indices in 2x2 block:
    # 00 -> 0, 01 -> 1, 10 -> 2, 11 -> 3 (row-major)
    
    def add_strassen(ai, aj, bk, bl, ci, cl):
        nonlocal col
        # ai, aj: block row, col of A
        # bk, bl: block row, col of B
        # ci, cl: block row, col of C (should match ai, bl)
        
        # Local to global index mapping
        # A local (r, c) -> global row 2*ai + r, col 2*aj + c
        # U index: (2*ai + r) * m + (2*aj + c)
        
        # B local (r, c) -> global row 2*bk + r, col 2*bl + c
        # V index: (2*bk + r) * p + (2*bl + c)
        
        # C local (r, c) -> global row 2*ci + r, col 2*cl + c
        # W index: (2*cl + c) * n + (2*ci + r)  <-- Note: W[k*n+i, r] convention?
        # Task says: W[k*n+i, r] where k is col of C, i is row of C?
        # "output factor follows the official trace convention W[k*n+i,r]"
        # Usually C[i, k] = sum_r U[i, r] V[k, r].
        # Here C is p x n? No, C is n x p.
        # Task: "multiplying a 4 by 4 matrix by a 4 by 6 matrix".
        # A is 4x4 (n x m). B is 4x6 (m x p). C is 4x6 (n x p).
        # W shape is (p*n, rank).
        # Index in W: k*n + i. k is col of C (0..p-1), i is row of C (0..n-1).
        # So W[k*n+i, r] corresponds to C[i, k].
        
        # Let's verify the seed program's W indexing.
        # Seed: W[k * n + i][column] = 1.
        # Loop: i in range(n), j in range(m), k in range(p).
        # U[i*m+j] = 1. V[j*p+k] = 1. W[k*n+i] = 1.
        # C[i, k] += U[i, m+j] V[j, p+k] ...
        # Yes, W index is k*n + i.
        
        # So for C block (ci, cl), local (r, c):
        # Global row i = 2*ci + r
        # Global col k = 2*cl + c
        # W index = k * n + i = (2*cl + c) * n + (2*ci + r)
        
        # Strassen M1
        # (A00+A11) * (B01+B11)
        # U: A00, A11
        # V: B01, B11
        # W: C00 (+1), C01 (-1? No, M1 is in C00 and C11)
        # C00 += M1
        # C11 += M1
        
        # Let's define the 7 terms explicitly.
        
        # M1: (A00+A11)(B01+B11)
        # U: A00(1), A11(1)
        # V: B01(1), B11(1)
        # W: C00(1), C11(1)
        
        # M2: (A10+A11)B00
        # U: A10(1), A11(1)
        # V: B00(1)
        # W: C10(1), C11(-1)
        
        # M3: A00(B10-B11)
        # U: A00(1)
        # V: B10(1), B11(-1)
        # W: C01(1), C11(1)
        
        # M4: A11(B00-B01)
        # U: A11(1)
        # V: B00(1), B01(-1)
        # W: C00(1), C10(1)
        
        # M5: (A00+A01)B11
        # U: A00(1), A01(1)
        # V: B11(1)
        # W: C00(-1), C01(1)
        
        # M6: (A10-A00)(B00+B01)
        # U: A10(1), A00(-1)
        # V: B00(1), B01(1)
        # W: C11(1)
        
        # M7: (A01-A11)(B10+B11)
        # U: A01(1), A11(-1)
        # V: B10(1), B11(1)
        # W: C00(1), C11(-1)
        
        # We add 7 rank-1 components.
        
        # Helper to add a component
        def add_component(u_indices, u_vals, v_indices, v_vals, w_indices, w_vals):
            nonlocal col
            for idx, val in zip(u_indices, u_vals):
                U[idx][col] = val
            for idx, val in zip(v_indices, v_vals):
                V[idx][col] = val
            for idx, val in zip(w_indices, w_vals):
                W[idx][col] = val
            col += 1
        
        # Define local to global index functions
        def a_idx(r, c):
            return (2*ai + r) * m + (2*aj + c)
        def b_idx(r, c):
            return (2*bk + r) * p + (2*bl + c)
        def c_idx(r, c):
            return (2*cl + c) * n + (2*ci + r)
            
        # M1
        add_component(
            [a_idx(0,0), a_idx(1,1)], [1, 1],
            [b_idx(0,1), b_idx(1,1)], [1, 1],
            [c_idx(0,0), c_idx(1,1)], [1, 1]
        )
        # M2
        add_component(
            [a_idx(1,0), a_idx(1,1)], [1, 1],
            [b_idx(0,0)], [1],
            [c_idx(1,0), c_idx(1,1)], [1, -1]
        )
        # M3
        add_component(
            [a_idx(0,0)], [1],
            [b_idx(1,0), b_idx(1,1)], [1, -1],
            [c_idx(0,1), c_idx(1,1)], [1, 1]
        )
        # M4
        add_component(
            [a_idx(1,1)], [1],
            [b_idx(0,0), b_idx(0,1)], [1, -1],
            [c_idx(0,0), c_idx(1,0)], [1, 1]
        )
        # M5
        add_component(
            [a_idx(0,0), a_idx(0,1)], [1, 1],
            [b_idx(1,1)], [1],
            [c_idx(0,0), c_idx(0,1)], [-1, 1]
        )
        # M6
        add_component(
            [a_idx(1,0), a_idx(0,0)], [1, -1],
            [b_idx(0,0), b_idx(0,1)], [1, 1],
            [c_idx(1,1)], [1]
        )
        # M7
        add_component(
            [a_idx(0,1), a_idx(1,1)], [1, -1],
            [b_idx(1,0), b_idx(1,1)], [1, 1],
            [c_idx(0,0), c_idx(1,1)], [1, -1]
        )
        
    # Iterate over all block terms
    for ai in range(2):
        for bk in range(2):
            for bl in range(3):
                # C block is at (ai, bl)
                add_strassen(ai, bk, bk, bl, ai, bl)
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
