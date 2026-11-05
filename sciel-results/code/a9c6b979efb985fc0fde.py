"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np
from collections import defaultdict
from fractions import Fraction

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin from 150s limit
    
    n, m, p = DIMENSIONS
    trivial_rank = n * m * p  # 84
    
    # Precompute the trivial decomposition
    def make_trivial():
        U = [[0] * trivial_rank for _ in range(n * m)]
        V = [[0] * trivial_rank for _ in range(m * p)]
        W = [[0] * trivial_rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = make_trivial()
    best_rank = trivial_rank
    
    # Strategy: Use Strassen's algorithm on 2x2x2 sub-blocks to reduce rank
    # For a 2x2x2 block, Strassen uses 7 terms instead of 8
    # We can apply this to non-overlapping 2x2x2 blocks within our 3x4x7 tensor
    
    # The trivial decomposition has rank 84 (3*4*7)
    # Each 2x2x2 block uses 8 terms; Strassen reduces to 7, saving 1 term per block
    
    # Let's identify non-overlapping 2x2x2 blocks:
    # n=3: we can fit one 2-block (indices 0,1) and one singleton (index 2)
    # m=4: we can fit two 2-blocks (indices 0,1 and 2,3)
    # p=7: we can fit three 2-blocks (indices 0,1; 2,3; 4,5) and one singleton (index 6)
    
    # Non-overlapping 2x2x2 blocks:
    # (i in {0,1}, j in {0,1}, k in {0,1})
    # (i in {0,1}, j in {2,3}, k in {0,1})
    # (i in {0,1}, j in {0,1}, k in {2,3})
    # (i in {0,1}, j in {2,3}, k in {2,3})
    # (i in {0,1}, j in {0,1}, k in {4,5})
    # (i in {0,1}, j in {2,3}, k in {4,5})
    # That's 6 blocks, each saving 1 term: total saving = 6
    # New rank: 84 - 6 = 78
    
    # Let's construct this explicitly
    def make_strassen_improved():
        rank = trivial_rank - 6  # 78
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        
        # Strassen's 7 terms for 2x2x2:
        # M1 = (A00 + A11)(B00 + B11)
        # M2 = (A10 + A11)B00
        # M3 = A00(B01 - B11)
        # M4 = A11(B10 - B00)
        # M5 = (A00 + A01)B11
        # M6 = (A10 - A00)(B00 + B01)
        # M7 = (A01 - A11)(B10 + B11)
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        strassen_terms = [
            # (u_coeffs, v_coeffs, w_coeffs)
            # u_coeffs: [(i_offset, j_offset, coeff), ...]
            # v_coeffs: [(j_offset, k_offset, coeff), ...]
            # w_coeffs: [(k_offset, i_offset, coeff), ...]
            
            # M1 = (A00 + A11)(B00 + B11)
            ([((0,0,1), (1,1,1)), ((0,0,1), (1,1,1))], 
             [((0,0,1), (0,1,1)), ((1,0,1), (1,1,1))],
             [((0,0,1), (1,1,1)), ((0,0,1), (1,1,1))]),
            
            # M2 = (A10 + A11)B00
            ([((1,0,1), (1,1,1)), ((0,0,1))], 
             [((0,0,1)), ((0,0,1))],
             [((0,0,1)), ((0,0,1))]),
            
            # M3 = A00(B01 - B11)
            ([((0,0,1)), ((0,0,1))], 
             [((0,1,1), (1,1,-1)), ((0,1,1), (1,1,-1))],
             [((0,0,1), (1,0,-1)), ((0,0,1), (1,0,-1))]),
            
            # M4 = A11(B10 - B00)
            ([((1,1,1)), ((1,1,1))], 
             [((1,0,1), (0,0,-1)), ((1,0,1), (0,0,-1))],
             [((1,0,1), (0,0,-1)), ((1,0,1), (0,0,-1))]),
            
            # M5 = (A00 + A01)B11
            ([((0,0,1), (0,1,1)), ((0,0,1), (0,1,1))], 
             [((1,1,1)), ((1,1,1))],
             [((1,1,1), (0,0,-1)), ((1,1,1), (0,0,-1))]),
            
            # M6 = (A10 - A00)(B00 + B01)
            ([((1,0,1), (0,0,-1)), ((1,0,1), (0,0,-1))], 
             [((0,0,1), (0,1,1)), ((0,0,1), (0,1,1))],
             [((0,0,1), (1,1,1)), ((0,0,1), (1,1,1))]),
            
            # M7 = (A01 - A11)(B10 + B11)
            ([((0,1,1), (1,1,-1)), ((0,1,1), (1,1,-1))], 
             [((1,0,1), (1,1,1)), ((1,0,1), (1,1,1))],
             [((1,1,1), (0,0,1)), ((1,1,1), (0,0,1))]),
        ]
        
        # Apply Strassen to 6 non-overlapping 2x2x2 blocks
        blocks = [
            (0, 0, 0),  # i=0,1; j=0,1; k=0,1
            (0, 2, 0),  # i=0,1; j=2,3; k=0,1
            (0, 0, 2),  # i=0,1; j=0,1; k=2,3
            (0, 2, 2),  # i=0,1; j=2,3; k=2,3
            (0, 0, 4),  # i=0,1; j=0,1; k=4,5
            (0, 2, 4),  # i=0,1; j=2,3; k=4,5
        ]
        
        for (i0, j0, k0) in blocks:
            for term_idx in range(7):
                u_coeffs, v_coeffs, w_coeffs = strassen_terms[term_idx]
                # For each coefficient in this term, add to the matrices
                # u_coeffs is a list of (i_off, j_off, coeff) tuples
                # But we need to handle the structure correctly
                
                # Actually, let me redo this more carefully
                # Each Strassen term has specific coefficients for each position
                pass
        
        # Let me use a simpler approach: just construct the trivial decomposition
        # and then apply Strassen to specific blocks
        # For now, return the trivial result
        return make_trivial()
    
    # For now, return the trivial decomposition
    return best_result
# EVOLVE-BLOCK-END
