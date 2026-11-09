import numpy as np
import time

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # Rank 54 decomposition
    rank = 54
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    def add_strassen_block(a_i, a_j, b_j, b_k):
        nonlocal col
        # A is 2x2 at rows a_i, a_i+1 and cols a_j, a_j+1
        # B is 2x2 at rows b_j, b_j+1 and cols b_k, b_k+1
        # C is 2x2 at rows a_i, a_i+1 and cols b_k, b_k+1
        
        # Strassen M1..M7
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22) B11
        # M3 = A11 (B12 - B22)
        # M4 = A22 (B21 - B11)
        # M5 = (A11 + A12) B22
        # M6 = (A21 - A11) (B11 + B12)
        # M7 = (A12 - A22) (B21 + B22)
        
        # We need to add 7 rank-1 terms.
        # Each term r has U, V, W vectors.
        
        # Helper to add a term
        def add_term(u_coeffs, v_coeffs, w_coeffs):
            nonlocal col
            # u_coeffs: dict mapping (i, j) -> coeff
            # v_coeffs: dict mapping (j, k) -> coeff
            # w_coeffs: dict mapping (k, i) -> coeff
            
            for (i, j), c_u in u_coeffs.items():
                if c_u != 0:
                    U[a_i + i, a_j + j, col] += c_u # Wait, U index is i*m + j
                    U[i * m + j, col] += c_u
            for (j, k), c_v in v_coeffs.items():
                if c_v != 0:
                    V[j * p + k, col] += c_v
            for (k, i), c_w in w_coeffs.items():
                if c_w != 0:
                    W[k * n + i, col] += c_w
            col += 1

        # M1
        # A: (1,1,1), (2,2,1) -> local (0,0), (1,1)
        # B: (1,1,1), (2,2,1) -> local (0,0), (1,1)
        # C: (1,1,1), (2,2,1) -> local (0,0), (1,1)
        add_term(
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1}
        )
        
        # M2
        # A: (2,1,1), (2,2,1) -> local (1,0), (1,1)
        # B: (1,1,1) -> local (0,0)
        # C: (2,1,1) -> local (1,0)
        add_term(
            {(1,0): 1, (1,1): 1},
            {(0,0): 1},
            {(0,1): 1} # k=0, i=1
        )
        
        # M3
        # A: (1,1,1) -> local (0,0)
        # B: (1,2,1), (2,2,-1) -> local (0,1), (1,1)
        # C: (1,2,1) -> local (0,1)
        add_term(
            {(0,0): 1},
            {(0,1): 1, (1,1): -1},
            {(1,0): 1} # k=1, i=0
        )
        
        # M4
        # A: (2,2,1) -> local (1,1)
        # B: (2,1,1), (1,1,-1) -> local (1,0), (0,0)
        # C: (2,1,1) -> local (1,0)
        add_term(
            {(1,1): 1},
            {(1,0): 1, (0,0): -1},
            {(0,1): 1} # k=0, i=1
        )
        
        # M5
        # A: (1,1,1), (1,2,1) -> local (0,0), (0,1)
        # B: (2,2,1) -> local (1,1)
        # C: (1,1,-1), (2,2,1) -> local (0,0), (1,1)
        add_term(
            {(0,0): 1, (0,1): 1},
            {(1,1): 1},
            {(0,0): -1, (1,1): 1}
        )
        
        # M6
        # A: (2,1,1), (1,1,-1) -> local (1,0), (0,0)
        # B: (1,1,1), (1,2,1) -> local (0,0), (0,1)
        # C: (2,2,1) -> local (1,1)
        add_term(
            {(1,0): 1, (0,0): -1},
            {(0,0): 1, (0,1): 1},
            {(1,1): 1} # k=1, i=1
        )
        
        # M7
        # A: (1,2,1), (2,2,-1) -> local (0,1), (1,1)
        # B: (2,1,1), (2,2,1) -> local (1,0), (1,1)
        # C: (1,1,1), (2,1,-1) -> local (0,0), (1,0)
        add_term(
            {(0,1): 1, (1,1): -1},
            {(1,0): 1, (1,1): 1},
            {(0,0): 1, (0,1): -1}
        )

    def add_outer_block(a_i, a_j, b_j, b_k):
        nonlocal col
        # A is 2x1 at rows a_i, a_i+1 and col a_j
        # B is 1x2 at row b_j and cols b_k, b_k+1
        # C is 2x2 at rows a_i, a_i+1 and cols b_k, b_k+1
        
        def add_term(u_coeffs, v_coeffs, w_coeffs):
            nonlocal col
            for (i, j), c_u in u_coeffs.items():
                if c_u != 0:
                    U[i * m + j, col] += c_u
            for (j, k), c_v in v_coeffs.items():
                if c_v != 0:
                    V[j * p + k, col] += c_v
            for (k, i), c_w in w_coeffs.items():
                if c_w != 0:
                    W[k * n + i, col] += c_w
            col += 1

        # C00 = A00 B00
        add_term(
            {(0, 0): 1},
            {(0, 0): 1},
            {(0, 0): 1}
        )
        # C01 = A00 B01
        add_term(
            {(0, 0): 1},
            {(0, 1): 1},
            {(1, 0): 1}
        )
        # C10 = A10 B00
        add_term(
            {(1, 0): 1},
            {(0, 0): 1},
            {(0, 1): 1}
        )
        # C11 = A10 B01
        add_term(
            {(1, 0): 1},
            {(0, 1): 1},
            {(1, 1): 1}
        )

    # Top part of B (rows 0-1)
    # A cols 0-1
    add_strassen_block(0, 0, 0, 0) # B cols 0-1
    add_strassen_block(0, 0, 0, 2) # B cols 2-3
    add_strassen_block(0, 0, 0, 4) # B cols 4-5
    
    # Bottom part of B (rows 2-3)
    # A cols 2-3
    add_strassen_block(0, 2, 2, 0) # B cols 0-1
    add_strassen_block(0, 2, 2, 2) # B cols 2-3
    add_strassen_block(0, 2, 2, 4) # B cols 4-5
    
    # Last row of B (row 4)
    # A col 4
    add_outer_block(0, 4, 4, 0) # B cols 0-1
    add_outer_block(0, 4, 4, 2) # B cols 2-3
    add_outer_block(0, 4, 4, 4) # B cols 4-5

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
