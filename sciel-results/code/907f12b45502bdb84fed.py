"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=7
    
    # We will build the tensor decomposition term by term.
    # U: (n*m, rank) = (16, rank)
    # V: (m*p, rank) = (28, rank)
    # W: (p*n, rank) = (28, rank)
    
    # We use a list of tuples to store terms, then convert to matrices.
    # Term: (u_dict, v_dict, w_dict) where keys are flattened indices.
    # u_dict: {i*m + j: coeff}
    # v_dict: {j*p + k: coeff}
    # w_dict: {k*n + i: coeff}
    
    terms = []

    def add_term(u_dict, v_dict, w_dict):
        terms.append((u_dict, v_dict, w_dict))

    # Helper to add a schoolbook term for a specific block multiplication
    # X is a 2x2 block of A, Y is a 2x2 or 2x3 block of B, Z is a 2x2 or 2x3 block of C
    # block_i, block_j: position of X in A (0,0 to 1,1)
    # block_j2, block_k: position of Y in B (0,0 to 1,1 for left, 0,0 to 1,0 for right? No, B is 4x7)
    # Let's handle B indices carefully.
    # B is 4x7.
    # Left part (cols 0-3): blocks (0,0), (0,1), (1,0), (1,1) of size 2x2.
    # Right part (cols 4-6): blocks (0,0), (1,0) of size 2x3.
    
    def add_schoolbook_2x2x2(block_i, block_j, block_j2, block_k, coeff=1):
        """
        Adds terms for X(block_i, block_j) * Y(block_j2, block_k) -> Z
        X is A[2*block_i:2*block_i+2, 2*block_j:2*block_j+2]
        Y is B[2*block_j2:2*block_j2+2, 2*block_k:2*block_k+2]
        Z is C[2*block_i:2*block_i+2, 2*block_k:2*block_k+2]
        """
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    i = 2*block_i + di
                    j = 2*block_j + dj
                    k = 2*block_k + dk
                    
                    # Wait, Y index j2 must match X index j?
                    # In matrix mult C_ik = sum_j A_ij B_jk.
                    # Here A is 4x4, B is 4x7.
                    # A_ij: i in 0..3, j in 0..3.
                    # B_jk: j in 0..3, k in 0..6.
                    # So j is the shared index.
                    # block_j for A and block_j2 for B must be the same block row index.
                    if block_j != block_j2:
                        return # Mismatch, shouldn't happen in correct usage
                    
                    # U index
                    u_idx = i * m + j
                    # V index
                    v_idx = j * p + k
                    # W index
                    w_idx = k * n + i
                    
                    u_dict = {u_idx: coeff}
                    v_dict = {v_idx: coeff}
                    w_dict = {w_idx: coeff}
                    add_term(u_dict, v_dict, w_dict)

    def add_schoolbook_2x2x3(block_i, block_j, block_j2, block_k, coeff=1):
        """
        X is 2x2, Y is 2x3, Z is 2x3.
        Y is B[2*block_j2:2*block_j2+2, 4+block_k:4+block_k+3] ?
        Let's assume block_k is the column index within the 3-column part.
        Actually, B_right is 4x3.
        Block (0,0) is B[0:2, 4:7].
        Block (1,0) is B[2:4, 4:7].
        So block_k is always 0 for the right part?
        And the columns are 4,5,6.
        """
        if block_j != block_j2:
            return
            
        for di in range(2):
            for dj in range(2):
                for dk in range(3):
                    i = 2*block_i + di
                    j = 2*block_j + dj
                    k = 4 + dk # Columns 4,5,6
                    
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    
                    u_dict = {u_idx: coeff}
                    v_dict = {v_idx: coeff}
                    w_dict = {w_idx: coeff}
                    add_term(u_dict, v_dict, w_dict)

    # --- Part 1: 4x4 * 4x4 -> 4x4 (Left part of C) ---
    # Use Strassen for the 4x4x4 multiplication.
    # Blocks:
    # A11 (0,0), A12 (0,1), A21 (1,0), A22 (1,1)
    # B11 (0,0), B12 (0,1), B21 (1,0), B22 (1,1)
    # C11 (0,0), C12 (0,1), C21 (1,0), C22 (1,1)
    
    # Strassen Steps:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # We need a function to add the 7-term decomposition of 2x2x2
    # given linear combinations of A and B blocks.
    
    def add_strassen_step(a_coeffs, b_coeffs, c_coeffs):
        """
        a_coeffs: dict { (bi, bj): coeff } for A blocks
        b_coeffs: dict { (bj, bk): coeff } for B blocks
        c_coeffs: dict { (bi, bk): coeff } for C blocks (output)
        """
        # The 7 terms of Strassen for 2x2x2
        # Each term t has u_t (4 coeffs for X), v_t (4 coeffs for Y), w_t (4 coeffs for Z)
        # X is 2x2, Y is 2x2, Z is 2x2.
        
        strassen_terms = [
            # u, v, w
            ([1, 0, 1, 0], [1, 0, 1, 0], [1, 0, 0, 1]), # M1
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]), # M2
            ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),  # M3
            ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),  # M4
            ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),  # M5
            ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),  # M6
            ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])   # M7
        ]
        
        for u_t, v_t, w_t in strassen_terms:
            u_dict = {}
            v_dict = {}
            w_dict = {}
            
            # Expand u_t (coeffs for X) to A blocks
            # X[di, dj] = sum_{bi, bj} a_coeffs[(bi,bj)] * A[2bi+di, 2bj+dj]
            # u_t is indexed by X's flattened index: di*2 + dj
            for idx, coeff_x in enumerate(u_t):
                if coeff_x == 0: continue
                di = idx // 2
                dj = idx % 2
                for (bi, bj), coeff_a in a_coeffs.items():
                    if coeff_a == 0: continue
                    i = 2*bi + di
                    j = 2*bj + dj
                    u_idx = i * m + j
                    u_dict[u_idx] = u_dict.get(u_idx, 0) + coeff_x * coeff_a
            
            # Expand v_t (coeffs for Y) to B blocks
            for idx, coeff_y in enumerate(v_t):
                if coeff_y == 0: continue
                dj = idx // 2
                dk = idx % 2
                for (bj, bk), coeff_b in b_coeffs.items():
                    if coeff_b == 0: continue
                    j = 2*bj + dj
                    k = 2*bk + dk
                    v_idx = j * p + k
                    v_dict[v_idx] = v_dict.get(v_idx, 0) + coeff_y * coeff_b
            
            # Expand w_t (coeffs for Z) to C blocks
            for idx, coeff_z in enumerate(w_t):
                if coeff_z == 0: continue
                di = idx // 2
                dk = idx % 2
                for (bi, bk), coeff_c in c_coeffs.items():
                    if coeff_c == 0: continue
                    i = 2*bi + di
                    k = 2*bk + dk
                    w_idx = k * n + i
                    w_dict[w_idx] = w_dict.get(w_idx, 0) + coeff_z * coeff_c
            
            add_term(u_dict, v_dict, w_dict)

    # Define Strassen steps for 4x4x4
    # M1 = (A11+A22)(B11+B22) -> C11 (+1), C22 (+1)
    add_strassen_step(
        {(0,0): 1, (1,1): 1},
        {(0,0): 1, (1,1): 1},
        {(0,0): 1, (1,1): 1}
    )
    
    # M2 = (A21+A22)B11 -> C21 (+1), C22 (-1)
    add_strassen_step(
        {(1,0): 1, (1,1): 1},
        {(0,0): 1},
        {(1,0): 1, (1,1): -1}
    )
    
    # M3 = A11(B12-B22) -> C12 (+1), C22 (+1)
    add_strassen_step(
        {(0,0): 1},
        {(0,1): 1, (1,1): -1},
        {(0,1): 1, (1,1): 1}
    )
    
    # M4 = A22(B21-B11) -> C11 (+1), C21 (+1)
    add_strassen_step(
        {(1,1): 1},
        {(1,0): 1, (0,0): -1},
        {(0,0): 1, (1,0): 1}
    )
    
    # M5 = (A11+A12)B22 -> C12 (+1), C11 (-1)
    add_strassen_step(
        {(0,0): 1, (0,1): 1},
        {(1,1): 1},
        {(0,1): 1, (0,0): -1}
    )
    
    # M6 = (A21-A11)(B11+B12) -> C22 (+1)
    add_strassen_step(
        {(1,0): 1, (0,0): -1},
        {(0,0): 1, (0,1): 1},
        {(1,1): 1}
    )
    
    # M7 = (A12-A22)(B21+B22) -> C11 (+1)
    add_strassen_step(
        {(0,1): 1, (1,1): -1},
        {(1,0): 1, (1,1): 1},
        {(0,0): 1}
    )

    # --- Part 2: 4x4 * 4x3 -> 4x3 (Right part of C) ---
    # Schoolbook
    # C13 = A11 B13 + A12 B23
    # C23 = A21 B13 + A22 B23
    
    # B13 is B[0:2, 4:7] -> block (0,0) of right part
    # B23 is B[2:4, 4:7] -> block (1,0) of right part
    
    # A11 B13 -> C13
    add_schoolbook_2x2x3(0, 0, 0, 0, 1)
    # A12 B23 -> C13
    add_schoolbook_2x2x3(0, 1, 1, 0, 1)
    
    # A21 B13 -> C23
    add_schoolbook_2x2x3(1, 0, 0, 0, 1)
    # A22 B23 -> C23
    add_schoolbook_2x2x3(1, 1, 1, 0, 1)

    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, (u_dict, v_dict, w_dict) in enumerate(terms):
        for idx, coeff in u_dict.items():
            U[idx][r] = coeff
        for idx, coeff in v_dict.items():
            V[idx][r] = coeff
        for idx, coeff in w_dict.items():
            W[idx][r] = coeff
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
