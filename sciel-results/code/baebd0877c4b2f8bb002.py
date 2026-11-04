"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We will construct the decomposition by summing rank-1 tensors
    # corresponding to the blocks defined by the split 5 = 2 + 3.
    
    # We use a list of tuples (u_indices, u_coeffs, v_indices, v_coeffs, w_indices, w_coeffs)
    # to represent rank-1 tensors.
    # Each rank-1 tensor is sum of products.
    # Actually, it's easier to just accumulate into U, V, W arrays.
    
    # Max rank estimate: 46
    # We'll use a dictionary or list to store the rank-1 terms, then convert to matrices.
    # Term: (u_dict, v_dict, w_dict)
    # u_dict: {index: coeff}
    # v_dict: {index: coeff}
    # w_dict: {index: coeff}
    
    terms = []
    
    def add_term(u_dict, v_dict, w_dict):
        terms.append((u_dict, v_dict, w_dict))
        
    # Helper to add a term from linear combinations
    # u: list of (index, coeff)
    # v: list of (index, coeff)
    # w: list of (index, coeff)
    def add_lu_term(u, v, w):
        u_dict = {}
        for idx, c in u:
            u_dict[idx] = u_dict.get(idx, 0) + c
        v_dict = {}
        for idx, c in v:
            v_dict[idx] = v_dict.get(idx, 0) + c
        w_dict = {}
        for idx, c in w:
            w_dict[idx] = w_dict.get(idx, 0) + c
            
        # Prune zeros
        u_dict = {k: v for k, v in u_dict.items() if v != 0}
        v_dict = {k: v for k, v in v_dict.items() if v != 0}
        w_dict = {k: v for k, v in w_dict.items() if v != 0}
        
        if u_dict and v_dict and w_dict:
            add_term(u_dict, v_dict, w_dict)

    # Block 1: j in 0..1, k in 0..1 (2x2x2)
    # Strassen
    # M1 = (A00+A11)(B00+B11) -> W00, W11
    add_lu_term(
        [(0, 1), (3, 1)],  # A00, A11 (indices: 0*2+0=0, 1*2+1=3)
        [(0, 1), (3, 1)],  # B00, B11 (indices: 0*5+0=0, 1*5+1=6? No, V index is j*p+k. j=0..1, k=0..1. 0*5+0=0, 1*5+1=6)
        [(0, 1), (1, 1)]   # C00, C11 (indices: 0*2+0=0, 1*2+1=3)
    )
    # Wait, V index for B11 is j=1, k=1. 1*5+1 = 6.
    # W index for C11 is k=1, i=1. 1*2+1 = 3.
    
    # Let's re-verify indices.
    # U: i*m + j. i in 0..1, j in 0..4.
    # V: j*p + k. j in 0..4, k in 0..4.
    # W: k*n + i. k in 0..4, i in 0..1.
    
    # Strassen M1: (A00+A11)(B00+B11)
    # A00: i=0, j=0 -> U[0]
    # A11: i=1, j=1 -> U[1*5+1] = U[6]
    # B00: j=0, k=0 -> V[0]
    # B11: j=1, k=1 -> V[1*5+1] = V[6]
    # C00: k=0, i=0 -> W[0]
    # C11: k=1, i=1 -> W[1*2+1] = W[3]
    
    add_lu_term(
        [(0, 1), (6, 1)],
        [(0, 1), (6, 1)],
        [(0, 1), (3, 1)]
    )
    
    # M2 = (A10+A11)B00 -> C10
    # A10: i=1, j=0 -> U[5]
    # A11: i=1, j=1 -> U[6]
    # B00: j=0, k=0 -> V[0]
    # C10: k=0, i=1 -> W[2]
    add_lu_term(
        [(5, 1), (6, 1)],
        [(0, 1)],
        [(2, 1)]
    )
    
    # M3 = A00(B01-B11) -> C01
    # A00: U[0]
    # B01: j=0, k=1 -> V[1]
    # B11: j=1, k=1 -> V[6]
    # C01: k=1, i=0 -> W[1]
    add_lu_term(
        [(0, 1)],
        [(1, 1), (6, -1)],
        [(1, 1)]
    )
    
    # M4 = A11(B10-B00) -> C00, C10
    # A11: U[6]
    # B10: j=1, k=0 -> V[5]
    # B00: j=0, k=0 -> V[0]
    # C00: W[0]
    # C10: W[2]
    add_lu_term(
        [(6, 1)],
        [(5, 1), (0, -1)],
        [(0, 1), (2, 1)]
    )
    
    # M5 = (A00+A01)B11 -> C00, C01
    # A00: U[0]
    # A01: i=0, j=1 -> U[1]
    # B11: V[6]
    # C00: W[0]
    # C01: W[1]
    add_lu_term(
        [(0, 1), (1, 1)],
        [(6, 1)],
        [(0, 1), (1, 1)]
    )
    
    # M6 = (A10-A00)(B00+B01) -> C11
    # A10: U[5]
    # A00: U[0]
    # B00: V[0]
    # B01: V[1]
    # C11: W[3]
    add_lu_term(
        [(5, 1), (0, -1)],
        [(0, 1), (1, 1)],
        [(3, 1)]
    )
    
    # M7 = (A01-A11)(B10+B11) -> C00, C11
    # A01: U[1]
    # A11: U[6]
    # B10: V[5]
    # B11: V[6]
    # C00: W[0]
    # C11: W[3]
    add_lu_term(
        [(1, 1), (6, -1)],
        [(5, 1), (6, 1)],
        [(0, 1), (3, 1)]
    )
    
    # Block 2: j in 0..1, k in 2..4 (2x2x3)
    # Split k into 2, 3, 4.
    # This is 3 copies of 2x2x1? No, 2x2x3.
    # We can just use schoolbook for 2x2x3? Rank 12.
    # Or use the decomposition: J(2,2,1) + J(2,2,1) + J(2,2,1)?
    # Rank 4+4+4 = 12.
    # My previous estimate was 11.
    # Let's stick to 12 for simplicity unless I can find the 11-rank one.
    # Actually, J(2,2,3) rank is 11?
    # Let's use 12 to be safe.
    
    for k_offset in [2, 3, 4]:
        k = k_offset
        # C0k = A00 B0k + A01 B1k
        # C1k = A10 B0k + A11 B1k
        
        # A00 B0k
        add_lu_term(
            [(0, 1)],
            [(0*5+k, 1)], # B0k
            [(k*2+0, 1)]  # C0k
        )
        # A01 B1k
        add_lu_term(
            [(1, 1)],
            [(1*5+k, 1)], # B1k
            [(k*2+0, 1)]  # C0k
        )
        # A10 B0k
        add_lu_term(
            [(5, 1)],
            [(0*5+k, 1)], # B0k
            [(k*2+1, 1)]  # C1k
        )
        # A11 B1k
        add_lu_term(
            [(6, 1)],
            [(1*5+k, 1)], # B1k
            [(k*2+1, 1)]  # C1k
        )
        
    # Block 3: j in 2..4, k in 0..1 (2x3x2)
    # Split j into 2, 3, 4.
    # 3 copies of 2x1x2? Rank 4 each. Total 12.
    
    for j_offset in [2, 3, 4]:
        j = j_offset
        # C00 = A0j B00
        # C01 = A0j B01
        # C10 = A1j B00
        # C11 = A1j B01
        
        # A0j B00
        add_lu_term(
            [(0*5+j, 1)], # A0j
            [(0, 1)],     # B00
            [(0*2+0, 1)]  # C00
        )
        # A0j B01
        add_lu_term(
            [(0*5+j, 1)], # A0j
            [(1, 1)],     # B01
            [(1*2+0, 1)]  # C01
        )
        # A1j B00
        add_lu_term(
            [(1*5+j, 1)], # A1j
            [(0, 1)],     # B00
            [(0*2+1, 1)]  # C10
        )
        # A1j B01
        add_lu_term(
            [(1*5+j, 1)], # A1j
            [(1, 1)],     # B01
            [(1*2+1, 1)]  # C11
        )

    # Block 4: j in 2..4, k in 2..4 (2x3x3)
    # Split j into 2,3,4 and k into 2,3,4.
    # 9 copies of 2x1x1? Rank 2 each. Total 18.
    # Previous estimate 17.
    
    for j_offset in [2, 3, 4]:
        for k_offset in [2, 3, 4]:
            j = j_offset
            k = k_offset
            # C0k = A0j B0k? No, B is j x k.
            # C0k = A0j B_jk
            # C1k = A1j B_jk
            
            # A0j B_jk
            add_lu_term(
                [(0*5+j, 1)], # A0j
                [(j*5+k, 1)], # B_jk
                [(k*2+0, 1)]  # C0k
            )
            # A1j B_jk
            add_lu_term(
                [(1*5+j, 1)], # A1j
                [(j*5+k, 1)], # B_jk
                [(k*2+1, 1)]  # C1k
            )

    # Now convert terms to U, V, W matrices
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c, (u_dict, v_dict, w_dict) in enumerate(terms):
        for idx, coeff in u_dict.items():
            U[idx][c] = coeff
        for idx, coeff in v_dict.items():
            V[idx][c] = coeff
        for idx, coeff in w_dict.items():
            W[idx][c] = coeff
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
