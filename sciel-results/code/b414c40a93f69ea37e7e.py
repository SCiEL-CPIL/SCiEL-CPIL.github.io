import numpy as np
import time

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Block 1: Strassen (j=0,1; k=0,1) -> rank 7
    # Block 2: Strassen (j=0,1; k=2,3) -> rank 7
    # Block 3: Schoolbook (j=0,1; k=4,5,6) -> rank 6
    # Block 4: Schoolbook (j=2; k=0..6) -> rank 14
    # Total: 7 + 7 + 6 + 14 = 34
    
    rank = 34
    
    U = [[0] * rank for _ in range(n * m)]  # 6 x 34
    V = [[0] * rank for _ in range(m * p)]  # 21 x 34
    W = [[0] * rank for _ in range(p * n)]  # 14 x 34
    
    col = 0
    
    def add_term(u_indices, v_indices, w_indices, signs=None):
        nonlocal col
        if signs is None:
            signs = [1] * len(u_indices)
        for idx, s in zip(u_indices, signs):
            U[idx][col] += s
        # Need separate handling for V and W signs
        col += 1
    
    # Let me use a cleaner approach
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def set_term(u_vals, v_vals, w_vals):
        """u_vals: dict {index: coeff}, same for v_vals, w_vals"""
        nonlocal col
        for idx, coeff in u_vals.items():
            U[idx][col] += coeff
        for idx, coeff in v_vals.items():
            V[idx][col] += coeff
        for idx, coeff in w_vals.items():
            W[idx][col] += coeff
        col += 1
    
    # Strassen block for (j_start, k_start) where j is 0 or 1, k is 0,1,2,3
    # A indices: i*3 + j for i in {0,1}, j in {j_start, j_start+1}
    # B indices: j*7 + k for j in {j_start, j_start+1}, k in {k_start, k_start+1}
    # C indices: k*2 + i for k in {k_start, k_start+1}, i in {0,1}
    
    def add_strassen(j_start, k_start):
        nonlocal col
        # A indices
        a00 = 0 * 3 + j_start      # A[0, j_start]
        a01 = 0 * 3 + (j_start + 1) # A[0, j_start+1]
        a10 = 1 * 3 + j_start      # A[1, j_start]
        a11 = 1 * 3 + (j_start + 1) # A[1, j_start+1]
        
        # B indices
        b00 = j_start * 7 + k_start
        b01 = j_start * 7 + (k_start + 1)
        b10 = (j_start + 1) * 7 + k_start
        b11 = (j_start + 1) * 7 + (k_start + 1)
        
        # C indices: C[i, k] -> index k*2 + i
        c00 = k_start * 2 + 0
        c01 = (k_start + 1) * 2 + 0
        c10 = k_start * 2 + 1
        c11 = (k_start + 1) * 2 + 1
        
        # M1: (A00+A11)(B00+B11) -> +C00, +C11
        set_term({a00: 1, a11: 1}, {b00: 1, b11: 1}, {c00: 1, c11: 1})
        
        # M2: (A10+A11)B00 -> +C10, -C11
        set_term({a10: 1, a11: 1}, {b00: 1}, {c10: 1, c11: -1})
        
        # M3: A00(B01-B11) -> +C01, +C11
        set_term({a00: 1}, {b01: 1, b11: -1}, {c01: 1, c11: 1})
        
        # M4: A11(B10-B00) -> +C00, +C10
        set_term({a11: 1}, {b10: 1, b00: -1}, {c00: 1, c10: 1})
        
        # M5: (A00+A01)B11 -> -C00, +C01
        set_term({a00: 1, a01: 1}, {b11: 1}, {c00: -1, c01: 1})
        
        # M6: (A10-A00)(B00+B01) -> +C11
        set_term({a10: 1, a00: -1}, {b00: 1, b01: 1}, {c11: 1})
        
        # M7: (A01-A11)(B10+B11) -> +C00
        set_term({a01: 1, a11: -1}, {b10: 1, b11: 1}, {c00: 1})
    
    # Block 1: Strassen for j=0,1; k=0,1
    add_strassen(0, 0)
    
    # Block 2: Strassen for j=0,1; k=2,3
    add_strassen(0, 2)
    
    # Block 3: Schoolbook for j=0,1; k=4,5,6
    # C[i, k] += A[i, j] * B[j, k] for j in {0,1}, k in {4,5,6}
    for k in [4, 5, 6]:
        for j in [0, 1]:
            a_idx = 0 * 3 + j  # A[0, j]
            a_idx1 = 1 * 3 + j # A[1, j]
            b_idx = j * 7 + k
            c_idx0 = k * 2 + 0 # C[0, k]
            c_idx1 = k * 2 + 1 # C[1, k]
            
            # Term for C[0,k]: A[0,j] * B[j,k]
            set_term({a_idx: 1}, {b_idx: 1}, {c_idx0: 1})
            
            # Term for C[1,k]: A[1,j] * B[j,k]
            set_term({a_idx1: 1}, {b_idx: 1}, {c_idx1: 1})
    
    # Block 4: Schoolbook for j=2; k=0..6
    # C[i, k] += A[i, 2] * B[2, k] for i in {0,1}, k in 0..6
    for k in range(7):
        a_idx0 = 0 * 3 + 2  # A[0, 2]
        a_idx1 = 1 * 3 + 2  # A[1, 2]
        b_idx = 2 * 7 + k
        c_idx0 = k * 2 + 0  # C[0, k]
        c_idx1 = k * 2 + 1  # C[1, k]
        
        # Term for C[0,k]: A[0,2] * B[2,k]
        set_term({a_idx0: 1}, {b_idx: 1}, {c_idx0: 1})
        
        # Term for C[1,k]: A[1,2] * B[2,k]
        set_term({a_idx1: 1}, {b_idx: 1}, {c_idx1: 1})
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
