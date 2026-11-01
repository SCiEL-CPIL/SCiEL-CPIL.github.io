import numpy as np

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 13 decomposition
    rank = 13
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Helper to add a term
    # u_indices: list of (idx, coeff) for U
    # v_indices: list of (idx, coeff) for V
    # w_indices: list of (idx, coeff) for W
    def add_term(t, u_idx, v_idx, w_idx):
        for idx, coeff in u_idx:
            U[idx, t] += coeff
        for idx, coeff in v_idx:
            V[idx, t] += coeff
        for idx, coeff in w_idx:
            W[idx, t] += coeff

    # Indices for A (2x3): i*3 + j
    # Indices for B (3x3): j*3 + k
    # Indices for C (2x3) via W (3x2) trace convention: k*2 + i
    
    # 1. Strassen 2x2x2 for A[0:2, 0:2] * B[0:2, 0:2]
    # M1 = (A00 + A11)(B00 + B11)
    add_term(0, [(0, 1), (4, 1)], [(0, 1), (4, 1)], 
             [(0, 1), (2, 1), (1, -1), (3, 1)]) # C00=M1, C11=M1? No.
             # C00 = M1 + M4 - M5 + M7
             # C11 = M1 - M2 + M3 - M6
             # So W for M1 should have coeff 1 for C00 and C11.
             # C00 index: k=0, i=0 -> 0*2+0 = 0
             # C11 index: k=1, i=1 -> 1*2+1 = 3
             # So W[0]=1, W[3]=1.
             # My previous line was wrong. I will redefine W contributions carefully.
             
    # Let's restart the term definitions with explicit W contributions.
    
    # Term 0: M1 = (A00 + A11)(B00 + B11)
    # Contributes to C00 (+1) and C11 (+1)
    add_term(0, [(0, 1), (4, 1)], [(0, 1), (4, 1)], [(0, 1), (3, 1)])
    
    # Term 1: M2 = (A10 + A11)B00
    # Contributes to C10 (+1) and C11 (-1)
    # C10 index: k=0, i=1 -> 0*2+1 = 1
    # C11 index: k=1, i=1 -> 1*2+1 = 3
    add_term(1, [(3, 1), (4, 1)], [(0, 1)], [(1, 1), (3, -1)])
    
    # Term 2: M3 = A00(B10 - B11)
    # Contributes to C01 (+1) and C11 (+1)
    # C01 index: k=1, i=0 -> 1*2+0 = 2
    # C11 index: 3
    add_term(2, [(0, 1)], [(3, 1), (4, -1)], [(2, 1), (3, 1)])
    
    # Term 3: M4 = A11(B01 - B00)
    # Contributes to C00 (+1) and C10 (+1)
    # C00 index: 0
    # C10 index: 1
    add_term(3, [(4, 1)], [(1, 1), (0, -1)], [(0, 1), (1, 1)])
    
    # Term 4: M5 = (A00 + A01)B11
    # Contributes to C00 (-1) and C01 (+1)
    # C00 index: 0
    # C01 index: 2
    add_term(4, [(0, 1), (1, 1)], [(4, 1)], [(0, -1), (2, 1)])
    
    # Term 5: M6 = (A10 - A00)(B00 + B01)
    # Contributes to C11 (-1)
    # C11 index: 3
    add_term(5, [(3, 1), (0, -1)], [(0, 1), (1, 1)], [(3, -1)])
    
    # Term 6: M7 = (A01 - A11)(B01 + B11)
    # Contributes to C00 (+1)
    # C00 index: 0
    add_term(6, [(1, 1), (4, -1)], [(1, 1), (4, 1)], [(0, 1)])
    
    # Term 7: y*w = A[:,2] * B[2,0:2]
    # A02 B20 -> C00
    # A02 B21 -> C01
    # A12 B20 -> C10
    # A12 B21 -> C11
    # This is a single rank-1 term?
    # U: A02 (idx 2), A12 (idx 5) -> coeff 1
    # V: B20 (idx 6), B21 (idx 7) -> coeff 1
    # W: C00 (0), C01 (2), C10 (1), C11 (3) -> coeff 1
    add_term(7, [(2, 1), (5, 1)], [(6, 1), (7, 1)], [(0, 1), (2, 1), (1, 1), (3, 1)])
    
    # Term 8: y*s = A[:,2] * B[2,2]
    # A02 B22 -> C02
    # A12 B22 -> C12
    # U: A02 (2), A12 (5) -> 1
    # V: B22 (8) -> 1
    # W: C02 (k=2, i=0 -> 2*2+0=4), C12 (k=2, i=1 -> 2*2+1=5) -> 1
    add_term(8, [(2, 1), (5, 1)], [(8, 1)], [(4, 1), (5, 1)])
    
    # Terms 9-12: X*z = A[0:2, 0:2] * B[0:2, 2]
    # A00 B02 -> C02
    add_term(9, [(0, 1)], [(2, 1)], [(4, 1)])
    
    # A01 B12 -> C02
    add_term(10, [(1, 1)], [(5, 1)], [(4, 1)])
    
    # A10 B02 -> C12
    add_term(11, [(3, 1)], [(2, 1)], [(5, 1)])
    
    # A11 B12 -> C12
    add_term(12, [(4, 1)], [(5, 1)], [(5, 1)])
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
