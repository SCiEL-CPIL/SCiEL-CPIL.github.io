"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from collections import defaultdict

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Helper to create sparse matrices
    def make_sparse(shape, entries):
        mat = np.zeros(shape, dtype=np.int32)
        for r, c, val in entries:
            mat[r, c] = val
        return mat

    # Strassen 2x2x2 decomposition (Rank 7)
    # A indices: 00,01,10,11 -> 0,1,2,3
    # B indices: 00,01,10,11 -> 0,1,2,3
    # C indices: 00,01,10,11 -> 0,1,2,3
    U_222_entries = [
        (0, 0, 1), (3, 0, 1), # M1: (A00+A11)
        (2, 1, 1), (3, 1, 1), # M2: (A10+A11)
        (0, 2, 1),            # M3: A00
        (3, 3, 1),            # M4: A11
        (0, 4, 1), (1, 4, 1), # M5: (A00+A01)
        (2, 5, 1), (0, 5, -1),# M6: (A10-A00)
        (1, 6, 1), (3, 6, -1) # M7: (A01-A11)
    ]
    V_222_entries = [
        (0, 0, 1), (3, 0, 1), # M1: (B00+B11)
        (0, 1, 1),            # M2: B00
        (1, 2, 1), (3, 2, -1),# M3: (B01-B11)
        (2, 3, 1), (0, 3, -1),# M4: (B10-B00)
        (3, 4, 1),            # M5: B11
        (0, 5, 1), (1, 5, 1), # M6: (B00+B01)
        (2, 6, 1), (3, 6, 1)  # M7: (B10+B11)
    ]
    W_222_entries = [
        (0, 0, 1), (3, 0, 1), # C00, C11 get M1
        (2, 1, 1), (3, 1, -1),# C10 gets M2, C11 gets -M2
        (1, 2, 1),            # C01 gets M3
        (0, 3, 1), (2, 3, 1), # C00 gets M4, C10 gets M4
        (0, 4, -1), (1, 4, 1),# C00 gets -M5, C01 gets M5
        (3, 5, 1),            # C11 gets M6
        (0, 6, 1)             # C00 gets M7
    ]
    
    U_222 = make_sparse((4, 7), U_222_entries)
    V_222 = make_sparse((4, 7), V_222_entries)
    W_222 = make_sparse((4, 7), W_222_entries)
    
    # 2x2x1 decomposition (Rank 4)
    # A indices: 0,1,2,3
    # B indices: 0,1 (since p=1, B is 2x1, indices 0,1)
    # C indices: 0,1 (since n=2, C is 2x1, indices 0,1)
    U_221_entries = [
        (0, 0, 1), # A00
        (1, 1, 1), # A01
        (2, 2, 1), # A10
        (3, 3, 1)  # A11
    ]
    V_221_entries = [
        (0, 0, 1), # B00
        (1, 1, 1), # B10
        (0, 2, 1), # B00
        (1, 3, 1)  # B10
    ]
    W_221_entries = [
        (0, 0, 1), # C00
        (0, 1, 1), # C00
        (1, 2, 1), # C10
        (1, 3, 1)  # C10
    ]
    
    U_221 = make_sparse((4, 4), U_221_entries)
    V_221 = make_sparse((2, 4), V_221_entries)
    W_221 = make_sparse((2, 4), W_221_entries)
    
    # Combine to 2x2x5 (Rank 7+7+4=18)
    # Split B (2x5) into B1 (2x2), B2 (2x2), B3 (2x1)
    # U is same for all parts (A is 2x2)
    # V is concatenated
    # W is concatenated
    
    U_225 = np.zeros((4, 18), dtype=np.int32)
    V_225 = np.zeros((2, 18), dtype=np.int32)
    W_225 = np.zeros((2, 18), dtype=np.int32)
    
    # Part 1: 2x2x2 (cols 0-6)
    U_225[:, 0:7] = U_222
    V_225[:, 0:7] = V_222
    W_225[:, 0:7] = W_222
    
    # Part 2: 2x2x2 (cols 7-13)
    U_225[:, 7:14] = U_222
    V_225[:, 7:14] = V_222
    W_225[:, 7:14] = W_222
    
    # Part 3: 2x2x1 (cols 14-17)
    U_225[:, 14:18] = U_221
    V_225[:, 14:18] = V_221
    W_225[:, 14:18] = W_221
    
    # Combine to 2x4x5 (Rank 18+18=36)
    # A (2x4) split into A_L (2x2), A_R (2x2)
    # B (4x5) split into B_T (2x5), B_B (2x5)
    # C (2x5) is sum of A_L B_T and A_R B_B
    
    U_245 = np.zeros((8, 36), dtype=np.int32)
    V_245 = np.zeros((20, 36), dtype=np.int32)
    W_245 = np.zeros((10, 36), dtype=np.int32)
    
    # Block 1: A_L (rows 0-3 of U) * B_T (rows 0-1 of V) -> C (rows 0-1 of W)
    U_245[0:4, 0:18] = U_225
    V_245[0:2, 0:18] = V_225
    W_245[0:2, 0:18] = W_225
    
    # Block 2: A_R (rows 4-7 of U) * B_B (rows 2-3 of V) -> C (rows 0-1 of W)
    U_245[4:8, 18:36] = U_225
    V_245[2:4, 18:36] = V_225
    W_245[0:2, 18:36] = W_225
    
    return {"U": U_245, "V": V_245, "W": W_245}
# EVOLVE-BLOCK-END
