"""Program for matrix multiplication 2x3x7 tensor decomposition."""
import time
import numpy as np
from fractions import Fraction
import random

DIMENSIONS = [2, 3, 7]

def _verify(U, V, W, n=2, m=3, p=7):
    """Verify the decomposition. U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)"""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                # sum_r U[idx_u, r] * V[idx_v, r] * W[idx_w, r]
                s = np.sum(U[idx_u] * V[idx_v] * W[idx_w])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _build_strassen_222():
    """
    Returns U, V, W for 2x2x2 matrix multiplication using Strassen's algorithm.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indices:
    A: [[0, 1], [2, 3]] -> U rows 0,1,2,3
    B: [[4, 5], [6, 7]] -> V rows 0,1,2,3 (local)
    C: [[8, 9], [10, 11]] -> W rows 0,1,2,3 (local)
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Strassen's 7 products:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # U entries (A)
    U[0, 0] = 1; U[3, 0] = 1 # A00, A11
    U[2, 1] = 1; U[3, 1] = 1 # A10, A11
    U[0, 2] = 1              # A00
    U[3, 3] = 1              # A11
    U[0, 4] = 1; U[1, 4] = 1 # A00, A01
    U[2, 5] = 1; U[0, 5] = -1 # A10, A00
    U[1, 6] = 1; U[3, 6] = -1 # A01, A11
    
    # V entries (B)
    # B indices: 0:00, 1:01, 2:10, 3:11
    V[0, 0] = 1; V[3, 0] = 1 # B00, B11
    V[0, 1] = 1              # B00
    V[1, 2] = 1; V[3, 2] = -1 # B01, B11
    V[2, 3] = 1; V[0, 3] = -1 # B10, B00
    V[3, 4] = 1              # B11
    V[0, 5] = 1; V[1, 5] = 1 # B00, B01
    V[2, 6] = 1; V[3, 6] = 1 # B10, B11
    
    # W entries (C)
    # C indices: 0:00, 1:01, 2:10, 3:11
    W[0, 0] = 1; W[3, 0] = 1 # C00, C11
    W[2, 1] = 1              # C10
    W[1, 2] = 1              # C01
    W[3, 3] = 1              # C11
    W[3, 4] = 1              # C11
    W[2, 5] = 1; W[0, 5] = 1 # C10, C00
    W[1, 6] = 1; W[3, 6] = 1 # C01, C11
    
    return U, V, W

def _build_schoolbook_221():
    """
    Returns U, V, W for 2x2x1 matrix multiplication (schoolbook).
    U: (4, 4), V: (2, 4), W: (2, 4)
    A: 2x2, B: 2x1, C: 2x1
    """
    U = np.zeros((4, 4))
    V = np.zeros((2, 4))
    W = np.zeros((2, 4))
    
    # C0 = A00*B0 + A01*B1
    # C1 = A10*B0 + A11*B1
    # Rank 4
    
    # Term 0: A00 * B0 -> C0
    U[0, 0] = 1
    V[0, 0] = 1
    W[0, 0] = 1
    
    # Term 1: A01 * B1 -> C0
    U[1, 1] = 1
    V[1, 1] = 1
    W[0, 1] = 1
    
    # Term 2: A10 * B0 -> C1
    U[2, 2] = 1
    V[0, 2] = 1
    W[1, 2] = 1
    
    # Term 3: A11 * B1 -> C1
    U[3, 3] = 1
    V[1, 3] = 1
    W[1, 3] = 1
    
    return U, V, W

def _build_block_232():
    """
    Builds decomposition for 2x3x2 multiplication.
    A: 2x3, B: 3x2, C: 2x2.
    Rank 11.
    """
    # Split A into A_L (2x2) and A_R (2x1)
    # Split B into B_T (2x2) and B_B (1x2)
    # C = A_L B_T + A_R B_B
    
    U1, V1, W1 = _build_strassen_222() # Rank 7
    U2, V2, W2 = _build_schoolbook_221() # Rank 4
    
    U = np.zeros((6, 11))
    V = np.zeros((6, 11)) # B is 3x2 -> 6 entries
    W = np.zeros((4, 11)) # C is 2x2 -> 4 entries
    
    # Map indices for A_L B_T (Strassen part)
    # A_L rows: 0,1,2,3 (local) -> Global A rows: 0,1,3,4 (cols 0,1)
    # A indices: 0:00, 1:01, 2:10, 3:11
    # Global A: i*3+j
    # 0:0, 1:1, 3:3, 4:4
    a_map_1 = [0, 1, 3, 4]
    
    # B_T cols: 0,1,2,3 (local) -> Global B rows: 0,1 (cols 0,1)
    # B indices: 0:00, 1:01, 2:10, 3:11
    # Global B: j*2+k
    # 0:0, 1:1, 2:2, 3:3
    b_map_1 = [0, 1, 2, 3]
    
    # C rows: 0,1,2,3 (local) -> Global C rows: 0,1,2,3
    # C indices: 0:00, 1:01, 2:10, 3:11
    # Global C: k*2+i
    # 0:0, 1:1, 2:2, 3:3
    c_map_1 = [0, 1, 2, 3]
    
    for r in range(7):
        for l, g in enumerate(a_map_1):
            U[g, r] = U1[l, r]
        for l, g in enumerate(b_map_1):
            V[g, r] = V1[l, r]
        for l, g in enumerate(c_map_1):
            W[g, r] = W1[l, r]
    
    # Map indices for A_R B_B (Schoolbook part)
    # A_R rows: 0,1,2,3 (local) -> Global A rows: 2,5 (col 2)
    # A_R is 2x1. Local indices 0,1 for rows.
    # But _build_schoolbook_221 expects 2x2 A.
    # We only use the first column of A in U2?
    # U2 is 4x4. Rows 0,1,2,3 correspond to A00, A01, A10, A11.
    # We only want A00 (local 0) and A10 (local 2) to map to A_R[0] and A_R[1].
    # A_R[0] is A[0,2] (global 2). A_R[1] is A[1,2] (global 5).
    
    # V2 is 2x4. Rows 0,1 correspond to B0, B1.
    # B_B is 1x2. Local indices 0,1 for cols.
    # Global B_B rows: j=2, k=0,1. Indices 4, 5.
    
    # W2 is 2x4. Rows 0,1 correspond to C0, C1.
    # Global C is 2x2.
    # C = A_R B_B.
    # C0 (global 0) = A_R0 * B_B0
    # C1 (global 1) = A_R0 * B_B1
    # C2 (global 2) = A_R1 * B_B0
    # C3 (global 3) = A_R1 * B_B1
    
    # We need to map U2, V2, W2 to U, V, W correctly.
    # U2[r] corresponds to A_local[r].
    # r=0 -> A00 -> Global 2
    # r=1 -> A01 -> Unused (0)
    # r=2 -> A10 -> Global 5
    # r=3 -> A11 -> Unused (0)
    
    # V2[r] corresponds to B_local[r].
    # r=0 -> B0 -> Global 4
    # r=1 -> B1 -> Global 5
    
    # W2[r] corresponds to C_local[r].
    # r=0 -> C0 -> Global 0
    # r=1 -> C1 -> Global 2 (Wait, C is 2x1 in schoolbook, but here C is 2x2?)
    # No, A_R B_B is 2x2.
    # A_R (2x1) * B_B (1x2) = 2x2.
    # Schoolbook for 2x1x2 is rank 4.
    # My _build_schoolbook_221 is for 2x2x1.
    # I need a 2x1x2 decomposition.
    
    # Let's build 2x1x2 explicitly.
    # A: 2x1, B: 1x2, C: 2x2.
    # C = A B.
    # C00 = A0 B0
    # C01 = A0 B1
    # C10 = A1 B0
    # C11 = A1 B1
    # Rank 4.
    
    # U_212 (2, 4)
    # V_212 (2, 4)
    # W_212 (4, 4)
    
    U_212 = np.zeros((2, 4))
    V_212 = np.zeros((2, 4))
    W_212 = np.zeros((4, 4))
    
    # r=0: A0 B0 -> C00
    U_212[0, 0] = 1
    V_212[0, 0] = 1
    W_212[0, 0] = 1
    
    # r=1: A0 B1 -> C01
    U_212[0, 1] = 1
    V_212[1, 1] = 1
    W_212[1, 1] = 1
    
    # r=2: A1 B0 -> C10
    U_212[1, 2] = 1
    V_212[0, 2] = 1
    W_212[2, 2] = 1
    
    # r=3: A1 B1 -> C11
    U_212[1, 3] = 1
    V_212[1, 3] = 1
    W_212[3, 3] = 1
    
    # Map to global
    # A_R rows: 0->2, 1->5
    a_map_2 = [2, 5]
    # B_B rows: 0->4, 1->5
    b_map_2 = [4, 5]
    # C rows: 0->0, 1->1, 2->2, 3->3
    c_map_2 = [0, 1, 2, 3]
    
    for r in range(4):
        for l, g in enumerate(a_map_2):
            U[g, 7 + r] = U_212[l, r]
        for l, g in enumerate(b_map_2):
            V[g, 7 + r] = V_212[l, r]
        for l, g in enumerate(c_map_2):
            W[g, 7 + r] = W_212[l, r]
            
    return U, V, W

def _build_block_233():
    """
    Builds decomposition for 2x3x3 multiplication.
    A: 2x3, B: 3x3, C: 2x3.
    Rank 17.
    C = A_L B_T + A_R B_B.
    A_L (2x2), B_T (2x3) -> Rank 11.
    A_R (2x1), B_B (1x3) -> Rank 6.
    """
    # Part 1: A_L B_T (2x2x3) Rank 11
    # Split B_T into B_T1 (2x2) and B_T2 (2x1).
    # C_T = A_L B_T1 + A_L B_T2.
    # A_L B_T1: 2x2x2 Rank 7.
    # A_L B_T2: 2x2x1 Rank 4.
    
    U1, V1, W1 = _build_strassen_222() # 7
    U2, V2, W2 = _build_schoolbook_221() # 4
    
    # Part 2: A_R B_B (2x1x3) Rank 6
    # A_R (2x1), B_B (1x3).
    # C = A_R B_B.
    # C00 = A0 B0, C01 = A0 B1, C02 = A0 B2
    # C10 = A1 B0, C11 = A1 B1, C12 = A1 B2
    # Rank 6.
    
    U3 = np.zeros((2, 6))
    V3 = np.zeros((3, 6))
    W3 = np.zeros((6, 6))
    
    for i in range(2):
        for k in range(3):
            r = i * 3 + k
            U3[i, r] = 1
            V3[k, r] = 1
            W3[k * 2 + i, r] = 1 # C index k*2+i? No, C is 2x3. k is col, i is row.
            # C index: k * 2 + i?
            # In 2x3x3, C is 2x3.
            # Global C for block 233:
            # We are building 2x3x3.
            # C indices: i (row 0,1), k (col 0,1,2).
            # Index i * 3 + k?
            # Wait, the interface is W[k*n+i, r].
            # For 2x3x3, n=2, p=3.
            # W index: k * n + i = k * 2 + i.
            
    # Total U, V, W for 2x3x3
    U = np.zeros((6, 17))
    V = np.zeros((9, 17))
    W = np.zeros((6, 17))
    
    # Map Part 1 (A_L B_T)
    # A_L: 0,1,3,4
    # B_T: 0,1,3,4,6,7 (Rows 0,1; Cols 0,1,2)
    # B indices: j*3+k.
    # j=0: 0,1,2
    # j=1: 3,4,5
    # B_T1 (2x2): j=0,1; k=0,1 -> 0,1,3,4
    # B_T2 (2x1): j=0,1; k=2 -> 2,5
    
    a_map_1 = [0, 1, 3, 4]
    b_map_1a = [0, 1, 3, 4] # For Strassen (2x2)
    b_map_1b = [2, 5]       # For Schoolbook (2x1)
    c_map_1 = [0, 1, 2, 3, 4, 5] # C is 2x2? No, C_T is 2x2?
    # A_L (2x2) * B_T (2x3) = C_T (2x3).
    # C_T cols 0,1 come from B_T1. C_T col 2 comes from B_T2.
    # C_T1 (2x2) indices: 0,1,2,3 (k=0,1)
    # C_T2 (2x1) indices: 4,5 (k=2)
    
    # Strassen part (2x2x2)
    for r in range(7):
        for l, g in enumerate(a_map_1):
            U[g, r] = U1[l, r]
        for l, g in enumerate(b_map_1a):
            V[g, r] = V1[l, r]
        # W1 is 4x4. C_T1 indices 0,1,2,3.
        for l, g in enumerate([0, 1, 2, 3]):
            W[g, r] = W1[l, r]
            
    # Schoolbook part (2x2x1)
    # U2 (4x4), V2 (2x4), W2 (2x4)
    # U2 rows 0,1,2,3 -> A_L 0,1,2,3
    # V2 rows 0,1 -> B_T2 0,1 (local) -> Global 2,5
    # W2 rows 0,1 -> C_T2 0,1 (local) -> Global 4,5
    
    for r in range(4):
        for l, g in enumerate(a_map_1):
            U[g, 7 + r] = U2[l, r]
        for l, g in enumerate(b_map_1b):
            V[g, 7 + r] = V2[l, r]
        for l, g in enumerate([4, 5]):
            W[g, 7 + r] = W2[l, r]
            
    # Map Part 2 (A_R B_B)
    # A_R: 2,5
    # B_B: 6,7,8 (Row 2; Cols 0,1,2)
    # C_B: 4,5,6,7,8,9? No, C is 2x3.
    # C indices: k*2+i.
    # k=0: 0,1
    # k=1: 2,3
    # k=2: 4,5
    # B_B contributes to all columns?
    # C = A_L B_T + A_R B_B.
    # A_R B_B is 2x3.
    # So C_B has indices 0..5.
    
    a_map_2 = [2, 5]
    b_map_2 = [6, 7, 8]
    c_map_2 = [0, 1, 2, 3, 4, 5]
    
    for r in range(6):
        for l, g in enumerate(a_map_2):
            U[g, 11 + r] = U3[l, r]
        for l, g in enumerate(b_map_2):
            V[g, 11 + r] = V3[l, r]
        for l, g in enumerate(c_map_2):
            W[g, 11 + r] = W3[l, r]
            
    return U, V, W

def construct():
    """Construct the tensor decomposition for 2x3x7 matrix multiplication."""
    # Block 1: 2x3x2 (Rank 11)
    U1, V1, W1 = _build_block_232()
    
    # Block 2: 2x3x2 (Rank 11)
    U2, V2, W2 = _build_block_232()
    
    # Block 3: 2x3x3 (Rank 17)
    U3, V3, W3 = _build_block_233()
    
    # Combine
    U = np.hstack([U1, U2, U3])
    
    # V and W need careful indexing for the full 2x3x7 problem.
    # Full V: (3, 7) -> 21 entries.
    # Full W: (7, 2) -> 14 entries.
    
    # Block 1 uses B cols 0,1. C cols 0,1.
    # Block 2 uses B cols 2,3. C cols 2,3.
    # Block 3 uses B cols 4,5,6. C cols 4,5,6.
    
    V_full = np.zeros((21, 39))
    W_full = np.zeros((14, 39))
    
    # Map V1 (6x11) to V_full
    # V1 rows 0..5 correspond to B[0,0], B[0,1], B[1,0], B[1,1], B[2,0], B[2,1]
    # Global V index: j*7+k.
    # j=0: 0,1
    # j=1: 7,8
    # j=2: 14,15
    v_map_1 = [0, 1, 7, 8, 14, 15]
    for r in range(11):
        for l, g in enumerate(v_map_1):
            V_full[g, r] = V1[l, r]
            
    # Map W1 (4x11) to W_full
    # W1 rows 0..3 correspond to C[0,0], C[0,1], C[1,0], C[1,1]
    # Global W index: k*2+i.
    # k=0: 0,1
    # k=1: 2,3
    w_map_1 = [0, 1, 2, 3]
    for r in range(11):
        for l, g in enumerate(w_map_1):
            W_full[g, r] = W1[l, r]
            
    # Map V2 (6x11) to V_full
    # B cols 2,3.
    # j=0: 2,3
    # j=1: 9,10
    # j=2: 16,17
    v_map_2 = [2, 3, 9, 10, 16, 17]
    for r in range(11):
        for l, g in enumerate(v_map_2):
            V_full[g, 11 + r] = V2[l, r]
            
    # Map W2 (4x11) to W_full
    # C cols 2,3.
    # k=2: 4,5
    # k=3: 6,7
    w_map_2 = [4, 5, 6, 7]
    for r in range(11):
        for l, g in enumerate(w_map_2):
            W_full[g, 11 + r] = W2[l, r]
            
    # Map V3 (9x17) to V_full
    # B cols 4,5,6.
    # j=0: 4,5,6
    # j=1: 11,12,13
    # j=2: 18,19,20
    v_map_3 = [4, 5, 6, 11, 12, 13, 18, 19, 20]
    for r in range(17):
        for l, g in enumerate(v_map_3):
            V_full[g, 22 + r] = V3[l, r]
            
    # Map W3 (6x17) to W_full
    # C cols 4,5,6.
    # k=4: 8,9
    # k=5: 10,11
    # k=6: 12,13
    w_map_3 = [8, 9, 10, 11, 12, 13]
    for r in range(17):
        for l, g in enumerate(w_map_3):
            W_full[g, 22 + r] = W3[l, r]
            
    result = {
        "U": U.tolist(),
        "V": V_full.tolist(),
        "W": W_full.tolist()
    }
    
    return result

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
