"""Matrix multiplication 2,4,5 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

_START_TIME = time.time()
_TIME_LIMIT = 140.0

def _strassen_222():
    """
    Returns U, V, W for 2x2x2 multiplication using Strassen's algorithm (rank 7).
    U: (4, 7)
    V: (4, 7)
    W: (4, 7)
    """
    U = np.zeros((4, 7), dtype=np.int8)
    V = np.zeros((4, 7), dtype=np.int8)
    W = np.zeros((4, 7), dtype=np.int8)
    
    # M1 = (A00 + A11) * (B00 + B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10 + A11) * B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    
    # M3 = A00 * (B01 - B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1
    
    # M4 = A11 * (B10 - B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[2, 3] = 1; W[0, 3] = 1
    
    # M5 = (A00 + A01) * B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[3, 4] = 1
    
    # M6 = (A10 - A00) * (B00 + B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A01 - A11) * (B10 + B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1; W[1, 6] = -1
    
    return U, V, W

def _schoolbook_221():
    """
    Returns U, V, W for 2x2x1 multiplication using schoolbook (rank 4).
    A is 2x2, B is 2x1, C is 2x1.
    U: (4, 4)
    V: (2, 4)
    W: (2, 4)
    """
    U = np.zeros((4, 4), dtype=np.int8)
    V = np.zeros((2, 4), dtype=np.int8)
    W = np.zeros((2, 4), dtype=np.int8)
    
    # C0 = A00*B00 + A01*B10
    # C1 = A10*B00 + A11*B10
    
    # Term 1: A00 * B00 -> C0
    U[0, 0] = 1
    V[0, 0] = 1
    W[0, 0] = 1
    
    # Term 2: A01 * B10 -> C0
    U[1, 1] = 1
    V[1, 1] = 1
    W[0, 1] = 1
    
    # Term 3: A10 * B00 -> C1
    U[2, 2] = 1
    V[0, 2] = 1
    W[1, 2] = 1
    
    # Term 4: A11 * B10 -> C1
    U[3, 3] = 1
    V[1, 3] = 1
    W[1, 3] = 1
    
    return U, V, W

def _construct_225():
    """
    Constructs 2x2x5 decomposition.
    A: 2x2, B: 2x5, C: 2x5.
    Splits B into B1 (2x2) and B2 (2x3).
    B2 splits into B2a (2x2) and B2b (2x1).
    Rank = 7 (for 2x2x2) + 7 (for 2x2x2) + 4 (for 2x2x1) = 18.
    """
    U_blocks = []
    V_blocks = []
    W_blocks = []
    
    # Block 1: A * B1 (2x2 * 2x2 -> 2x2)
    U1, V1, W1 = _strassen_222()
    # V1 is 2x2 part of B. B1 indices 0,1,2,3.
    # W1 is 2x2 part of C. C1 indices 0,1,2,3.
    U_blocks.append(U1)
    V_blocks.append(V1)
    W_blocks.append(W1)
    
    # Block 2: A * B2a (2x2 * 2x2 -> 2x2)
    # B2a is indices 4,5,6,7 of B.
    # C2a is indices 4,5,6,7 of C.
    U2, V2, W2 = _strassen_222()
    # Shift V2 and W2 indices by 4
    V2_shifted = np.zeros((10, 7), dtype=np.int8) # B is 2x5=10 elements? No, B is 2x5, so 10 elements.
    # Wait, B is 2x5. Indices 0..9.
    # B1: 0..3. B2a: 4..7. B2b: 8..9.
    # C is 2x5. Indices 0..9.
    # C1: 0..3. C2a: 4..7. C2b: 8..9.
    
    # V2 is 4x7. We need to place it in rows 4..7 of a 10x7 matrix.
    V2_full = np.zeros((10, 7), dtype=np.int8)
    V2_full[4:8, :] = V2
    W2_full = np.zeros((10, 7), dtype=np.int8)
    W2_full[4:8, :] = W2
    U_blocks.append(U2)
    V_blocks.append(V2_full)
    W_blocks.append(W2_full)
    
    # Block 3: A * B2b (2x2 * 2x1 -> 2x1)
    # B2b is indices 8,9 of B.
    # C2b is indices 8,9 of C.
    U3, V3, W3 = _schoolbook_221()
    # V3 is 2x4. Place in rows 8,9 of 10x4 matrix.
    V3_full = np.zeros((10, 4), dtype=np.int8)
    V3_full[8:10, :] = V3
    W3_full = np.zeros((10, 4), dtype=np.int8)
    W3_full[8:10, :] = W3
    U_blocks.append(U3)
    V_blocks.append(V3_full)
    W_blocks.append(W3_full)
    
    U_total = np.hstack(U_blocks)
    V_total = np.hstack(V_blocks)
    W_total = np.hstack(W_blocks)
    
    return U_total, V_total, W_total

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Construct 2x4x5 using two 2x2x5 blocks.
    # A is 2x4. Split into A1 (2x2, indices 0..3) and A2 (2x2, indices 4..7).
    # B is 4x5. Split into B1 (2x5, indices 0..9) and B2 (2x5, indices 10..19).
    # C is 5x2. Indices 0..9.
    # C = A1*B1 + A2*B2.
    
    U_blocks = []
    V_blocks = []
    W_blocks = []
    
    # Block 1: A1 * B1
    U1, V1, W1 = _construct_225()
    # U1 is 4x18. Place in rows 0..3 of 8x18 matrix.
    U1_full = np.zeros((8, 18), dtype=np.int8)
    U1_full[0:4, :] = U1
    V_blocks.append(V1)
    W_blocks.append(W1)
    U_blocks.append(U1_full)
    
    # Block 2: A2 * B2
    U2, V2, W2 = _construct_225()
    # U2 is 4x18. Place in rows 4..7 of 8x18 matrix.
    U2_full = np.zeros((8, 18), dtype=np.int8)
    U2_full[4:8, :] = U2
    # V2 is 10x18. Place in rows 10..19 of 20x18 matrix.
    V2_full = np.zeros((20, 18), dtype=np.int8)
    V2_full[10:20, :] = V2
    # W2 is 10x18. Same C indices, so just add to W1?
    # No, W is a matrix (10, rank).
    # The terms are independent. So we just concatenate the columns.
    # But wait, W1 and W2 operate on the same C indices.
    # So W_total = hstack(W1, W2).
    W_blocks.append(W2)
    U_blocks.append(U2_full)
    V_blocks.append(V2_full)
    
    U_total = np.hstack(U_blocks)
    V_total = np.hstack(V_blocks)
    W_total = np.hstack(W_blocks)
    
    # Convert to lists of lists as required by interface
    U_list = U_total.tolist()
    V_list = V_total.tolist()
    W_list = W_total.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
