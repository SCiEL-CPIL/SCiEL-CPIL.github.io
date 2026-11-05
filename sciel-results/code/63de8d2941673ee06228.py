"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3 by 3 matrix by a 3 by 7 matrix.
    Uses Strassen's algorithm for the 2x2 blocks to reduce rank from 63 to 60.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Safety margin from 150s limit
    
    # Total rank calculated as:
    # 3 Strassen blocks (7 each) + 1 Schoolbook block (4) + 14 (a*b) + 14 (a^T*B) + 7 (alpha*b)
    rank = 3 * 7 + 4 + 14 + 14 + 7
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # Helper to add a rank-1 term
    def add_term(u_idx, v_idx, w_idx, coeff):
        nonlocal col
        U[u_idx, col] += coeff
        V[v_idx, col] += coeff
        W[w_idx, col] += coeff
        col += 1

    # 1. A[0:2, 0:2] * B[0:2, 0:7] -> C[0:2, 0:7]
    # A indices: 0,1,3,4 (i*m+j)
    # B indices: 0..6 (j=0), 7..13 (j=1)
    # C indices: k*3 + i
    
    # Strassen for columns 0,1
    # A block: A00(0), A01(1), A10(3), A11(4)
    # B block: B00(0), B01(1), B10(7), B11(8) -- Wait, B index is j*p+k.
    # B[0,0]=0, B[0,1]=1, B[1,0]=7, B[1,1]=8.
    # C block: C[0,0]=0, C[1,0]=1, C[0,1]=3, C[1,1]=4.
    
    # M1 = (A00+A11)(B00+B11)
    add_term(0, 0, 0, 1) # A00 B00 -> C00
    add_term(0, 7, 0, 1) # A00 B10 -> C00
    add_term(4, 0, 0, 1) # A11 B00 -> C00
    add_term(4, 7, 0, 1) # A11 B10 -> C00
    # Wait, Strassen terms are products of sums.
    # We need to construct the factors for the sums.
    # This manual addition of terms is for schoolbook.
    # For Strassen, we must add the 7 specific rank-1 tensors.
    
    # Let's restart the Strassen addition logic properly.
    # Rank-1 tensor for M1: (A00+A11) * (B00+B11)
    # U vector: has 1 at A00, 1 at A11.
    # V vector: has 1 at B00, 1 at B11.
    # W vector: has coefficients for C.
    
    # It is easier to just define the U, V, W columns for the 7 Strassen mults.
    
    def add_strassen_block(a_00, a_01, a_10, a_11, b_00, b_01, b_10, b_11, c_00, c_01, c_10, c_11):
        nonlocal col
        
        # M1 = (A00+A11)(B00+B11)
        U[a_00, col] = 1; U[a_11, col] = 1
        V[b_00, col] = 1; V[b_11, col] = 1
        W[c_00, col] = 1; W[c_11, col] = 1
        col += 1
        
        # M2 = (A10+A11)B00
        U[a_10, col] = 1; U[a_11, col] = 1
        V[b_00, col] = 1
        W[c_10, col] = 1; W[c_11, col] = -1
        col += 1
        
        # M3 = A00(B10-B11)
        U[a_00, col] = 1
        V[b_10, col] = 1; V[b_11, col] = -1
        W[c_01, col] = 1; W[c_11, col] = 1
        col += 1
        
        # M4 = A11(B01-B00)
        U[a_11, col] = 1
        V[b_01, col] = 1; V[b_00, col] = -1
        W[c_00, col] = 1; W[c_10, col] = 1
        col += 1
        
        # M5 = (A00+A01)B11
        U[a_00, col] = 1; U[a_01, col] = 1
        V[b_11, col] = 1
        W[c_00, col] = -1; W[c_01, col] = 1
        col += 1
        
        # M6 = (A10-A00)(B00+B01)
        U[a_10, col] = 1; U[a_00, col] = -1
        V[b_00, col] = 1; V[b_01, col] = 1
        W[c_11, col] = 1
        col += 1
        
        # M7 = (A01-A11)(B10+B11)
        U[a_01, col] = 1; U[a_11, col] = -1
        V[b_10, col] = 1; V[b_11, col] = 1
        W[c_00, col] = 1; W[c_11, col] = -1
        col += 1

    # Block 1: Cols 0,1
    # A: 0,1,3,4
    # B: 0,1,7,8
    # C: 0,3,1,4 (c00=0, c01=3, c10=1, c11=4)
    add_strassen_block(0, 1, 3, 4, 0, 1, 7, 8, 0, 3, 1, 4)
    
    # Block 2: Cols 2,3
    # B: 2,3,9,10
    # C: 6,9,7,10
    add_strassen_block(0, 1, 3, 4, 2, 3, 9, 10, 6, 9, 7, 10)
    
    # Block 3: Cols 4,5
    # B: 4,5,11,12
    # C: 12,15,13,16
    add_strassen_block(0, 1, 3, 4, 4, 5, 11, 12, 12, 15, 13, 16)
    
    # Block 4: Col 6 (Schoolbook)
    # C[0,6] = A00 B06 + A01 B16
    # C[1,6] = A10 B06 + A11 B16
    # A: 0,1,3,4
    # B: 6, 13
    # C: 18, 19
    
    # Term 1: A00 B06 -> C06
    U[0, col] = 1; V[6, col] = 1; W[18, col] = 1; col += 1
    # Term 2: A01 B16 -> C06
    U[1, col] = 1; V[13, col] = 1; W[18, col] = 1; col += 1
    # Term 3: A10 B06 -> C16
    U[3, col] = 1; V[6, col] = 1; W[19, col] = 1; col += 1
    # Term 4: A11 B16 -> C16
    U[4, col] = 1; V[13, col] = 1; W[19, col] = 1; col += 1

    # 2. a * b^T
    # a = [A02, A12] -> indices 2, 5
    # b^T = [B20, ..., B26] -> indices 14, 15, 16, 17, 18, 19, 20
    # C[0, k] += A02 B2k
    # C[1, k] += A12 B2k
    
    for k in range(7):
        b_idx = 14 + k
        c0_idx = k * 3 + 0
        c1_idx = k * 3 + 1
        
        # A02 B2k -> C0k
        U[2, col] = 1; V[b_idx, col] = 1; W[c0_idx, col] = 1; col += 1
        # A12 B2k -> C1k
        U[5, col] = 1; V[b_idx, col] = 1; W[c1_idx, col] = 1; col += 1

    # 3. a^T * B[0:2, 0:7]
    # a^T = [A20, A21] -> indices 6, 7
    # B[0, k] -> k
    # B[1, k] -> 7 + k
    # C[2, k] += A20 B0k + A21 B1k
    
    for k in range(7):
        b0_idx = k
        b1_idx = 7 + k
        c2_idx = k * 3 + 2
        
        # A20 B0k -> C2k
        U[6, col] = 1; V[b0_idx, col] = 1; W[c2_idx, col] = 1; col += 1
        # A21 B1k -> C2k
        U[7, col] = 1; V[b1_idx, col] = 1; W[c2_idx, col] = 1; col += 1

    # 4. alpha * b^T
    # alpha = A22 -> index 8
    # b^T = [B20, ..., B26] -> indices 14..20
    # C[2, k] += A22 B2k
    
    for k in range(7):
        b_idx = 14 + k
        c2_idx = k * 3 + 2
        
        U[8, col] = 1; V[b_idx, col] = 1; W[c2_idx, col] = 1; col += 1

    # Check time
    if time.time() > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
