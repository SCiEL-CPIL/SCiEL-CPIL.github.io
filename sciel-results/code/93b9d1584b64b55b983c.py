"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START

def _strassen_2x2x2():
    """Return (U, V, W) for 2x2x2 matrix multiplication using Strassen.
    U shape: (4, 7), V shape: (4, 7), W shape: (4, 7)
    Convention: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c] where n=m=p=2
    """
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # M1 = (A11 + A22)(B11 + B22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    # C11 += M1, C22 -= M1
    W[0, 0] = 1; W[3, 0] = -1
    
    # M2 = (A21 + A22) B11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    # C21 += M2, C22 -= M2
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3 = A11 (B12 - B22)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    # C12 += M3, C22 += M3
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4 = A22 (B21 - B11)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    # C11 += M4, C21 += M4
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5 = (A11 + A12) B22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    # C11 -= M5, C12 += M5
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6 = (A21 - A11)(B11 + B12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    # C22 += M6
    W[3, 5] = 1
    
    # M7 = (A12 - A22)(B21 + B22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    # C11 += M7
    W[0, 6] = 1
    
    return U, V, W

def _schoolbook_nm_p(n, m, p):
    """Return (U, V, W) for n x m times m x p using schoolbook.
    U shape: (n*m, n*m*p), V shape: (m*p, n*m*p), W shape: (p*n, n*m*p)
    """
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _combine_decompositions(decomps):
    """Combine a list of (U, V, W) decompositions that operate on disjoint
    row/col blocks. Each decomp has its own local dimensions.
    Returns (U, V, W) for the combined problem.
    
    This is for block-diagonal-like combinations where each decomp handles
    a subset of the input/output indices independently.
    """
    # This function is a placeholder; we'll build the full decomposition manually.
    raise NotImplementedError("Use manual construction for 3x4x5")

def construct():
    start_time = time.time()
    deadline = 140.0  # 20s margin before 150s limit
    
    n, m, p = DIMENSIONS  # 3, 4, 5
    
    # Strategy: Split the 3x4 matrix into:
    #   A_top: 2x4 (rows 0,1)
    #   A_bot: 1x4 (row 2)
    # And the 4x5 matrix B is full.
    # C_top (2x5) = A_top * B  -- this is 2x4x5
    # C_bot (1x5) = A_bot * B  -- this is 1x4x5
    
    # For 2x4x5: Split B into B_left (4x2) and B_right (4x3)
    #   C_top_left (2x2) = A_top * B_left  -- 2x4x2
    #   C_top_right (2x3) = A_top * B_right -- 2x4x3
    # For 2x4x2: Split A_top into A_TL (2x2) and A_TR (2x2)
    #   C_TL = A_TL * B_TL (2x2x2) + A_TR * B_TL (2x2x2) -- two 2x2x2
    # For 2x4x3: Split A_top into A_TL (2x2) and A_TR (2x2)
    #   C_TR = A_TL * B_TR (2x2x3) + A_TR * B_TR (2x2x3) -- two 2x2x3
    
    # Let's use a more systematic approach with explicit index tracking.
    # We'll build the full U, V, W matrices for the 3x4x5 problem.
    
    # Total rank estimate:
    # 2x2x2 (Strassen): rank 7
    # 2x2x3 (Strassen 2x2x2 + schoolbook 2x2x1): rank 7 + 2 = 9
    # 1x4x5 (schoolbook): rank 5
    # Total: 7 + 9 + 5 = 21? No, that's not right.
    
    # Let me reconsider. The decomposition should be:
    # C = A * B where A is 3x4, B is 4x5, C is 3x5.
    
    # Block decomposition:
    # A = [A1 (2x4); A2 (1x4)]
    # B = [B1 (4x5)]
    # C = [A1*B (2x5); A2*B (1x5)]
    
    # For A1*B (2x4x5):
    #   Split B into B_L (4x2) and B_R (4x3)
    #   A1*B_L (2x4x2) and A1*B_R (2x4x3)
    #   
    #   For A1*B_L (2x4x2):
    #     Split A1 into A1_L (2x2) and A1_R (2x2)
    #     A1_L * B_L (2x2x2) + A1_R * B_L (2x2x2)
    #     Each 2x2x2 uses Strassen (rank 7), total rank 14
    #   
    #   For A1*B_R (2x4x3):
    #     Split A1 into A1_L (2x2) and A1_R (2x2)
    #     A1_L * B_R (2x2x3) + A1_R * B_R (2x2x3)
    #     Each 2x2x3: use Strassen on first 2 cols (rank 7) + schoolbook on last col (rank 2) = rank 9
    #     Total rank: 9 + 9 = 18
    #   
    # For A2*B (1x4x5):
    #   Schoolbook: rank 5
    
    # Total rank: 14 + 18 + 5 = 37
    
    # Now let's build the actual matrices.
    # We need to carefully track which local indices map to which global indices.
    
    # Global dimensions: n=3, m=4, p=5
    # U: (12, rank), V: (20, rank), W: (15, rank)
    
    # Let's define the sub-decompositions and their index mappings.
    
    # Sub-decomp 1: A1_L * B_L (2x2x2)
    #   A rows: 0,1; A cols: 0,1
    #   B rows: 0,1; B cols: 0,1
    #   C rows: 0,1; C cols: 0,1
    #   U indices: A[i*4+j] for i in {0,1}, j in {0,1} -> {0,1,4,5}
    #   V indices: B[j*5+k] for j in {0,1}, k in {0,1} -> {0,1,5,6}
    #   W indices: C[k*3+i] for k in {0,1}, i in {0,1} -> {0,1,3,4}
    
    # Sub-decomp 2: A1_R * B_L (2x2x2)
    #   A rows: 0,1; A cols: 2,3
    #   B rows: 0,1; B cols: 0,1
    #   C rows: 0,1; C cols: 0,1
    #   U indices: A[i*4+j] for i in {0,1}, j in {2,3} -> {2,3,6,7}
    #   V indices: B[j*5+k] for j in {0,1}, k in {0,1} -> {0,1,5,6}
    #   W indices: C[k*3+i] for k in {0,1}, i in {0,1} -> {0,1,3,4}
    
    # Sub-decomp 3: A1_L * B_R (2x2x3)
    #   A rows: 0,1; A cols: 0,1
    #   B rows: 0,1; B cols: 2,3,4
    #   C rows: 0,1; C cols: 2,3,4
    #   U indices: A[i*4+j] for i in {0,1}, j in {0,1} -> {0,1,4,5}
    #   V indices: B[j*5+k] for j in {0,1}, k in {2,3,4} -> {2,3,4,7,8,9}
    #   W indices: C[k*3+i] for k in {2,3,4}, i in {0,1} -> {6,7,8,9,10,11}
    
    # Sub-decomp 4: A1_R * B_R (2x2x3)
    #   A rows: 0,1; A cols: 2,3
    #   B rows: 0,1; B cols: 2,3,4
    #   C rows: 0,1; C cols: 2,3,4
    #   U indices: A[i*4+j] for i in {0,1}, j in {2,3} -> {2,3,6,7}
    #   V indices: B[j*5+k] for j in {0,1}, k in {2,3,4} -> {2,3,4,7,8,9}
    #   W indices: C[k*3+i] for k in {2,3,4}, i in {0,1} -> {6,7,8,9,10,11}
    
    # Sub-decomp 5: A2 * B (1x4x5)
    #   A rows: 2; A cols: 0,1,2,3
    #   B rows: 0,1,2,3; B cols: 0,1,2,3,4
    #   C rows: 2; C cols: 0,1,2,3,4
    #   U indices: A[2*4+j] for j in {0,1,2,3} -> {8,9,10,11}
    #   V indices: B[j*5+k] for j in {0,1,2,3}, k in {0,1,2,3,4} -> all 20
    #   W indices: C[k*3+2] for k in {0,1,2,3,4} -> {2,5,8,11,14}
    
    # Wait, I need to be more careful. The issue is that sub-decomps 1 and 2 both
    # use the same B rows {0,1} and C rows {0,1}, cols {0,1}. They are not independent
    # in the B and C spaces. I need to combine them properly.
    
    # Actually, let me reconsider the decomposition. The standard way to decompose
    # a larger matrix multiplication is to use a Kronecker product or block structure.
    
    # Let me try a different approach: directly construct the decomposition using
    # known results. For 3x4x5, the rank is known to be at most 37 (from the
    # block decomposition above), but I should verify this is achievable.
    
    # Actually, let me just implement the schoolbook decomposition first to ensure
    # correctness, then try to improve.
    
    # For now, let's return the schoolbook decomposition which is guaranteed correct.
    rank = n * m * p  # 60
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
