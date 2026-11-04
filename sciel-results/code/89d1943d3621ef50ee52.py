import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [5, 5, 5]

def _schoolbook(n, m, p):
    """Construct the standard rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _strassen_2x2():
    """
    Return the classic Strassen 2x2x2 decomposition with 7 terms.
    U, V, W are 4x7 matrices.
    Indexing: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A11+A22)(B11+B22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A21+A22)B11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    
    # M3 = A11(B21-B22)
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1
    
    # M4 = A22(B12-B11)
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[2, 3] = 1
    
    # M5 = (A11+A12)B22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6 = (A21-A11)(B11+B12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A12-A22)(B21+B22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _construct_4x4x4():
    """
    Construct 4x4x4 decomposition using Strassen recursively.
    Returns U, V, W of shape (16, 49).
    """
    U2, V2, W2 = _strassen_2x2()
    
    # We need to assemble 4x4 from 2x2 blocks.
    # The Strassen algorithm computes 7 intermediate 2x2 products.
    # Each product is a rank 7 decomposition.
    # We need to map the input/output indices of the 7 products to the 4x4 indices.
    
    # Let's define the 7 "super-terms".
    # Each super-term corresponds to one of the 7 Strassen products.
    # For each super-term t, we have a 2x2x2 decomposition (U_t, V_t, W_t).
    # We need to place these into the 4x4x4 U, V, W.
    
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    # Mapping of Strassen products to 4x4 indices
    # M1: (A11+A22)(B11+B22) -> contributes to C11, C12, C21, C22?
    # Actually, the Strassen formulas are:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Wait, the standard Strassen formulas for C = AB:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We need to construct the tensor for C11, C12, C21, C22.
    # Each C block is 2x2.
    # So C11 corresponds to indices i in {0,1}, k in {0,1} in the 4x4 output?
    # No, C11 is the top-left 2x2 block.
    # So for C11, i in {0,1}, k in {0,1}.
    # For C12, i in {0,1}, k in {2,3}.
    # For C21, i in {2,3}, k in {0,1}.
    # For C22, i in {2,3}, k in {2,3}.
    
    # The input A is 4x4.
    # A11: i in {0,1}, j in {0,1}.
    # A12: i in {0,1}, j in {2,3}.
    # A21: i in {2,3}, j in {0,1}.
    # A22: i in {2,3}, j in {2,3}.
    
    # The input B is 4x4.
    # B11: j in {0,1}, k in {0,1}.
    # B12: j in {0,1}, k in {2,3}.
    # B21: j in {2,3}, k in {0,1}.
    # B22: j in {2,3}, k in {2,3}.
    
    # We iterate over the 7 Strassen products.
    # For each product t (0..6), we have a 2x2x2 decomposition.
    # We need to determine which A, B, C indices are involved in product t.
    
    # This is complex to map directly. Instead, we can use the fact that
    # the 4x4x4 tensor is the sum of 7 rank-7 tensors.
    # Let's just use the schoolbook for 4x4x4 if the mapping is too error-prone?
    # No, rank 49 is better than 64.
    
    # Let's try to implement the mapping carefully.
    
    # Strassen Product 0: M1 = (A11+A22)(B11+B22)
    # A factors: A11 (1,1), A22 (1,1)
    # B factors: B11 (1,1), B22 (1,1)
    # C factors: C11 (1,1), C22 (1,1) with coeffs?
    # M1 contributes +1 to C11 and +1 to C22.
    
    # This is getting complicated. Let's use a simpler approach:
    # Use the known rank 49 decomposition for 4x4x4 from literature or derive it.
    # Or, just use schoolbook for 4x4 (rank 64) and see if the total rank is better than 125.
    # 4x4 schoolbook: 64.
    # 4x1 schoolbook: 4.
    # 1x4 schoolbook: 4.
    # 1x1 schoolbook: 1.
    # Total: 64 + 4 + 4 + 1 + (cross terms)
    # C11: 64 + 4 = 68
    # C12: 16 + 4 = 20
    # C21: 16 + 4 = 20
    # C22: 4 + 1 = 5
    # Total: 113.
    
    # 113 is better than 125.
    # Implementing Strassen 4x4 correctly is risky in 120s.
    # Let's stick to the 4+1 partition with schoolbook 4x4.
    
    return None # Placeholder

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    
    # Partition 5 = 4 + 1
    n1, n2 = 4, 1
    m1, m2 = 4, 1
    p1, p2 = 4, 1
    
    # We need to construct U, V, W for the 5x5x5 tensor.
    # U shape: (25, Rank)
    # V shape: (25, Rank)
    # W shape: (25, Rank)
    
    # We will use schoolbook for all sub-blocks for simplicity and reliability.
    # Rank 4x4x4: 64
    # Rank 4x1x1: 4
    # Rank 1x4x1: 4
    # Rank 1x1x4: 4
    # Rank 1x1x1: 1
    
    # Total Rank = 64 + 4 + 4 + 1 + 4 + 4 + 4 + 1 = 86?
    # Wait, the blocks are:
    # C11 = A11 B11 + A12 B21
    # C12 = A11 B12 + A12 B22
    # C21 = A21 B11 + A22 B21
    # C22 = A21 B12 + A22 B22
    
    # A11: 4x4, B11: 4x4 -> Rank 64
    # A12: 4x1, B21: 1x4 -> Rank 4*1*4 = 16?
    # No, A12 is 4x1 (i in 0..3, j=4). B21 is 1x4 (j=4, k in 0..3).
    # Product A12 B21 is 4x4.
    # Rank of 4x1x4 tensor is 4*1*4 = 16.
    
    # Let's re-calculate with schoolbook for all sub-blocks.
    # C11:
    # A11(4x4) B11(4x4): Rank 64
    # A12(4x1) B21(1x4): Rank 16
    # Total C11: 80
    
    # C12:
    # A11(4x4) B12(4x1): Rank 16
    # A12(4x1) B22(1x1): Rank 4
    # Total C12: 20
    
    # C21:
    # A21(1x4) B11(4x4): Rank 16
    # A22(1x1) B21(1x4): Rank 4
    # Total C21: 20
    
    # C22:
    # A21(1x4) B12(4x1): Rank 4
    # A22(1x1) B22(1x1): Rank 1
    # Total C22: 5
    
    # Total Rank = 80 + 20 + 20 + 5 = 125.
    # It's the same as schoolbook!
    
    # Why? Because schoolbook is optimal for these rectangular blocks if we don't use Strassen.
    # We MUST use Strassen for the 4x4x4 block to get rank 49 instead of 64.
    # Then C11 becomes 49 + 16 = 65.
    # Total = 65 + 20 + 20 + 5 = 110.
    
    # So we need Strassen 4x4x4.
    
    # Let's implement Strassen 4x4x4 carefully.
    
    return _schoolbook(n, m, p) # Fallback

# EVOLVE-BLOCK-END
