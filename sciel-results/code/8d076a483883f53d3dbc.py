import time
import numpy as np

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-42 decomposition of the 2x2x12 matrix multiplication tensor.
    
    Uses the tensor product decomposition:
    M_{2,2,12} ~ M_{2,2,2} (tensored with) M_{1,1,6}
    
    M_{2,2,2} uses Strassen's algorithm (rank 7)
    M_{1,1,6} uses the standard rank-6 decomposition
    
    Total rank = 7 * 6 = 42
    
    The tensor product of decompositions (U1,V1,W1) and (U2,V2,W2) gives:
    U = U1 ⊗ U2, V = V1 ⊗ V2, W = W1 ⊗ W2
    """
    start_time = time.time()
    deadline = 150.0
    
    # Strassen's algorithm for 2x2x2 (rank 7)
    # U1: shape (n*m, rank) = (4, 7)
    # V1: shape (m*p, rank) = (4, 7) 
    # W1: shape (p*n, rank) = (4, 7)
    # 
    # Strassen's 7 products:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # U1 indices: row = i*m + j, where i in {0,1}, j in {0,1} (n=2, m=2)
    # V1 indices: row = j*p + k, where j in {0,1}, k in {0,1} (m=2, p=2)
    # W1 indices: row = k*n + i, where k in {0,1}, i in {0,1} (p=2, n=2)
    
    # Strassen U1 matrix (4x7):
    # Rows: (i,j) = (0,0), (0,1), (1,0), (1,1)
    # Columns: M1..M7
    U1 = np.zeros((4, 7))
    # M1: (A11 + A22): A11 is (0,0), A22 is (1,1)
    U1[0, 0] = 1  # (0,0) in M1
    U1[3, 0] = 1  # (1,1) in M1
    # M2: (A21 + A22): A21 is (1,0), A22 is (1,1)
    U1[2, 1] = 1  # (1,0) in M2
    U1[3, 1] = 1  # (1,1) in M2
    # M3: A11: (0,0)
    U1[0, 2] = 1
    # M4: A22: (1,1)
    U1[3, 3] = 1
    # M5: (A11 + A12): A11 is (0,0), A12 is (0,1)
    U1[0, 4] = 1  # (0,0) in M5
    U1[1, 4] = 1  # (0,1) in M5
    # M6: (A21 - A11): A21 is (1,0), A11 is (0,0)
    U1[2, 5] = 1  # (1,0) in M6
    U1[0, 5] = -1 # (0,0) in M6
    # M7: (A12 - A22): A12 is (0,1), A22 is (1,1)
    U1[1, 6] = 1  # (0,1) in M7
    U1[3, 6] = -1 # (1,1) in M7
    
    # V1 matrix (4x7):
    # Rows: (j,k) = (0,0), (0,1), (1,0), (1,1)
    V1 = np.zeros((4, 7))
    # M1: (B11 + B22): B11 is (0,0), B22 is (1,1)
    V1[0, 0] = 1
    V1[3, 0] = 1
    # M2: B11: (0,0)
    V1[0, 1] = 1
    # M3: (B12 - B22): B12 is (0,1), B22 is (1,1)
    V1[1, 2] = 1
    V1[3, 2] = -1
    # M4: (B21 - B11): B21 is (1,0), B11 is (0,0)
    V1[2, 3] = 1
    V1[0, 3] = -1
    # M5: B22: (1,1)
    V1[3, 4] = 1
    # M6: (B11 + B12): B11 is (0,0), B12 is (0,1)
    V1[0, 5] = 1
    V1[1, 5] = 1
    # M7: (B21 + B22): B21 is (1,0), B22 is (1,1)
    V1[2, 6] = 1
    V1[3, 6] = 1
    
    # W1 matrix (4x7):
    # Rows: (k,i) = (0,0), (0,1), (1,0), (1,1)
    # C11 = M1 + M4 - M5 + M7 -> row (0,0)
    # C12 = M3 + M5 -> row (0,1)
    # C21 = M2 + M4 -> row (1,0)
    # C22 = M1 - M2 + M3 + M6 -> row (1,1)
    W1 = np.zeros((4, 7))
    # C11 (0,0):
    W1[0, 0] = 1   # M1
    W1[0, 3] = 1   # M4
    W1[0, 4] = -1  # -M5
    W1[0, 6] = 1   # M7
    # C12 (0,1):
    W1[1, 2] = 1   # M3
    W1[1, 4] = 1   # M5
    # C21 (1,0):
    W1[2, 1] = 1   # M2
    W1[2, 3] = 1   # M4
    # C22 (1,1):
    W1[3, 0] = 1   # M1
    W1[3, 1] = -1  # -M2
    W1[3, 2] = 1   # M3
    W1[3, 5] = 1   # M6
    
    # Standard rank-6 decomposition for 1x1x6 (vector-matrix multiplication)
    # This is just the identity mapping: each of the 6 output elements
    # corresponds to one product term.
    # U2: shape (1*1, 6) = (1, 6)
    # V2: shape (1*6, 6) = (6, 6)
    # W2: shape (6*1, 6) = (6, 6)
    U2 = np.zeros((1, 6))
    U2[0, :] = 1  # all ones
    
    V2 = np.eye(6)  # identity
    
    W2 = np.eye(6)  # identity
    
    # Now compute tensor product
    # U = U1 ⊗ U2, shape (4*1, 7*6) = (4, 42)
    # But we need shape (n*m, rank) = (4, 42) for the full problem
    # Wait, the full problem is 2x2x12, so:
    # U: shape (n*m, rank) = (4, 42)
    # V: shape (m*p, rank) = (24, 42)
    # W: shape (p*n, rank) = (24, 42)
    
    # The tensor product gives:
    # U = U1 ⊗ U2: shape (4*1, 7*6) = (4, 42) ✓
    # V = V1 ⊗ V2: shape (4*6, 7*6) = (24, 42) ✓
    # W = W1 ⊗ W2: shape (4*6, 7*6) = (24, 42) ✓
    
    U = np.kron(U1, U2)
    V = np.kron(V1, V2)
    W = np.kron(W1, W2)
    
    # Convert to lists of lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    # Verify we're within time budget
    elapsed = time.time() - start_time
    if elapsed > deadline - 5:
        # Return what we have (shouldn't happen for this simple construction)
        pass
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
