"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _construct_strassen_2x2_block():
    """
    Returns U, V, W for a 2x2 * 2x2 -> 2x2 multiplication using Strassen (rank 7).
    U shape: (4, 7)  indexed by (i*2+j)
    V shape: (4, 7)  indexed by (j*2+k)
    W shape: (4, 7)  indexed by (k*2+i)
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Strassen formulas:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # U entries (A indices: 00, 01, 10, 11)
    # V entries (B indices: 00, 01, 10, 11)
    # W entries (C indices: 00, 01, 10, 11)
    
    # M1: U[00]+=1, U[11]+=1; V[00]+=1, V[11]+=1; W[00]+=1, W[11]-=1
    U[0, 0] += 1; U[3, 0] += 1
    V[0, 0] += 1; V[3, 0] += 1
    W[0, 0] += 1; W[3, 0] -= 1
    
    # M2: U[10]+=1, U[11]+=1; V[00]+=1; W[10]+=1, W[11]-=1
    U[2, 1] += 1; U[3, 1] += 1
    V[0, 1] += 1
    W[2, 1] += 1; W[3, 1] -= 1
    
    # M3: U[00]+=1; V[10]+=1, V[11]-=1; W[01]+=1, W[11]+=1
    U[0, 2] += 1
    V[2, 2] += 1; V[3, 2] -= 1
    W[1, 2] += 1; W[3, 2] += 1
    
    # M4: U[11]+=1; V[01]+=1, V[00]-=1; W[00]+=1, W[10]+=1
    U[3, 3] += 1
    V[1, 3] += 1; V[0, 3] -= 1
    W[0, 3] += 1; W[2, 3] += 1
    
    # M5: U[00]+=1, U[01]+=1; V[11]+=1; W[00]-=1, W[01]+=1
    U[0, 4] += 1; U[1, 4] += 1
    V[3, 4] += 1
    W[0, 4] -= 1; W[1, 4] += 1
    
    # M6: U[10]+=1, U[00]-=1; V[00]+=1, V[01]+=1; W[11]+=1
    U[2, 5] += 1; U[0, 5] -= 1
    V[0, 5] += 1; V[1, 5] += 1
    W[3, 5] += 1
    
    # M7: U[01]+=1, U[11]-=1; V[10]+=1, V[11]+=1; W[00]+=1
    U[1, 6] += 1; U[3, 6] -= 1
    V[2, 6] += 1; V[3, 6] += 1
    W[0, 6] += 1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 5
    
    # Strategy: Use Strassen for the 2x2 part of A (columns 0,1) with first 2 columns of B.
    # Then schoolbook for the rest.
    # 
    # A is 2x3: A[:, 0:2] is 2x2, A[:, 2] is 2x1
    # B is 3x5: B[0:2, 0:2] is 2x2, B[0:2, 2:5] is 2x3, B[2, :] is 1x5
    #
    # C = A[:, 0:2] @ B[0:2, :] + A[:, 2] @ B[2, :]
    # 
    # Let C1 = A[:, 0:2] @ B[0:2, :] (2x5)
    # Let C2 = A[:, 2] @ B[2, :] (2x5)
    #
    # For C1: Split B[0:2, :] into B[0:2, 0:2] (2x2) and B[0:2, 2:5] (2x3)
    # C1_left = A[:, 0:2] @ B[0:2, 0:2] -> 2x2, use Strassen (rank 7)
    # C1_right = A[:, 0:2] @ B[0:2, 2:5] -> 2x3, schoolbook (rank 2*2*3=12)
    # C2 = A[:, 2] @ B[2, :] -> 2x5, schoolbook (rank 1*1*5=5)
    #
    # Total rank = 7 + 12 + 5 = 24
    
    rank = 7 + 12 + 5
    
    U = np.zeros((n * m, rank))  # (6, 24)
    V = np.zeros((m * p, rank))  # (15, 24)
    W = np.zeros((p * n, rank))  # (10, 24)
    
    # Part 1: Strassen for A[:, 0:2] @ B[0:2, 0:2] -> C[:, 0:2]
    U1, V1, W1 = _construct_strassen_2x2_block()
    # U1 shape (4, 7): indices (i*2+j) for i in {0,1}, j in {0,1}
    # In full A (2x3), these correspond to rows i, columns j (0,1) -> index i*3 + j
    # V1 shape (4, 7): indices (j*2+k) for j in {0,1}, k in {0,1}
    # In full B (3x5), these correspond to rows j (0,1), columns k (0,1) -> index j*5 + k
    # W1 shape (4, 7): indices (k*2+i) for k in {0,1}, i in {0,1}
    # In full C (2x5), output is at (i, k) -> index k*2 + i (same convention)
    
    for c in range(7):
        for idx in range(4):
            # U1[idx, c] corresponds to A row idx//2, col idx%2
            a_row = idx // 2
            a_col = idx % 2
            U[a_row * 3 + a_col, c] += U1[idx, c]
            
            # V1[idx, c] corresponds to B row idx//2, col idx%2
            b_row = idx // 2
            b_col = idx % 2
            V[b_row * 5 + b_col, c] += V1[idx, c]
            
            # W1[idx, c] corresponds to C output at (i=idx%2, k=idx//2)
            # Index in W: k*n + i = idx//2 * 2 + idx%2
            W[idx, c] += W1[idx, c]
    
    # Part 2: Schoolbook for A[:, 0:2] @ B[0:2, 2:5] -> C[:, 2:5]
    # A columns 0,1 (2x2) times B rows 0,1 columns 2,3,4 (2x3)
    col_offset = 7
    for i in range(2):
        for j in range(2):
            for k in range(3):
                U[i * 3 + j, col_offset] = 1.0
                V[j * 5 + (2 + k), col_offset] = 1.0
                W[(2 + k) * 2 + i, col_offset] = 1.0
                col_offset += 1
    
    # Part 3: Schoolbook for A[:, 2] @ B[2, :] -> C[:, :]
    # A column 2 (2x1) times B row 2 (1x5)
    for i in range(2):
        for k in range(5):
            U[i * 3 + 2, col_offset] = 1.0
            V[2 * 5 + k, col_offset] = 1.0
            W[k * 2 + i, col_offset] = 1.0
            col_offset += 1
    
    # Convert to lists of lists for output
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
