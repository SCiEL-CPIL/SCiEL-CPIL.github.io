"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Split n=3 into 2+1
    # Part 1: 2x4 * 4x7 -> split 4 into 2+2, 7 into 4+3
    #   2x2 * 2x4: Strassen rank 7
    #   2x2 * 2x3: schoolbook rank 12
    # Part 2: 1x4 * 4x7: schoolbook rank 28
    # Total rank: 7 + 12 + 28 = 47
    
    # Strassen for 2x2 * 2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B11-B21)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    rank = 47
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Part 1: rows 0,1 of A (2x4) times 4x7
    # A[0:2, 0:4] * B[0:4, 0:7]
    # Split B columns: B[:, 0:4] and B[:, 4:7]
    # Split A rows: A[0:2, :] is 2x4, split into A[0:2, 0:2] and A[0:2, 2:4]
    
    # Strassen for A[0:2, 0:2] * B[0:2, 0:4] (2x2 * 2x4)
    # This is 2x2 * 2x2 (left half of B) + 2x2 * 2x2 (right half of B)
    # Actually, 2x2 * 2x4 can be done as two 2x2*2x2 multiplications
    # Each 2x2*2x2 uses Strassen with rank 7
    # So 2x2 * 2x4 uses rank 14? No wait...
    
    # Let me reconsider. 2x2 * 2x4:
    # A is 2x2, B is 2x4
    # C = A * B is 2x4
    # We can split B into B1 (2x2) and B2 (2x2)
    # C1 = A * B1 (2x2 * 2x2, Strassen rank 7)
    # C2 = A * B2 (2x2 * 2x2, Strassen rank 7)
    # Total rank: 14
    
    # Hmm, but the observation says 2x2 * 2x4 uses Strassen rank 7.
    # That doesn't seem right for standard Strassen. Let me re-read.
    
    # Actually, maybe they mean something different. Let me just use schoolbook
    # for all sub-blocks to be safe, then see if we can improve.
    
    # Schoolbook for 2x4 * 4x7: rank = 2*4*7 = 56
    # Schoolbook for 1x4 * 4x7: rank = 1*4*7 = 28
    # Total: 84 (same as full schoolbook)
    
    # Let me try the Strassen approach properly.
    # For 2x2 * 2x2, Strassen uses 7 multiplications.
    
    # Let me build this step by step.
    
    # Reset
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Helper: add a Strassen 2x2*2x2 block
    # A is at rows a0:a0+2, cols b0:b0+2 of the full A matrix
    # B is at rows b0:b0+2, cols c0:c0+2 of the full B matrix
    # C is at rows a0:a0+2, cols c0:c0+2 of the full C matrix
    def add_strassen(a0, b0, c0, start_col):
        nonlocal col
        # M1 = (A11+A22)(B11+B22)
        # A11 = A[a0, b0], A22 = A[a0+1, b0+1]
        # B11 = B[b0, c0], B22 = B[b0+1, c0+1]
        
        # M1
        U[a0, start_col] = 1  # A11
        U[a0+1, start_col] = 1  # A22
        V[b0, start_col] = 1  # B11
        V[b0+1, start_col] = 1  # B22
        W[c0, a0] = 1  # C11 +=
        W[c0+1, a0] = 1  # C12 +=
        W[c0, a0+1] = 1  # C21 +=
        W[c0+1, a0+1] = 1  # C22 +=
        col += 1
        
        # M2 = (A21+A22)B11
        U[a0+1, start_col+1] = 1  # A21
        U[a0+1, start_col+2] = 1  # A22 (separate term)
        V[b0, start_col+1] = 1  # B11
        W[c0, a0+1] = 1  # C21 +=
        W[c0+1, a0+1] = 1  # C22 +=
        col += 1
        
        # Wait, this is getting complicated. Let me use a cleaner approach.
        # Each Strassen multiplication M_k = (sum of A terms) * (sum of B terms)
        # contributes to specific C entries with coefficients.
        
        # Let me restart with a cleaner formulation.
        pass
    
    # Let me use a different approach: directly construct the decomposition.
    
    # For the full 3x4*4x7 problem:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # Split i into {0,1} and {2}
    # Split j into {0,1} and {2,3}
    # Split k into {0,1,2,3} and {4,5,6}
    
    # Block (i in {0,1}, j in {0,1}, k in {0,1,2,3}): 2x2 * 2x4
    # Block (i in {0,1}, j in {2,3}, k in {0,1,2,3}): 2x2 * 2x4
    # Block (i in {0,1}, j in {0,1}, k in {4,5,6}): 2x2 * 2x3
    # Block (i in {0,1}, j in {2,3}, k in {4,5,6}): 2x2 * 2x3
    # Block (i in {2}, j in {0,1,2,3}, k in {0,1,2,3}): 1x4 * 4x4
    # Block (i in {2}, j in {0,1,2,3}, k in {4,5,6}): 1x4 * 4x3
    
    # For 2x2 * 2x2: Strassen rank 7
    # For 2x2 * 2x4: two 2x2*2x2, rank 14
    # For 2x2 * 2x3: 2x2*2x2 + 2x2*2x1, rank 7 + 4 = 11
    # For 1x4 * 4x4: 4*4*1 = 16 schoolbook, or 4*(1x1*1x1) = 16
    # For 1x4 * 4x3: 4*3*1 = 12 schoolbook
    
    # Total: 14 + 14 + 11 + 11 + 16 + 12 = 78
    
    # That matches the observation! Rank 78.
    
    # Let me implement this.
    
    U = np.zeros((n * m, 78))
    V = np.zeros((m * p, 78))
    W = np.zeros((p * n, 78))
    
    col = 0
    
    # Strassen 2x2*2x2 helper
    # A rows: a0, a0+1; A cols: b0, b0+1
    # B rows: b0, b0+1; B cols: c0, c0+1
    # C rows: a0, a0+1; C cols: c0, c0+1
    def strassen_2x2(a0, b0, c0):
        nonlocal col
        # M1 = (A11+A22)(B11+B22)
        U[a0, col] = 1; U[a0+1, col] = 1
        V[b0, col] = 1; V[b0+1, col] = 1
        W[c0, a0] = 1; W[c0+1, a0] = 1
        W[c0, a0+1] = 1; W[c0+1, a0+1] = 1
        col += 1
        
        # M2 = (A21+A22)B11
        U[a0+1, col] = 1
        V[b0, col] = 1
        W[c0, a0+1] = 1; W[c0+1, a0+1] = 1
        col += 1
        
        # M3 = A11(B21-B22)
        U[a0, col] = 1
        V[b0+1, col] = 1
        W[c0+1, a0] = 1
        V[b0+1, col+1] = -1  # Wait, I can't have two columns in one M
        # I need to be more careful.
        pass
    
    # This is getting messy. Let me use a cleaner method.
    # Each "multiplication" in the tensor decomposition is a rank-1 term:
    # U[i,j] * V[j,k] * W[k,i]
    
    # For Strassen, each M_k is a product of a linear combination of A entries
    # and a linear combination of B entries, contributing to linear combinations of C entries.
    
    # Let me just use schoolbook for everything and get rank 84, then try to improve.
    
    U = np.zeros((n * m, 84))
    V = np.zeros((m * p, 84))
    W = np.zeros((p * n, 84))
    
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
