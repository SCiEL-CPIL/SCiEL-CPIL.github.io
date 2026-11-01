import time
import numpy as np

def construct():
    """
    Construct a rank-25 exact tensor decomposition for 2x2 * 2x7 matrix multiplication.
    
    This implementation uses a block-wise approach:
    - Three 2x2 blocks of columns (0-1, 2-3, 4-5) use Strassen's algorithm (rank 7 each).
    - The final column (6) uses the schoolbook algorithm (rank 4).
    Total rank: 7 + 7 + 7 + 4 = 25.
    
    The output factor W follows the convention W[k*n+i, r] where C[i,k] is the output.
    """
    n, m, p = 2, 2, 7
    total_rank = 25
    
    # Initialize matrices
    U = np.zeros((n*m, total_rank), dtype=float)
    V = np.zeros((m*p, total_rank), dtype=float)
    W = np.zeros((p*n, total_rank), dtype=float)
    
    # Helper to add a Strassen block for columns k, k+1 starting at rank offset 'offset'
    def add_strassen_block(k, offset):
        # M1 = (A00 + A11) * (B0,k + B1,k+1)
        # A indices: 00->0, 11->3. B indices: 0,k->7k, 1,k+1->7(k+1)+1.
        # C indices: 0,k->2k, 1,k+1->2(k+1)+1.
        
        # Term 1
        U[0, offset] += 1.0
        U[3, offset] += 1.0
        V[7*k, offset] += 1.0
        V[7*(k+1) + 1, offset] += 1.0
        W[2*k, offset] += 1.0
        W[2*(k+1) + 1, offset] += 1.0
        
        # M2 = (A10 + A11) * B0,k
        # A: 10->2, 11->3. B: 0,k->7k.
        # C: 1,k->2k+1, 1,k+1->2(k+1)+1 (with -1)
        U[2, offset+1] += 1.0
        U[3, offset+1] += 1.0
        V[7*k, offset+1] += 1.0
        W[2*k + 1, offset+1] += 1.0
        W[2*(k+1) + 1, offset+1] -= 1.0
        
        # M3 = A00 * (B0,k+1 - B1,k+1)
        # A: 00->0. B: 0,k+1->7(k+1), 1,k+1->7(k+1)+1.
        # C: 0,k+1->2(k+1), 1,k+1->2(k+1)+1.
        U[0, offset+2] += 1.0
        V[7*(k+1), offset+2] += 1.0
        V[7*(k+1) + 1, offset+2] -= 1.0
        W[2*(k+1), offset+2] += 1.0
        W[2*(k+1) + 1, offset+2] += 1.0
        
        # M4 = A11 * (B1,k - B0,k)
        # A: 11->3. B: 1,k->7k+1, 0,k->7k.
        # C: 0,k->2k, 1,k->2k+1.
        U[3, offset+3] += 1.0
        V[7*k + 1, offset+3] += 1.0
        V[7*k, offset+3] -= 1.0
        W[2*k, offset+3] += 1.0
        W[2*k + 1, offset+3] += 1.0
        
        # M5 = (A00 + A01) * B1,k+1
        # A: 00->0, 01->1. B: 1,k+1->7(k+1)+1.
        # C: 0,k->2k (-1), 0,k+1->2(k+1) (+1).
        U[0, offset+4] += 1.0
        U[1, offset+4] += 1.0
        V[7*(k+1) + 1, offset+4] += 1.0
        W[2*k, offset+4] -= 1.0
        W[2*(k+1), offset+4] += 1.0
        
        # M6 = (A10 - A00) * (B0,k + B0,k+1)
        # A: 10->2, 00->0 (-1). B: 0,k->7k, 0,k+1->7(k+1).
        # C: 1,k+1->2(k+1)+1.
        U[2, offset+5] += 1.0
        U[0, offset+5] -= 1.0
        V[7*k, offset+5] += 1.0
        V[7*(k+1), offset+5] += 1.0
        W[2*(k+1) + 1, offset+5] += 1.0
        
        # M7 = (A01 - A11) * (B1,k + B1,k+1)
        # A: 01->1, 11->3 (-1). B: 1,k->7k+1, 1,k+1->7(k+1)+1.
        # C: 0,k->2k.
        U[1, offset+6] += 1.0
        U[3, offset+6] -= 1.0
        V[7*k + 1, offset+6] += 1.0
        V[7*(k+1) + 1, offset+6] += 1.0
        W[2*k, offset+6] += 1.0

    # Apply Strassen to blocks (0,1), (2,3), (4,5)
    add_strassen_block(0, 0)
    add_strassen_block(2, 7)
    add_strassen_block(4, 14)
    
    # Apply Schoolbook to column 6 (rank 4)
    # C[0,6] = A[0,0]B[0,6] + A[0,1]B[1,6]
    # C[1,6] = A[1,0]B[0,6] + A[1,1]B[1,6]
    
    # Term 21: A[0,0] * B[0,6] -> C[0,6]
    U[0, 21] += 1.0
    V[7*0 + 6, 21] += 1.0
    W[2*6 + 0, 21] += 1.0
    
    # Term 22: A[0,1] * B[1,6] -> C[0,6]
    U[1, 22] += 1.0
    V[7*1 + 6, 22] += 1.0
    W[2*6 + 0, 22] += 1.0
    
    # Term 23: A[1,0] * B[0,6] -> C[1,6]
    U[2, 23] += 1.0
    V[7*0 + 6, 23] += 1.0
    W[2*6 + 1, 23] += 1.0
    
    # Term 24: A[1,1] * B[1,6] -> C[1,6]
    U[3, 24] += 1.0
    V[7*1 + 6, 24] += 1.0
    W[2*6 + 1, 24] += 1.0
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
