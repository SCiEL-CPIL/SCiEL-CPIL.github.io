import numpy as np

DIMENSIONS = [2, 2, 12]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=12
    # We use Strassen's algorithm for 2x2 matrix multiplication.
    # The B matrix is 2x12, which can be viewed as 6 blocks of 2x2.
    # Each block requires 7 multiplications.
    # Total rank = 6 * 7 = 42.
    
    rank = 42
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # U indices: 0=A00, 1=A01, 2=A10, 3=A11
    # V indices: 0..11 are row 0 of B, 12..23 are row 1 of B
    # W indices: 0..11 are row 0 of C, 12..23 are row 1 of C
    # W convention: W[k*n + i, r] where C = A @ B
    # C[i, k] = sum_j A[i, j] B[j, k]
    # In the task description: "output factor follows the official trace convention W[k*n+i,r]"
    # This usually means the index in W corresponds to the position in C.
    # If C is 2x12, and we flatten it, the index is k*2 + i? Or i*12 + k?
    # "W[k*n+i,r]" implies column-major or specific ordering.
    # Let's check the current program's W indexing.
    # Current program:
    # c00 = 4*l -> W[4*l, col]
    # c01 = 4*l + 2 -> W[4*l+2, col]
    # c10 = 4*l + 1 -> W[4*l+1, col]
    # c11 = 4*l + 3 -> W[4*l+3, col]
    # Here l is the block index (0..5).
    # Block l corresponds to B columns 2l, 2l+1.
    # C columns 2l, 2l+1.
    # C[0, 2l] is c00. Index 4l.
    # C[1, 2l] is c10. Index 4l+1.
    # C[0, 2l+1] is c01. Index 4l+2.
    # C[1, 2l+1] is c11. Index 4l+3.
    # This mapping is: Index = (2l)*2 + i? No.
    # For l=0: C[0,0]->0, C[1,0]->1, C[0,1]->2, C[1,1]->3.
    # This is column-major order for the 2x2 block: [C00, C10, C01, C11].
    # So W index = k*n + i is NOT what is used in the current program?
    # k is column, i is row.
    # For C[0,0] (i=0, k=0): k*n+i = 0*2+0=0. Matches.
    # For C[1,0] (i=1, k=0): k*n+i = 0*2+1=1. Matches.
    # For C[0,1] (i=0, k=1): k*n+i = 1*2+0=2. Matches.
    # For C[1,1] (i=1, k=1): k*n+i = 1*2+1=3. Matches.
    # So the convention is W[k*n + i, r] where i is row index, k is column index.
    # This is consistent.

    for l in range(6):
        base_col = l * 7
        
        # B block indices for block l
        # B[0, 2l] -> V[2l]
        # B[0, 2l+1] -> V[2l+1]
        # B[1, 2l] -> V[12+2l]
        # B[1, 2l+1] -> V[12+2l+1]
        
        b00_idx = 2*l
        b01_idx = 2*l + 1
        b10_idx = 12 + 2*l
        b11_idx = 12 + 2*l + 1
        
        # C block indices for block l
        # C[0, 2l] -> W[2l*2 + 0] = 4l
        # C[1, 2l] -> W[2l*2 + 1] = 4l+1
        # C[0, 2l+1] -> W[(2l+1)*2 + 0] = 4l+2
        # C[1, 2l+1] -> W[(2l+1)*2 + 1] = 4l+3
        
        c00_idx = 4*l
        c10_idx = 4*l + 1
        c01_idx = 4*l + 2
        c11_idx = 4*l + 3
        
        # Strassen Multiplications
        # M1 = (A00+A11)(B00+B11)
        col = base_col
        U[0, col] = 1
        U[3, col] = 1
        V[b00_idx, col] = 1
        V[b11_idx, col] = 1
        W[c00_idx, col] = 1
        W[c11_idx, col] = 1
        
        # M2 = (A10+A11)B00
        col = base_col + 1
        U[2, col] = 1
        U[3, col] = 1
        V[b00_idx, col] = 1
        W[c10_idx, col] = 1
        W[c11_idx, col] = -1
        
        # M3 = A00(B01-B11)
        col = base_col + 2
        U[0, col] = 1
        V[b01_idx, col] = 1
        V[b11_idx, col] = -1
        W[c01_idx, col] = 1
        W[c11_idx, col] = 1
        
        # M4 = A11(B10-B00)
        col = base_col + 3
        U[3, col] = 1
        V[b10_idx, col] = 1
        V[b00_idx, col] = -1
        W[c00_idx, col] = 1
        W[c10_idx, col] = 1
        
        # M5 = (A00+A01)B11
        col = base_col + 4
        U[0, col] = 1
        U[1, col] = 1
        V[b11_idx, col] = 1
        W[c00_idx, col] = -1
        W[c01_idx, col] = 1
        
        # M6 = (A10-A00)(B00+B01)
        col = base_col + 5
        U[2, col] = 1
        U[0, col] = -1
        V[b00_idx, col] = 1
        V[b01_idx, col] = 1
        W[c11_idx, col] = 1
        
        # M7 = (A01-A11)(B10+B11)
        col = base_col + 6
        U[1, col] = 1
        U[3, col] = -1
        V[b10_idx, col] = 1
        V[b11_idx, col] = 1
        W[c00_idx, col] = 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
