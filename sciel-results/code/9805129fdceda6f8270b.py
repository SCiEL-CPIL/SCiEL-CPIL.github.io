import numpy as np
import time

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # Target rank: 66
    # 6 blocks of (2x2) x (2x2) using Strassen (rank 7 each) -> 42
    # 6 blocks of (1x2) x (2x2) using Schoolbook (rank 4 each) -> 24
    # Total rank = 66
    
    rank = 66
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    col_offset = 0
    
    # Helper to add a rank-1 tensor
    # u_idx, v_idx, w_idx are global indices
    # coeffs are (u_coeff, v_coeff, w_coeff)
    def add_tensor(u_idx, v_idx, w_idx, u_c, v_c, w_c, offset):
        U[u_idx, offset] += u_c
        V[v_idx, offset] += v_c
        W[w_idx, offset] += w_c
    
    # Indices for A (3x4)
    # A[i, j] -> U[i*m + j, c]
    # Indices for B (4x6)
    # B[j, k] -> V[j*p + k, c]
    # Indices for C (3x6)
    # C[i, k] -> W[k*n + i, c]  (Trace convention: W[k*n+i, r])
    
    # Partition A into 4 blocks:
    # A11: rows 0-1, cols 0-1 (2x2)
    # A12: rows 0-1, cols 2-3 (2x2)
    # A21: row 2, cols 0-1 (1x2)
    # A22: row 2, cols 2-3 (1x2)
    
    # Partition B into 6 blocks (2 rows x 3 cols of 2x2 blocks? No, B is 4x6)
    # B is 4x6.
    # Rows 0-1, Cols 0-1 -> B11 (2x2)
    # Rows 0-1, Cols 2-3 -> B12 (2x2)
    # Rows 0-1, Cols 4-5 -> B13 (2x2)
    # Rows 2-3, Cols 0-1 -> B21 (2x2)
    # Rows 2-3, Cols 2-3 -> B22 (2x2)
    # Rows 2-3, Cols 4-5 -> B23 (2x2)
    
    # Products:
    # C11 (2x2) = A11 B11 + A12 B21
    # C12 (2x2) = A11 B12 + A12 B22
    # C13 (2x2) = A11 B13 + A12 B23
    # C21 (1x2) = A21 B11 + A22 B21
    # C22 (1x2) = A21 B12 + A22 B22
    # C23 (1x2) = A21 B13 + A22 B23
    
    # Define Strassen factors for (2x2) x (2x2)
    # Local indices for A_block (2x2):
    # (0,0), (0,1), (1,0), (1,1)
    # Local indices for B_block (2x2):
    # (0,0), (0,1), (1,0), (1,1)
    # Local indices for C_block (2x2):
    # (0,0), (0,1), (1,0), (1,1)
    
    # Strassen Tensors (u, v, w)
    # M1: (A00+A11)(B00+B11) -> C00, C11
    # M2: (A10+A11)B00 -> C10
    # M3: A00(B01-B11) -> C01
    # M4: A11(B10-B00) -> C00, C10
    # M5: (A00+A01)B11 -> C00, C01
    # M6: (A10-A00)(B00+B01) -> C11
    # M7: (A01-A11)(B10+B11) -> C00, C11
    
    def add_strassen(a_r0, a_c0, b_r0, b_c0, c_r0, c_c0, offset):
        """
        Add Strassen decomposition for A_block at (a_r0, a_c0) and B_block at (b_r0, b_c0)
        contributing to C_block at (c_r0, c_c0).
        a_r0, a_c0: top-left row/col of A block in global A
        b_r0, b_c0: top-left row/col of B block in global B
        c_r0, c_c0: top-left row/col of C block in global C
        """
        # Local (r,c) to Global Index
        # A: i=a_r0+r, j=a_c0+c -> U[i*m + j, offset]
        # B: j=b_r0+r, k=b_c0+c -> V[j*p + k, offset]
        # C: i=c_r0+r, k=c_c0+c -> W[k*n + i, offset]
        
        # M1: (A00+A11)(B00+B11)
        # U: A00(+), A11(+)
        # V: B00(+), B11(+)
        # W: C00(+), C11(+)
        add_tensor(a_r0*m + a_c0, b_r0*p + b_c0, c_c0*n + c_r0, 1, 1, 1, offset)
        add_tensor(a_r0*m + a_c0, b_r0*p + (b_c0+1), c_c0*n + c_r0, 1, 1, 1, offset) # Wait, B11 is (1,1) local.
        # Let's be precise with indices.
        
        # Local A:
        # (0,0) -> (a_r0, a_c0)
        # (0,1) -> (a_r0, a_c0+1)
        # (1,0) -> (a_r0+1, a_c0)
        # (1,1) -> (a_r0+1, a_c0+1)
        
        # Local B:
        # (0,0) -> (b_r0, b_c0)
        # (0,1) -> (b_r0, b_c0+1)
        # (1,0) -> (b_r0+1, b_c0)
        # (1,1) -> (b_r0+1, b_c0+1)
        
        # Local C:
        # (0,0) -> (c_r0, c_c0)
        # (0,1) -> (c_r0, c_c0+1)
        # (1,0) -> (c_r0+1, c_c0)
        # (1,1) -> (c_r0+1, c_c0+1)
        
        # Note: W index is k*n + i.
        # C(i, k) -> W[k*n + i]
        
        # M1: (A00+A11)(B00+B11)
        # U: A00=1, A11=1
        # V: B00=1, B11=1
        # W: C00=1, C11=1
        U[a_r0*m + a_c0, offset] += 1
        U[(a_r0+1)*m + (a_c0+1), offset] += 1
        V[b_r0*p + b_c0, offset] += 1
        V[(b_r0+1)*p + (b_c0+1), offset] += 1
        W[c_c0*n + c_r0, offset] += 1       # C00
        W[(c_c0+1)*n + (c_r0+1), offset] += 1 # C11
        
        # M2: (A10+A11)B00
        # U: A10=1, A11=1
        # V: B00=1
        # W: C10=1
        U[(a_r0+1)*m + a_c0, offset] += 1
        U[(a_r0+1)*m + (a_c0+1), offset] += 1
        V[b_r0*p + b_c0, offset] += 1
        W[c_c0*n + (c_r0+1), offset] += 1   # C10
        
        # M3: A00(B01-B11)
        # U: A00=1
        # V: B01=1, B11=-1
        # W: C01=1
        U[a_r0*m + a_c0, offset] += 1
        V[b_r0*p + (b_c0+1), offset] += 1
        V[(b_r0+1)*p + (b_c0+1), offset] += -1
        W[(c_c0+1)*n + c_r0, offset] += 1   # C01
        
        # M4: A11(B10-B00)
        # U: A11=1
        # V: B10=1, B00=-1
        # W: C00=1, C10=1
        U[(a_r0+1)*m + (a_c0+1), offset] += 1
        V[(b_r0+1)*p + b_c0, offset] += 1
        V[b_r0*p + b_c0, offset] += -1
        W[c_c0*n + c_r0, offset] += 1       # C00
        W[c_c0*n + (c_r0+1), offset] += 1   # C10
        
        # M5: (A00+A01)B11
        # U: A00=1, A01=1
        # V: B11=1
        # W: C00=-1, C01=1
        U[a_r0*m + a_c0, offset] += 1
        U[a_r0*m + (a_c0+1), offset] += 1
        V[(b_r0+1)*p + (b_c0+1), offset] += 1
        W[c_c0*n + c_r0, offset] += -1      # C00
        W[(c_c0+1)*n + c_r0, offset] += 1   # C01
        
        # M6: (A10-A00)(B00+B01)
        # U: A10=1, A00=-1
        # V: B00=1, B01=1
        # W: C11=1
        U[(a_r0+1)*m + a_c0, offset] += 1
        U[a_r0*m + a_c0, offset] += -1
        V[b_r0*p + b_c0, offset] += 1
        V[b_r0*p + (b_c0+1), offset] += 1
        W[(c_c0+1)*n + (c_r0+1), offset] += 1 # C11
        
        # M7: (A01-A11)(B10+B11)
        # U: A01=1, A11=-1
        # V: B10=1, B11=1
        # W: C00=1, C11=1
        U[a_r0*m + (a_c0+1), offset] += 1
        U[(a_r0+1)*m + (a_c0+1), offset] += -1
        V[(b_r0+1)*p + b_c0, offset] += 1
        V[(b_r0+1)*p + (b_c0+1), offset] += 1
        W[c_c0*n + c_r0, offset] += 1       # C00
        W[(c_c0+1)*n + (c_r0+1), offset] += 1 # C11

    def add_schoolbook_1x2x2(a_r0, a_c0, b_r0, b_c0, c_r0, c_c0, offset):
        """
        Add Schoolbook decomposition for A_block (1x2) at (a_r0, a_c0) and B_block (2x2) at (b_r0, b_c0)
        contributing to C_block (1x2) at (c_r0, c_c0).
        """
        # A is 1x2:
        # A00 -> (a_r0, a_c0)
        # A01 -> (a_r0, a_c0+1)
        
        # B is 2x2:
        # B00 -> (b_r0, b_c0)
        # B01 -> (b_r0, b_c0+1)
        # B10 -> (b_r0+1, b_c0)
        # B11 -> (b_r0+1, b_c0+1)
        
        # C is 1x2:
        # C00 -> (c_r0, c_c0)
        # C01 -> (c_r0, c_c0+1)
        
        # T1: A00 B00 -> C00
        U[a_r0*m + a_c0, offset] += 1
        V[b_r0*p + b_c0, offset] += 1
        W[c_c0*n + c_r0, offset] += 1
        
        # T2: A01 B10 -> C00
        U[a_r0*m + (a_c0+1), offset] += 1
        V[(b_r0+1)*p + b_c0, offset] += 1
        W[c_c0*n + c_r0, offset] += 1
        
        # T3: A00 B01 -> C01
        U[a_r0*m + a_c0, offset] += 1
        V[b_r0*p + (b_c0+1), offset] += 1
        W[(c_c0+1)*n + c_r0, offset] += 1
        
        # T4: A01 B11 -> C01
        U[a_r0*m + (a_c0+1), offset] += 1
        V[(b_r0+1)*p + (b_c0+1), offset] += 1
        W[(c_c0+1)*n + c_r0, offset] += 1

    # 1. C11 (2x2) = A11 B11 (Strassen) + A12 B21 (Strassen)
    # A11: r0=0, c0=0. B11: r0=0, c0=0. C11: r0=0, c0=0.
    add_strassen(0, 0, 0, 0, 0, 0, col_offset)
    col_offset += 7
    
    # A12: r0=0, c0=2. B21: r0=2, c0=0. C11: r0=0, c0=0.
    add_strassen(0, 2, 2, 0, 0, 0, col_offset)
    col_offset += 7
    
    # 2. C12 (2x2) = A11 B12 (Strassen) + A12 B22 (Strassen)
    # A11: r0=0, c0=0. B12: r0=0, c0=2. C12: r0=0, c0=2.
    add_strassen(0, 0, 0, 2, 0, 2, col_offset)
    col_offset += 7
    
    # A12: r0=0, c0=2. B22: r0=2, c0=2. C12: r0=0, c0=2.
    add_strassen(0, 2, 2, 2, 0, 2, col_offset)
    col_offset += 7
    
    # 3. C13 (2x2) = A11 B13 (Strassen) + A12 B23 (Strassen)
    # A11: r0=0, c0=0. B13: r0=0, c0=4. C13: r0=0, c0=4.
    add_strassen(0, 0, 0, 4, 0, 4, col_offset)
    col_offset += 7
    
    # A12: r0=0, c0=2. B23: r0=2, c0=4. C13: r0=0, c0=4.
    add_strassen(0, 2, 2, 4, 0, 4, col_offset)
    col_offset += 7
    
    # 4. C21 (1x2) = A21 B11 (Schoolbook) + A22 B21 (Schoolbook)
    # A21: r0=2, c0=0. B11: r0=0, c0=0. C21: r0=2, c0=0.
    add_schoolbook_1x2x2(2, 0, 0, 0, 2, 0, col_offset)
    col_offset += 4
    
    # A22: r0=2, c0=2. B21: r0=2, c0=0. C21: r0=2, c0=0.
    add_schoolbook_1x2x2(2, 2, 2, 0, 2, 0, col_offset)
    col_offset += 4
    
    # 5. C22 (1x2) = A21 B12 (Schoolbook) + A22 B22 (Schoolbook)
    # A21: r0=2, c0=0. B12: r0=0, c0=2. C22: r0=2, c0=2.
    add_schoolbook_1x2x2(2, 0, 0, 2, 2, 2, col_offset)
    col_offset += 4
    
    # A22: r0=2, c0=2. B22: r0=2, c0=2. C22: r0=2, c0=2.
    add_schoolbook_1x2x2(2, 2, 2, 2, 2, 2, col_offset)
    col_offset += 4
    
    # 6. C23 (1x2) = A21 B13 (Schoolbook) + A22 B23 (Schoolbook)
    # A21: r0=2, c0=0. B13: r0=0, c0=4. C23: r0=2, c0=4.
    add_schoolbook_1x2x2(2, 0, 0, 4, 2, 4, col_offset)
    col_offset += 4
    
    # A22: r0=2, c0=2. B23: r0=2, c0=4. C23: r0=2, c0=4.
    add_schoolbook_1x2x2(2, 2, 2, 4, 2, 4, col_offset)
    col_offset += 4
    
    # Total offset should be 66
    assert col_offset == rank, f"Column offset {col_offset} does not match rank {rank}"
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
