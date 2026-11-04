"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import time

def _strassen_2x2_rank7():
    """
    Returns (U, V, W) for 2x2 * 2x2 -> 2x2 with rank 7 (Strassen).
    U: shape (4, 7)  -- indices: [i*2+j] for i in {0,1}, j in {0,1}
    V: shape (4, 7)  -- indices: [j*2+k] for j in {0,1}, k in {0,1}
    W: shape (4, 7)  -- indices: [k*2+i] for i in {0,1}, k in {0,1}
    
    The 7 products are:
    M1 = A11*(B11+B22)
    M2 = (A11+A12)*B22
    M3 = (A21+A22)*B11
    M4 = A22*(B12-B21)
    M5 = (A11+A21)*(B11+B21)
    M6 = (A12-A22)*(B21+B22)
    M7 = (A11-A12)*(B11-B12)
    
    Results:
    C11 = M1 + M4 - M5 + M7
    C12 = M2 + M4 - M6 + M7
    C21 = M3 + M5 - M1 + M6
    C22 = M1 - M2 + M3 + M4
    """
    # Map Strassen products to U, V, W vectors
    # Each product c corresponds to:
    #   U[c] is a vector in R^4 (input A flattening: [A11, A12, A21, A22])
    #   V[c] is a vector in R^4 (input B flattening: [B11, B12, B21, B22])
    #   W[c] is a vector in R^4 (output C flattening: [C11, C12, C21, C22])
    
    # U vectors (for A flattening [A11, A12, A21, A22]):
    # M1: A11          -> [1, 0, 0, 0]
    # M2: A11+A12      -> [1, 1, 0, 0]
    # M3: A21+A22      -> [0, 0, 1, 1]
    # M4: A22          -> [0, 0, 0, 1]
    # M5: A11+A21      -> [1, 0, 1, 0]
    # M6: A12-A22      -> [0, 1, 0, -1]
    # M7: A11-A12      -> [1, -1, 0, 0]
    
    U = [
        [1,  1,  0,  0,  1,  0,  1],   # M1..M7 for A11
        [0,  1,  0,  0,  0,  1, -1],   # M1..M7 for A12
        [0,  0,  1,  0,  1,  0,  0],   # M1..M7 for A21
        [0,  0,  1,  1,  0, -1, 0],    # M1..M7 for A22
    ]
    
    # V vectors (for B flattening [B11, B12, B21, B22]):
    # M1: B11+B22      -> [1, 0, 0, 1]
    # M2: B22          -> [0, 0, 0, 1]
    # M3: B11          -> [1, 0, 0, 0]
    # M4: B12-B21      -> [0, 1, -1, 0]
    # M5: B11+B21      -> [1, 0, 1, 0]
    # M6: B21+B22      -> [0, 0, 1, 1]
    # M7: B11-B12      -> [1, -1, 0, 0]
    
    V = [
        [1,  0,  1,  0,  1,  0,  1],   # M1..M7 for B11
        [0,  0,  0,  1,  0,  0, -1],   # M1..M7 for B12
        [0,  0,  0, -1, 1,  1, 0],     # M1..M7 for B21
        [1,  1,  0,  0,  0,  1, 0],    # M1..M7 for B22
    ]
    
    # W vectors (for C flattening [C11, C12, C21, C22]):
    # C11 = M1 + M4 - M5 + M7  -> [1, 0, 0, 0] gets contributions: M1:1, M4:1, M5:-1, M7:1
    # C12 = M2 + M4 - M6 + M7  -> [0, 1, 0, 0] gets: M2:1, M4:1, M6:-1, M7:1
    # C21 = M3 + M5 - M1 + M6  -> [0, 0, 1, 0] gets: M3:1, M5:1, M1:-1, M6:1
    # C22 = M1 - M2 + M3 + M4  -> [0, 0, 0, 1] gets: M1:1, M2:-1, M3:1, M4:1
    
    # W[k*n+i, c] where output is 2x2, so W index is [k*2+i]
    # C11: i=0,k=0 -> idx 0
    # C12: i=0,k=1 -> idx 1
    # C21: i=1,k=0 -> idx 2
    # C22: i=1,k=1 -> idx 3
    
    W = [
        [ 1,  0, -1,  1],   # C11 from M1,M2,M3,M4? No:
        # Let me redo: W[idx_out][c]
        # For each output element, which products contribute with what sign:
        # C11 = M1*1 + M4*1 + M5*(-1) + M7*1
        # C12 = M2*1 + M4*1 + M6*(-1) + M7*1
        # C21 = M3*1 + M5*1 + M1*(-1) + M6*1
        # C22 = M1*1 + M2*(-1) + M3*1 + M4*1
        
        # So W row 0 (C11): [1, 0, 0, 1, -1, 0, 1]
        # W row 1 (C12): [0, 1, 0, 1, 0, -1, 1]
        # W row 2 (C21): [-1, 0, 1, 0, 1, 1, 0]
        # W row 3 (C22): [1, -1, 1, 1, 0, 0, 0]
    ]
    
    W = [
        [ 1,  0,  0,  1, -1,  0,  1],
        [ 0,  1,  0,  1,  0, -1,  1],
        [-1,  0,  1,  0,  1,  1,  0],
        [ 1, -1,  1,  1,  0,  0,  0],
    ]
    
    return U, V, W

def _verify_2x2(U, V, W):
    """Verify that the 2x2 Strassen decomposition is correct."""
    import numpy as np
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Test with random matrices
    for _ in range(20):
        A = np.random.randint(-10, 10, size=(2,2)).astype(float)
        B = np.random.randint(-10, 10, size=(2,2)).astype(float)
        C_true = A @ B
        
        A_flat = A.flatten()  # [A11, A12, A21, A22]
        B_flat = B.flatten()  # [B11, B12, B21, B22]
        
        # Compute products
        prods = np.array([np.dot(U_arr[:, c], A_flat) * np.dot(V_arr[:, c], B_flat) for c in range(7)])
        
        C_computed = np.array([np.dot(W_arr[r, :], prods) for r in range(4)])
        C_computed = C_computed.reshape(2, 2)
        
        if not np.allclose(C_true, C_computed, atol=1e-10):
            return False
    return True

def construct():
    """
    Construct a rank-28 exact tensor decomposition for 2x4 * 4x4 -> 2x4.
    
    Uses block structure: split into 2x2 blocks.
    A is 2x4 = [A1 | A2] where A1, A2 are 2x2.
    B is 4x4 = [[B11, B12], [B21, B22]] where each is 2x2.
    C is 2x4 = [C1 | C2] where C1 = A1*B11 + A2*B21, C2 = A1*B12 + A2*B22.
    
    This requires 4 products of 2x2 matrices, each done via Strassen (rank 7).
    Total rank = 4 * 7 = 28.
    """
    deadline = time.time() + 140  # safety margin
    
    U2, V2, W2 = _strassen_2x2_rank7()
    
    # Verify correctness
    if not _verify_2x2(U2, V2, W2):
        # Fallback to schoolbook if verification fails
        n, m, p = DIMENSIONS
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    # Build the full 2x4x4 decomposition
    # n=2, m=4, p=4
    # A is 2x4: flatten to 8 entries [A[0,0], A[0,1], A[0,2], A[0,3], A[1,0], A[1,1], A[1,2], A[1,3]]
    # B is 4x4: flatten to 16 entries
    # C is 2x4: flatten to 8 entries (output convention W[k*n+i] -> for 2x4, output is 2x4)
    
    # Block structure:
    # A = [A1 | A2] where A1 = A[:, 0:2], A2 = A[:, 2:4]
    # B = [[B11, B12], [B21, B22]] where B11 = B[0:2, 0:2], etc.
    # C1 = A1 * B11 + A2 * B21 (2x2 result)
    # C2 = A1 * B12 + A2 * B22 (2x2 result)
    
    # We need 4 Strassen decompositions:
    # Product 1: A1 * B11 -> contributes to C1
    # Product 2: A2 * B21 -> contributes to C1
    # Product 3: A1 * B12 -> contributes to C2
    # Product 4: A2 * B22 -> contributes to C2
    
    # For each product, we have 7 terms.
    # Total rank = 28.
    
    rank = 28
    n, m, p = 2, 4, 4
    
    # U: shape (n*m, rank) = (8, 28)
    # V: shape (m*p, rank) = (16, 28)
    # W: shape (p*n, rank) = (8, 28)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define the 4 products and their block mappings
    # Product type: (A_block, B_block, C_block)
    # A_block: which 2x2 block of A (col 0:1 or 2:3)
    # B_block: which 2x2 block of B (row/col 0:1 or 2:3)
    # C_block: which 2x2 block of C (col 0:1 or 2:3)
    
    products = [
        # (A_col_block, B_row_block, B_col_block, C_col_block)
        (0, 0, 0, 0),  # A1 * B11 -> C1
        (1, 1, 0, 0),  # A2 * B21 -> C1
        (0, 0, 1, 1),  # A1 * B12 -> C2
        (1, 1, 1, 1),  # A2 * B22 -> C2
    ]
    
    for prod_idx, (a_col, b_row, b_col, c_col) in enumerate(products):
        # Offset into the rank-28 vector
        offset = prod_idx * 7
        
        # A block: A[:, a_col*2 : (a_col+1)*2]
        # A flattening: [A[0,0], A[0,1], A[0,2], A[0,3], A[1,0], A[1,1], A[1,2], A[1,3]]
        # A1 = A[:, 0:2] -> indices 0,1,4,5 in flattening
        # A2 = A[:, 2:4] -> indices 2,3,6,7 in flattening
        # For A block at col a_col:
        #   A[0, a_col*2]     -> index a_col*2
        #   A[0, a_col*2+1]   -> index a_col*2+1
        #   A[1, a_col*2]     -> index 4 + a_col*2
        #   A[1, a_col*2+1]   -> index 4 + a_col*2+1
        
        a_indices = [a_col*2, a_col*2+1, 4 + a_col*2, 4 + a_col*2+1]
        
        # B block: B[b_row*2:(b_row+1)*2, b_col*2:(b_col+1)*2]
        # B flattening: row-major, so B[i,j] is at index i*4+j
        # B block indices:
        #   B[b_row*2, b_col*2]     -> (b_row*2)*4 + b_col*2
        #   B[b_row*2, b_col*2+1]   -> (b_row*2)*4 + b_col*2+1
        #   B[b_row*2+1, b_col*2]   -> (b_row*2+1)*4 + b_col*2
        #   B[b_row*2+1, b_col*2+1] -> (b_row*2+1)*4 + b_col*2+1
        
        b_indices = [
            (b_row*2)*4 + b_col*2,
            (b_row*2)*4 + b_col*2+1,
            (b_row*2+1)*4 + b_col*2,
            (b_row*2+1)*4 + b_col*2+1
        ]
        
        # C block: C[:, c_col*2 : (c_col+1)*2]
        # Output convention: W[k*n+i] where output is 2x4
        # C is 2x4, so C[i, j] for i in {0,1}, j in {0,1,2,3}
        # W index for C[i,j] is j*2 + i (since p*n = 4*2 = 8, and convention is W[k*n+i])
        # Wait, the problem says "output factor follows the official trace convention W[k*n+i,r]"
        # For C of shape (n, p) = (2, 4), the entries are C[i, j] where i in [0,n), j in [0,p)
        # The convention W[k*n+i] suggests the output is indexed as if it were (p, n)?
        # Let me re-read: "output factor follows the official trace convention W[k*n+i,r]"
        # For matrix multiplication C = A*B where A is n*m, B is m*p, C is n*p.
        # The standard tensor is T[i, j, k] = sum C[i,k] * A[i,j] * B[j,k]
        # W has shape (p*n, rank), indexed as W[k*n+i, c]
        # So W[k*n+i, c] contributes to C[i, k] with coefficient W[k*n+i, c]
        
        # For our 2x4 output C:
        # C[0, 0] -> W[0*2+0] = W[0]
        # C[0, 1] -> W[1*2+0] = W[2]
        # C[0, 2] -> W[2*2+0] = W[4]
        # C[0, 3] -> W[3*2+0] = W[6]
        # C[1, 0] -> W[0*2+1] = W[1]
        # C[1, 1] -> W[1*2+1] = W[3]
        # C[1, 2] -> W[2*2+1] = W[5]
        # C[1, 3] -> W[3*2+1] = W[7]
        
        # So C[i, j] -> W[j*2 + i]
        
        # C block at col c_col:
        #   C[0, c_col*2]     -> W[c_col*2*2 + 0] = W[4*c_col]
        #   C[0, c_col*2+1]   -> W[c_col*2*2 + 1] = W[4*c_col+1]  wait no
        #   C[0, j] -> W[j*2 + 0]
        #   C[1, j] -> W[j*2 + 1]
        
        # For C block at col c_col (columns c_col*2 and c_col*2+1):
        #   C[0, c_col*2]   -> W[(c_col*2)*2 + 0] = W[4*c_col]
        #   C[0, c_col*2+1] -> W[(c_col*2+1)*2 + 0] = W[4*c_col+2]
        #   C[1, c_col*2]   -> W[(c_col*2)*2 + 1] = W[4*c_col+1]
        #   C[1, c_col*2+1] -> W[(c_col*2+1)*2 + 1] = W[4*c_col+3]
        
        c_indices = [
            (c_col*2)*2 + 0,      # C[0, c_col*2]
            (c_col*2+1)*2 + 0,    # C[0, c_col*2+1]
            (c_col*2)*2 + 1,      # C[1, c_col*2]
            (c_col*2+1)*2 + 1     # C[1, c_col*2+1]
        ]
        
        # Now fill in U, V, W for this product's 7 terms
        for c in range(7):
            rank_idx = offset + c
            
            # U: for each of the 4 A entries, add U2[a_idx, c]
            for a_idx, a_flat_idx in enumerate(a_indices):
                U[a_flat_idx][rank_idx] = U2[a_idx][c]
            
            # V: for each of the 4 B entries, add V2[b_idx, c]
            for b_idx, b_flat_idx in enumerate(b_indices):
                V[b_flat_idx][rank_idx] = V2[b_idx][c]
            
            # W: for each of the 4 C entries, add W2[c_idx, c]
            for c_idx, c_flat_idx in enumerate(c_indices):
                W[c_flat_idx][rank_idx] = W2[c_idx][c]
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
