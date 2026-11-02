"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-28 tensor decomposition for 2x2 * 2x8 matrix multiplication
    using Strassen's algorithm on four 2x2 blocks of the second matrix.
    """
    n, m, p = DIMENSIONS
    rank = 28
    
    # U: (n*m, rank) = (4, 28)
    # V: (m*p, rank) = (16, 28)
    # W: (p*n, rank) = (16, 28)
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Indices for A (n*m = 4):
    # A[0,0] -> 0, A[0,1] -> 1, A[1,0] -> 2, A[1,1] -> 3
    idx_A00 = 0
    idx_A01 = 1
    idx_A10 = 2
    idx_A11 = 3
    
    # Indices for B (m*p = 16):
    # B[j, k] -> j*p + k
    # Block i (0..3) uses columns 2i and 2i+1.
    # B[0, 2i] -> 0*8 + 2i = 2i
    # B[0, 2i+1] -> 0*8 + 2i+1 = 2i+1
    # B[1, 2i] -> 1*8 + 2i = 8+2i
    # B[1, 2i+1] -> 1*8 + 2i+1 = 8+2i+1
    
    # Indices for C (p*n = 16):
    # C[i, k] -> k*n + i
    # Block i produces C[0, 2i], C[0, 2i+1], C[1, 2i], C[1, 2i+1]
    # C[0, 2i] -> 2i*2 + 0 = 4i
    # C[1, 2i] -> 2i*2 + 1 = 4i+1
    # C[0, 2i+1] -> (2i+1)*2 + 0 = 4i+2
    # C[1, 2i+1] -> (2i+1)*2 + 1 = 4i+3
    
    col = 0
    for i in range(4):
        # B indices for this block
        b00_idx = 2 * i
        b01_idx = 2 * i + 1
        b10_idx = 8 + 2 * i
        b11_idx = 8 + 2 * i + 1
        
        # C indices for this block
        c00_idx = 4 * i
        c10_idx = 4 * i + 1
        c01_idx = 4 * i + 2
        c11_idx = 4 * i + 3
        
        # --- Strassen Term 1: P1 = (A00+A11) * (B00+B11) ---
        U[idx_A00][col] = 1.0
        U[idx_A11][col] = 1.0
        V[b00_idx][col] = 1.0
        V[b11_idx][col] = 1.0
        W[c00_idx][col] = 1.0  # C11
        W[c11_idx][col] = 1.0  # C22
        col += 1
        
        # --- Strassen Term 2: P2 = (A10+A11) * B00 ---
        U[idx_A10][col] = 1.0
        U[idx_A11][col] = 1.0
        V[b00_idx][col] = 1.0
        W[c10_idx][col] = 1.0  # C21
        W[c11_idx][col] = -1.0 # C22
        col += 1
        
        # --- Strassen Term 3: P3 = A00 * (B01 - B11) ---
        U[idx_A00][col] = 1.0
        V[b01_idx][col] = 1.0
        V[b11_idx][col] = -1.0
        W[c01_idx][col] = 1.0  # C12
        W[c11_idx][col] = 1.0  # C22
        col += 1
        
        # --- Strassen Term 4: P4 = A11 * (B10 - B00) ---
        U[idx_A11][col] = 1.0
        V[b10_idx][col] = 1.0
        V[b00_idx][col] = -1.0
        W[c00_idx][col] = 1.0  # C11
        W[c10_idx][col] = 1.0  # C21
        col += 1
        
        # --- Strassen Term 5: P5 = (A00+A01) * B11 ---
        U[idx_A00][col] = 1.0
        U[idx_A01][col] = 1.0
        V[b11_idx][col] = 1.0
        W[c00_idx][col] = -1.0 # C11
        W[c01_idx][col] = 1.0  # C12
        col += 1
        
        # --- Strassen Term 6: P6 = (A10-A00) * (B00+B01) ---
        U[idx_A10][col] = 1.0
        U[idx_A00][col] = -1.0
        V[b00_idx][col] = 1.0
        V[b01_idx][col] = 1.0
        W[c11_idx][col] = 1.0  # C22
        col += 1
        
        # --- Strassen Term 7: P7 = (A01-A11) * (B10+B11) ---
        U[idx_A01][col] = 1.0
        U[idx_A11][col] = -1.0
        V[b10_idx][col] = 1.0
        V[b11_idx][col] = 1.0
        W[c00_idx][col] = 1.0  # C11
        col += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
