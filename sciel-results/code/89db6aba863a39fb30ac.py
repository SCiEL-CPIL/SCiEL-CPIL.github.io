import time
import numpy as np

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-42 exact tensor decomposition for multiplying a 2x2 matrix by a 2x12 matrix.
    Uses Strassen's algorithm for each 2x2x2 block (6 blocks total).
    The rank is optimal as the tensor decomposes into a direct sum of 6 M_{2,2,2} tensors
    with disjoint V and W supports, preventing cross-block rank reduction.
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=12
    
    num_blocks = p // 2  # 6 blocks of width 2
    rank = num_blocks * 7  # 42
    
    # Initialize matrices
    # U shape: (n*m, rank) = (4, 42)
    # V shape: (m*p, rank) = (24, 42)
    # W shape: (p*n, rank) = (24, 42)
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Define Strassen components for 2x2x2 multiplication
    # We compute C = A * B where A, B, C are 2x2.
    # Indices:
    # A: [[A00, A01], [A10, A11]] -> flat indices 0,1,2,3 (row-major)
    # B: [[B00, B01], [B10, B11]]
    # C: [[C00, C01], [C10, C11]]
    
    # Strassen Formulas:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # C00 = M1 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For each block t (0 to 5):
    for t in range(num_blocks):
        base = t * 7
        
        # B indices (V rows)
        # B is 2x12. Block t corresponds to columns 2*t and 2*t+1.
        # Row 0 of B: indices 0..11. Row 1 of B: indices 12..23.
        # V index for B[r, col] is r * p + col.
        v_b00 = 0 * p + 2 * t
        v_b01 = 0 * p + (2 * t + 1)
        v_b10 = 1 * p + 2 * t
        v_b11 = 1 * p + (2 * t + 1)
        
        # C indices (W rows)
        # C is 2x12. Output factor follows W[k*n+i, r].
        # C[r, col]. W index is col * n + r.
        w_c00 = (2 * t) * n + 0
        w_c01 = (2 * t + 1) * n + 0
        w_c10 = (2 * t) * n + 1
        w_c11 = (2 * t + 1) * n + 1
        
        # A indices (U rows)
        # A is 2x2. Flat index: row * 2 + col.
        u_a00 = 0
        u_a01 = 1
        u_a10 = 2
        u_a11 = 3
        
        # Term 0: M1
        c = base + 0
        U[u_a00, c] = 1
        U[u_a11, c] = 1
        V[v_b00, c] = 1
        V[v_b11, c] = 1
        W[w_c00, c] = 1   # C00 += M1
        W[w_c11, c] = 1   # C11 += M1
        
        # Term 1: M2
        c = base + 1
        U[u_a10, c] = 1
        U[u_a11, c] = 1
        V[v_b00, c] = 1
        W[w_c10, c] = 1   # C10 += M2
        W[w_c11, c] = -1  # C11 -= M2
        
        # Term 2: M3
        c = base + 2
        U[u_a00, c] = 1
        V[v_b01, c] = 1
        V[v_b11, c] = -1
        W[w_c01, c] = 1   # C01 += M3
        W[w_c11, c] = 1   # C11 += M3
        
        # Term 3: M4
        c = base + 3
        U[u_a11, c] = 1
        V[v_b10, c] = 1
        V[v_b00, c] = -1
        W[w_c00, c] = 1   # C00 += M4
        W[w_c10, c] = 1   # C10 += M4
        
        # Term 4: M5
        c = base + 4
        U[u_a00, c] = 1
        U[u_a01, c] = 1
        V[v_b11, c] = 1
        W[w_c00, c] = -1  # C00 -= M5
        W[w_c01, c] = 1   # C01 += M5
        
        # Term 5: M6
        c = base + 5
        U[u_a10, c] = 1
        U[u_a00, c] = -1
        V[v_b00, c] = 1
        V[v_b01, c] = 1
        W[w_c11, c] = 1   # C11 += M6
        
        # Term 6: M7
        c = base + 6
        U[u_a01, c] = 1
        U[u_a11, c] = -1
        V[v_b10, c] = 1
        V[v_b11, c] = 1
        W[w_c00, c] = 1   # C00 += M7

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
