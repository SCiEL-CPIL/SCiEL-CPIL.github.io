"""Matrix multiplication 3,4,8 - rank-minimal exact tensor decomposition."""
import time

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import time

DEADLINE = time.time() + 145  # Safety margin before 150s limit

def _add_strassen_2x2x2(U, V, W, i0, j0, k0, offset):
    """
    Adds a rank-7 Strassen decomposition for a 2x2x2 sub-block.
    A rows: i0, i0+1
    A cols: j0, j0+1
    B rows: j0, j0+1
    B cols: k0, k0+1
    C rows: i0, i0+1
    C cols: k0, k0+1
    
    U index: i * 4 + j
    V index: j * 8 + k
    W index: k * 3 + i
    """
    u00 = i0 * 4 + j0
    u01 = i0 * 4 + j0 + 1
    u10 = (i0 + 1) * 4 + j0
    u11 = (i0 + 1) * 4 + j0 + 1
    
    v00 = j0 * 8 + k0
    v01 = j0 * 8 + k0 + 1
    v10 = (j0 + 1) * 8 + k0
    v11 = (j0 + 1) * 8 + k0 + 1
    
    w00 = k0 * 3 + i0
    w01 = (k0 + 1) * 3 + i0
    w10 = k0 * 3 + (i0 + 1)
    w11 = (k0 + 1) * 3 + (i0 + 1)
    
    # M1 = (A00 + A11) * (B00 + B11)
    U[u00][offset] += 1.0
    U[u11][offset] += 1.0
    V[v00][offset] += 1.0
    V[v11][offset] += 1.0
    W[w00][offset] += 1.0
    W[w11][offset] += 1.0
    
    # M2 = (A10 + A11) * B00
    U[u10][offset + 1] += 1.0
    U[u11][offset + 1] += 1.0
    V[v00][offset + 1] += 1.0
    W[w10][offset + 1] += 1.0
    W[w11][offset + 1] -= 1.0
    
    # M3 = A00 * (B01 - B11)
    U[u00][offset + 2] += 1.0
    V[v01][offset + 2] += 1.0
    V[v11][offset + 2] -= 1.0
    W[w01][offset + 2] += 1.0
    
    # M4 = A11 * (B10 - B00)
    U[u11][offset + 3] += 1.0
    V[v10][offset + 3] += 1.0
    V[v00][offset + 3] -= 1.0
    W[w00][offset + 3] += 1.0
    W[w10][offset + 3] += 1.0
    
    # M5 = (A00 + A01) * B11
    U[u00][offset + 4] += 1.0
    U[u01][offset + 4] += 1.0
    V[v11][offset + 4] += 1.0
    W[w00][offset + 4] -= 1.0
    W[w01][offset + 4] += 1.0
    
    # M6 = (A10 - A00) * (B00 + B01)
    U[u10][offset + 5] += 1.0
    U[u00][offset + 5] -= 1.0
    V[v00][offset + 5] += 1.0
    V[v01][offset + 5] += 1.0
    W[w11][offset + 5] += 1.0
    
    # M7 = (A01 - A11) * (B10 + B11)
    U[u01][offset + 6] += 1.0
    U[u11][offset + 6] -= 1.0
    V[v10][offset + 6] += 1.0
    V[v11][offset + 6] += 1.0
    W[w00][offset + 6] += 1.0
    W[w11][offset + 6] += 1.0

def _add_schoolbook_1x2x2(U, V, W, i0, j0, k0, offset):
    """
    Adds a rank-4 schoolbook decomposition for a 1x2x2 sub-block.
    A row: i0
    A cols: j0, j0+1
    B rows: j0, j0+1
    B cols: k0, k0+1
    C row: i0
    C cols: k0, k0+1
    """
    u0 = i0 * 4 + j0
    u1 = i0 * 4 + j0 + 1
    
    v00 = j0 * 8 + k0
    v01 = j0 * 8 + k0 + 1
    v10 = (j0 + 1) * 8 + k0
    v11 = (j0 + 1) * 8 + k0 + 1
    
    w0 = k0 * 3 + i0
    w1 = (k0 + 1) * 3 + i0
    
    # C0 = A0 B00 + A1 B10
    U[u0][offset] += 1.0
    V[v00][offset] += 1.0
    W[w0][offset] += 1.0
    
    U[u1][offset + 1] += 1.0
    V[v10][offset + 1] += 1.0
    W[w0][offset + 1] += 1.0
    
    # C1 = A0 B01 + A1 B11
    U[u0][offset + 2] += 1.0
    V[v01][offset + 2] += 1.0
    W[w1][offset + 2] += 1.0
    
    U[u1][offset + 3] += 1.0
    V[v11][offset + 3] += 1.0
    W[w1][offset + 3] += 1.0

def _add_2x4x4(U, V, W, i0, j0, k0, offset):
    """
    Decomposes a 2x4x4 block using two Strassen 2x2x2 operations.
    A rows: i0, i0+1
    A cols: j0, j0+1, j0+2, j0+3
    B rows: j0, j0+1, j0+2, j0+3
    B cols: k0, k0+1, k0+2, k0+3
    C rows: i0, i0+1
    C cols: k0, k0+1, k0+2, k0+3
    
    Split into four 2x2x2 blocks:
    1. A[:2,:2] @ B[:2,:2] -> C[:2,:2]  (j0..j0+1, k0..k0+1)
    2. A[:2,2:4] @ B[2:4,:2] -> C[:2,:2]  (j0+2..j0+3, k0..k0+1)
    3. A[:2,:2] @ B[:2,2:4] -> C[:2,2:4]  (j0..j0+1, k0+2..k0+3)
    4. A[:2,2:4] @ B[2:4,2:4] -> C[:2,2:4]  (j0+2..j0+3, k0+2..k0+3)
    """
    _add_strassen_2x2x2(U, V, W, i0, j0, k0, offset)
    offset += 7
    _add_strassen_2x2x2(U, V, W, i0, j0 + 2, k0, offset)
    offset += 7
    _add_strassen_2x2x2(U, V, W, i0, j0, k0 + 2, offset)
    offset += 7
    _add_strassen_2x2x2(U, V, W, i0, j0 + 2, k0 + 2, offset)
    offset += 7
    return offset

def _add_1x4x4(U, V, W, i0, j0, k0, offset):
    """
    Decomposes a 1x4x4 block using four 1x2x2 schoolbook operations.
    A row: i0
    A cols: j0, j0+1, j0+2, j0+3
    B rows: j0, j0+1, j0+2, j0+3
    B cols: k0, k0+1, k0+2, k0+3
    C row: i0
    C cols: k0, k0+1, k0+2, k0+3
    """
    _add_schoolbook_1x2x2(U, V, W, i0, j0, k0, offset)
    offset += 4
    _add_schoolbook_1x2x2(U, V, W, i0, j0 + 2, k0, offset)
    offset += 4
    _add_schoolbook_1x2x2(U, V, W, i0, j0, k0 + 2, offset)
    offset += 4
    _add_schoolbook_1x2x2(U, V, W, i0, j0 + 2, k0 + 2, offset)
    offset += 4
    return offset

def construct():
    n, m, p = 3, 4, 8
    # Rank calculation:
    # Rows 0,1 of A (2x4) @ B (4x8) -> C rows 0,1 (2x8)
    #   Split B into two 4x4 blocks (k=0..3, k=4..7)
    #   For each 4x4 block: 4 Strassen 2x2x2 = 4*7 = 28 rank
    #   Two 4x4 blocks: 2*28 = 56 rank
    # Row 2 of A (1x4) @ B (4x8) -> C row 2 (1x8)
    #   Split B into two 4x4 blocks
    #   For each 4x4 block: 4 schoolbook 1x2x2 = 4*4 = 16 rank
    #   Two 4x4 blocks: 2*16 = 32 rank
    # Total: 56 + 32 = 88
    rank = 88
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    offset = 0
    
    # For each 4x4 block of B (k=0..3 and k=4..7)
    for k_base in [0, 4]:
        # Top 2 rows of A (rows 0,1) @ 4x4 block of B
        # This gives C rows 0,1 @ cols k_base..k_base+3
        offset = _add_2x4x4(U, V, W, 0, 0, k_base, offset)
        
        if time.time() > DEADLINE:
            break
        
        # Bottom row of A (row 2) @ 4x4 block of B
        # This gives C row 2 @ cols k_base..k_base+3
        offset = _add_1x4x4(U, V, W, 2, 0, k_base, offset)
        
        if time.time() > DEADLINE:
            break
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
