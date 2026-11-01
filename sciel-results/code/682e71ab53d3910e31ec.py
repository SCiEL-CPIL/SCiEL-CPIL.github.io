"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
import time

def construct():
    """
    Return a tensor decomposition for 2x2x5 matrix multiplication.
    
    Uses a hybrid approach:
    - Strassen's algorithm (rank 7) for the first two 2x2 blocks of B.
    - Schoolbook algorithm (rank 4) for the remaining 2x1 block of B.
    Total rank = 7 + 7 + 4 = 18.
    """
    n, m, p = 2, 2, 5
    rank = 18
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x 18
    V = [[0] * rank for _ in range(m * p)]  # 10 x 18
    W = [[0] * rank for _ in range(p * n)]  # 10 x 18
    
    col = 0
    
    # Helper to add a Strassen block
    # B indices for this block: j in {0, 1}, k_offset
    # A indices: i in {0, 1}, j in {0, 1}
    def add_strassen(k_offset):
        nonlocal col
        # M1: (A11+A22)(B11+B22)
        U[0][col] += 1; U[3][col] += 1
        V[0 + k_offset*2][col] += 1; V[1 + k_offset*2][col] += 1
        # W: C11+=1, C22+=1, C12-=-1, C21-=-1
        # W index: k*n + i
        # C11: k=0, i=0 -> 0
        # C22: k=1, i=1 -> 1*2+1=3
        # C12: k=1, i=0 -> 1*2+0=2
        # C21: k=0, i=1 -> 0*2+1=1
        W[0][col] += 1
        W[3][col] += 1
        W[2][col] += -1
        W[1][col] += -1
        col += 1
        
        # M2: (A21+A22)B11
        U[2][col] += 1; U[3][col] += 1
        V[0 + k_offset*2][col] += 1
        # W: C21+=1, C22-=-1
        W[1][col] += 1
        W[3][col] += -1
        col += 1
        
        # M3: A11(B12-B22)
        U[0][col] += 1
        V[1 + k_offset*2][col] += 1
        V[3 + k_offset*2][col] += -1 # Wait, B indices for 2x2 block k_offset
        # B is 2x5.
        # Block 1: k=0,1. B indices in V: j*p + k.
        # j=0, k=0 -> 0. j=0, k=1 -> 1.
        # j=1, k=0 -> 2. j=1, k=1 -> 3.
        # So for k_offset=0:
        # B11 is V[0], B12 is V[1], B21 is V[2], B22 is V[3].
        # My previous V indices were wrong in the helper logic if not careful.
        # Let's define V indices explicitly.
        # j=0, k=0 -> 0
        # j=0, k=1 -> 1
        # j=1, k=0 -> 2
        # j=1, k=1 -> 3
        # So B11=V[0], B12=V[1], B21=V[2], B22=V[3].
        
        # Correction for M3: A11(B12-B22)
        # V[1] += 1, V[3] += -1
        V[1 + k_offset*2][col] = 1 # This is B12 for k_offset=0?
        # If k_offset=0, k=1 is index 1. Correct.
        # If k_offset=2, k=0 is index 2, k=1 is index 3.
        # So B12 is V[1 + k_offset*2]?
        # j=0, k=1 -> 0*5 + 1 = 1.
        # j=0, k=3 -> 0*5 + 3 = 3.
        # So yes, B12 index is 1 + k_offset*2?
        # No, k_offset is the start column index.
        # If k_offset=0, cols are 0,1. B12 is col 1. Index 1.
        # If k_offset=2, cols are 2,3. B12 is col 3. Index 3.
        # So index is 1 + k_offset*2?
        # 1 + 0*2 = 1.
        # 1 + 2*2 = 5? No, 1 + 2 = 3.
        # The formula for V index is j*p + k.
        # j=0. k = k_offset + 1.
        # Index = 0*5 + (k_offset + 1) = k_offset + 1.
        # My previous `1 + k_offset*2` is wrong.
        
        # Let's restart the helper with correct indices.
        
        return

    # Clear U, V, W
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0

    def add_strassen_block(k_start):
        nonlocal col
        # k_start is 0 or 2
        # B indices:
        # B11: j=0, k=k_start -> 0*5 + k_start = k_start
        # B12: j=0, k=k_start+1 -> k_start+1
        # B21: j=1, k=k_start -> 1*5 + k_start = 5 + k_start
        # B22: j=1, k=k_start+1 -> 5 + k_start + 1
        
        # M1: (A11+A22)(B11+B22)
        U[0][col] += 1; U[3][col] += 1
        V[k_start][col] += 1
        V[5 + k_start + 1][col] += 1
        W[0][col] += 1  # C11
        W[3][col] += 1  # C22
        W[2][col] += -1 # C12
        W[1][col] += -1 # C21
        col += 1
        
        # M2: (A21+A22)B11
        U[2][col] += 1; U[3][col] += 1
        V[k_start][col] += 1
        W[1][col] += 1  # C21
        W[3][col] += -1 # C22
        col += 1
        
        # M3: A11(B12-B22)
        U[0][col] += 1
        V[k_start+1][col] += 1
        V[5 + k_start + 1][col] += -1
        W[2][col] += 1  # C12
        W[3][col] += 1  # C22
        col += 1
        
        # M4: A22(B21-B11)
        U[3][col] += 1
        V[5 + k_start][col] += 1
        V[k_start][col] += -1
        W[0][col] += 1  # C11
        W[1][col] += 1  # C21
        col += 1
        
        # M5: (A11+A12)B22
        U[0][col] += 1; U[1][col] += 1
        V[5 + k_start + 1][col] += 1
        W[0][col] += -1 # C11
        W[2][col] += 1  # C12
        col += 1
        
        # M6: (A21-A11)(B11+B12)
        U[2][col] += 1; U[0][col] += -1
        V[k_start][col] += 1
        V[k_start+1][col] += 1
        W[3][col] += 1  # C22
        col += 1
        
        # M7: (A12-A22)(B21+B22)
        U[1][col] += 1; U[3][col] += -1
        V[5 + k_start][col] += 1
        V[5 + k_start + 1][col] += 1
        W[0][col] += 1  # C11
        W[1][col] += 1  # C21
        col += 1

    # Add Strassen block for k=0,1
    add_strassen_block(0)
    # Add Strassen block for k=2,3
    add_strassen_block(2)
    
    # Schoolbook for k=4
    # C11 += A11 B14 (j=0, k=4)
    U[0][col] += 1
    V[4][col] += 1
    W[0][col] += 1
    col += 1
    
    # C12 += A12 B24 (j=1, k=4)
    U[1][col] += 1
    V[5 + 4][col] += 1
    W[2][col] += 1
    col += 1
    
    # C21 += A21 B14 (j=0, k=4)
    U[2][col] += 1
    V[4][col] += 1
    W[1][col] += 1
    col += 1
    
    # C22 += A22 B24 (j=1, k=4)
    U[3][col] += 1
    V[5 + 4][col] += 1
    W[3][col] += 1
    col += 1

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
