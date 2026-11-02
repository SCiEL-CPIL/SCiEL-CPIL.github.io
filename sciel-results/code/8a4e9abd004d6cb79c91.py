import time
import sys

DIMENSIONS = [2, 2, 7]

def construct():
    start_time = time.time()
    deadline = 150.0 # Safety margin
    
    n, m, p = DIMENSIONS
    # We will construct a decomposition of rank 26
    # Block 1: cols 0,1 of B (2x2x2 Strassen, rank 7)
    # Block 2: cols 2,3 of B (2x2x2 Strassen, rank 7)
    # Block 3: cols 4,5,6 of B (2x2x3 Schoolbook, rank 12)
    
    total_rank = 26
    
    # Initialize matrices
    # U: (n*m, rank) = (4, 26)
    # V: (m*p, rank) = (14, 26)
    # W: (p*n, rank) = (14, 26)
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col_offset = 0
    
    # Helper to add Strassen block
    def add_strassen_block(k_start):
        nonlocal col_offset
        # k_start is the global column index of B for this block (0 or 2)
        # Local k is 0, 1. Global k = k_start + local_k
        # Local C col is 0, 1. Global C col = k_start + local_k
        
        # P1: (A00+A11)(B00+B11)
        # U: A00, A11
        U[0][col_offset] = 1
        U[3][col_offset] = 1
        # V: B00, B11 (local) -> Global j=0,k=k_start; j=1,k=k_start
        V[0 * 7 + k_start][col_offset] = 1
        V[1 * 7 + k_start][col_offset] = 1
        # W: C00 (coeff 1), C11 (coeff 1)
        # C index: k*n + i
        # C00: k=0, i=0 -> 0*2+0=0
        # C11: k=1, i=1 -> 1*2+1=3
        # Global C col for block is k_start.
        # So C00 global is (k_start)*2 + 0
        # C11 global is (k_start+1)*2 + 1
        W[k_start * 2 + 0][col_offset] = 1
        W[(k_start + 1) * 2 + 1][col_offset] = 1
        col_offset += 1
        
        # P2: (A10+A11)B00
        U[2][col_offset] = 1 # A10 is i=1, j=0 -> 1*2+0=2
        U[3][col_offset] = 1 # A11 is i=1, j=1 -> 1*2+1=3
        V[0 * 7 + k_start][col_offset] = 1 # B00 local
        # W: C10 (coeff 1)
        # C10: k=0, i=1 -> 0*2+1=1
        W[k_start * 2 + 1][col_offset] = 1
        col_offset += 1
        
        # P3: A00(B01-B11)
        U[0][col_offset] = 1
        V[0 * 7 + (k_start + 1)][col_offset] = 1 # B01 local (k=1)
        V[1 * 7 + (k_start + 1)][col_offset] = -1 # B11 local
        # W: C01 (coeff 1)
        # C01: k=1, i=0 -> 1*2+0=2
        W[(k_start + 1) * 2 + 0][col_offset] = 1
        col_offset += 1
        
        # P4: A11(B10-B00)
        U[3][col_offset] = 1
        V[1 * 7 + k_start][col_offset] = 1 # B10 local (k=0)
        V[0 * 7 + k_start][col_offset] = -1 # B00 local
        # W: C10 (coeff 1)
        W[k_start * 2 + 1][col_offset] = 1
        col_offset += 1
        
        # P5: (A00+A01)B11
        U[0][col_offset] = 1
        U[1][col_offset] = 1 # A01 is i=0, j=1 -> 0*2+1=1
        V[1 * 7 + (k_start + 1)][col_offset] = 1 # B11 local
        # W: C00 (coeff -1), C01 (coeff 1)
        W[k_start * 2 + 0][col_offset] = -1
        W[(k_start + 1) * 2 + 0][col_offset] = 1
        col_offset += 1
        
        # P6: (A10-A00)(B00+B01)
        U[2][col_offset] = 1
        U[0][col_offset] = -1
        V[0 * 7 + k_start][col_offset] = 1
        V[0 * 7 + (k_start + 1)][col_offset] = 1
        # W: C11 (coeff 1)
        W[(k_start + 1) * 2 + 1][col_offset] = 1
        col_offset += 1
        
        # P7: (A01-A11)(B10+B11)
        U[1][col_offset] = 1
        U[3][col_offset] = -1
        V[1 * 7 + k_start][col_offset] = 1
        V[1 * 7 + (k_start + 1)][col_offset] = 1
        # W: C00 (coeff 1)
        W[k_start * 2 + 0][col_offset] = 1
        col_offset += 1
        
    # Helper to add Schoolbook block
    def add_schoolbook_block(k_start, p_block):
        nonlocal col_offset
        for i in range(n):
            for j in range(m):
                for k_loc in range(p_block):
                    k_glob = k_start + k_loc
                    # U
                    U[i * m + j][col_offset] = 1
                    # V
                    V[j * p + k_glob][col_offset] = 1
                    # W
                    W[k_glob * n + i][col_offset] = 1
                    col_offset += 1

    # Execute
    add_strassen_block(0) # Cols 0,1
    if time.time() > deadline - 5:
        # Fallback if too slow (unlikely)
        pass
    add_strassen_block(2) # Cols 2,3
    add_schoolbook_block(4, 3) # Cols 4,5,6
    
    return {"U": U, "V": V, "W": W}
