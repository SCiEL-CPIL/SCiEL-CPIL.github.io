import time
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    U = []
    V = []
    W = []
    
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        # u_idx is index in U (n*m), v_idx in V (m*p), w_idx in W (p*n)
        # We append to the current rank
        pass # This approach is tricky with dynamic sizes. 
              # Better to pre-calculate the rank and fill matrices.
        
    # Let's pre-calculate the rank.
    # Rank calculation:
    # Block 1: A1 (3x2) * B1 (2x6)
    # Split n=3 into 1+2.
    # A1a (1x2) * B1 (2x6) -> Rank 12 (3 blocks of 1x2x2)
    # A1b (2x2) * B1 (2x6) -> Rank 21 (3 blocks of 2x2x2 Strassen)
    # Total Block 1: 33
    
    # Block 2: A2 (3x3) * B2 (3x6)
    # Split n=3 into 1+2.
    # A2a (1x3) * B2 (3x6) -> Rank 18 (schoolbook 1x3x6)
    # A2b (2x3) * B2 (3x6)
    #   Split m=3 into 1+2.
    #   A2b1 (2x1) * B21 (1x6) -> Rank 12 (schoolbook 2x1x6)
    #   A2b2 (2x2) * B22 (2x6)
    #     Split p=6 into 2+2+2.
    #     A2b2 * B221 (2x2) -> Rank 7
    #     A2b2 * B222 (2x2) -> Rank 7
    #     A2b2 * B223 (2x2) -> Rank 7
    #     Total A2b2*B22: 21
    #   Total A2b*B2: 12 + 21 = 33
    # Total Block 2: 18 + 33 = 51
    
    # Total Rank: 33 + 51 = 84
    
    rank = 84
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a schoolbook block
    def add_schoolbook(n_b, m_b, p_b, i_off, j_off, k_off):
        nonlocal col
        for i in range(n_b):
            for j in range(m_b):
                for k in range(p_b):
                    U[i_off + i][col] = 1.0
                    V[j_off + j][col] = 1.0
                    W[k_off + k][col] = 1.0
                    col += 1
                    if col >= rank:
                        raise ValueError("Rank exceeded")

    # Helper to add a Strassen block (2x2x2)
    def add_strassen(i_off, j_off, k_off):
        nonlocal col
        # Indices:
        # A: (i_off, j_off), (i_off, j_off+1), (i_off+1, j_off), (i_off+1, j_off+1)
        # B: (j_off, k_off), (j_off, k_off+1), (j_off+1, k_off), (j_off+1, k_off+1)
        # W: (k_off, i_off), (k_off, i_off+1), (k_off+1, i_off), (k_off+1, i_off+1)
        
        # M1 = (A11+A22)(B11+B22)
        U[i_off][col] += 1.0
        U[i_off+1][col] += 1.0
        V[j_off][col] += 1.0
        V[j_off+1][col] += 1.0
        W[k_off][col] += 1.0
        W[k_off+1][col] += 1.0
        W[k_off][col] += 1.0 # C11 gets M1
        W[k_off+1][col] += 1.0 # C12 gets M1? No.
        
        # Let's trace the W contributions for Strassen carefully.
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # We need to add 7 columns.
        
        # Col 1: M1
        # U: A11, A22
        # V: B11, B22
        # W: C11, C22
        U[i_off][col] += 1.0
        U[i_off+1][col] += 1.0
        V[j_off][col] += 1.0
        V[j_off+1][col] += 1.0
        W[k_off][col] += 1.0       # C11
        W[k_off+1][col] += 1.0     # C22
        col += 1
        
        # Col 2: M2
        # U: A21, A22
        # V: B11
        # W: C21
        U[i_off+1][col] += 1.0     # A21 is at (i_off+1, j_off)
        U[i_off+1][col] += 1.0     # A22 is at (i_off+1, j_off+1)
        V[j_off][col] += 1.0       # B11
        W[k_off][col] += 1.0       # C21 is at (k_off, i_off+1)
        col += 1
        
        # Col 3: M3
        # U: A11
        # V: B21, -B22
        # W: C12, C22
        U[i_off][col] += 1.0       # A11
        V[j_off+1][col] += 1.0     # B21
        V[j_off+1][col] -= 1.0     # -B22
        W[k_off+1][col] += 1.0     # C12
        W[k_off+1][col] += 1.0     # C22
        col += 1
        
        # Col 4: M4
        # U: A22
        # V: B12, -B11
        # W: C11, C21
        U[i_off+1][col] += 1.0     # A22
        V[j_off][col] += 1.0       # B12
        V[j_off][col] -= 1.0       # -B11
        W[k_off][col] += 1.0       # C11
        W[k_off][col] += 1.0       # C21
        col += 1
        
        # Col 5: M5
        # U: A11, A12
        # V: B22
        # W: -C11, C12
        U[i_off][col] += 1.0       # A11
        U[i_off][col] += 1.0       # A12
        V[j_off+1][col] += 1.0     # B22
        W[k_off][col] -= 1.0       # -C11
        W[k_off+1][col] += 1.0     # C12
        col += 1
        
        # Col 6: M6
        # U: A21, -A11
        # V: B11, B12
        # W: C22
        U[i_off+1][col] += 1.0     # A21
        U[i_off][col] -= 1.0       # -A11
        V[j_off][col] += 1.0       # B11
        V[j_off][col] += 1.0       # B12
        W[k_off+1][col] += 1.0     # C22
        col += 1
        
        # Col 7: M7
        # U: A12, -A22
        # V: B21, B22
        # W: C11
        U[i_off][col] += 1.0       # A12
        U[i_off+1][col] -= 1.0     # -A22
        V[j_off+1][col] += 1.0     # B21
        V[j_off+1][col] += 1.0     # B22
        W[k_off][col] += 1.0       # C11
        col += 1

    # Block 1: A1 (3x2) * B1 (2x6)
    # i_off=0, j_off=0, k_off=0
    # A1 is rows 0-2 of A, cols 0-1.
    # B1 is rows 0-1 of B, cols 0-5.
    
    # A1a (1x2) * B1 (2x6)
    # i_off=0, j_off=0, k_off=0
    # Split B1 into 3 blocks of 2x2: B11, B12, B13
    # B11: j=0,1 k=0,1
    # B12: j=0,1 k=2,3
    # B13: j=0,1 k=4,5
    for k_block in range(3):
        add_schoolbook(1, 2, 2, 0, 0, k_block * 2)
        
    # A1b (2x2) * B1 (2x6)
    # i_off=1, j_off=0, k_off=0
    # Split B1 into 3 blocks of 2x2
    for k_block in range(3):
        add_strassen(1, 0, k_block * 2)

    # Block 2: A2 (3x3) * B2 (3x6)
    # i_off=0, j_off=2, k_off=0
    # A2 is rows 0-2 of A, cols 2-4.
    # B2 is rows 2-4 of B, cols 0-5.
    
    # A2a (1x3) * B2 (3x6)
    # i_off=0, j_off=2, k_off=0
    # Schoolbook 1x3x6
    add_schoolbook(1, 3, 6, 0, 2, 0)
    
    # A2b (2x3) * B2 (3x6)
    # i_off=1, j_off=2, k_off=0
    # Split m=3 into 1+2.
    # A2b1 (2x1) * B21 (1x6)
    # i_off=1, j_off=2, k_off=0
    # Schoolbook 2x1x6
    add_schoolbook(2, 1, 6, 1, 2, 0)
    
    # A2b2 (2x2) * B22 (2x6)
    # i_off=1, j_off=3, k_off=0
    # Split p=6 into 2+2+2
    for k_block in range(3):
        add_strassen(1, 3, k_block * 2)

    if col != rank:
        print(f"Rank mismatch: expected {rank}, got {col}")
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
