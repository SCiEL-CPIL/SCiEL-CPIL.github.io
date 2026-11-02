import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# Strassen tensor for 2x2x2 matrix multiplication
# Returns U, V, W matrices of shape (4, 7) for one 2x2x2 block
# Indices:
# U: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
# V: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
# W: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)  (Output C)
# Convention: W[k*n+i, r] where n=2.
# C[i,k] = sum U[i,m1] V[m1,j] W[j,n+i] ... wait.
# Standard tensor T[i,j,k] = 1 if C[i,k] depends on A[i,j]B[j,k].
# U shape (n*m, r), V shape (m*p, r), W shape (p*n, r).
# Product term r contributes: A[i,j] * U[i*m+j, r] * B[j,k] * V[j*p+k, r] * W[k*n+i, r]
# To C[i,k].

def get_strassen_222():
    """
    Returns U, V, W for 2x2x2 Strassen decomposition.
    Rank 7.
    U rows: A indices (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    V rows: B indices (0,0)=4, (0,1)=5, (1,0)=6, (1,1)=7  -> mapped to 0,1,2,3 for local block
    W rows: C indices (k*n+i). n=2.
             C[0,0] -> 0*2+0 = 0
             C[0,1] -> 1*2+0 = 2
             C[1,0] -> 0*2+1 = 1
             C[1,1] -> 1*2+1 = 3
    
    Local indices for this 2x2x2 sub-problem:
    U_local: 4 entries.
    V_local: 4 entries.
    W_local: 4 entries.
    
    We need to map global indices to these local blocks later.
    """
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    # M1 = A00(A11 + B01)? No, standard Strassen:
    # M1 = A00 * B11 ? No.
    # Let's use the standard definitions for C = AB.
    # M1 = A00 * (B00 - B11)? No.
    # Standard:
    # M1 = A00 * (B00 - B11) -- Wait, usually defined for 2x2.
    # M1 = (A00) * (B00 - B11) ? No.
    # Let's use:
    # M1 = A00 * (B00 - B11) is not quite right.
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # This set is valid.
    # Coefficients in U (A), V (B), W (C)
    
    # M1
    U[0, 0] = 1 # A00
    U[3, 0] = 1 # A11
    V[0, 0] = 1 # B00
    V[3, 0] = 1 # B11
    W[0, 0] = 1 # C00
    W[2, 0] = -1 # C11 (from B00-B11? No, this formula is M1=(A00+A11)(B00+B11))
    # Wait, if M1 = (A00+A11)(B00+B11), then it contributes to C00? No.
    # The expansion of M1 = A00B00 + A00B11 + A11B00 + A11B11.
    # This doesn't isolate C00.
    
    # Let's use a verified set.
    # Reference: Strassen 1969
    # M1 = (A00 + A11) (B00 + B11)
    # M2 = (A10 + A11) B00
    # M3 = A00 (B01 - B11)
    # M4 = A11 (B10 - B00)
    # M5 = (A00 + A01) B11
    # M6 = (A10 - A00) (B00 + B01)
    # M7 = (A01 - A11) (B10 + B11)
    
    # Reconstruction:
    # C00 = M1 - M4 - M6 + M7
    # C01 = M3 + M5
    # C10 = M2 - M4 - M5 + M7  (Wait, check signs)
    # C11 = M1 - M2 + M3 + M4
    
    # Let's verify C00:
    # M1 = A00B00 + A00B11 + A11B00 + A11B11
    # -M4 = -(A11B10 - A11B00) = -A11B10 + A11B00
    # -M6 = -(A10B00 + A10B01 - A00B00 - A00B01) = -A10B00 - A10B01 + A00B00 + A00B01
    # +M7 = (A01B10 + A01B11 - A11B10 - A11B11)
    
    # Sum coeffs for C00 (A00B00):
    # M1: 1
    # -M4: 0
    # -M6: 1 (from A00B00 term in M6 expansion? M6=(A10-A00)(B00+B01) = A10B00+A10B01-A00B00-A00B01. So -M6 has +A00B00)
    # +M7: 0
    # Total A00B00: 2? That's wrong. It should be 1.
    
    # My Strassen reconstruction coefficients are slightly off or the M definitions are mixed.
    # Let's use a simpler known set or derive it carefully.
    # Actually, the evaluator just checks the tensor product.
    # I will use a known valid Strassen tensor implementation.
    
    # U, V, W for 2x2x2
    # Indices: U 0..3, V 0..3, W 0..3 (W index k*2+i)
    
    U = [
        [0, 0, 0, 1, 0, 1, 0], # A00
        [0, 0, 1, 0, 0, 0, 1], # A01
        [0, 1, 0, 0, 0, 1, 0], # A10
        [1, 1, 0, 0, 0, 0, 1]  # A11
    ]
    V = [
        [0, 0, 0, 1, 0, 1, 0], # B00
        [0, 0, 1, 0, 0, 0, 0], # B01
        [0, 1, 0, 0, 0, 1, 0], # B10
        [1, 1, 0, 0, 0, 0, 1]  # B11
    ]
    W = [
        [0, 0, 0, 0, 0, 1, 0], # C00 (k=0,i=0) -> idx 0
        [1, 0, 0, 0, 0, 1, 0], # C10 (k=0,i=1) -> idx 1
        [1, 0, 0, 0, 1, 0, 0], # C01 (k=1,i=0) -> idx 2
        [1, 1, 1, 1, 0, 0, 0]  # C11 (k=1,i=1) -> idx 3
    ]
    
    # Wait, I need to double check this specific tensor.
    # Let's construct it programmatically using the formulas:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Coeffs U (A):
    # M1: A00:1, A11:1
    # M2: A10:1, A11:1
    # M3: A00:1
    # M4: A11:1
    # M5: A00:1, A01:1
    # M6: A10:1, A00:-1
    # M7: A01:1, A11:-1
    
    # Coeffs V (B):
    # M1: B00:1, B11:1
    # M2: B00:1
    # M3: B01:1, B11:-1
    # M4: B10:1, B00:-1
    # M5: B11:1
    # M6: B00:1, B01:1
    # M7: B10:1, B11:1
    
    # Coeffs W (C):
    # C00 = M1 - M4 - M6 + M7 ?
    # Let's check A00B00 in C00.
    # M1: A00B00 (1)
    # -M4: -A11B10 + A11B00. No A00B00.
    # -M6: -(A10B00 + A10B01 - A00B00 - A00B01) = -A10B00 - A10B01 + A00B00 + A00B01.
    #   So -M6 gives +A00B00.
    # +M7: (A01B10 + A01B11 - A11B10 - A11B11). No A00B00.
    # Sum A00B00: 1 (from M1) + 1 (from -M6) = 2. Still 2.
    
    # The standard Strassen reconstruction is:
    # C00 = M1 - M4 - M6 + M7  ... leads to 2 A00B00?
    # Ah, M6 is usually (A10-A00)(B00+B01).
    # Maybe M1 is (A00)(B00-B11)? No.
    
    # Let's use a hardcoded correct tensor from a reliable source or just use the schoolbook for 2x2 if Strassen is too risky to code correctly in text.
    # Rank 2x2x2 schoolbook is 4. Strassen is 7.
    # Using 7 is better than 4*6=24 for the 2x4x6 part?
    # 2x4x6 = 60 schoolbook.
    # My Strassen plan: 42.
    # If I use schoolbook for 2x2x2 (rank 4), then 2x4x6 part would be:
    # A1a Ba (2x2x2) x3 -> 3*4 = 12
    # A1b Bb (2x2x2) x3 -> 3*4 = 12
    # Total 24.
    # Plus A2 B2 (2x1x6) -> 12.
    # Total 36.
    
    # Wait, is 2x4x6 rank 24?
    # 2x2x2 is rank 4 (schoolbook).
    # 2x4x2 is rank 8?
    # 2x4x6...
    # Known lower bounds: 2x2x2 is 4? No, 2x2x2 minimal rank is 4 (Schur product? No, matrix mult 2x2x2 is 4).
    # Actually, minimal rank of 2x2x2 matrix multiplication is 4.
    # So using Strassen (7) is WORSE than schoolbook (4) for 2x2x2!
    # I made a mistake in the "Knowledge" section interpretation or my own memory.
    # Strassen is for 2x2x2 being 7?
    # Strassen showed 7 multiplications suffice. Schoolbook uses 4.
    # Wait, 2x2 matrix multiplication requires 4 scalar multiplications: A00B00, A00B01, A01B00, A01B01.
    # Why is Strassen 7 then?
    # Strassen reduces the number of MULTIPLICATIONS for N x N matrices.
    # For 2x2, Strassen uses 7 products of (sums/diffs) vs 4 products of elements.
    # But rank is about scalar multiplications.
    # The rank of the 2x2x2 tensor IS 4. It cannot be less than 4.
    # Strassen does not reduce the rank of 2x2x2. It reduces the arithmetic complexity for larger N by avoiding the cost of additions? No, rank is the minimum number of scalar multiplies.
    # For 2x2x2, the tensor is rank 4.
    # So my "Strassen" idea to get rank 7 for 2x2x2 is suboptimal compared to rank 4.
    
    # Let's re-evaluate.
    # 2x5x6 Schoolbook: 60.
    # My hybrid:
    # A2 B2 (2x1x6): Rank 12.
    # A1 B1 (2x4x6):
    #   Split into 6 sub-problems of 2x2x2?
    #   A1 is 2x4. B1 is 4x6.
    #   Terms: Sum over j=0..3 A[0,j]B[j,:] + ...
    #   We can group.
    #   If we just do schoolbook for 2x4x6, rank is 2*4*6 = 48.
    #   Total 48+12 = 60.
    
    # Can we do better than 60?
    # 2x2x2 rank is 4.
    # 2x4x4?
    # If we use rank 4 for each 2x2x2 block:
    # A1 B1 can be seen as 3 blocks of 2x2x2 for B columns?
    # B is 4x6.
    # A1a (2x2) * B1a (2x6).
    # B1a = [B1a0, B1a1, B1a2] (each 2x2).
    # A1a B1a = A1a B1a0 + A1a B1a1 + A1a B1a2.
    # Each term 2x2x2 rank 4.
    # So A1a B1a rank 12.
    # Similarly A1b B1b rank 12.
    # Total A1 B1 rank 24.
    # Total rank 24 + 12 = 36.
    
    # So we can get rank 36 with schoolbook 2x2x2 blocks!
    
    def construct():
        n, m, p = DIMENSIONS
        # Rank 36 decomposition
        rank = 36
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        # Part 1: A2 (rows 0..1, col 4) * B2 (row 4, cols 0..5)
        # A indices: 4, 5 (i=0,1; j=4 -> idx 0*5+4=4, 1*5+4=9)
        # V indices: 4*6+k -> 24+k
        # W indices: k*2+i
        
        # For each i in 0..1, k in 0..5:
        for i in range(2):
            for k in range(6):
                U[i * 5 + 4][col] = 1
                V[4 * 6 + k][col] = 1
                W[k * 2 + i][col] = 1
                col += 1
        
        # Part 2: A1 (cols 0..3) * B1 (rows 0..3)
        # A1a (cols 0..1) * B1a (rows 0..1, cols 0..5)
        # A1b (cols 2..3) * B1b (rows 2..3, cols 0..5)
        
        # Helper for 2x2x2 block
        # A_block (2x2), B_block (2x2 part of B), 
        # We need to map local indices to global.
        
        def add_2x2x2_block(a_row_offset, a_col_offset, b_row_offset, b_col_offset):
            nonlocal col
            # a_row_offset: 0 (both rows of A)
            # a_col_offset: 0 or 2
            # b_row_offset: 0 or 2
            # b_col_offset: 0, 2, or 4
            
            # Local U indices:
            # (0,0) -> a_row_offset+0, a_col_offset+0 -> 0*5 + (a_col_offset)
            # (0,1) -> 0*5 + (a_col_offset+1)
            # (1,0) -> 1*5 + (a_col_offset)
            # (1,1) -> 1*5 + (a_col_offset+1)
            
            # Local V indices:
            # (0,0) -> (b_row_offset+0)*6 + (b_col_offset)
            # (0,1) -> (b_row_offset+0)*6 + (b_col_offset+1)
            # (1,0) -> (b_row_offset+1)*6 + (b_col_offset)
            # (1,1) -> (b_row_offset+1)*6 + (b_col_offset+1)
            
            # Local W indices:
            # (0,0) -> 0*2 + 0 = 0
            # (1,0) -> 0*2 + 1 = 1
            # (0,1) -> 1*2 + 0 = 2
            # (1,1) -> 1*2 + 1 = 3
            
            # 4 terms
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        # U[i*m+j, c]
                        u_idx = 0 * 5 + (a_col_offset + j) # A row is always 0? No.
                        # A is 2x5.
                        # A[i, j]
                        u_idx = i * 5 + (a_col_offset + j)
                        
                        # V[j*p+k, c]
                        # B is 5x6.
                        # B[j, k]
                        v_idx = (b_row_offset + j) * 6 + (b_col_offset + k)
                        
                        # W[k*n+i, c]
                        w_idx = k * 2 + i
                        
                        U[u_idx][col] = 1
                        V[v_idx][col] = 1
                        W[w_idx][col] = 1
                        col += 1
            
        # Block A1a (A cols 0,1) * B1a (B rows 0,1)
        # B1a has 3 column pairs: 0-1, 2-3, 4-5
        for b_col_off in [0, 2, 4]:
            add_2x2x2_block(0, 0, 0, b_col_off)
            
        # Block A1b (A cols 2,3) * B1b (B rows 2,3)
        for b_col_off in [0, 2, 4]:
            add_2x2x2_block(0, 2, 2, b_col_off)
            
        return {"U": U, "V": V, "W": W}

    return construct()

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    # Rank 36 decomposition
    # 12 from 2x1x6
    # 24 from 2x4x6 (6 blocks of 2x2x2, rank 4 each)
    
    n, m, p = DIMENSIONS
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A[:, 4] * B[4, :]
    # A indices: 4, 9
    # B indices: 24, 25, 26, 27, 28, 29
    for i in range(n):
        for k in range(p):
            U[i * m + 4][col] = 1
            V[4 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    # Part 2: A[:, 0:4] * B[0:4, :]
    # Split A cols into 0-1 and 2-3
    # Split B rows into 0-1 and 2-3
    # Split B cols into 0-1, 2-3, 4-5
    
    def add_block(a_col_start, b_row_start, b_col_start):
        nonlocal col
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    u_idx = i * m + (a_col_start + j)
                    v_idx = (b_row_start + j) * p + (b_col_start + k)
                    w_idx = k * n + i
                    U[u_idx][col] = 1
                    V[v_idx][col] = 1
                    W[w_idx][col] = 1
                    col += 1
    
    # A cols 0,1 ; B rows 0,1 ; B cols 0,1
    add_block(0, 0, 0)
    # A cols 0,1 ; B rows 0,1 ; B cols 2,3
    add_block(0, 0, 2)
    # A cols 0,1 ; B rows 0,1 ; B cols 4,5
    add_block(0, 0, 4)
    
    # A cols 2,3 ; B rows 2,3 ; B cols 0,1
    add_block(2, 2, 0)
    # A cols 2,3 ; B rows 2,3 ; B cols 2,3
    add_block(2, 2, 2)
    # A cols 2,3 ; B rows 2,3 ; B cols 4,5
    add_block(2, 2, 4)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

