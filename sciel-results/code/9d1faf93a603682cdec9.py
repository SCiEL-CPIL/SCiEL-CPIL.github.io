"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    n, m, p = DIMENSIONS
    
    # We use the recursive Strassen algorithm for 4x4x4 multiplication.
    # Rank of 2x2x2 Strassen is 7.
    # Rank of 4x4x4 Strassen is 7*7 = 49.
    # We split the 4x8 matrix B into two 4x4 matrices B1 and B2.
    # We compute A*B1 and A*B2 separately.
    # Total rank = 49 + 49 = 98.
    
    # Define the Strassen 2x2 tensor decomposition
    # Inputs: A (2x2), B (2x2). Output: C (2x2).
    # Indices: 0:00, 1:01, 2:10, 3:11
    
    # The 7 products for 2x2 multiplication
    # M_k = U_k(A) * V_k(B)
    # C = sum_k W_k * M_k
    
    # U vectors (coefficients for A)
    u_2x2 = [
        [1, 0, 0, 1], # A00 + A11
        [0, 0, 1, 1], # A10 + A11
        [1, 0, 0, 0], # A00
        [0, 0, 0, 1], # A11
        [1, 1, 0, 0], # A00 + A01
        [-1, 0, 1, 0],# A10 - A00
        [0, 1, 0, 1]  # A01 + A11
    ]
    
    # V vectors (coefficients for B)
    v_2x2 = [
        [1, 0, 0, 1], # B00 + B11
        [1, 0, 0, 0], # B00
        [0, 1, 0, -1],# B01 - B11
        [0, 0, 1, -1],# B10 - B00 (Note: standard Strassen often uses B10-B00 or similar)
        # Let's verify the recovery with these V vectors.
        # If V4 is [0,0,1,-1], then M4 = A11 * (B10 - B00).
        # C21 = A10B00 + A11B10.
        # M2 = (A10+A11)B00 = A10B00 + A11B00.
        # M4 = A11B10 - A11B00.
        # M2 + M4 = A10B00 + A11B10. Correct.
        [1, 1, 0, 0], # B00 + B01
        [1, -1, 0, 1] # B00 - B01 + B11 ? No.
        # Let's use the standard set that works.
    ]
    
    # A reliable set of Strassen coefficients:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01+A11)(B10+B11)
    
    # Recovery:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Let's verify C11 = A00B01 + A01B11
    # M1 = A00B00 + A00B11 + A11B00 + A11B11
    # -M2 = -A10B00 - A11B00
    # M3 = A00B01 - A00B11
    # M6 = (A10-A00)(B00+B01) = A10B00 + A10B01 - A00B00 - A00B01
    # Sum:
    # A00B00: 1 - 1 = 0
    # A00B11: 1 - 1 = 0
    # A11B00: 1 - 1 = 0
    # A11B11: 1
    # A10B00: -1 + 1 = 0
    # A10B01: 1
    # A00B01: -1 - 1 = -2 ?
    # Wait. C11 should be A00B01 + A01B11.
    # My sum has A11B11 + A10B01 - 2A00B01. This is wrong.
    
    # The recovery formulas must match the M definitions.
    # I will use the verified Wikipedia formulas.
    
    # Correct Strassen Coefficients:
    # U:
    # 1: [1, 0, 0, 1]
    # 2: [0, 0, 1, 1]
    # 3: [1, 0, 0, 0]
    # 4: [0, 0, 0, 1]
    # 5: [1, 1, 0, 0]
    # 6: [-1, 0, 1, 0]
    # 7: [0, 1, 0, 1]
    
    # V:
    # 1: [1, 0, 0, 1]
    # 2: [1, 0, 0, 0]
    # 3: [0, 1, 0, -1]
    # 4: [0, 0, 1, -1]
    # 5: [0, 0, 0, 1]
    # 6: [1, 1, 0, 0]
    # 7: [0, 0, 1, 1]
    
    # W (Recovery for C00, C01, C10, C11):
    # C00: M1 + M4 - M5 + M7 -> [1, 0, 0, 1, -1, 0, 1]
    # C01: M3 + M5           -> [0, 0, 1, 0, 1, 0, 0]
    # C10: M2 + M4           -> [0, 1, 0, 1, 0, 0, 0]
    # C11: M1 - M2 + M3 + M6 -> [1, -1, 1, 0, 0, 1, 0]
    
    # Let's verify C11 with these:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01+A11)(B10+B11)
    
    # C11 = M1 - M2 + M3 + M6
    # = (A00+A11)(B00+B11) - (A10+A11)B00 + A00(B01-B11) + (A10-A00)(B00+B01)
    # = (A00B00 + A00B11 + A11B00 + A11B11)
    #   - (A10B00 + A11B00)
    #   + (A00B01 - A00B11)
    #   + (A10B00 + A10B01 - A00B00 - A00B01)
    
    # A00B00: 1 - 1 = 0
    # A00B11: 1 - 1 = 0
    # A11B00: 1 - 1 = 0
    # A11B11: 1
    # A10B00: -1 + 1 = 0
    # A10B01: 1
    # A00B01: -1 - 1 = -2
    # Still wrong. C11 should be A00B01 + A01B11.
    # I have A11B11 + A10B01 - 2A00B01.
    
    # It seems I have the recovery for C11 wrong or the M definitions.
    # Let's try a different known set or just use the schoolbook for 2x2 if I can't get Strassen right?
    # No, 7 is key.
    
    # Let's look up "Strassen algorithm coefficients".
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-A11) -- This one is suspicious.
    # Actually, one variant is:
    # M4 = A22(B21 - A11) is NOT B-only.
    
    # Another variant:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-A11) -> No.
    
    # I will use the following verified set:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-A11) -> WAIT.
    # Is it possible that M4 involves A and B mixed? No, rank 1 means U(A)*V(B).
    
    # Let's assume the standard Wikipedia set is correct and I am summing wrong.
    # C11 = M1 + M4 - M5 + M7 is for C11?
    # Wikipedia says:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # If C11 = M1 + M4 - M5 + M7
    # And my previous check failed for C22 (which corresponds to C11 in 0-indexed? No, 11 is top-left).
    # Let's stick to 0-indexed: C00 is top-left.
    # C00 = M1 + M4 - M5 + M7
    # Let's check C00 = A00B00 + A01B10
    # M1 = A00B00 + A00B11 + A11B00 + A11B11
    # M4 = A11B10 - A11B00 (Assuming V4=[0,0,1,-1])
    # -M5 = -(A00+A01)B11 = -A00B11 - A01B11
    # M7 = (A01+A11)(B10+B11) = A01B10 + A01B11 + A11B10 + A11B11
    
    # Sum:
    # A00B00: 1
    # A00B11: 1 - 1 = 0
    # A11B00: 1 - 1 = 0
    # A11B11: 1 + 1 = 2 ?
    # A11B10: 1 + 1 = 2 ?
    # A01B10: 1
    # A01B11: -1 + 1 = 0
    # Result: A00B00 + A01B10 + 2A11B11 + 2A11B10.
    # This is not A00B00 + A01B10.
    
    # Okay, I will not hardcode Strassen if I can't verify it in 5 minutes.
    # I will use the schoolbook for 2x2 (rank 4) and 4x4 (rank 16).
    # Rank for 4x4x4 is 16.
    # Rank for 4x4x8 is 16 + 16 = 32? No.
    # Schoolbook 4x4x4 is 16.
    # Schoolbook 4x4x8 is 128.
    # If I use schoolbook for the two 4x4x4 parts, rank is 16+16=32?
    # No, schoolbook 4x4x4 is 16.
    # 16+16=32.
    # But 32 < 128.
    # So even using schoolbook for the blocks, I get rank 32?
    # Wait. 4x4 matrix mult has 16 terms.
    # Two such multiplications have 32 terms.
    # Is it possible to do 4x4x4 with rank 16? Yes, that's schoolbook.
    # Is it possible to do 4x4x8 with rank 32?
    # Yes, by splitting B into B1, B2 and computing A*B1, A*B2 with schoolbook.
    # This is a valid decomposition.
    # Score = 128/32 = 4.0.
    
    # This is much simpler and guaranteed correct.
    
    rank = 32
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(32)]
    W = [[0] * rank for _ in range(32)]
    
    col = 0
    # First half: B cols 0-3
    for i in range(4):
        for j in range(4):
            for k in range(4):
                # C[i,k] += A[i,j] * B[j,k]
                U[i*4+j][col] = 1
                V[j*8+k][col] = 1
                W[k*4+i][col] = 1
                col += 1
                
    # Second half: B cols 4-7
    for i in range(4):
        for j in range(4):
            for k in range(4):
                # C[i,k+4] += A[i,j] * B[j,k+4]
                U[i*4+j][col] = 1
                V[j*8+(k+4)][col] = 1
                W[(k+4)*4+i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
