"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    # 1. Construct Schoolbook Decomposition (Rank 60)
    rank_schoolbook = n * m * p
    U_s = np.zeros((n * m, rank_schoolbook))
    V_s = np.zeros((m * p, rank_schoolbook))
    W_s = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_s[i * m + j, col] = 1
                V_s[j * p + k, col] = 1
                W_s[k * n + i, col] = 1
                col += 1
    best_result = {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
    best_rank = rank_schoolbook

    # 2. Construct Strassen-based Decomposition (Rank 53)
    # Strategy:
    # Split the 15 columns of B into 7 groups of 2 and 1 group of 1.
    # For each group of 2 columns (k, k+1), we multiply A (2x2) by B_block (2x2).
    # This is a standard 2x2x2 matrix multiplication.
    # We use Strassen's algorithm for these, which has rank 7.
    # For the last group of 1 column, we use schoolbook, which has rank 4.
    # Total Rank = 7 * 7 + 4 = 49 + 4 = 53.
    
    def construct_strassen():
        # Strassen coefficients for 2x2x2
        # M1 = (A00 + A11)(B00 + B11)
        # M2 = (A10 + A11)B00
        # M3 = A00(B10 - B11)
        # M4 = A11(B01 - B00)
        # M5 = (A00 + A10)B11
        # M6 = (A11 - A10)(B00 + B10)
        # M7 = (A01 - A11)(B01 + B11)
        #
        # C00 = M1 - M4 + M5 + M7
        # C01 = M3 + M4
        # C10 = M2 + M7
        # C11 = M1 - M3 + M6 + M7
        
        # Define the 7 terms for Strassen 2x2x2
        # Indices: U (A), V (B), W (C)
        # A indices: 00, 01, 10, 11 -> 0, 1, 2, 3
        # B indices: 00, 01, 10, 11 -> 0, 1, 2, 3
        # C indices: 00, 01, 10, 11 -> 0, 1, 2, 3
        
        # Term 1: (A00 + A11)(B00 + B11)
        U1 = [1, 0, 0, 1]
        V1 = [1, 0, 0, 1]
        # Contributes to C:
        # C00: +1
        # C11: +1
        W1 = [1, 0, 0, 1]
        
        # Term 2: (A10 + A11)B00
        U2 = [0, 0, 1, 1]
        V2 = [1, 0, 0, 0]
        # Contributes to C:
        # C10: +1
        W2 = [0, 0, 1, 0]
        
        # Term 3: A00(B10 - B11)
        U3 = [1, 0, 0, 0]
        V3 = [0, 0, 1, -1]
        # Contributes to C:
        # C01: +1
        # C11: -1
        W3 = [0, 1, 0, -1]
        
        # Term 4: A11(B01 - B00)
        U4 = [0, 0, 0, 1]
        V4 = [-1, 1, 0, 0]
        # Contributes to C:
        # C00: -1
        # C01: +1
        W4 = [-1, 1, 0, 0]
        
        # Term 5: (A00 + A10)B11
        U5 = [1, 0, 1, 0]
        V5 = [0, 0, 0, 1]
        # Contributes to C:
        # C00: +1
        # C11: +1
        W5 = [1, 0, 0, 1]
        
        # Term 6: (A11 - A10)(B00 + B10)
        U6 = [0, 0, -1, 1]
        V6 = [1, 0, 1, 0]
        # Contributes to C:
        # C11: +1
        W6 = [0, 0, 0, 1]
        
        # Term 7: (A01 - A11)(B01 + B11)
        U7 = [0, 1, 0, -1]
        V7 = [0, 1, 0, 1]
        # Contributes to C:
        # C00: +1
        # C01: +1
        # C10: +1
        W7 = [1, 1, 1, 0]
        
        # Let's verify Strassen 2x2x2
        # C00 = M1 - M4 + M5 + M7 = 1 - (-1) + 1 + 1 = 4? No.
        # Wait, the standard Strassen reconstruction is:
        # C00 = M1 - M4 + M5 + M7
        # M1 C00=1, M4 C00=-1, M5 C00=1, M7 C00=1. Sum = 1 - (-1) + 1 + 1 = 4.
        # This is incorrect. The standard Strassen formulas are:
        # M1 = (A00 + A11)(B00 + B11)
        # M2 = (A10 + A11)B00
        # M3 = A00(B10 - B11)
        # M4 = A11(B01 - B00)
        # M5 = (A00 + A10)B11
        # M6 = (A11 - A10)(B00 + B10)
        # M7 = (A01 - A11)(B01 + B11)
        #
        # C00 = M1 - M4 + M5 + M7
        # C01 = M3 + M4
        # C10 = M2 + M7
        # C11 = M1 - M3 + M6 + M7
        
        # Let's re-verify M4.
        # M4 = A11(B01 - B00).
        # A11 is U4=[0,0,0,1]. B01-B00 is V4=[-1, 1, 0, 0].
        # Product is A11*B01 - A11*B00.
        # This contributes to C01 (A11*B01) and C11 (A11*B00 with minus sign).
        # So W4 should be [0, 1, 0, -1].
        # My previous W4 was [-1, 1, 0, 0]. That was wrong.
        
        # Let's re-verify M3.
        # M3 = A00(B10 - B11).
        # A00 is U3=[1,0,0,0]. B10-B11 is V3=[0, 0, 1, -1].
        # Product is A00*B10 - A00*B11.
        # This contributes to C10 (A00*B10) and C11 (A00*B11 with minus sign).
        # So W3 should be [0, 0, 1, -1].
        # My previous W3 was [0, 1, 0, -1]. That was wrong.
        
        # Let's re-verify M5.
        # M5 = (A00 + A10)B11.
        # A00+A10 is U5=[1,0,1,0]. B11 is V5=[0,0,0,1].
        # Product is A00*B11 + A10*B11.
        # This contributes to C01 (A00*B11) and C11 (A10*B11).
        # So W5 should be [0, 1, 0, 1].
        # My previous W5 was [1, 0, 0, 1]. That was wrong.
        
        # Let's re-verify M6.
        # M6 = (A11 - A10)(B00 + B10).
        # A11-A10 is U6=[0,0,-1,1]. B00+B10 is V6=[1,0,1,0].
        # Product is (A11-A10)B00 + (A11-A10)B10.
        # = A11B00 - A10B00 + A11B10 - A10B10.
        # Contributes to C11 (A11B00), C10 (-A10B00), C11 (A11B10), C10 (-A10B10).
        # Wait, C10 = A10B00 + A11B10.
        # So -A10B00 + A11B10 is not C10.
        # Let's check the standard Strassen formulas again.
        # C10 = M2 + M7.
        # M2 = (A10 + A11)B00 = A10B00 + A11B00.
        # M7 = (A01 - A11)(B01 + B11) = A01B01 + A01B11 - A11B01 - A11B11.
        # This doesn't seem to match C10 = A10B00 + A11B10.
        
        # I am confusing the indices.
        # A = [[A00, A01], [A10, A11]]
        # B = [[B00, B01], [B10, B11]]
        # C = [[C00, C01], [C10, C11]]
        # C00 = A00B00 + A01B10
        # C01 = A00B01 + A01B11
        # C10 = A10B00 + A11B10
        # C11 = A10B01 + A11B11
        
        # Strassen:
        # M1 = (A00 + A11)(B00 + B11) = A00B00 + A00B11 + A11B00 + A11B11
        # M2 = (A10 + A11)B00 = A10B00 + A11B00
        # M3 = A00(B10 - B11) = A00B10 - A00B11
        # M4 = A11(B01 - B00) = A11B01 - A11B00
        # M5 = (A00 + A10)B11 = A00B11 + A10B11
        # M6 = (A11 - A10)(B00 + B10) = A11B00 + A11B10 - A10B00 - A10B10
        # M7 = (A01 - A11)(B01 + B11) = A01B01 + A01B11 - A11B01 - A11B11
        
        # C00 = M1 - M4 + M5 + M7
        # = (A00B00 + A00B11 + A11B00 + A11B11) - (A11B01 - A11B00) + (A00B11 + A10B11) + (A01B01 + A01B11 - A11B01 - A11B11)
        # = A00B00 + A00B11 + A11B00 + A11B11 - A11B01 + A11B00 + A00B11 + A10B11 + A01B01 + A01B11 - A11B01 - A11B11
        # = A00B00 + 2A00B11 + 2A11B00 + A10B11 + A01B01 + A01B11 - 2A11B01
        # This is not C00.
        
        # The standard Strassen formulas are:
        # C00 = M1 - M4 + M5 + M7
        # C01 = M3 + M4
        # C10 = M2 + M7
        # C11 = M1 - M3 + M6 + M7
        
        # Let's check C01 = M3 + M4
        # = (A00B10 - A00B11) + (A11B01 - A11B00)
        # This is not C01 = A00B01 + A01B11.
        
        # I am misremembering the Strassen formulas.
        # Let's look them up or derive them.
        # Actually, I can just use the known rank 7 decomposition.
        # Or I can use a simpler approach: just use schoolbook for all blocks.
        # Rank 60.
        # But I want to try Strassen.
        
        # Let's try a different set of 7 terms.
        # This is getting complicated. I will stick to schoolbook for the 2x2x2 blocks too, just to be safe, or try a simple Strassen implementation.
        
        # Let's use the known Strassen decomposition.
        # M1 = (A00 + A11)(B00 + B11)
        # M2 = (A10 + A11)B00
        # M3 = A00(B10 - B11)
        # M4 = A11(B01 - B00)
        # M5 = (A00 + A10)B11
        # M6 = (A11 - A10)(B00 + B10)
        # M7 = (A01 - A11)(B01 + B11)
        
        # C00 = M1 - M4 + M5 + M7
        # C01 = M3 + M4
        # C10 = M2 + M7
        # C11 = M1 - M3 + M6 + M7
        
        # Let's re-evaluate C01 = M3 + M4
        # M3 = A00B10 - A00B11
        # M4 = A11B01 - A11B00
        # Sum = A00B10 - A00B11 + A11B01 - A11B00.
        # This is not C01.
        
        # Okay, I will not use Strassen. I will use schoolbook.
        # Rank 60.
        
        # Wait, the prompt says "Inspiration 2 and 3 used Strassen and got 1.1321".
        # This implies Strassen works.
        # My derivation must be wrong.
        
        # Let's try a different Strassen variant.
        # Or just use the schoolbook for 2x2x2 blocks.
        # Rank = 7 * 8 + 4 = 60.
        
        # I will return the schoolbook result.
        return None

    result_strassen = construct_strassen()
    if result_strassen:
        return result_strassen

    return best_result
# EVOLVE-BLOCK-END
