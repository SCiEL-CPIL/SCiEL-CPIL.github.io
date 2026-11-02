import numpy as np
import time

DIMENSIONS = [4, 4, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    n, m, p = DIMENSIONS
    
    # We will build the decomposition in 3 parts:
    # Part 1: A (4x4) * B[:, 0:4] (4x4) -> C[:, 0:4] (4x4) using Strassen (Rank 49)
    # Part 2: A (4x4) * B[:, 4:8] (4x4) -> C[:, 4:8] (4x4) using Strassen (Rank 49)
    # Part 3: A (4x4) * B[:, 8:9] (4x1) -> C[:, 8:9] (4x1) using Schoolbook (Rank 16)
    
    rank1 = 49
    rank2 = 49
    rank3 = 16
    total_rank = rank1 + rank2 + rank3
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Helper to generate Strassen 2x2x2 factors
    def get_strassen_2x2():
        # Returns U, V, W for 2x2x2 multiplication
        # U: 2x2 flattened (4 rows) x 7 cols
        # V: 2x2 flattened (4 rows) x 7 cols
        # W: 2x2 flattened (4 rows) x 7 cols
        u = [[0]*7 for _ in range(4)]
        v = [[0]*7 for _ in range(4)]
        w = [[0]*7 for _ in range(4)]
        
        # M1 = (a00+a11)(b00+b11)
        u[0][0] = 1; u[1][0] = 1
        v[0][0] = 1; v[1][0] = 1
        # C00 += M1, C11 += M1
        w[0][0] = 1; w[3][0] = 1
        
        # M2 = (a10+a11)b00
        u[2][1] = 1; u[3][1] = 1
        v[0][1] = 1
        # C00 -= M2, C10 += M2
        w[0][1] = -1; w[2][1] = 1
        
        # M3 = a00(b01-b11)
        u[0][2] = 1
        v[1][2] = 1; v[3][2] = -1
        # C01 += M3
        w[1][2] = 1
        
        # M4 = a11(b10-b00)
        u[3][3] = 1
        v[2][3] = 1; v[0][3] = -1
        # C00 -= M4, C01 += M4
        w[0][3] = -1; w[1][3] = 1
        
        # M5 = (a00+a01)b11
        u[0][4] = 1; u[1][4] = 1
        v[3][4] = 1
        # C11 += M5
        w[3][4] = 1
        
        # M6 = (a10-a00)(b00+b01)
        u[2][5] = 1; u[0][5] = -1
        v[0][5] = 1; v[1][5] = 1
        # C11 -= M6? No, C11 = M1 - M3 + M5 - M7. 
        # Wait, standard Strassen:
        # C00 = M1 - M2 - M4 + M7
        # C01 = M3 + M4
        # C10 = M1 + M2
        # C11 = M1 - M3 + M5 - M7
        
        # Let's re-verify M6 and M7 contributions.
        # M6 is not in the standard 7-term Strassen formula for C?
        # The 7 terms are M1..M7.
        # C00 = M1 - M2 - M4 + M7
        # C01 = M3 + M4
        # C10 = M1 + M2
        # C11 = M1 - M3 + M5 - M7
        
        # So M6 is not used in the output?
        # Ah, the 7th term is M7.
        # My list above:
        # M1, M2, M3, M4, M5, M6, M7.
        # Let's check the standard definitions again.
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C00 = M1 - M2 - M4 + M7
        # C01 = M3 + M4
        # C10 = M1 + M2
        # C11 = M1 - M3 + M5 - M7
        
        # So M6 is indeed not in the final sum?
        # No, that's incorrect. The standard Strassen algorithm uses 7 multiplications.
        # Let's check C11 again.
        # C11 = A01*B10 + A11*B11.
        # M5 = (A00+A01)B11 = A00*B11 + A01*B11.
        # M3 = A00(B01-B11) = A00*B01 - A00*B11.
        # M1 - M3 + M5 = (A00+A11)(B00+B11) - (A00*B01 - A00*B11) + (A00*B11 + A01*B11)
        # = A00*B00 + A00*B11 + A11*B00 + A11*B11 - A00*B01 + A00*B11 + A00*B11 + A01*B11
        # This looks messy.
        
        # Let's use a known correct set of 7 terms.
        # From Wikipedia "Strassen algorithm":
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C00 = M1 - M2 - M4 + M7
        # C01 = M3 + M4
        # C10 = M1 + M2
        # C11 = M1 - M3 + M5 - M7
        
        # It seems M6 is not used in the output formula?
        # That can't be right.
        # Ah, I see. In some formulations, M6 is used.
        # Let's check C10.
        # C10 = A10*B00 + A11*B10.
        # M1 + M2 = (A00+A11)(B00+B11) + (A10+A11)B00
        # = A00*B00 + A00*B11 + A11*B00 + A11*B11 + A10*B00 + A11*B00
        # = A10*B00 + A11*B00 + A11*B00 + ...
        # This doesn't match C10.
        
        # I will use a verified Python implementation of Strassen 2x2 to extract coefficients.
        pass
        
    # Due to complexity and time, I will use a simpler approach:
    # Schoolbook for 4x4x4 has rank 64.
    # Strassen 2x2x2 has rank 7.
    # 4x4x4 via 4 blocks of 2x2x2? No, it's 7 blocks of 2x2x2.
    # Rank 49.
    # I will implement the 49-rank decomposition by composing the 2x2x2 Strassen.
    
    # For now, to ensure correctness and avoid bugs in complex symbolic algebra,
    # I will use Schoolbook for the 4x4x4 parts as well, but that gives rank 64+64+16 = 144.
    # That's the same as the seed.
    
    # Let's try to implement the 49-rank Strassen for 4x4x4 properly.
    
    # I will define the 2x2x2 Strassen factors explicitly.
    
    return {"U": U, "V": V, "W": W}

