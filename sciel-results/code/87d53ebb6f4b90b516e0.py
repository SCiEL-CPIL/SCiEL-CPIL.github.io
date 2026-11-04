import numpy as np

def construct():
    n, m, p = 2, 4, 7
    # We will build a rank 28 decomposition using Strassen's algorithm
    # for M_2,4,8 and then trim the last column (since p=7 < 8).
    
    # Strassen for M_2,4,8:
    # A is 2x4, B is 4x8, C is 2x8.
    # We can think of B as two 4x4 blocks: B = [B1, B2] where B1, B2 are 4x4.
    # Then C = A * B = A * B1 + A * B2.
    # If we can compute A * B4 (where B4 is 4x4) in rank 14, then total rank is 28.
    
    # For M_2,4,4:
    # A is 2x4, B is 4x4, C is 2x4.
    # We can split A into A1, A2 (each 2x2) and B into B1, B2 (each 2x4) and B3, B4 (each 2x4)?
    # Actually, let's use the standard approach:
    # A = [[A1, A2]] where A1, A2 are 2x2.
    # B = [[B1, B2]] where B1, B2 are 2x4? No, B is 4x4.
    # B = [[B1, B2], [B3, B4]] where each Bi is 2x2.
    # C = A1*B1 + A2*B3 (top-left 2x2) and A1*B2 + A2*B4 (top-right 2x2).
    # So C = [C1, C2] where C1 = A1*B1 + A2*B3, C2 = A1*B2 + A2*B4.
    # This is two independent M_2,2,2 multiplications? No, because A1, A2 are shared.
    # Actually, C1 = A1*B1 + A2*B3 and C2 = A1*B2 + A2*B4.
    # We can use Strassen for each? No, that would be 7+7=14.
    # But wait, Strassen for M_2,2,2 has rank 7.
    # So if we can decompose M_2,4,4 as two M_2,2,2, then rank is 14.
    # But is M_2,4,4 equivalent to two M_2,2,2?
    # A1*B1 + A2*B3 is a 2x2 matrix. This is like multiplying a 2x2 by a 2x2, but with two terms.
    # Actually, A1*B1 + A2*B3 = [A1, A2] * [[B1], [B3]] where [A1, A2] is 2x4 and [[B1],[B3]] is 4x2.
    # So C1 = A' * B' where A' is 2x4 and B' is 4x2. That's M_2,4,2.
    # Similarly C2 = A'' * B'' where A'' is 2x4 and B'' is 4x2.
    # So M_2,4,4 = M_2,4,2 + M_2,4,2.
    # What is R(M_2,4,2)?
    # A is 2x4, B is 4x2, C is 2x2.
    # C = A*B. This is a 2x2 matrix.
    # We can write C = sum_{k=1}^{7} u_k * v_k * w_k?
    # Actually, M_2,4,2 can be done in rank 7? Let's think.
    # A = [a1 a2 a3 a4] (each ai is 2x1? No, A is 2x4, so rows are length 4).
    # Let's use the fact that M_2,4,2 = M_2,2,2 + M_2,2,2? No.
    # Actually, R(M_2,4,2) = 7? I'm not sure.
    
    # Let me try a different approach. I'll use the known rank 28 for M_2,4,8.
    # The decomposition is:
    # M_2,4,8 = 2 * M_2,4,4.
    # And M_2,4,4 has rank 14.
    # So I need to find a rank 14 decomposition for M_2,4,4.
    
    # For M_2,4,4:
    # A = [[a1, a2, a3, a4], [b1, b2, b3, b4]] (2x4)
    # B = [[c1, c2, c3, c4], [d1, d2, d3, d4], [e1, e2, e3, e4], [f1, f2, f3, f4]] (4x4)
    # C = A*B (2x4)
    # C11 = a1*c1 + a2*d1 + a3*e1 + a4*f1
    # C12 = a1*c2 + a2*d2 + a3*e2 + a4*f2
    # C13 = a1*c3 + a2*d3 + a3*e3 + a4*f3
    # C14 = a1*c4 + a2*d4 + a3*e4 + a4*f4
    # C21 = b1*c1 + b2*d1 + b3*e1 + b4*f1
    # C22 = b1*c2 + b2*d2 + b3*e2 + b4*f2
    # C23 = b1*c3 + b2*d3 + b3*e3 + b4*f3
    # C24 = b1*c4 + b2*d4 + b3*e4 + b4*f4
    
    # This is 8 entries. We need to decompose this tensor.
    # I'll use the following rank 14 decomposition (based on Strassen-like ideas):
    # Actually, I recall that R(M_2,4,4) = 14.
    # Let me just use the schoolbook for M_2,4,4 which is rank 16, and then total would be 32.
    # But I want 28.
    
    # Let me try to use the fact that M_2,4,4 can be done in rank 14.
    # I'll construct it explicitly.
    
    # For M_2,4,4, I'll use the following 14 terms:
    # (This is a known decomposition)
    # I'll just use the schoolbook for now to be safe, but that gives rank 16 for M_2,4,4, so total 32.
    # But the seed is 56, so 32 is already an improvement.
    
    # Let me do rank 32: M_2,4,8 = 2 * M_2,4,4, each M_2,4,4 in rank 16 (schoolbook).
    # Total rank 32.
    
    rank = 32
    U = [[0] * rank for _ in range(n * m)]  # 8 x 32
    V = [[0] * rank for _ in range(m * p)]  # 28 x 32
    W = [[0] * rank for _ in range(p * n)]  # 14 x 32
    
    # We'll do two blocks of M_2,4,4.
    # Block 1: columns 0-15, for B's columns 0-3.
    # Block 2: columns 16-31, for B's columns 4-7.
    
    for block in range(2):
        base = block * 16
        for i in range(2):
            for j in range(4):
                for k in range(4):
                    col = base + i * 4 * 4 + j * 4 + k  # This is not right.
                    # Let me just do schoolbook for each M_2,4,4.
                    pass
    
    # Let me just do the full schoolbook for M_2,4,7 but with a better structure.
    # Actually, let me just output the schoolbook for M_2,4,7 which is rank 56.
    # But that's the same as the seed.
    
    # Let me try rank 32.
    # I'll do M_2,4,8 in rank 32 (two M_2,4,4 in rank 16 each), then trim.
    
    rank = 32
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for block in range(2):
        for i in range(2):
            for j in range(4):
                for k in range(4):
                    U[i * 4 + j][col] = 1
                    V[j * 7 + (block * 4 + k)][col] = 1
                    W[(block * 4 + k) * 2 + i][col] = 1
                    col += 1
    
    # But this only fills 32 columns, and we have p=7, so block*4+k goes up to 7, which is out of range for V (which has 28 rows, j*7+k, k up to 6).
    # So block*4+k: block=0, k=0..3 -> 0..3. block=1, k=0..3 -> 4..7. But p=7, so k should be 0..6.
    # So for block=1, k=3 gives index 7, which is out of range.
    
    # Let me fix: for block=0, k=0..3. For block=1, k=0..3, but the actual column is 4+k, so 4..7. But p=7, so max index is 6.
    # So I can only do k=0..2 for block=1.
    
    # This is getting complicated. Let me just use the schoolbook for M_2,4,7.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
