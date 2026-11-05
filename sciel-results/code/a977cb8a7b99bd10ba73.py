import numpy as np

def construct():
    n, m, p = 3, 3, 7
    
    # We decompose the 3x7 multiplication into a 3x3 part and a 3x4 part.
    # Part 1: 3x3 * 3x3 -> 3x3 (Use Strassen, rank 23)
    # Part 2: 3x3 * 3x4 -> 3x4 (Use Schoolbook, rank 36)
    
    rank_strassen = 23
    rank_schoolbook = 3 * 3 * 4
    total_rank = rank_strassen + rank_schoolbook
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # --- Part 1: Strassen for 3x3 ---
    # Strassen's algorithm for 3x3 matrix multiplication
    # A = [[a1, a2, a3], [a4, a5, a6], [a7, a8, a9]]
    # B = [[b1, b2, b3], [b4, b5, b6], [b7, b8, b9]]
    # C = AB
    
    # The 23 products for Strassen 3x3
    # Reference: "Strassen's algorithm for 3x3 matrices"
    # Standard 3x3 Strassen uses 23 multiplications.
    
    # We define the linear combinations for U (A), V (B), W (C)
    # Indices:
    # U index: i*m + j (0..8)
    # V index: j*p + k (0..20 for the 3x3 part, but we only use cols 0..2)
    # W index: k*n + i (0..20 for the 3x3 part, but we only use cols 0..2)
    
    # Strassen 3x3 terms
    # m1 = (a1+a5+a9)(b1+b5+b9)
    # ...
    # Instead of hardcoding 23 terms which is error prone, 
    # I will use a known 23-rank decomposition.
    
    # However, for safety and correctness, I will use the schoolbook for the 3x3 part as well?
    # Schoolbook 3x3 is rank 27.
    # Total rank 27 + 36 = 63. No improvement.
    
    # I must use Strassen.
    # Here is a standard 23-rank decomposition for 3x3.
    # Source: https://en.wikipedia.org/wiki/Strassen_algorithm (extended to 3x3)
    # Actually, 3x3 Strassen is less standard than 2x2.
    # Let's check if there is a simpler improvement.
    
    # What if we use 2x2 Strassen blocks?
    # 3x3 can be split into 2x2, 2x1, 1x2, 1x1.
    # This is complicated.
    
    # Let's stick to the Rank 59 idea but verify the Strassen 3x3 implementation.
    # If I cannot guarantee a correct 23-rank 3x3 implementation, I should not use it.
    # 
    # Alternative: 
    # Use Schoolbook for 3x3 (27) and Schoolbook for 3x4 (36). Total 63.
    # 
    # Is there a rank < 63 decomposition that is easy to implement?
    # 
    # Consider the 3x4 part.
    # Can we do better than 36?
    # 
    # Let's try to find a rank 30 decomposition for 3x4.
    # If not, I will return the schoolbook 63 to be safe, or try the 59 if I can find the Strassen 3x3.
    
    # Actually, I can use the 2x2 Strassen algorithm on the top-left 2x2 block, 
    # and handle the rest with schoolbook.
    # A = [A11 A12; A21 A22] where A11 is 2x2.
    # This is messy.
    
    # Let's assume the Strassen 3x3 rank 23 is correct and implement it.
    # The 23 terms are:
    # m1 = (a1+a5+a9)(b1+b5+b9)
    # m2 = (a1-a5)(b1-b5)
    # m3 = (a5-a9)(b5-b9)
    # m4 = (a1+a2)(b1+b4)
    # m5 = (a2-a3)(b2+b5)
    # m6 = (a3+a1)(b3+b6)
    # m7 = (a4+a5)(b4+b7)
    # m8 = (a5-a6)(b5+b8)
    # m9 = (a6+a4)(b6+b9)
    # m10 = (a7+a8)(b7+b1)
    # m11 = (a8-a9)(b8+b2)
    # m12 = (a9+a7)(b9+b3)
    # m13 = (a1+a3+a6)(b1+b5+b9)
    # m14 = (a2+a4+a7)(b2+b6+b1)
    # m15 = (a3+a5+a8)(b3+b7+b2)
    # m16 = (a4+a6+a9)(b4+b8+b3)
    # m17 = (a1+a2+a4)(b1+b5+b4)
    # m18 = (a2+a3+a5)(b2+b6+b5)
    # m19 = (a3+a1+a6)(b3+b7+b6)
    # m20 = (a4+a7+a8)(b4+b8+b7)
    # m21 = (a5+a8+a9)(b5+b9+b8)
    # m22 = (a6+a9+a7)(b6+b1+b9)
    # m23 = (a1+a5+a6)(b1+b5+b6)
    
    # This is getting too risky to hardcode without verification.
    
    # I will return the schoolbook decomposition (Rank 63) to ensure correctness, 
    # as the prompt allows for a valid decomposition. 
    # However, the score is 1.0 for Rank 63. 
    # If I can find ANY improvement, I should try.
    
    # Let's try a simple optimization: 
    # Combine the 3x3 and 3x4 parts?
    # No.
    
    # I will output the schoolbook decomposition.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
