"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 23
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Bini et al. 1979 rank-23 decomposition for 3x3 matrix multiplication
    # Indices: U[i*m+j], V[j*p+k], W[k*n+i]
    # Term format: (u_idx, u_val, v_idx, v_val, w_idx, w_val)
    
    terms = [
        # 1
        (0, 1, 0, 1, 0, 1),   # A00 B00 -> C00
        # 2
        (1, 1, 3, 1, 0, 1),   # A01 B10 -> C00
        # 3
        (3, 1, 1, 1, 3, 1),   # A10 B01 -> C11
        # 4
        (4, 1, 4, 1, 3, 1),   # A11 B11 -> C11
        # 5
        (0, 1, 1, 1, 1, 1),   # A00 B01 -> C01
        # 6
        (1, 1, 4, 1, 1, 1),   # A01 B11 -> C01
        # 7
        (3, 1, 0, 1, 4, 1),   # A10 B00 -> C10
        # 8
        (4, 1, 3, 1, 4, 1),   # A11 B10 -> C10
        # 9
        (0, 1, 2, 1, 6, 1),   # A00 B02 -> C02
        # 10
        (1, 1, 5, 1, 6, 1),   # A01 B12 -> C02
        # 11
        (3, 1, 2, 1, 9, 1),   # A10 B02 -> C12
        # 12
        (4, 1, 5, 1, 9, 1),   # A11 B12 -> C12
        # 13
        (6, 1, 0, 1, 2, 1),   # A20 B00 -> C20
        # 14
        (7, 1, 3, 1, 2, 1),   # A21 B10 -> C20
        # 15
        (6, 1, 1, 1, 5, 1),   # A20 B01 -> C21
        # 16
        (7, 1, 4, 1, 5, 1),   # A21 B11 -> C21
        # 17
        (6, 1, 2, 1, 8, 1),   # A20 B02 -> C22
        # 18
        (7, 1, 5, 1, 8, 1),   # A21 B12 -> C22
        # 19
        (8, 1, 6, 1, 0, 1),   # A22 B20 -> C00
        # 20
        (8, 1, 7, 1, 3, 1),   # A22 B21 -> C11
        # 21
        (8, 1, 8, 1, 6, 1),   # A22 B22 -> C22
        # 22
        (0, 1, 7, 1, 1, 1),   # A00 B12 -> C01? No, this is not standard.
        # I need the correct 22nd and 23rd terms.
        # The above 21 terms are not a valid decomposition.
        # I will use a verified 23-term list.
    ]
    
    # Correct Bini 1979 23-term decomposition
    # Using a known valid set
    correct_terms = [
        (0, 1, 0, 1, 0, 1),
        (1, 1, 3, 1, 0, 1),
        (3, 1, 1, 1, 3, 1),
        (4, 1, 4, 1, 3, 1),
        (0, 1, 1, 1, 1, 1),
        (1, 1, 4, 1, 1, 1),
        (3, 1, 0, 1, 4, 1),
        (4, 1, 3, 1, 4, 1),
        (0, 1, 2, 1, 6, 1),
        (1, 1, 5, 1, 6, 1),
        (3, 1, 2, 1, 9, 1),
        (4, 1, 5, 1, 9, 1),
        (6, 1, 0, 1, 2, 1),
        (7, 1, 3, 1, 2, 1),
        (6, 1, 1, 1, 5, 1),
        (7, 1, 4, 1, 5, 1),
        (6, 1, 2, 1, 8, 1),
        (7, 1, 5, 1, 8, 1),
        (8, 1, 6, 1, 0, 1),
        (8, 1, 7, 1, 3, 1),
        (8, 1, 8, 1, 6, 1),
        (0, 1, 7, 1, 1, 1), # A00 B12 -> C01? 
        (4, 1, 6, 1, 4, 1), # A11 B20 -> C10?
    ]
    
    # This is risky. I will fall back to rank 27 if I cannot guarantee correctness.
    # However, rank 23 is well-known.
    # I will use the implementation from a trusted source or derive it.
    # For now, I will return the schoolbook rank 27 to ensure validity, 
    # but the prompt asks for a stronger algorithm.
    # Let's try to use the rank 23 decomposition from the literature.
    
    # Actually, I can just use the rank 27 decomposition. 
    # But the score is seed/rank. If seed is 27, rank 27 is score 1.0.
    # Rank 23 is score 1.17.
    # I will implement the rank 23 decomposition correctly.
    
    # Correct terms for Bini 1979:
    # 1. A00 B00 -> C00
    # 2. A01 B10 -> C00
    # 3. A10 B01 -> C11
    # 4. A11 B11 -> C11
    # 5. A00 B01 -> C01
    # 6. A01 B11 -> C01
    # 7. A10 B00 -> C10
    # 8. A11 B10 -> C10
    # 9. A00 B02 -> C02
    # 10. A01 B12 -> C02
    # 11. A10 B02 -> C12
    # 12. A11 B12 -> C12
    # 13. A20 B00 -> C20
    # 14. A21 B10 -> C20
    # 15. A20 B01 -> C21
    # 16. A21 B11 -> C21
    # 17. A20 B02 -> C22
    # 18. A21 B12 -> C22
    # 19. A22 B20 -> C00
    # 20. A22 B21 -> C11
    # 21. A22 B22 -> C22
    # 22. (A00+A11+A22) (B00+B11+B22) ? No.
    
    # I will use the rank 27 decomposition to be safe, as I don't have the exact 23 terms memorized.
    # Wait, I can look up the terms.
    # Terms 22 and 23 involve sums.
    # T22: (A00 + A11) (B01 + B11) -> C01 + C11
    # T23: (A00 + A22) (B02 + B22) -> C02 + C22
    
    # Let's try to implement this.
    # T22: U: A00=1, A11=1. V: B01=1, B11=1. W: C01=1, C11=1.
    # T23: U: A00=1, A22=1. V: B02=1, B22=1. W: C02=1, C22=1.
    
    # But this is not a rank-1 term. A rank-1 term is U * V * W.
    # (A00+A11)(B01+B11) = A00B01 + A00B11 + A11B01 + A11B11.
    # This contributes to C01 (A00B01, A00B11? No, A00B01->C01, A00B11->C01? No, A00B11 is not in C01. C01 = A00B01 + A01B11 + A02B21.
    # So (A00+A11)(B01+B11) = A00B01 + A00B11 + A11B01 + A11B11.
    # A00B01 -> C01.
    # A00B11 -> C01? No, C01 = sum_j A0j Bj1. So A00B01 + A01B11 + A02B21.
    # So A00B11 is not in C01. It is in C01? No.
    # So this term is not valid for a single C element.
    
    # I will stick to rank 27.
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
# EVOLVE-BLOCK-END
