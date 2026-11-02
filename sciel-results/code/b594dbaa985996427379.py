"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 23 decomposition for 3x3x3
    # The 23 terms are defined by (u, v, w) where u, v, w are indices in 0..8
    # u corresponds to A[i,j] where i = u//3, j = u%3
    # v corresponds to B[j,k] where j = v//3, k = v%3
    # w corresponds to C[i,k] where i = w//3, k = w%3
    terms_23 = [
        (0, 0, 0),   # A00 B00 -> C00
        (4, 4, 4),   # A11 B11 -> C11
        (8, 8, 8),   # A22 B22 -> C22
        (0, 3, 3),   # A00 B10 -> C10
        (4, 6, 6),   # A11 B21 -> C21
        (8, 2, 2),   # A22 B02 -> C02
        (1, 3, 3),   # A01 B10 -> C10
        (5, 6, 6),   # A12 B21 -> C21
        (9, 2, 2),   # A20 B02 -> C02
        (1, 0, 0),   # A01 B00 -> C00
        (5, 4, 4),   # A12 B11 -> C11
        (9, 8, 8),   # A20 B22 -> C22
        (2, 1, 1),   # A02 B01 -> C01
        (6, 5, 5),   # A10 B12 -> C12
        (10, 9, 9),  # A21 B20 -> C20
        (2, 0, 0),   # A02 B00 -> C00
        (6, 4, 4),   # A10 B11 -> C11
        (10, 8, 8),  # A21 B22 -> C22
        (3, 1, 1),   # A10 B01 -> C01
        (7, 5, 5),   # A11 B12 -> C12
        (11, 9, 9),  # A22 B20 -> C20
        (3, 0, 0),   # A10 B00 -> C00
        (7, 4, 4),   # A11 B11 -> C11
        (11, 8, 8),  # A22 B22 -> C22
    ]
    
    # The above list is likely incorrect. I will use a known valid rank 23 decomposition.
    # Actually, I will use a rank 30 decomposition to be safe? No, 24 is better.
    # I will use the 24-rank decomposition.
    # The 23 terms for 3x3x3 are:
    # 1. (0,0,0)
    # 2. (1,1,1)
    # 3. (2,2,2)
    # 4. (0,1,1)
    # 5. (1,2,2)
    # 6. (2,0,0)
    # 7. (0,0,1)
    # 8. (0,1,2)
    # 9. (1,0,0)
    # 10. (1,2,0)
    # 11. (2,1,1)
    # 12. (2,2,0)
    # 13. (0,2,1)
    # 14. (1,0,2)
    # 15. (2,0,1)
    # 16. (0,1,0)
    # 17. (1,1,0)
    # 18. (2,1,2)
    # 19. (0,2,0)
    # 20. (1,2,1)
    # 21. (2,2,1)
    # 22. (0,0,2)
    # 23. (1,1,2)
    
    # I will use a simple rank 36 decomposition if I cannot verify the 23 terms.
    # But I will try the 24-rank decomposition.
    # The 23 terms are:
    terms = [
        (0, 0, 0),
        (4, 4, 4),
        (8, 8, 8),
        (0, 3, 3),
        (4, 6, 6),
        (8, 2, 2),
        (1, 3, 3),
        (5, 6, 6),
        (9, 2, 2),
        (1, 0, 0),
        (5, 4, 4),
        (9, 8, 8),
        (2, 1, 1),
        (6, 5, 5),
        (10, 9, 9),
        (2, 0, 0),
        (6, 4, 4),
        (10, 8, 8),
        (3, 1, 1),
        (7, 5, 5),
        (11, 9, 9),
        (3, 0, 0),
        (7, 4, 4),
        (11, 8, 8),
    ]
    
    # This is likely wrong. I will use the schoolbook rank 36.
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
