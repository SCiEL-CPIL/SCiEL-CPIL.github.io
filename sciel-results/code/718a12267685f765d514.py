import time
import numpy as np

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank is 72.
    # We aim for rank 44 using Laderman for 3x3x3 parts and schoolbook for the rest.
    # Split B (3x8) into two 3x4 blocks.
    # Block 1: cols 0-3. Use Laderman for cols 0-2 (19 terms), Schoolbook for col 3 (3 terms). Total 22.
    # Block 2: cols 4-7. Use Laderman for cols 4-6 (19 terms), Schoolbook for col 7 (3 terms). Total 22.
    # Total rank 44.
    
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Laderman coefficients for 3x3 multiplication
    # Each term is a list of (i, j) for A and (j, k) for B and (i, k) for C
    # We need the linear forms.
    # I will use a known set of Laderman coefficients.
    # If I don't have them, I will fallback to schoolbook.
    # For now, I will use schoolbook to ensure correctness, but this scores 1.0.
    # I need to improve.
    
    # Let's try to implement the Laderman coefficients.
    # The 19 terms are:
    # 1. A00*B00
    # 2. A01*B10
    # 3. A02*B20
    # 4. A10*B01
    # 5. A11*B11
    # 6. A12*B21
    # 7. A20*B02
    # 8. A21*B12
    # 9. A22*B22
    # 10. (A00+A11)*(B00+B11)
    # 11. (A01+A12)*(B10+B21)
    # 12. (A10+A21)*(B01+B12)
    # 13. (A02+A10+A21)*(B20+B01+B12)
    # 14. (A00+A01+A10)*(B00+B10+B01)
    # 15. (A11+A12+A21)*(B11+B21+B12)
    # 16. (A22+A20+A02)*(B22+B02+B20)
    # 17. (A00+A11+A22)*(B00+B11+B22)
    # 18. (A01+A12+A20)*(B10+B21+B02)
    # 19. (A10+A21+A02)*(B01+B12+B20)
    
    # I will define the coefficients for A and B for each of the 19 terms.
    # A coeffs: list of (i, j, coeff)
    # B coeffs: list of (j, k, coeff)
    
    laderman_terms = [
        # 1
        ([ (0,0,1) ], [ (0,0,1) ]),
        # 2
        ([ (0,1,1) ], [ (1,0,1) ]),
        # 3
        ([ (0,2,1) ], [ (2,0,1) ]),
        # 4
        ([ (1,0,1) ], [ (0,1,1) ]),
        # 5
        ([ (1,1,1) ], [ (1,1,1) ]),
        # 6
        ([ (1,2,1) ], [ (2,1,1) ]),
        # 7
        ([ (2,0,1) ], [ (0,2,1) ]),
        # 8
        ([ (2,1,1) ], [ (1,2,1) ]),
        # 9
        ([ (2,2,1) ], [ (2,2,1) ]),
        # 10
        ([ (0,0,1), (1,1,1) ], [ (0,0,1), (1,1,1) ]),
        # 11
        ([ (0,1,1), (1,2,1) ], [ (1,0,1), (2,1,1) ]),
        # 12
        ([ (1,0,1), (2,1,1) ], [ (0,1,1), (1,2,1) ]),
        # 13
        ([ (0,2,1), (1,0,1), (2,1,1) ], [ (2,0,1), (0,1,1), (1,2,1) ]),
        # 14
        ([ (0,0,1), (0,1,1), (1,0,1) ], [ (0,0,1), (1,0,1), (0,1,1) ]),
        # 15
        ([ (1,1,1), (1,2,1), (2,1,1) ], [ (1,1,1), (2,1,1), (1,2,1) ]),
        # 16
        ([ (2,2,1), (2,0,1), (0,2,1) ], [ (2,2,1), (0,2,1), (2,0,1) ]),
        # 17
        ([ (0,0,1), (1,1,1), (2,2,1) ], [ (0,0,1), (1,1,1), (2,2,1) ]),
        # 18
        ([ (0,1,1), (1,2,1), (2,0,1) ], [ (1,0,1), (2,1,1), (0,2,1) ]),
        # 19
        ([ (1,0,1), (2,1,1), (0,2,1) ], [ (0,1,1), (1,2,1), (2,0,1) ]),
    ]
    
    # This is a guess. I will use schoolbook to be safe.
    # If I use schoolbook, rank is 72.
    # I will try to use the Laderman terms if I can verify them.
    # Since I cannot verify, I will use schoolbook.
    
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
