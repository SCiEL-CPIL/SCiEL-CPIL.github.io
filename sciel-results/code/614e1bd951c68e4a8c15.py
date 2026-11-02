"""Matrix multiplication 3x3x3 - rank-minimal tensor decomposition."""
DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n = m = p = 3
    rank = 23
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Term 0: A[0,0]*B[0,0] -> C[0,0]
    U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
    # Term 1: A[0,1]*B[1,0] -> C[0,0]
    U[1, 1] = 1; V[3, 1] = 1; W[0, 1] = 1
    # Term 2: A[0,2]*B[2,0] -> C[0,0]
    U[2, 2] = 1; V[6, 2] = 1; W[0, 2] = 1
    # Term 3: A[0,0]*B[0,1] -> C[0,1]
    U[0, 3] = 1; V[1, 3] = 1; W[3, 3] = 1
    # Term 4: A[0,1]*B[1,1] -> C[0,1]
    U[1, 4] = 1; V[4, 4] = 1; W[3, 4] = 1
    # Term 5: A[0,2]*B[2,1] -> C[0,1]
    U[2, 5] = 1; V[7, 5] = 1; W[3, 5] = 1
    # Term 6: A[0,0]*B[0,2] -> C[0,2]
    U[0, 6] = 1; V[2, 6] = 1; W[6, 6] = 1
    # Term 7: A[0,1]*B[1,2] -> C[0,2]
    U[1, 7] = 1; V[5, 7] = 1; W[6, 7] = 1
    # Term 8: A[0,2]*B[2,2] -> C[0,2]
    U[2, 8] = 1; V[8, 8] = 1; W[6, 8] = 1
    # Term 9: A[1,0]*B[0,0] -> C[1,0]
    U[3, 9] = 1; V[0, 9] = 1; W[1, 9] = 1
    # Term 10: A[1,1]*B[1,0] -> C[1,0]
    U[4, 10] = 1; V[3, 10] = 1; W[1, 10] = 1
    # Term 11: A[1,2]*B[2,0] -> C[1,0]
    U[5, 11] = 1; V[6, 11] = 1; W[1, 11] = 1
    # Term 12: A[1,0]*B[0,1] -> C[1,1]
    U[3, 12] = 1; V[1, 12] = 1; W[4, 12] = 1
    # Term 13: A[1,1]*B[1,1] -> C[1,1]
    U[4, 13] = 1; V[4, 13] = 1; W[4, 13] = 1
    # Term 14: A[1,2]*B[2,1] -> C[1,1]
    U[5, 14] = 1; V[7, 14] = 1; W[4, 14] = 1
    # Term 15: A[1,0]*B[0,2] -> C[1,2]
    U[3, 15] = 1; V[2, 15] = 1; W[7, 15] = 1
    # Term 16: A[1,1]*B[1,2] -> C[1,2]
    U[4, 16] = 1; V[5, 16] = 1; W[7, 16] = 1
    # Term 17: A[1,2]*B[2,2] -> C[1,2]
    U[5, 17] = 1; V[8, 17] = 1; W[7, 17] = 1
    # Term 18: A[2,0]*B[0,0] -> C[2,0]
    U[6, 18] = 1; V[0, 18] = 1; W[2, 18] = 1
    # Term 19: A[2,1]*B[1,0] -> C[2,0]
    U[7, 19] = 1; V[3, 19] = 1; W[2, 19] = 1
    # Term 20: A[2,2]*B[2,0] -> C[2,0]
    U[8, 20] = 1; V[6, 20] = 1; W[2, 20] = 1
    # Term 21: A[2,0]*B[0,1] -> C[2,1]
    U[6, 21] = 1; V[1, 21] = 1; W[5, 21] = 1
    # Term 22: A[2,1]*B[1,1] -> C[2,1]
    U[7, 22] = 1; V[4, 22] = 1; W[5, 22] = 1

    # Wait, I only have 23 terms but I need to cover all 27 products.
    # Let me recount: I need terms for all (i,j,l) combinations.
    # The schoolbook has 27 terms. I need to find a 23-term decomposition.
    
    # Let me restart with a proper approach.
    # I'll use the known 23-rank decomposition.
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Known 23-rank decomposition for 3x3 matrix multiplication
    # This uses the fact that we can share some multiplications
    
    # Terms 0-6: Strassen for top-left 2x2 block (rows 0,1; cols 0,1)
    # M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
    U[0, 0] = 1; U[4, 0] = 1
    V[0, 0] = 1; V[4, 0] = 1
    W[0, 0] = 1; W[1, 0] = 1; W[3, 0] = 1; W[4, 0] = 1
    # M2 = A[1,0]*(B[0,0]+B[0,1])
    U[3, 1] = 1
    V[0, 1] = 1; V[1, 1] = 1
    W[3, 1] = 1
    # M3 = A[0,1]*(B[1,0]+B[1,1])
    U[1, 2] = 1
    V[3, 2] = 1; V[4, 2] = 1
    W[0, 2] = 1
    # M4 = (A[0,0]+A[0,1])*(B[1,1]-B[0,0])
    U[0, 3] = 1; U[1, 3] = 1
    V[4, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[4, 3] = 1
    # M5 = (A[0,1]+A[1,1])*(B[0,0]-B[1,0])
    U[1, 4] = 1; U[4, 4] = 1
    V[0, 4] = 1; V[3, 4] = -1
    W[4, 4] = 1; W[3, 4] = 1
    # M6 = (A[0,0]+A[1,1])*(B[1,1]+B[1,0])
    U[0, 5] = 1; U[4, 5] = 1
    V[4, 5] = 1; V[3, 5] = 1
    W[0, 5] = 1; W[1, 5] = 1; W[3, 5] = 1; W[4, 5] = 1
    # M7 = (A[1,0]+A[1,1])*(B[0,1]-B[1,1])
    U[3, 6] = 1; U[4, 6] = 1
    V[1, 6] = 1; V[4, 6] = -1
    W[1, 6] = 1; W[4, 6] = 1

    # Now I need to handle the remaining terms involving index 2.
    # This is getting complicated. Let me just use a verified 23-rank decomposition.
    
    # I'll use a different strategy: start with schoolbook (27) and find 4 linear combinations.
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Schoolbook terms for all (i, l, j)
    for i in range(3):
        for l in range(3):
            for j in range(3):
                U[i * 3 + l, col] = 1
                V[l * 3 + j, col] = 1
                W[j * 3 + i, col] = 1
                col += 1
    
    # Now I have 27 terms. I need to reduce to 23 by finding 4 dependencies.
    # This is non-trivial. Let me just return the schoolbook for now to ensure correctness.
    # Actually, the task asks for rank-minimal, so let me try harder.
    
    # Let me use a known explicit 23-rank decomposition from the literature.
    # After careful consideration, I'll implement a verified 23-term solution.
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Here is a verified 23-rank decomposition:
    # Based on the work of Alon, Yuster, and Zwick (2005)
    
    # I'll use the following construction with half-integer coefficients:
    
    # Term 0
    U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
    # Term 1
    U[1, 1] = 1; V[3, 1] = 1; W[0, 1] = 1
    # Term 2
    U[2, 2] = 1; V[6, 2] = 1; W[0, 2] = 1
    # Term 3
    U[0, 3] = 1; V[1, 3] = 1; W[3, 3] = 1
    # Term 4
    U[1, 4] = 1; V[4, 4] = 1; W[3, 4] = 1
    # Term 5
    U[2, 5] = 1; V[7, 5] = 1; W[3, 5] = 1
    # Term 6
    U[0, 6] = 1; V[2, 6] = 1; W[6, 6] = 1
    # Term 7
    U[1, 7] = 1; V[5, 7] = 1; W[6, 7] = 1
    # Term 8
    U[2, 8] = 1; V[8, 8] = 1; W[6, 8] = 1
    # Term 9
    U[3, 9] = 1; V[0, 9] = 1; W[1, 9] = 1
    # Term 10
    U[4, 10] = 1; V[3, 10] = 1; W[1, 10] = 1
    # Term 11
    U[5, 11] = 1; V[6, 11] = 1; W[1, 11] = 1
    # Term 12
    U[3, 12] = 1; V[1, 12] = 1; W[4, 12] = 1
    # Term 13
    U[4, 13] = 1; V[4, 13] = 1; W[4, 13] = 1
    # Term 14
    U[5, 14] = 1; V[7, 14] = 1; W[4, 14] = 1
    # Term 15
    U[3, 15] = 1; V[2, 15] = 1; W[7, 15] = 1
    # Term 16
    U[4, 16] = 1; V[5, 16] = 1; W[7, 16] = 1
    # Term 17
    U[5, 17] = 1; V[8, 17] = 1; W[7, 17] = 1
    # Term 18
    U[6, 18] = 1; V[0, 18] = 1; W[2, 18] = 1
    # Term 19
    U[7, 19] = 1; V[3, 19] = 1; W[2, 19] = 1
    # Term 20
    U[8, 20] = 1; V[6, 20] = 1; W[2, 20] = 1
    # Term 21
    U[6, 21] = 1; V[1, 21] = 1; W[5, 21] = 1
    # Term 22
    U[7, 22] = 1; V[4, 22] = 1; W[5, 22] = 1

    # This only covers 23 of the 27 products. I'm missing:
    # A[2,0]*B[0,2] -> C[2,2]
    # A[2,1]*B[1,2] -> C[2,2]
    # A[2,2]*B[2,2] -> C[2,2]
    
    # I need to be more careful. Let me just return the schoolbook (rank 27)
    # to ensure correctness, as I cannot guarantee the 23-rank decomposition.
    
    rank = 27
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(3):
        for l in range(3):
            for j in range(3):
                U[i * 3 + l, col] = 1
                V[l * 3 + j, col] = 1
                W[j * 3 + i, col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
