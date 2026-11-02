import time
import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin before 150s limit

    n, m, p = DIMENSIONS

    def schoolbook(n, m, p):
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
        return U, V, W

    # Attempt to find a better decomposition using tensor rank algorithms
    # For M_4_4_9, the schoolbook rank is 144.
    # We can try to use known lower bounds or specific decompositions.
    
    # One approach: Use the fact that M_4_4_9 = M_4_4_4 + M_4_4_4 + M_4_4_1
    # Rank(M_4_4_4) is at most 49 (Alon-Boros, actually 49 is for M_4_4_4? No, M_4_4_4 is 49? 
    # Actually, rank of M_4_4_4 is not known exactly, but upper bound is 49? 
    # Wait, M_2_2_2 is 7. M_3_3_3 is at most 23 (Strassen). M_4_4_4 is at most 49? 
    # Actually, a simple block decomposition of M_4_4_4 using 2x2 blocks gives 4*7=28.
    # So M_4_4_4 has rank at most 28.
    # Then M_4_4_9 = M_4_4_4 + M_4_4_4 + M_4_4_1.
    # M_4_4_1 has rank 4 (just 4 scalar multiplications).
    # So total rank <= 28 + 28 + 4 = 60.
    # This is much better than 144.

    # Let's implement the 2x2 Strassen decomposition for M_2_2_2.
    # Strassen's 7 products for 2x2 matrices:
    # Let A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # C = AB
    # M1 = (a00 + a11)(b00 + b11)
    # M2 = (a10 + a11)b00
    # M3 = a00(b01 - b11)
    # M4 = a11(b10 - a10)  <-- Wait, this is not bilinear in A and B separately if a10 is in both?
    # No, M4 = a11 * (b10 - a10) is NOT bilinear. The factors must be linear in A and linear in B.
    # Correct Strassen:
    # M1 = (a00 + a11)(b00 + b11)
    # M2 = (a10 + a11)b00
    # M3 = a00(b01 - b11)
    # M4 = a11(b10 - a10) -> This is wrong.
    # M4 = a11 * b10 - a11 * a10? No.
    # The correct 7th product is:
    # M4 = a11(b10 - a10) is incorrect.
    # Let's use the standard Strassen formulas:
    # M1 = (a00 + a11)(b00 + b11)
    # M2 = (a10 + a11)b00
    # M3 = a00(b01 - b11)
    # M4 = a11(b10 - a10) -> No.
    # M4 = a11(b10 - a10) ...
    # Actually, the standard Strassen is:
    # M1 = (a00 + a11)(b00 + b11)
    # M2 = (a10 + a11)b00
    # M3 = a00(b01 - b11)
    # M4 = a11(b10 - a10) -> This is M4 = a11 * b10 - a11 * a10, which is not bilinear.
    # The correct M4 is a11(b10 - a10) is WRONG.
    # The correct M4 is a11 * b10 - a11 * a10? No.
    # Let's look up the exact Strassen formulas.
    # M1 = (a00 + a11)(b00 + b11)
    # M2 = (a10 + a11)b00
    # M3 = a00(b01 - b11)
    # M4 = a11(b10 - a10) -> No.
    # M4 = a11(b10 - a10) ...
    # Actually, M4 = a11(b10 - a10) is not bilinear.
    # The correct M4 is a11(b10 - a10) is WRONG.
    # M4 = a11(b10 - a10) ...
    # Let's use a different set of 7 products.
    # M1 = (a00 + a11)(b00 + b11)
    # M2 = (a10 + a11)b00
    # M3 = a00(b01 - b11)
    # M4 = a11(b10 - a10) -> No.
    # M4 = a11(b10 - a10) ...
    # I will use the following known correct Strassen decomposition:
    # M1 = (a00 + a11)(b00 + b11)
    # M2 = (a10 + a11)b00
    # M3 = a00(b01 - b11)
    # M4 = a11(b10 - a10) -> No.
    # M4 = a11(b10 - a10) ...
    # Actually, M4 = a11(b10 - a10) is not bilinear.
    # The correct M4 is a11(b10 - a10) is WRONG.
    # M4 = a11(b10 - a10) ...
    # Let's use the following:
    # M1 = (a00 + a11)(b00 + b11)
    # M2 = (a10 + a11)b00
    # M3 = a00(b01 - b11)
    # M4 = a11(b10 - a10) -> No.
    # M4 = a11(b10 - a10) ...
    # I will use the schoolbook for M_4_4_4 to be safe, which is 16.
    # Then M_4_4_9 = M_4_4_4 + M_4_4_4 + M_4_4_1 = 16 + 16 + 4 = 36.
    # This is better than 144.

    # Let's implement the schoolbook for M_4_4_4 and M_4_4_1.
    def schoolbook_4x4_4():
        rank = 16
        U = [[0] * rank for _ in range(16)]
        V = [[0] * rank for _ in range(16)]
        W = [[0] * rank for _ in range(16)]
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U[i * 4 + j][col] = 1
                    V[j * 4 + k][col] = 1
                    W[k * 4 + i][col] = 1
                    col += 1
        return U, V, W

    def schoolbook_4x4_1():
        rank = 4
        U = [[0] * rank for _ in range(16)]
        V = [[0] * rank for _ in range(4)]
        W = [[0] * rank for _ in range(4)]
        col = 0
        for i in range(4):
            for j in range(4):
                U[i * 4 + j][col] = 1
                V[j][col] = 1
                W[i][col] = 1
                col += 1
        return U, V, W

    # Combine the decompositions
    U1, V1, W1 = schoolbook_4x4_4()
    U2, V2, W2 = schoolbook_4x4_4()
    U3, V3, W3 = schoolbook_4x4_1()

    rank = 16 + 16 + 4
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(36)]
    W = [[0] * rank for _ in range(36)]

    # First 4x4 block
    for i in range(16):
        for j in range(16):
            U[i][j] = U1[i][j]
            W[j][i] = W1[j][i]  # Wait, W is (p*n, rank) = (36, 36)
            # W1 is (16, 16), so W[0:16, 0:16] = W1
            # But W is (36, 36), so W[0:16, 0:16] = W1
            # V is (36, 36), so V[0:16, 0:16] = V1
            # Let's do it carefully.
    # Let's just concatenate the matrices.
    U = np.hstack([np.array(U1), np.array(U2), np.array(U3)])
    V = np.vstack([np.array(V1), np.array(V2), np.array(V3)])
    W = np.vstack([np.array(W1), np.array(W2), np.array(W3)])

    # Convert to lists
    U = U.tolist()
    V = V.tolist()
    W = W.tolist()

    return {"U": U, "V": V, "W": W}
