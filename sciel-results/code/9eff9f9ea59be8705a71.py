import time

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen(u11, u12, u21, u22, b11, b12, b21, b22, w11, w12, w21, w22, c):
        # M1
        U[u11][c] += 1
        U[u22][c] += 1
        V[b11][c] += 1
        V[b22][c] += 1
        W[w11][c] += 1
        W[w22][c] -= 1
        c += 1
        # M2
        U[u21][c] += 1
        U[u22][c] += 1
        V[b11][c] += 1
        W[w21][c] += 1
        W[w22][c] -= 1
        c += 1
        # M3
        U[u11][c] += 1
        V[b21][c] += 1
        V[b22][c] -= 1
        W[w12][c] += 1
        W[w22][c] += 1
        c += 1
        # M4
        U[u22][c] += 1
        V[b12][c] += 1
        V[b11][c] -= 1
        W[w11][c] += 1
        W[w21][c] += 1
        c += 1
        # M5
        U[u11][c] += 1
        U[u12][c] += 1
        V[b22][c] += 1
        W[w11][c] -= 1
        W[w12][c] -= 1
        c += 1
        # M6
        U[u21][c] += 1
        U[u11][c] -= 1
        V[b11][c] += 1
        V[b12][c] += 1
        W[w22][c] += 1
        c += 1
        # M7
        U[u12][c] += 1
        U[u22][c] -= 1
        V[b21][c] += 1
        V[b22][c] += 1
        W[w11][c] += 1
        W[w12][c] += 1
        c += 1
        return c

    def add_schoolbook(u11, u12, u21, u22, b1, b2, w1, w2, c):
        # M1
        U[u11][c] += 1
        V[b1][c] += 1
        W[w1][c] += 1
        c += 1
        # M2
        U[u12][c] += 1
        V[b2][c] += 1
        W[w1][c] += 1
        c += 1
        # M3
        U[u21][c] += 1
        V[b1][c] += 1
        W[w2][c] += 1
        c += 1
        # M4
        U[u22][c] += 1
        V[b2][c] += 1
        W[w2][c] += 1
        c += 1
        return c

    # Block 1: A1 (U 0,1,2,3), B1 (V 0..9), W1 (W 0..9)
    # B1 Row 0: V[0..4], Row 1: V[5..9]
    # W1 Col 0: W[0,2], Col 1: W[1,3], Col 2: W[2,4], Col 3: W[3,5], Col 4: W[4,6]
    # Wait, W index k*n + i.
    # k=0, i=0 -> 0. k=0, i=1 -> 1? No, n=2.
    # k=0, i=0 -> 0. k=0, i=1 -> 1.
    # k=1, i=0 -> 2. k=1, i=1 -> 3.
    # So W[0] is C[0,0]. W[1] is C[1,0].
    
    # Strassen for B1 cols 0,1
    # B11=V[0], B12=V[1], B21=V[5], B22=V[6]
    # W11=W[0], W12=W[2], W21=W[1], W22=W[3]
    col = add_strassen(0, 1, 2, 3, 0, 1, 5, 6, 0, 2, 1, 3, col)
    
    # Strassen for B1 cols 2,3
    # B11=V[2], B12=V[3], B21=V[7], B22=V[8]
    # W11=W[4], W12=W[6], W21=W[5], W22=W[7]
    col = add_strassen(0, 1, 2, 3, 2, 3, 7, 8, 4, 6, 5, 7, col)
    
    # Schoolbook for B1 col 4
    # B1=V[4], B2=V[9]
    # W1=W[8], W2=W[9]
    col = add_schoolbook(0, 1, 2, 3, 4, 9, 8, 9, col)
    
    # Block 2: A2 (U 4,5,6,7), B2 (V 10..19), W2 (W 10..19)
    # B2 Row 0: V[10..14], Row 1: V[15..19]
    # W2 Col 0: W[10,12], Col 1: W[11,13], etc.
    
    # Strassen for B2 cols 0,1
    col = add_strassen(4, 5, 6, 7, 10, 11, 15, 16, 10, 12, 11, 13, col)
    
    # Strassen for B2 cols 2,3
    col = add_strassen(4, 5, 6, 7, 12, 13, 17, 18, 14, 16, 15, 17, col)
    
    # Schoolbook for B2 col 4
    col = add_schoolbook(4, 5, 6, 7, 14, 19, 18, 19, col)
    
    return {"U": U, "V": V, "W": W}
