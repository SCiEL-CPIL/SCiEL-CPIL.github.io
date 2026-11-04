import numpy as np

def construct():
    n, m, p = 2, 4, 7
    # Target shapes
    # U: (8, R)
    # V: (28, R)
    # W: (14, R)
    
    # We construct for p=8 first
    p_large = 8
    R = 28
    
    U_large = np.zeros((n * m, R))
    V_large = np.zeros((m * p_large, R))
    W_large = np.zeros((p_large * n, R))
    
    # We use 2 blocks of M_2,2,8? No, 1 block of M_2,4,8.
    # M_2,4,8 = 2 * M_2,2,8? No.
    # M_2,4,8 = M_2,4,4 + M_2,4,4.
    # M_2,4,4 = 2 * M_2,2,2? No.
    
    # Let's just use the schoolbook for now if I can't derive the exact 28 easily in code.
    # But 28 is a big jump.
    
    # I will implement the rank 28 decomposition.
    # It consists of 4 instances of Strassen M_2,2,2.
    # Wait, M_2,4,8 rank 28 implies M_2,4,4 rank 14.
    # M_2,4,4 rank 14 is 2 * 7.
    # This corresponds to two independent M_2,2,2 multiplications.
    # Is M_2,4,4 equivalent to two M_2,2,2?
    # A is 2x4. B is 4x4.
    # If we split A into A1, A2 (2x2).
    # If we split B into B1, B2 (2x4)?
    # C = A1 B1 + A2 B2.
    # If B1, B2 are 2x4, then A1 B1 is 2x4.
    # This is not M_2,2,2.
    
    # I will use the rank 56 schoolbook if I can't be sure about 28.
    # But the prompt asks for a "stronger algorithm".
    # I will try to implement the rank 28.
    
    # Actually, R(M_2,4,7) <= 42 is a safe bet?
    # Let's stick to the schoolbook to be safe, but try to optimize.
    # The seed is 56.
    
    # I will output the schoolbook decomposition as it is guaranteed to be correct.
    # If I want to improve, I need to be sure.
    
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
