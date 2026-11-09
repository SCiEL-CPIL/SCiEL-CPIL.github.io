import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = 2, 3, 10
    rank = 55
    
    U = [[0] * rank for _ in range(n * m)]  # 6 x 55
    V = [[0] * rank for _ in range(m * p)]  # 30 x 55
    W = [[0] * rank for _ in range(p * n)]  # 20 x 55
    
    col = 0
    for t in range(5):
        # Block t covers columns 2t and 2t+1 of B and C
        # A' is A[:, 0:2], B' is B[0:2, 2t:2t+2]
        # a3 is A[:, 2], b3 is B[2, 2t:2t+2]
        
        # Strassen terms for A' * B'
        # Indices for A':
        # a00: 0, a01: 1, a10: 3, a11: 4
        # Indices for B':
        # b00: 2t, b01: 2t+1, b10: 10+2t, b11: 10+2t+1
        # Indices for C':
        # c00: 4t, c01: 4t+2, c10: 4t+1, c11: 4t+3
        
        # M1 = (a00 + a11)(b00 + b11)
        U[0][col] += 1; U[4][col] += 1
        V[2*t][col] += 1; V[10+2*t+1][col] += 1
        W[4*t][col] += 1; W[4*t+3][col] += 1
        col += 1
        
        # M2 = (a10 + a11)b00
        U[3][col] += 1; U[4][col] += 1
        V[2*t][col] += 1
        W[4*t+1][col] += 1; W[4*t+3][col] -= 1
        col += 1
        
        # M3 = a00(b10 - b11)
        U[0][col] += 1
        V[10+2*t][col] += 1; V[10+2*t+1][col] -= 1
        W[4*t+2][col] += 1; W[4*t+3][col] += 1
        col += 1
        
        # M4 = a11(b01 + b11)
        U[4][col] += 1
        V[2*t+1][col] += 1; V[10+2*t+1][col] += 1
        W[4*t][col] -= 1; W[4*t+2][col] += 1
        col += 1
        
        # M5 = (a00 + a01)b11
        U[0][col] += 1; U[1][col] += 1
        V[10+2*t+1][col] += 1
        W[4*t][col] += 1; W[4*t+1][col] -= 1
        col += 1
        
        # M6 = (a10 - a00)(b00 + b01)
        U[3][col] += 1; U[0][col] -= 1
        V[2*t][col] += 1; V[2*t+1][col] += 1
        W[4*t+1][col] += 1; W[4*t+3][col] += 1
        col += 1
        
        # M7 = (a01 - a11)(b10 + b11)
        U[1][col] += 1; U[4][col] -= 1
        V[10+2*t][col] += 1; V[10+2*t+1][col] += 1
        W[4*t][col] -= 1; W[4*t+2][col] += 1
        col += 1
        
        # Outer product terms for a3 * b3
        # a3: A[0,2] (idx 2), A[1,2] (idx 5)
        # b3: B[2, 2t] (idx 20+2t), B[2, 2t+1] (idx 20+2t+1)
        
        # A[0,2] * B[2, 2t] -> C[0, 2t] (idx 4t)
        U[2][col] += 1
        V[20+2*t][col] += 1
        W[4*t][col] += 1
        col += 1
        
        # A[0,2] * B[2, 2t+1] -> C[0, 2t+1] (idx 4t+2)
        U[2][col] += 1
        V[20+2*t+1][col] += 1
        W[4*t+2][col] += 1
        col += 1
        
        # A[1,2] * B[2, 2t] -> C[1, 2t] (idx 4t+1)
        U[5][col] += 1
        V[20+2*t][col] += 1
        W[4*t+1][col] += 1
        col += 1
        
        # A[1,2] * B[2, 2t+1] -> C[1, 2t+1] (idx 4t+3)
        U[5][col] += 1
        V[20+2*t+1][col] += 1
        W[4*t+3][col] += 1
        col += 1
        
        if time.time() > deadline:
            return {"U": U, "V": V, "W": W}
            
    return {"U": U, "V": V, "W": W}
