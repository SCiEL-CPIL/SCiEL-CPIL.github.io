import time

DIMENSIONS = [2, 2, 10]

def construct():
    n, m, p = DIMENSIONS
    # We use Strassen's algorithm for 2x2 matrix multiplication.
    # B is 2x10, so we split it into 5 blocks of 2x2.
    # Each block multiplication A * B_t has rank 7.
    # Total rank = 5 * 7 = 35.
    
    rank = 35
    U = [[0] * rank for _ in range(n * m)] # n*m = 4
    V = [[0] * rank for _ in range(m * p)] # m*p = 20
    W = [[0] * rank for _ in range(p * n)] # p*n = 20
    
    # Strassen's 7 multiplications for 2x2 * 2x2
    # Let A = [[a11, a12], [a21, a22]]
    # Let B = [[b11, b12], [b21, b22]]
    # M1 = (a11 + a22) * (b11 + b22)
    # M2 = (a21 + a22) * b11
    # M3 = a11 * (b12 - b22)
    # M4 = a22 * (b21 - b11)
    # M5 = (a11 + a12) * b22
    # M6 = (a21 - a11) * (b11 + b12)
    # M7 = (a12 - a22) * (b21 + b22)
    
    # Indices for A: i in {0,1}, j in {0,1} -> idx = i*m + j
    # 0: a11, 1: a12, 2: a21, 3: a22
    
    # Indices for B: j in {0,1}, k in {0..9} -> idx = j*p + k
    # Block t (0..4) has columns 2t, 2t+1.
    # b11,t is at j=0, k=2t -> idx = 0*10 + 2t = 2t
    # b12,t is at j=0, k=2t+1 -> idx = 2t+1
    # b21,t is at j=1, k=2t -> idx = 10 + 2t
    # b22,t is at j=1, k=2t+1 -> idx = 10 + 2t + 1
    
    # Indices for C: k in {0..9}, i in {0,1} -> idx = k*n + i
    # c11,t is at k=2t, i=0 -> idx = 2t*2 + 0 = 4t
    # c12,t is at k=2t+1, i=0 -> idx = (2t+1)*2 + 0 = 4t+2
    # c21,t is at k=2t, i=1 -> idx = 2t*2 + 1 = 4t+1
    # c22,t is at k=2t+1, i=1 -> idx = (2t+1)*2 + 1 = 4t+3
    
    # Strassen reconstruction:
    # c11 = M1 + M4 - M5 + M7
    # c12 = M3 + M5
    # c21 = M2 + M4
    # c22 = M1 - M2 + M3 + M6
    
    for t in range(5):
        base = t * 7
        # M1
        U[0][base] += 1  # a11
        U[3][base] += 1  # a22
        V[2*t][base] += 1    # b11,t
        V[10+2*t+1][base] += 1 # b22,t
        W[4*t][base] += 1    # c11,t
        W[4*t+3][base] += 1  # c22,t
        
        # M2
        U[2][base+1] += 1  # a21
        U[3][base+1] += 1  # a22
        V[2*t][base+1] += 1    # b11,t
        W[4*t+1][base+1] += 1  # c21,t
        
        # M3
        U[0][base+2] += 1  # a11
        V[2*t+1][base+2] += 1   # b12,t
        V[10+2*t+1][base+2] -= 1 # b22,t
        W[4*t][base+2] += 1    # c11,t
        W[4*t+2][base+2] += 1  # c12,t
        
        # M4
        U[3][base+3] += 1  # a22
        V[10+2*t][base+3] += 1   # b21,t
        V[2*t][base+3] -= 1      # b11,t
        W[4*t][base+3] += 1    # c11,t
        W[4*t+1][base+3] += 1  # c21,t
        
        # M5
        U[0][base+4] += 1  # a11
        U[1][base+4] += 1  # a12
        V[10+2*t+1][base+4] += 1 # b22,t
        W[4*t][base+4] -= 1    # c11,t
        W[4*t+2][base+4] += 1  # c12,t
        
        # M6
        U[2][base+5] += 1  # a21
        U[0][base+5] -= 1  # a11
        V[2*t][base+5] += 1    # b11,t
        V[2*t+1][base+5] += 1  # b12,t
        W[4*t+3][base+5] += 1  # c22,t
        
        # M7
        U[1][base+6] += 1  # a12
        U[3][base+6] -= 1  # a22
        V[10+2*t][base+6] += 1   # b21,t
        V[10+2*t+1][base+6] += 1 # b22,t
        W[4*t][base+6] += 1    # c11,t
        W[4*t+3][base+6] += 1  # c22,t

    return {"U": U, "V": V, "W": W}
