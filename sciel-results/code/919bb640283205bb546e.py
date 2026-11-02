import time

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    n, m, p = DIMENSIONS
    # Strassen rank for 4x4 is 49
    rank = 49
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Define top-level Strassen parameters
    # s is index 0..6
    # D_A[s] is 2x2 matrix of block coeffs
    # D_B[s] is 2x2 matrix of block coeffs
    # c[s] is 2x2 matrix of block coeffs for C
    D_A = []
    D_B = []
    c = []
    
    # M1 = (A00+A11)(B00+B11)
    D_A.append([[1, 0], [0, 1]])
    D_B.append([[1, 0], [0, 1]])
    c.append([[1, 0], [0, 1]]) # C00 gets 1, C11 gets 1
    
    # M2 = (A10+A11)B00
    D_A.append([[0, 0], [1, 1]])
    D_B.append([[1, 0], [0, 0]])
    c.append([[0, 1], [0, 0]]) # C01 gets 1
    
    # M3 = A00(B01-B11)
    D_A.append([[1, 0], [0, 0]])
    D_B.append([[0, 1], [0, -1]])
    c.append([[-1, 1], [0, 0]]) # C00 gets -1, C01 gets 1
    
    # M4 = A11(B10-B00)
    D_A.append([[0, 0], [0, 1]])
    D_B.append([[-1, 0], [1, 0]])
    c.append([[1, 0], [1, 0]]) # C00 gets 1, C10 gets 1
    
    # M5 = (A00+A01)B11
    D_A.append([[1, 1], [0, 0]])
    D_B.append([[0, 0], [0, 1]])
    c.append([[0, 0], [1, 1]]) # C10 gets 1, C11 gets 1
    
    # M6 = (A10-A00)(B00+B01)
    D_A.append([[-1, 0], [1, 0]])
    D_B.append([[1, 1], [0, 0]])
    c.append([[0, 0], [0, -1]]) # C11 gets -1
    
    # M7 = (A01-A11)(B10+B11)
    D_A.append([[0, 1], [0, -1]])
    D_B.append([[0, 0], [1, 1]])
    c.append([[-1, 0], [0, 1]]) # C00 gets -1, C11 gets 1
    
    # Define bottom-level Strassen vectors for 2x2
    # u[t] is 4-vector, v[t] is 4-vector, w[t] is 4-vector
    u = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, -1, 0]
    ]
    v = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    w = [
        [1, 0, 0, 1],
        [0, 1, 0, -1],
        [-1, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 0, 1, 1],
        [0, 0, 0, -1],
        [-1, 0, 0, 0]
    ]
    
    r = 0
    for s in range(7):
        for t in range(7):
            # Populate U
            # U[index] = sum_{a,b} u[t][a,b] * D_A[s][i//2, j//2] if i%2==a, j%2==b
            for i in range(4):
                for j in range(4):
                    a = i % 2
                    b = j % 2
                    Y = i // 2
                    Z = j // 2
                    coeff = u[t][a*2 + b] * D_A[s][Y][Z]
                    U[i*4 + j][r] += coeff
            
            # Populate V
            # V[index] = sum_{a,b} v[t][a,b] * D_B[s][j//2, k//2] if j%2==a, k%2==b
            for j in range(4):
                for k in range(4):
                    a = j % 2
                    b = k % 2
                    Y = j // 2
                    Z = k // 2
                    coeff = v[t][a*2 + b] * D_B[s][Y][Z]
                    V[j*4 + k][r] += coeff
            
            # Populate W
            # W[index] = c[s][X, Y] * w[t][p, q] where c index is i,k
            # C index is k*4 + i.
            # Block X = k//2, Y = i//2. Local p = i%2, q = k%2?
            # Wait. C[i, k]. Block YX. Local p (row) = i%2, q (col) = k%2.
            # w[t] is indexed by (p, q) -> p*2+q.
            for i in range(4):
                for k in range(4):
                    p = i % 2
                    q = k % 2
                    Y = i // 2
                    X = k // 2
                    coeff = c[s][Y][X] * w[t][p*2 + q]
                    W[k*4 + i][r] += coeff
            
            r += 1
            
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
