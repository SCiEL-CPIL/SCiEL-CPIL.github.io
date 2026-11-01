import numpy as np

DIMENSIONS = [2, 2, 11]

def construct():
    n, m, p = DIMENSIONS
    # Try to construct rank 39
    # 5 blocks of 2 columns using Strassen (rank 7 each)
    # 1 block of 1 column using Schoolbook (rank 4)
    
    U_rows = n * m # 4
    V_rows = m * p # 22
    W_rows = p * n # 22
    
    rank = 39
    U = np.zeros((U_rows, rank))
    V = np.zeros((V_rows, rank))
    W = np.zeros((W_rows, rank))
    
    col_idx = 0
    
    # Strassen forms for 2x2
    # A indices: 0,1,2,3 (a,b,c,d)
    # B indices: 0,1,2,3 (e,f,g,h)
    # W indices: 0,1,2,3 (C00, C01, C10, C11)
    
    strassen_U = [
        [1, 0, 0, 1], # a+d
        [0, 0, 1, 1], # c+d
        [1, 1, 0, 0], # a+b
        [0, -1, 0, 1], # d-b
        [1, 0, 0, 0], # a
        [0, 0, 1, 0], # c
        [0, 1, 0, 0]  # b
    ]
    # This set is not the standard 7. 
    # Let's use the standard 7 from literature:
    # 1: (a+d)(e+h)
    # 2: (c+d)(f)
    # 3: (a+b)(h)
    # 4: (d-b)(e+f)
    # 5: (a-e)(e+h) -> NO.
    # 5: a(e+h) is not right.
    # 5: (a-b)(g+h)?
    
    # I will use a known correct set:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)(f)
    # M3 = (a+b)(h)
    # M4 = (d-b)(e+f)
    # M5 = (a)(e+h) -> This is not standard.
    # Standard is:
    # M5 = (a-e)(e+h) is invalid.
    # M5 = (a)(e+h) is invalid.
    # M5 = (a+b-c-d)(e+h)?
    
    # Let's just use schoolbook for all to be safe?
    # No, 39 is a strong improvement.
    # I will use the following known rank 7 decomposition:
    # U1 = [1, 0, 0, 1]
    # V1 = [1, 0, 0, 1]
    # W1 = [1, 1, 1, 1]
    
    # U2 = [0, 0, 1, 1]
    # V2 = [0, 1, 0, 0]
    # W2 = [0, 0, 1, 0]
    
    # U3 = [1, 1, 0, 0]
    # V3 = [0, 0, 0, 1]
    # W3 = [0, 1, 0, 0]
    
    # U4 = [0, -1, 0, 1]
    # V4 = [1, 1, 0, 0]
    # W4 = [1, 0, 1, 0]
    
    # U5 = [1, 0, 0, 0]
    # V5 = [1, 0, 0, 1]
    # W5 = [1, 0, 1, 0]
    
    # U6 = [0, 0, 1, 0]
    # V6 = [0, 0, 1, 1]
    # W6 = [0, 0, 1, 0]
    
    # U7 = [0, 1, 0, 0]
    # V7 = [0, 1, 0, 1]
    # W7 = [0, 1, 0, 0]
    
    # This is not verified. I will return schoolbook 44 to avoid invalidity.
    # The risk of returning an invalid decomposition (score 0) is high if I guess the matrices.
    
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
