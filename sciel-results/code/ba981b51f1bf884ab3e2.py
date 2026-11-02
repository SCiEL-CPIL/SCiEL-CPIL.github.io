import numpy as np

def construct():
    """
    Returns the rank-7 Strassen decomposition for 2x2 matrix multiplication.
    The tensor rank of 2x2 matmul is known to be 7, so this is the minimal rank.
    
    U: (4, 7) corresponding to A flattened (row-major)
    V: (4, 7) corresponding to B flattened (row-major)
    W: (4, 7) corresponding to C flattened with W[k*2 + i, r]
    """
    rank = 7
    # Initialize matrices with zeros
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(4)]
    W = [[0] * rank for _ in range(4)]
    
    # Strassen's 7 products:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # U coefficients for A:
    # Indices: 0=A00, 1=A01, 2=A10, 3=A11
    
    # M1
    U[0][0] = 1
    U[3][0] = 1
    
    # M2
    U[2][1] = 1
    U[3][1] = 1
    
    # M3
    U[0][2] = 1
    
    # M4
    U[3][3] = 1
    
    # M5
    U[0][4] = 1
    U[1][4] = 1
    
    # M6
    U[2][5] = 1
    U[0][5] = -1
    
    # M7
    U[1][6] = 1
    U[3][6] = -1
    
    # V coefficients for B:
    # Indices: 0=B00, 1=B01, 2=B10, 3=B11
    
    # M1
    V[0][0] = 1
    V[3][0] = 1
    
    # M2
    V[0][1] = 1
    
    # M3
    V[1][2] = 1
    V[3][2] = -1
    
    # M4
    V[2][3] = 1
    V[0][3] = -1
    
    # M5
    V[3][4] = 1
    
    # M6
    V[0][5] = 1
    V[1][5] = 1
    
    # M7
    V[2][6] = 1
    V[3][6] = 1
    
    # W coefficients for C:
    # Indices: 0=C00 (k=0,i=0), 1=C10 (k=0,i=1), 2=C01 (k=1,i=0), 3=C11 (k=1,i=1)
    
    # C00 = M1 + M4 - M5 + M7
    W[0][0] = 1
    W[0][3] = 1
    W[0][4] = -1
    W[0][6] = 1
    
    # C10 = M2 + M4
    W[1][1] = 1
    W[1][3] = 1
    
    # C01 = M3 + M5
    W[2][2] = 1
    W[2][4] = 1
    
    # C11 = M1 - M2 + M3 + M6
    W[3][0] = 1
    W[3][1] = -1
    W[3][2] = 1
    W[3][5] = 1
    
    # Convert to numpy arrays with integer dtype for strict type compliance
    U_np = np.array(U, dtype=int)
    V_np = np.array(V, dtype=int)
    W_np = np.array(W, dtype=int)
    
    return {"U": U_np, "V": V_np, "W": W_np}
