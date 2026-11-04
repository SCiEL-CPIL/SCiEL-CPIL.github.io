import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 10
    
    # Rank 55 decomposition:
    # 5 blocks of Strassen (2x2x2) for j in {0,1}, k pairs (0,1)..(8,9) -> 5 * 7 = 35
    # 1 block of Schoolbook (2x1x10) for j=2 -> 20
    # Total 55
    
    rank = 55
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Strassen for 2x2x2
    # A: 2x2, B: 2x2, C: 2x2
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    # M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
    # M3 = A00(B01-B11) -> C01 += M3, C11 += M3
    # M4 = A11(B10-B00) -> C00 += M4, C10 += M4
    # M5 = (A00+A01)B11 -> C00 -= M5, C01 += M5
    # M6 = (A11-A00)(B00+B01) -> C11 += M6
    # M7 = (A01-A11)(B10+B11) -> C00 += M7, C11 -= M7
    
    for pair in range(5):
        k0 = pair * 2
        k1 = pair * 2 + 1
        
        # M1
        U[0, col] = 1
        U[4, col] = 1
        V[k0, col] = 1
        V[10 + k1, col] = 1
        W[2 * k0, col] = 1
        W[2 * k1 + 1, col] = 1
        col += 1
        
        # M2
        U[3, col] = 1
        U[4, col] = 1
        V[k0, col] = 1
        W[2 * k0 + 1, col] = 1
        W[2 * k1 + 1, col] = -1
        col += 1
        
        # M3
        U[0, col] = 1
        V[10 + k1, col] = 1
        V[10 + k1 + 10, col] = -1
        W[2 * k1, col] = 1
        W[2 * k1 + 1, col] = 1
        col += 1
        
        # M4
        U[4, col] = 1
        V[10 + k0, col] = 1
        V[k0, col] = -1
        W[2 * k0, col] = 1
        W[2 * k0 + 1, col] = 1
        col += 1
        
        # M5
        U[0, col] = 1
        U[1, col] = 1
        V[10 + k1 + 10, col] = 1
        W[2 * k0, col] = -1
        W[2 * k1, col] = 1
        col += 1
        
        # M6
        U[4, col] = 1
        U[0, col] = -1
        V[k0, col] = 1
        V[k0 + 1, col] = 1
        W[2 * k1 + 1, col] = 1
        col += 1
        
        # M7
        U[1, col] = 1
        U[4, col] = -1
        V[10 + k0, col] = 1
        V[10 + k1, col] = 1
        W[2 * k0, col] = 1
        W[2 * k1 + 1, col] = -1
        col += 1
    
    # Schoolbook for j=2
    for i in range(2):
        for k in range(10):
            U[i * 3 + 2, col] = 1
            V[20 + k, col] = 1
            W[2 * k + i, col] = 1
            col += 1
            
    assert col == rank, f"Column count {col} != expected rank {rank}"
    
    # Convert to list format as required by the interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
