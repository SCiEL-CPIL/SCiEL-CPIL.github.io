import time

DIMENSIONS = [2, 2, 14]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=14
    # Rank 49
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # A indices: 0,1,2,3 correspond to A[0,0], A[0,1], A[1,0], A[1,1]
    # B indices: j*p + k
    # C indices: k*n + i
    
    for b in range(7):
        base_col = b * 7
        # B block columns: 2*b, 2*b+1
        # C block columns: 2*b, 2*b+1
        
        # Term 0: P1 = (A00 + A11) * (B0,2b + B1,2b+1)
        c = base_col
        U[0][c] = 1
        U[3][c] = 1
        V[0 * p + 2 * b][c] = 1
        V[1 * p + (2 * b + 1)][c] = 1
        W[(2 * b) * n + 0][c] = 1
        W[(2 * b + 1) * n + 1][c] = 1
        
        # Term 1: P2 = (A10 + A11) * B0,2b
        c = base_col + 1
        U[2][c] = 1
        U[3][c] = 1
        V[0 * p + 2 * b][c] = 1
        W[(2 * b) * n + 1][c] = 1
        W[(2 * b + 1) * n + 1][c] = -1
        
        # Term 2: P3 = A00 * (B0,2b+1 - B1,2b+1)
        c = base_col + 2
        U[0][c] = 1
        V[0 * p + (2 * b + 1)][c] = 1
        V[1 * p + (2 * b + 1)][c] = -1
        W[(2 * b + 1) * n + 0][c] = 1
        W[(2 * b + 1) * n + 1][c] = 1
        
        # Term 3: P4 = A11 * (B1,2b - B0,2b)
        c = base_col + 3
        U[3][c] = 1
        V[1 * p + 2 * b][c] = 1
        V[0 * p + 2 * b][c] = -1
        W[(2 * b) * n + 0][c] = 1
        W[(2 * b) * n + 1][c] = 1
        
        # Term 4: P5 = (A00 + A01) * B1,2b+1
        c = base_col + 4
        U[0][c] = 1
        U[1][c] = 1
        V[1 * p + (2 * b + 1)][c] = 1
        W[(2 * b) * n + 0][c] = -1
        W[(2 * b + 1) * n + 0][c] = 1
        
        # Term 5: P6 = (A10 - A00) * (B0,2b + B0,2b+1)
        c = base_col + 5
        U[2][c] = 1
        U[0][c] = -1
        V[0 * p + 2 * b][c] = 1
        V[0 * p + (2 * b + 1)][c] = 1
        W[(2 * b + 1) * n + 1][c] = 1
        
        # Term 6: P7 = (A01 - A11) * (B1,2b + B1,2b+1)
        c = base_col + 6
        U[1][c] = 1
        U[3][c] = -1
        V[1 * p + 2 * b][c] = 1
        V[1 * p + (2 * b + 1)][c] = 1
        W[(2 * b) * n + 0][c] = 1

    return {"U": U, "V": V, "W": W}
