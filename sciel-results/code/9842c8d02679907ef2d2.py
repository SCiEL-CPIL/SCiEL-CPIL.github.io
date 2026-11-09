import numpy as np

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    # Rank 28 decomposition
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a rank-1 component
    def add_component(u_indices, v_indices, w_indices, val=1):
        nonlocal col
        for i, j in u_indices:
            U[i * m + j][col] += val
        for j, k in v_indices:
            V[j * p + k][col] += val
        for k, i in w_indices:
            W[k * n + i][col] += val
        col += 1

    # 1. Strassen 2x2x2 block (i in 0,1; j in 0,1; k in 0,1)
    # M1 = A00 (B00 - B10)
    add_component([(0,0)], [(0,0), (1,0)], [(0,0), (0,1)], 1)
    U[0*m+0][col-1] = 1
    V[0*p+0][col-1] = 1
    V[1*p+0][col-1] = -1
    W[0*n+0][col-1] = 1 # C00? No, M1 is in C01 and C11.
    # Correct W indices for M1:
    # C01 = M1 + M7. W index for C01 is k=1, i=0 -> 1*2+0 = 2.
    # C11 = M5 + M1 - M3 - M6. W index for C11 is k=1, i=1 -> 1*2+1 = 3.
    # So W[2] += 1, W[3] += 1.
    # My helper added W[0,0] and W[0,1] which are indices 0 and 1.
    # I need to be careful with the helper. It takes (k,i) pairs.
    # Let's re-verify the helper.
    # W index is k*n + i.
    # For M1:
    # Contributes to C01 (k=1, i=0) -> idx 2.
    # Contributes to C11 (k=1, i=1) -> idx 3.
    # So w_indices should be [(1,0), (1,1)].
    
    # Let's restart the Strassen block implementation carefully using the helper.
    col = 0
    
    # M1 = A00 (B00 - B10)
    # U: (0,0)=1
    # V: (0,0)=1, (1,0)=-1
    # W: (1,0)=1, (1,1)=1
    add_component([(0,0)], [(0,0), (1,0)], [(1,0), (1,1)], 1)
    U[0*m+0][col-1] = 1
    V[0*p+0][col-1] = 1
    V[1*p+0][col-1] = -1
    W[1*n+0][col-1] = 1
    W[1*n+1][col-1] = 1
    col += 1
    
    # M2 = (A00 + A10) B10
    # U: (0,0)=1, (1,0)=1
    # V: (1,0)=1
    # W: (0,0)=-1 (C00 = ... - M2)
    add_component([(0,0), (1,0)], [(1,0)], [(0,0)], 1)
    U[0*m+0][col-1] = 1
    U[1*m+0][col-1] = 1
    V[1*p+0][col-1] = 1
    W[0*n+0][col-1] = -1
    col += 1
    
    # M3 = (A01 + A11) B01
    # U: (0,1)=1, (1,1)=1
    # V: (0,1)=1
    # W: (0,1)=1 (C10 = M3 + M6), (1,1)=-1 (C11 = ... - M3)
    add_component([(0,1), (1,1)], [(0,1)], [(0,1), (1,1)], 1)
    U[0*m+1][col-1] = 1
    U[1*m+1][col-1] = 1
    V[0*p+1][col-1] = 1
    W[0*n+1][col-1] = 1
    W[1*n+1][col-1] = -1
    col += 1
    
    # M4 = A11 (B11 - B01)
    # U: (1,1)=1
    # V: (1,1)=1, (0,1)=-1
    # W: (0,0)=1 (C00 = ... + M4)
    add_component([(1,1)], [(1,1), (0,1)], [(0,0)], 1)
    U[1*m+1][col-1] = 1
    V[1*p+1][col-1] = 1
    V[0*p+1][col-1] = -1
    W[0*n+0][col-1] = 1
    col += 1
    
    # M5 = (A00 + A11) (B00 + B11)
    # U: (0,0)=1, (1,1)=1
    # V: (0,0)=1, (1,1)=1
    # W: (0,0)=1 (C00), (1,1)=1 (C11)
    add_component([(0,0), (1,1)], [(0,0), (1,1)], [(0,0), (1,1)], 1)
    U[0*m+0][col-1] = 1
    U[1*m+1][col-1] = 1
    V[0*p+0][col-1] = 1
    V[1*p+1][col-1] = 1
    W[0*n+0][col-1] = 1
    W[1*n+1][col-1] = 1
    col += 1
    
    # M6 = (A10 - A00) (B00 + B01)
    # U: (1,0)=1, (0,0)=-1
    # V: (0,0)=1, (0,1)=1
    # W: (0,1)=1 (C10 = M3 + M6), (1,1)=-1 (C11 = ... - M6)
    add_component([(1,0), (0,0)], [(0,0), (0,1)], [(0,1), (1,1)], 1)
    U[1*m+0][col-1] = 1
    U[0*m+0][col-1] = -1
    V[0*p+0][col-1] = 1
    V[0*p+1][col-1] = 1
    W[0*n+1][col-1] = 1
    W[1*n+1][col-1] = -1
    col += 1
    
    # M7 = (A01 - A11) (B10 + B11)
    # U: (0,1)=1, (1,1)=-1
    # V: (1,0)=1, (1,1)=1
    # W: (0,0)=-1 (C00 = ... - M7), (0,1)=1 (C01 = M1 + M7)
    add_component([(0,1), (1,1)], [(1,0), (1,1)], [(0,0), (0,1)], 1)
    U[0*m+1][col-1] = 1
    U[1*m+1][col-1] = -1
    V[1*p+0][col-1] = 1
    V[1*p+1][col-1] = 1
    W[0*n+0][col-1] = -1
    W[0*n+1][col-1] = 1
    col += 1
    
    # 2. Strassen 2x2x2 block (i in 0,1; j in 0,1; k in 2,3)
    # Same structure, k offset by 2.
    # M1' = A00 (B02 - B12)
    add_component([(0,0)], [(0,2), (1,2)], [(2,0), (3,0)], 1)
    U[0*m+0][col-1] = 1
    V[0*p+2][col-1] = 1
    V[1*p+2][col-1] = -1
    W[2*n+0][col-1] = 1
    W[3*n+0][col-1] = 1 # Wait, M1' contributes to C02 and C12?
    # C02 = M1' + M7'
    # C12 = M5' + M1' - M3' - M6'
    # So W indices: (2,0) and (2,1)?
    # k=2, i=0 -> C02. k=2, i=1 -> C12.
    # Yes.
    col += 1
    
    # M2' = (A00 + A10) B12
    add_component([(0,0), (1,0)], [(1,2)], [(2,0)], 1)
    U[0*m+0][col-1] = 1
    U[1*m+0][col-1] = 1
    V[1*p+2][col-1] = 1
    W[2*n+0][col-1] = -1
    col += 1
    
    # M3' = (A01 + A11) B03
    add_component([(0,1), (1,1)], [(0,3)], [(2,1), (3,1)], 1)
    U[0*m+1][col-1] = 1
    U[1*m+1][col-1] = 1
    V[0*p+3][col-1] = 1
    W[2*n+1][col-1] = 1
    W[3*n+1][col-1] = -1
    col += 1
    
    # M4' = A11 (B13 - B03)
    add_component([(1,1)], [(1,3), (0,3)], [(2,0)], 1)
    U[1*m+1][col-1] = 1
    V[1*p+3][col-1] = 1
    V[0*p+3][col-1] = -1
    W[2*n+0][col-1] = 1
    col += 1
    
    # M5' = (A00 + A11) (B02 + B13)
    add_component([(0,0), (1,1)], [(0,2), (1,3)], [(2,0), (3,1)], 1)
    U[0*m+0][col-1] = 1
    U[1*m+1][col-1] = 1
    V[0*p+2][col-1] = 1
    V[1*p+3][col-1] = 1
    W[2*n+0][col-1] = 1
    W[3*n+1][col-1] = 1
    col += 1
    
    # M6' = (A10 - A00) (B02 + B03)
    add_component([(1,0), (0,0)], [(0,2), (0,3)], [(2,1), (3,1)], 1)
    U[1*m+0][col-1] = 1
    U[0*m+0][col-1] = -1
    V[0*p+2][col-1] = 1
    V[0*p+3][col-1] = 1
    W[2*n+1][col-1] = 1
    W[3*n+1][col-1] = -1
    col += 1
    
    # M7' = (A01 - A11) (B12 + B13)
    add_component([(0,1), (1,1)], [(1,2), (1,3)], [(2,0), (2,1)], 1)
    U[0*m+1][col-1] = 1
    U[1*m+1][col-1] = -1
    V[1*p+2][col-1] = 1
    V[1*p+3][col-1] = 1
    W[2*n+0][col-1] = -1
    W[2*n+1][col-1] = 1
    col += 1
    
    # 3. Schoolbook for j=2
    # Terms: A[i,2] * B[2,k] for i in 0,1 and k in 0..4
    for i in range(2):
        for k in range(5):
            U[i*m+2][col] = 1
            V[2*p+k][col] = 1
            W[k*n+i][col] = 1
            col += 1
            
    # 4. Schoolbook for k=4, j in 0,1
    # Terms: A[i,j] * B[j,4] for i in 0,1 and j in 0,1
    for i in range(2):
        for j in range(2):
            U[i*m+j][col] = 1
            V[j*p+4][col] = 1
            W[4*n+i][col] = 1
            col += 1

    return {"U": U, "V": V, "W": W}
