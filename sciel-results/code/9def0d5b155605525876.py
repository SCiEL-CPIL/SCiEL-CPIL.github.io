import numpy as np

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=5
    
    # Target Rank: 28
    # Decomposition:
    # 1. j=0: 2x1x5 -> Rank 10 (Schoolbook)
    # 2. j in {1,2}, k in {0,1}: 2x2x2 -> Rank 7 (Strassen)
    # 3. j in {1,2}, k=2: 2x2x1 -> Rank 4 (Schoolbook)
    # 4. j in {1,2}, k in {3,4}: 2x2x2 -> Rank 7 (Strassen)
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, val=1):
        nonlocal col
        if val != 0:
            U[u_idx][col] += val
            V[v_idx][col] += val
            W[w_idx][col] += val
        col += 1

    # Block 1: j=0
    # i in 0..1, k in 0..4
    # Term: A[i,0] * B[0,k] -> C[k,i]
    for i in range(n):
        for k in range(p):
            u_idx = i * m + 0
            v_idx = 0 * p + k
            w_idx = k * n + i
            add_term(u_idx, v_idx, w_idx)

    # Block 2: j in {1,2}
    # Map local j' in {0,1} to global j = j' + 1
    
    # Sub-block 2a: k in {0,1} (Strassen 2x2x2)
    # Local k' in {0,1} maps to global k = k'
    # Strassen terms for C = A*B where A,B are 2x2
    # M1: (A00+A11)(B00+B11) -> C00, C11
    # M2: (A10+A11)B00 -> C10, -C11
    # M3: A00(B01-B11) -> C01, C11
    # M4: A11(B10-B00) -> C00, C10
    # M5: (A00+A01)B11 -> -C00, C01
    # M6: (A10-A00)(B00+B01) -> C11
    # M7: (A01-A11)(B10+B11) -> C00
    
    # Indices:
    # U: i * m + (j' + 1)
    # V: (j' + 1) * p + k'
    # W: k' * n + i
    
    # M1
    # A: (0,1) and (1,2) -> U[1], U[4]
    # B: (1,0) and (2,1) -> V[5], V[11]
    # C: (0,0) and (1,1) -> W[0], W[3]
    add_term(1, 5, 0) # A00 part? No, Strassen mixes.
    # Let's use the explicit coefficients from the current program which were verified.
    # I will just copy the logic but clean it up.
    
    # Reset col to 10 (after block 1)
    col = 10
    
    # Strassen 1 (k=0,1)
    # M1
    U[1][col] = 1; U[4][col] = 1
    V[5][col] = 1; V[11][col] = 1
    W[0][col] = 1; W[3][col] = 1
    col += 1
    
    # M2
    U[4][col] = 1; U[5][col] = 1 # Wait, U index for j=1 is 1*3+1=4? No.
    # n=2, m=3.
    # U index = i * m + j.
    # i=0, j=1 -> 0*3+1 = 1.
    # i=1, j=1 -> 1*3+1 = 4.
    # i=0, j=2 -> 0*3+2 = 2.
    # i=1, j=2 -> 1*3+2 = 5.
    
    # Let's re-verify indices for U in current program.
    # Block 2a M1:
    # U[0 * m + 1] = U[1]. U[1 * m + 2] = U[5].
    # My manual calc: i=0,j=1 -> 1. i=1,j=2 -> 5.
    # Current program: U[0*m+1]=1, U[1*m+2]=5. Matches.
    
    # V index = j * p + k.
    # j=1, k=0 -> 1*5+0 = 5.
    # j=2, k=1 -> 2*5+1 = 11.
    # Current program: V[1*p+0]=5, V[2*p+1]=11. Matches.
    
    # W index = k * n + i.
    # k=0, i=0 -> 0.
    # k=1, i=1 -> 1*2+1 = 3.
    # Current program: W[0*n+0]=0, W[1*n+1]=3. Matches.
    
    # I will trust the current program's indices as they seem consistent.
    # I will just rewrite it cleanly.
    
    # Since I am rewriting from scratch, I will use the verified indices from the analysis above.
    
    # M2
    U[4][col] = 1; U[5][col] = 1 # (A10+A11) -> i=1,j=1 (U4) and i=1,j=2 (U5)?
    # Strassen M2: (A10+A11)B00.
    # Local A10 is i=1, j'=0 -> Global i=1, j=1 -> U[4].
    # Local A11 is i=1, j'=1 -> Global i=1, j=2 -> U[5].
    # Local B00 is j'=0, k'=0 -> Global j=1, k=0 -> V[5].
    # C10: Local C10 is i=1, k'=0 -> Global i=1, k=0 -> W[0*2+1]=1.
    # C11: Local C11 is i=1, k'=1 -> Global i=1, k=1 -> W[1*2+1]=3.
    # Current program:
    # U[1*m+1]=4, U[1*m+2]=5.
    # V[1*p+0]=5.
    # W[0*n+1]=1, W[1*n+1]=-1.
    # Matches.
    
    # M3
    # A00 -> U[1].
    # B01-B11 -> V[1*p+1]=6, V[2*p+1]=11.
    # C01 -> W[1*n+0]=2.
    # C11 -> W[1*n+1]=3.
    # Current:
    # U[0*m+1]=1.
    # V[1*p+1]=6, V[2*p+1]=-11.
    # W[1*n+0]=2, W[1*n+1]=3.
    # Matches.
    
    # M4
    # A11 -> U[5].
    # B10-B00 -> V[2*p+0]=10, V[1*p+0]=-5.
    # C00 -> W[0*n+0]=0.
    # C10 -> W[0*n+1]=1.
    # Current:
    # U[1*m+2]=5.
    # V[2*p+0]=10, V[1*p+0]=-5.
    # W[0*n+0]=0, W[0*n+1]=1.
    # Matches.
    
    # M5
    # A00+A01 -> U[1], U[2].
    # B11 -> V[2*p+1]=11.
    # -C00 -> W[0*n+0]=-0.
    # C01 -> W[1*n+0]=2.
    # Current:
    # U[0*m+1]=1, U[0*m+2]=2.
    # V[2*p+1]=11.
    # W[0*n+0]=-0, W[1*n+0]=2.
    # Matches.
    
    # M6
    # A10-A00 -> U[4], -U[1].
    # B00+B01 -> V[5], V[6].
    # C11 -> W[3].
    # Current:
    # U[1*m+1]=4, U[0*m+1]=-1.
    # V[1*p+0]=5, V[1*p+1]=6.
    # W[1*n+1]=3.
    # Matches.
    
    # M7
    # A01-A11 -> U[2], -U[5].
    # B10+B11 -> V[10], V[11].
    # C00 -> W[0].
    # Current:
    # U[0*m+2]=2, U[1*m+2]=-5.
    # V[2*p+0]=10, V[2*p+1]=11.
    # W[0*n+0]=0.
    # Matches.
    
    # Block 2b: k=2 (Schoolbook)
    # i=0,1; j=1,2.
    # A[i,j] B[j,2] -> C[2,i].
    for i in range(n):
        for j in [1, 2]:
            U[i * m + j][col] = 1
            V[j * p + 2][col] = 1
            W[2 * n + i][col] = 1
            col += 1
            
    # Block 2c: k=3,4 (Strassen 2x2x2)
    # Local k' in {0,1} maps to global k = k' + 3.
    # M1
    U[1][col] = 1; U[5][col] = 1
    V[1 * p + 3][col] = 1; V[2 * p + 4][col] = 1
    W[3 * n + 0][col] = 1; W[4 * n + 1][col] = 1
    col += 1
    
    # M2
    U[4][col] = 1; U[5][col] = 1
    V[1 * p + 3][col] = 1
    W[3 * n + 1][col] = 1; W[4 * n + 1][col] = -1
    col += 1
    
    # M3
    U[1][col] = 1
    V[1 * p + 4][col] = 1; V[2 * p + 4][col] = -1
    W[4 * n + 0][col] = 1; W[4 * n + 1][col] = 1
    col += 1
    
    # M4
    U[5][col] = 1
    V[2 * p + 3][col] = 1; V[1 * p + 3][col] = -1
    W[3 * n + 0][col] = 1; W[3 * n + 1][col] = 1
    col += 1
    
    # M5
    U[1][col] = 1; U[2][col] = 1
    V[2 * p + 4][col] = 1
    W[3 * n + 0][col] = -1; W[4 * n + 0][col] = 1
    col += 1
    
    # M6
    U[4][col] = 1; U[1][col] = -1
    V[1 * p + 3][col] = 1; V[1 * p + 4][col] = 1
    W[4 * n + 1][col] = 1
    col += 1
    
    # M7
    U[2][col] = 1; U[5][col] = -1
    V[2 * p + 3][col] = 1; V[2 * p + 4][col] = 1
    W[3 * n + 0][col] = 1
    col += 1
    
    assert col == rank, f"Column count mismatch: {col} != {rank}"
    
    return {"U": U, "V": V, "W": W}
