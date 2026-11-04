import numpy as np

def construct():
    n, m, p = 4, 4, 6
    rank = 77
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Define Strassen 2x2x2 patterns
    # U, V, W are 2x2 matrices of coefficients
    # Local indices i,j,k in 0..1
    patterns = []
    
    # M1 = (A11 + A22)(B11 + B22)
    # U: A11=1, A22=1
    # V: B11=1, B22=1
    # W: C11=1, C22=1
    patterns.append((
        np.array([[1, 0], [0, 1]]), # U: A11, A22
        np.array([[1, 0], [0, 1]]), # V: B11, B22
        np.array([[1, 0], [0, 1]])  # W: C11, C22 (W[k,i], so W[0,0]=1, W[1,1]=1)
    ))
    
    # M2 = (A21 + A22) B11
    # U: A21=1, A22=1
    # V: B11=1
    # W: C21=1, C22=-1 (From C22 = M1 - M2 + M3 - M6)
    patterns.append((
        np.array([[0, 0], [1, 1]]), # U: A21, A22
        np.array([[1, 0], [0, 0]]), # V: B11
        np.array([[0, 0], [1, -1]]) # W: C21 (k=1,i=0), C22 (k=1,i=1) -> W[1,0]=1, W[1,1]=-1
    ))
    
    # M3 = A11 (B12 - B22)
    # U: A11=1
    # V: B12=1, B22=-1
    # W: C12=1, C22=1 (From C12=M3+M5, C22=...+M3...)
    patterns.append((
        np.array([[1, 0], [0, 0]]), # U: A11
        np.array([[0, 1], [0, -1]]), # V: B12, B22
        np.array([[0, 1], [0, 1]])  # W: C12 (k=0,i=1), C22 (k=1,i=1) -> W[0,1]=1, W[1,1]=1
    ))
    
    # M4 = A22 (B21 - B11)
    # U: A22=1
    # V: B21=1, B11=-1
    # W: C11=1, C21=1, C22=1 (From C11=...+M4..., C21=M2+M4, C22=...+M4...)
    patterns.append((
        np.array([[0, 0], [0, 1]]), # U: A22
        np.array([[-1, 0], [1, 0]]), # V: B11, B21
        np.array([[1, 0], [1, 1]])  # W: C11 (k=0,i=0), C21 (k=1,i=0), C22 (k=1,i=1) -> W[0,0]=1, W[1,0]=1, W[1,1]=1
    ))
    
    # M5 = (A11 + A12) B22
    # U: A11=1, A12=1
    # V: B22=1
    # W: C11=-1, C12=1, C22=1 (From C11=...-M5..., C12=...+M5..., C22=...+M5...)
    patterns.append((
        np.array([[1, 1], [0, 0]]), # U: A11, A12
        np.array([[0, 0], [0, 1]]), # V: B22
        np.array([[-1, 1], [0, 1]]) # W: C11, C12, C22 -> W[0,0]=-1, W[0,1]=1, W[1,1]=1
    ))
    
    # M6 = (A21 - A11)(B11 + B12)
    # U: A21=1, A11=-1
    # V: B11=1, B12=1
    # W: C22=-1 (From C22=...-M6...)
    patterns.append((
        np.array([[-1, 0], [1, 0]]), # U: A11, A21
        np.array([[1, 1], [0, 0]]), # V: B11, B12
        np.array([[0, 0], [0, -1]]) # W: C22 -> W[1,1]=-1
    ))
    
    # M7 = (A12 - A22)(B21 + B22)
    # U: A12=1, A22=-1
    # V: B21=1, B22=1
    # W: C11=1 (From C11=...+M7...)
    patterns.append((
        np.array([[0, 1], [0, -1]]), # U: A12, A22
        np.array([[0, 0], [1, 1]]), # V: B21, B22
        np.array([[1, 0], [0, 0]])  # W: C11 -> W[0,0]=1
    ))
    
    col = 0
    
    # Part 1: 4x4x4 (Rank 49)
    # Macro Strassen terms
    macro_U = [
        np.array([[1, 0], [0, 1]]), # M1: A11, A22
        np.array([[0, 0], [1, 1]]), # M2: A21, A22
        np.array([[1, 0], [0, 0]]), # M3: A11
        np.array([[0, 0], [0, 1]]), # M4: A22
        np.array([[1, 1], [0, 0]]), # M5: A11, A12
        np.array([[-1, 0], [1, 0]]), # M6: A11, A21
        np.array([[0, 1], [0, -1]])  # M7: A12, A22
    ]
    macro_V = [
        np.array([[1, 0], [0, 1]]), # M1: B11, B22
        np.array([[1, 0], [0, 0]]), # M2: B11
        np.array([[0, 1], [0, -1]]), # M3: B12, B22
        np.array([[-1, 0], [1, 0]]), # M4: B11, B21
        np.array([[0, 0], [0, 1]]), # M5: B22
        np.array([[1, 1], [0, 0]]), # M6: B11, B12
        np.array([[0, 0], [1, 1]])  # M7: B21, B22
    ]
    macro_W = [
        np.array([[1, 0], [0, 1]]), # M1: C11, C22
        np.array([[0, 0], [1, -1]]), # M2: C21, C22
        np.array([[0, 1], [0, 1]]), # M3: C12, C22
        np.array([[1, 0], [1, 1]]), # M4: C11, C21, C22
        np.array([[-1, 1], [0, 1]]), # M5: C11, C12, C22
        np.array([[0, 0], [0, -1]]), # M6: C22
        np.array([[1, 0], [0, 0]])  # M7: C11
    ]
    
    for t in range(7):
        for s in range(7):
            u_micro, v_micro, w_micro = patterns[s]
            u_macro = macro_U[t]
            v_macro = macro_V[t]
            w_macro = macro_W[t]
            
            # Construct full 4x4 U, 4x4 V (for 4x4x4 part), 4x4 W
            # U_full[i, j] = u_macro[I, J] * u_micro[i, j] where i=I*2+... no
            # i is global row, j is global col.
            # I = i//2, J = j//2. i%2 is local row.
            U_full = np.zeros((4, 4))
            V_full = np.zeros((4, 4))
            W_full = np.zeros((4, 4))
            
            for i in range(4):
                for j in range(4):
                    I, J = i // 2, j // 2
                    li, lj = i % 2, j % 2
                    U_full[i, j] = u_macro[I, J] * u_micro[li, lj]
                    
            for j in range(4):
                for k in range(4):
                    J, K = j // 2, k // 2
                    lj, lk = j % 2, k % 2
                    V_full[j, k] = v_macro[J, K] * v_micro[lj, lk]
                    
            for k in range(4):
                for i in range(4):
                    K, I = k // 2, i // 2
                    lk, li = k % 2, i % 2
                    W_full[k, i] = w_macro[K, I] * w_micro[lk, li]
            
            # Add to global matrices
            # U index: i*m + j
            for i in range(4):
                for j in range(4):
                    U[i * 4 + j, col] = U_full[i, j]
                    
            # V index: j*p + k (p=6, but here we only use k<4)
            for j in range(4):
                for k in range(4):
                    V[j * 6 + k, col] = V_full[j, k]
                    
            # W index: k*n + i (n=4)
            for k in range(4):
                for i in range(4):
                    W[k * 4 + i, col] = W_full[k, i]
                    
            col += 1
            
    # Part 2: 4x4x2 (Rank 28)
    # 4 independent 2x2x2 multiplications
    # A blocks: 00, 01, 10, 11
    # B blocks: Top (rows 0,1), Bot (rows 2,3)
    # C blocks: Top (rows 0,1), Bot (rows 2,3)
    
    # Multiplications:
    # 1. A00 * B_top -> C_top
    # 2. A01 * B_bot -> C_top
    # 3. A10 * B_top -> C_bot
    # 4. A11 * B_bot -> C_bot
    
    mults = [
        (0, 0, 0), # A_block_idx (0=00, 1=01, 2=10, 3=11), B_block_idx (0=top, 1=bot), C_block_idx (0=top, 1=bot)
        (1, 1, 0),
        (2, 0, 1),
        (3, 1, 1)
    ]
    
    for a_idx, b_idx, c_idx in mults:
        for s in range(7):
            u_micro, v_micro, w_micro = patterns[s]
            
            U_full = np.zeros((4, 4))
            V_full = np.zeros((4, 2)) # B is 4x2
            W_full = np.zeros((2, 4)) # C is 2x4 (k=0,1; i=0..3)
            
            # A block
            I, J = a_idx // 2, a_idx % 2
            for i in range(2):
                for j in range(2):
                    U_full[I * 2 + i, J * 2 + j] = u_micro[i, j]
                    
            # B block
            # b_idx 0 -> rows 0,1. b_idx 1 -> rows 2,3.
            J_start = b_idx * 2
            for j in range(2):
                for k in range(2):
                    V_full[J_start + j, k] = v_micro[j, k]
                    
            # C block
            # c_idx 0 -> rows 0,1. c_idx 1 -> rows 2,3.
            I_start = c_idx * 2
            for k in range(2):
                for i in range(2):
                    W_full[k, I_start + i] = w_micro[k, i]
                    
            # Add to global
            for i in range(4):
                for j in range(4):
                    U[i * 4 + j, col] = U_full[i, j]
                    
            for j in range(4):
                for k in range(2):
                    V[j * 6 + k, col] = V_full[j, k]
                    
            for k in range(2):
                for i in range(4):
                    W[k * 4 + i, col] = W_full[k, i]
                    
            col += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
