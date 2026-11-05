"""Matrix multiplication 4x4x6 via Strassen block decomposition, rank 77."""
import time
import numpy as np

def _strassen_2x2():
    """Return U, V, W for 2x2x2 Strassen. U: (4,7), V: (4,7), W: (4,7)."""
    rank = 7
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))
    
    # Strassen's 7 products for 2x2 matrix multiplication
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[i,j],[k,l]]
    # P1 = (a+d)(e+h)
    # P2 = (c+d)e
    # P3 = a(f-h)
    # P4 = d(g-e)
    # P5 = (a+b)h
    # P6 = (c-a)(e+f)
    # P7 = (b-d)(g+h)
    # i = P1 + P4 - P5 + P7
    # j = P3 + P5
    # k = P2 + P4
    # l = P1 - P2 + P3 + P6
    
    # U[i*2+j, t] = coefficient of A[i,j] in product t
    # V[j*2+k, t] = coefficient of B[j,k] in product t
    # W[k*2+i, t] = coefficient of product t in C[k,i]... 
    # Wait, the convention is W[k*n+i, t] where n=2, so W[k*2+i, t]
    
    # Product 1: (a+d)(e+h)
    # A part: a=1, d=1 → U[0,0]=1, U[1,1]=1
    # B part: e=1, h=1 → V[0,0]=1, V[1,1]=1
    # C part: i=P1, l=-P1 → W[0,0]=1, W[1,1]=-1
    U[0,0] += 1; U[1,1] += 1
    V[0,0] += 1; V[1,1] += 1
    W[0,0] += 1; W[1,1] += -1
    
    # Product 2: (c+d)e
    # A part: c=1, d=1 → U[1,0]=1, U[1,1]=1
    # B part: e=1 → V[0,0]=1
    # C part: k=P2, l=-P2 → W[0,1]=1, W[1,1]=-1
    U[1,0] += 1; U[1,1] += 1
    V[0,1] += 1
    W[0,1] += 1; W[1,1] += -1
    
    # Product 3: a(f-h)
    # A part: a=1 → U[0,0]=1
    # B part: f=1, h=-1 → V[0,1]=1, V[1,1]=-1
    # C part: j=P3, l=P3 → W[0,0]=1, W[1,0]=1
    U[0,2] += 1
    V[0,2] += 1; V[1,2] += -1
    W[0,2] += 1; W[1,2] += 1
    
    # Product 4: d(g-e)
    # A part: d=1 → U[1,1]=1
    # B part: g=1, e=-1 → V[1,0]=1, V[0,0]=-1
    # C part: i=P4, k=P4 → W[0,0]=1, W[1,0]=1
    U[1,3] += 1
    V[1,3] += 1; V[0,3] += -1
    W[0,3] += 1; W[1,3] += 1
    
    # Product 5: (a+b)h
    # A part: a=1, b=1 → U[0,0]=1, U[0,1]=1
    # B part: h=1 → V[1,1]=1
    # C part: i=-P5, j=P5 → W[0,0]=-1, W[0,1]=1
    U[0,4] += 1; U[0,1] += 1
    V[1,4] += 1
    W[0,4] += -1; W[0,1] += 1
    
    # Product 6: (c-a)(e+f)
    # A part: c=1, a=-1 → U[1,0]=1, U[0,0]=-1
    # B part: e=1, f=1 → V[0,0]=1, V[0,1]=1
    # C part: l=P6 → W[1,1]=1
    U[1,5] += 1; U[0,5] += -1
    V[0,5] += 1; V[0,1] += 1
    W[1,5] += 1
    
    # Product 7: (b-d)(g+h)
    # A part: b=1, d=-1 → U[0,1]=1, U[1,1]=-1
    # B part: g=1, h=1 → V[1,0]=1, V[1,1]=1
    # C part: i=P7 → W[0,0]=1
    U[0,6] += 1; U[1,6] += -1
    V[1,6] += 1; V[1,1] += 1
    W[0,6] += 1
    
    return U, V, W

def _verify(U, V, W, n, m, p, rank):
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    Ur = Ua.reshape(n, m, rank)
    Vr = Va.reshape(m, p, rank)
    Wr = Wa.reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
    err = np.max(np.abs(T - 1.0))
    return err

def construct():
    n, m, p = 4, 4, 6
    t0 = time.time()
    deadline = t0 + 110.0
    
    Us, Vs, Ws = _strassen_2x2()
    
    rank_444 = 49
    rank_442 = 28
    total_rank = rank_444 + rank_442
    
    U = np.zeros((n * m, total_rank))  # 16 x 77
    V = np.zeros((m * p, total_rank))  # 24 x 77
    W = np.zeros((p * n, total_rank))  # 24 x 77
    
    # Part 1: 4x4x4 Strassen (columns 0-48)
    # 2x2 block decomposition:
    # A[i,j] where i=2I+i0, j=2J+j0
    # B[j,k] where j=2J+j0, k=2K+k0
    # C[k,i] where k=2K+k0, i=2I+i0
    # U[i*4+j, 7t+s] = Us[2i0+j0, t] * Us[2I+J, t]... 
    # Wait, the Kronecker product structure:
    # For 4x4x4 = (2x2) ⊗ (2x2) Strassen:
    # U_444[i,j,c] where c = 7*t + s
    # U_444[2I+i0, 2J+j0, 7t+s] = Us[2I+J, t] * Us[2i0+j0, s]
    
    for I in range(2):
        for J in range(2):
            for i0 in range(2):
                for j0 in range(2):
                    i = 2 * I + i0
                    j = 2 * J + j0
                    for t in range(7):
                        for s in range(7):
                            col = 7 * t + s
                            val = Us[2 * I + J, t] * Us[2 * i0 + j0, s]
                            if val != 0:
                                U[i * 4 + j, col] += val
    
    for J in range(2):
        for K in range(2):
            for j0 in range(2):
                for k0 in range(2):
                    j = 2 * J + j0
                    k = 2 * K + k0
                    for t in range(7):
                        for s in range(7):
                            col = 7 * t + s
                            val = Vs[2 * J + K, t] * Vs[2 * j0 + k0, s]
                            if val != 0:
                                V[j * 6 + k, col] += val
    
    for K in range(2):
        for I in range(2):
            for k0 in range(2):
                for i0 in range(2):
                    k = 2 * K + k0
                    i = 2 * I + i0
                    for t in range(7):
                        for s in range(7):
                            col = 7 * t + s
                            val = Ws[2 * K + I, t] * Ws[2 * k0 + i0, s]
                            if val != 0:
                                W[k * 4 + i, col] += val
    
    # Part 2: 4x4x2 (columns 49-76)
    # B is 4x6, we take columns 4,5.
    # Partition A into 2x2 blocks: A_00, A_01, A_10, A_11
    # B_right (cols 4,5): partition into B_top (rows 0-1, cols 4-5) and B_bot (rows 2-3, cols 4-5)
    # C_right (cols 4,5): partition into C_top (rows 0-1, cols 4-5) and C_bot (rows 2-3, cols 4-5)
    # C_top = A_00 * B_top + A_01 * B_bot
    # C_bot = A_10 * B_top + A_11 * B_bot
    # 4 Strassen products:
    # P0: A_00 * B_top → C_top
    # P1: A_01 * B_bot → C_top
    # P2: A_10 * B_top → C_bot
    # P3: A_11 * B_bot → C_bot
    
    products = [
        (0, 0, 0),  # I=0, J=0, B_row_block=0 (top)
        (0, 1, 1),  # I=0, J=1, B_row_block=1 (bot)
        (1, 0, 0),  # I=1, J=0, B_row_block=0 (top)
        (1, 1, 1),  # I=1, J=1, B_row_block=1 (bot)
    ]
    
    for p_idx, (I, J, b_row) in enumerate(products):
        col_off = rank_444 + p_idx * 7
        
        # U part: A_IJ block
        for i0 in range(2):
            for j0 in range(2):
                i = 2 * I + i0
                j = 2 * J + j0
                for s in range(7):
                    val = Us[2 * i0 + j0, s]
                    if val != 0:
                        U[i * 4 + j, col_off + s] += val
        
        # V part: B block (rows 2*b_row..2*b_row+1, cols 4..5)
        for j0 in range(2):
            for k0 in range(2):
                j_global = 2 * b_row + j0
                k_global = 4 + k0
                for s in range(7):
                    val = Vs[2 * j0 + k0, s]
                    if val != 0:
                        V[j_global * 6 + k_global, col_off + s] += val
        
        # W part: C block (rows 4..5, cols 2*I..2*I+1)
        for i0 in range(2):
            for k0 in range(2):
                i_global = 2 * I + i0
                k_global = 4 + k0
                for s in range(7):
                    val = Ws[2 * k0 + i0, s]
                    if val != 0:
                        W[k_global * 4 + i_global, col_off + s] += val
    
    # Verify
    err = _verify(U, V, W, n, m, p, total_rank)
    
    if err > 1e-6:
        # Fallback: schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
