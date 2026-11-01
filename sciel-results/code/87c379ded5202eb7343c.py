import numpy as np
from scipy.optimize import minimize

def construct():
    n, m, p = 2, 2, 12
    rank = 24
    
    # Initialize with schoolbook (rank 48) and then try to reduce
    # Actually, let me try a direct construction.
    
    # For each column k, use 2 terms (one for C0k, one for C1k)
    # C0k = A00*B0k + A01*B1k
    # C1k = A10*B0k + A11*B1k
    
    # Term 2k: u=(1,0,0,0), v=(e_{0,k}), w=(e_{0,k}) → A00*B0k to C0k
    # Term 2k+1: u=(0,0,1,0), v=(e_{0,k}), w=(e_{1,k}) → A10*B0k to C1k
    # But this misses A01*B1k and A11*B1k.
    
    # Let me try 2 terms per column:
    # Term 2k: u=(1,0,0,0), v=(e_{0,k}), w=(e_{0,k}) + u=(0,1,0,0), v=(e_{1,k}), w=(e_{0,k})
    # Can't combine into one rank-1 term.
    
    # OK, let me just try to solve the system numerically.
    
    # The tensor has 4*24*24 = 2304 entries.
    # Non-zero entries: T[i*m+j, j*p+k, k*n+i] = 1 for i=0,1; j=0,1; k=0..11
    # That's 2*2*12 = 48 non-zero entries.
    
    # For rank 24, I have 24 terms, each with 4+24+24 = 52 variables.
    # Total: 1248 unknowns.
    
    # Let me try a simple initialization and iterate.
    
    # Actually, let me try a different approach: use the "block" structure.
    # Split into 4 blocks of 3 columns each.
    # For each block, try to find a rank-6 decomposition.
    
    # For a block of 3 columns, the sub-tensor has 4*6*6 = 144 entries,
    # with 12 non-zero entries.
    
    # Let me try to find 6 terms for one block.
    
    # Block 0: columns 0, 1, 2
    # A indices: 0=A00, 1=A01, 2=A10, 3=A11
    # B indices: 0=B0,0, 1=B1,0, 2=B0,1, 3=B1,1, 4=B0,2, 5=B1,2
    # C indices: 0=C0,0, 1=C1,0, 2=C0,1, 3=C1,1, 4=C0,2, 5=C1,2
    
    # Non-zero entries:
    # (0,0,0), (1,1,0), (2,0,1), (3,1,1),
    # (0,2,2), (1,3,2), (2,2,3), (3,3,3),
    # (0,4,4), (1,5,4), (2,4,5), (3,5,5)
    
    # I'll try to solve this as a linear system by fixing some variables.
    
    # Let me try the following ansatz for 6 terms:
    # Term 0: u=(a0,0,0,0), v=(b0,0,0,0,0,0), w=(c0,0,0,0,0,0)
    # Term 1: u=(0,a1,0,0), v=(0,0,b1,0,0,0), w=(c1,0,0,0,0,0)
    # ...
    
    # This is getting too complex. Let me just try to use a numerical solver.
    
    # For now, let me keep the rank-42 Strassen solution and verify it works.
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    num_pairs = p // 2
    
    for t in range(num_pairs):
        base = t * 7
        for s in range(7):
            c = base + s
            if s == 0:
                U[0][c] = 1
                U[3][c] = 1
                V[0*p + 2*t][c] = 1
                V[1*p + 2*t+1][c] = 1
                W[2*t*n + 0][c] = 1
                W[(2*t+1)*n + 1][c] = 1
            elif s == 1:
                U[2][c] = 1
                U[3][c] = 1
                V[0*p + 2*t][c] = 1
                W[2*t*n + 1][c] = 1
                W[(2*t+1)*n + 1][c] = -1
            elif s == 2:
                U[0][c] = 1
                V[0*p + 2*t+1][c] = 1
                V[1*p + 2*t+1][c] = -1
                W[(2*t+1)*n + 0][c] = 1
                W[(2*t+1)*n + 1][c] = 1
            elif s == 3:
                U[3][c] = 1
                V[1*p + 2*t][c] = 1
                V[0*p + 2*t][c] = -1
                W[2*t*n + 0][c] = 1
                W[2*t*n + 1][c] = 1
            elif s == 4:
                U[0][c] = 1
                U[1][c] = 1
                V[1*p + 2*t+1][c] = 1
                W[2*t*n + 0][c] = -1
                W[(2*t+1)*n + 0][c] = 1
            elif s == 5:
                U[2][c] = 1
                U[0][c] = -1
                V[0*p + 2*t][c] = 1
                V[0*p + 2*t+1][c] = 1
                W[(2*t+1)*n + 1][c] = 1
            elif s == 6:
                U[1][c] = 1
                U[3][c] = -1
                V[1*p + 2*t][c] = 1
                V[1*p + 2*t+1][c] = 1
                W[2*t*n + 0][c] = 1
    
    # Verify
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                if abs(val - 1.0) > max_err:
                    max_err = abs(val - 1.0)
    
    if max_err > 1e-9:
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][idx] = 1
                    V[j*p+k][idx] = 1
                    W[k*n+i][idx] = 1
                    idx += 1
    
    return {"U": U, "V": V, "W": W}
