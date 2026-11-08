import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Total rank
    rank = 54
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Part 1: A1 (2x2) * B1 (2x6)
    # A1 is A[0:2, 0:2]
    # B1 is B[0:2, 0:6]
    # Split B1 into 3 blocks of 2x2
    for b in range(3):
        # Strassen for 2x2x2
        # A indices: i in 0..1, j in 0..1 (global A cols 0..1)
        # B indices: j in 0..1 (global B rows 0..1), k in 0..1 (global B cols 2*b..2*b+1)
        # C indices: i in 0..1, k in 0..1 (global C cols 2*b..2*b+1)
        
        # M1 = (A00+A11)(B00+B11)
        U[0*m+0, col] = 1; U[1*m+1, col] = 1
        V[0*p+2*b, col] = 1; V[1*p+2*b+1, col] = 1
        W[2*b*n+0, col] = 1; W[(2*b+1)*n+0, col] = 1
        W[2*b*n+1, col] = 1; W[(2*b+1)*n+1, col] = 1
        # Correction: W maps C[i,k]. C00 gets M1, C11 gets M1.
        # My W setup above sets W[0,0]=1, W[1,1]=1, W[0,1]=1, W[1,0]=1? No.
        # Let's be precise.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # We need to add coefficients to W for each M.
        # M1 contributes to C00 and C11.
        W[2*b*n+0, col] += 1
        W[(2*b+1)*n+1, col] += 1
        col += 1
        
        # M2 = (A10+A11)B00
        U[1*m+0, col] = 1; U[1*m+1, col] = 1
        V[0*p+2*b, col] = 1
        # M2 contributes to C10
        W[2*b*n+1, col] += 1
        col += 1
        
        # M3 = A00(B01-B11)
        U[0*m+0, col] = 1
        V[0*p+2*b+1, col] = 1; V[1*p+2*b+1, col] = -1
        # M3 contributes to C01
        W[2*b*n+0, col] += 1
        col += 1
        
        # M4 = A11(B10-B00)
        U[1*m+1, col] = 1
        V[1*p+2*b, col] = 1; V[0*p+2*b, col] = -1
        # M4 contributes to C00 and C10
        W[2*b*n+0, col] += 1
        W[2*b*n+1, col] += 1
        col += 1
        
        # M5 = (A00+A01)B11
        U[0*m+0, col] = 1; U[0*m+1, col] = 1
        V[1*p+2*b+1, col] = 1
        # M5 contributes to C00 (-1) and C01 (+1)
        W[2*b*n+0, col] += -1
        W[2*b*n+0, col] += 1 # Wait, C00 gets -M5, C01 gets +M5
        # C00 is W[0,0], C01 is W[0,1]
        W[2*b*n+0, col] = -1 # Overwrite? No, accumulate.
        # Let's restart the W accumulation logic clearly.
        
    # The above manual accumulation is error-prone. I will write a helper to add a rank-1 term.
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    
    def add_term(u_vec, v_vec, w_vec):
        nonlocal col
        U[:, col] = u_vec
        V[:, col] = v_vec
        W[:, col] = w_vec
        col += 1

    # Part 1: A1 (2x2) * B1 (2x6)
    for b in range(3):
        # Strassen
        # M1
        u = np.zeros(n*m); u[0]=1; u[1*m+1]=1
        v = np.zeros(m*p); v[0*p+2*b]=1; v[1*p+2*b+1]=1
        w = np.zeros(p*n); w[2*b*n+0]=1; w[(2*b+1)*n+1]=1
        add_term(u, v, w)
        
        # M2
        u = np.zeros(n*m); u[1*m+0]=1; u[1*m+1]=1
        v = np.zeros(m*p); v[0*p+2*b]=1
        w = np.zeros(p*n); w[2*b*n+1]=1
        add_term(u, v, w)
        
        # M3
        u = np.zeros(n*m); u[0*m+0]=1
        v = np.zeros(m*p); v[0*p+2*b+1]=1; v[1*p+2*b+1]=-1
        w = np.zeros(p*n); w[2*b*n+0]=1
        add_term(u, v, w)
        
        # M4
        u = np.zeros(n*m); u[1*m+1]=1
        v = np.zeros(m*p); v[1*p+2*b]=1; v[0*p+2*b]=-1
        w = np.zeros(p*n); w[2*b*n+0]=1; w[2*b*n+1]=1
        add_term(u, v, w)
        
        # M5
        u = np.zeros(n*m); u[0*m+0]=1; u[0*m+1]=1
        v = np.zeros(m*p); v[1*p+2*b+1]=1
        w = np.zeros(p*n); w[2*b*n+0]=-1; w[2*b*n+0]=1 # C00 gets -M5, C01 gets +M5
        # C00 is index 0, C01 is index 1 in the 2x2 block
        w[2*b*n+0] = -1
        w[2*b*n+0] = 1 # Wait. C00 = ... - M5 ... C01 = ... + M5 ...
        # C00 is W[2*b*n+0], C01 is W[2*b*n+0+1]? No.
        # C is 2x2. Indices (0,0), (0,1), (1,0), (1,1).
        # W index = k*n + i.
        # For block b, k=0,1. i=0,1.
        # C00 -> k=0, i=0 -> W[0*n+0] = W[0] (relative to block)
        # C01 -> k=1, i=0 -> W[1*n+0] = W[2] (relative to block, n=2)
        # C10 -> k=0, i=1 -> W[0*n+1] = W[1]
        # C11 -> k=1, i=1 -> W[1*n+1] = W[3]
        
        # Let's re-verify W indices.
        # W[k*n+i].
        # M5 contributes -1 to C00 and +1 to C01.
        # C00 is k=0, i=0. Index 0.
        # C01 is k=1, i=0. Index 1*n + 0 = 2.
        w = np.zeros(p*n)
        w[2*b*n + 0] = -1
        w[2*b*n + 0 + n] = 1
        add_term(u, v, w)
        
        # M6
        u = np.zeros(n*m); u[1*m+0]=1; u[0*m+0]=-1
        v = np.zeros(m*p); v[0*p+2*b]=1; v[0*p+2*b+1]=1
        w = np.zeros(p*n); w[2*b*n + 0 + n] = 1 # C11 is k=1, i=1. Index 1*n+1 = 3.
        w[2*b*n + 0 + n + 1] = 1
        add_term(u, v, w)
        
        # M7
        u = np.zeros(n*m); u[0*m+1]=1; u[1*m+1]=-1
        v = np.zeros(m*p); v[1*p+2*b]=1; v[1*p+2*b+1]=1
        w = np.zeros(p*n); w[2*b*n + 0] = 1 # C00
        add_term(u, v, w)

    # Part 2: A2 (2x3) * B2 (3x6)
    for b in range(3):
        # A2a (2x2, cols 2,3) * B2ba (2x2, rows 2,3)
        # Strassen
        # A indices: i=0,1. j=2,3 (global A)
        # B indices: j=2,3 (global B). k=2*b, 2*b+1
        # C indices: i=0,1. k=2*b, 2*b+1
        
        # M1
        u = np.zeros(n*m); u[0*m+2]=1; u[1*m+3]=1
        v = np.zeros(m*p); v[2*p+2*b]=1; v[3*p+2*b+1]=1
        w = np.zeros(p*n); w[2*b*n+0]=1; w[(2*b+1)*n+1]=1
        add_term(u, v, w)
        
        # M2
        u = np.zeros(n*m); u[1*m+2]=1; u[1*m+3]=1
        v = np.zeros(m*p); v[2*p+2*b]=1
        w = np.zeros(p*n); w[2*b*n+1]=1
        add_term(u, v, w)
        
        # M3
        u = np.zeros(n*m); u[0*m+2]=1
        v = np.zeros(m*p); v[2*p+2*b+1]=1; v[3*p+2*b+1]=-1
        w = np.zeros(p*n); w[2*b*n+0]=1
        add_term(u, v, w)
        
        # M4
        u = np.zeros(n*m); u[1*m+3]=1
        v = np.zeros(m*p); v[3*p+2*b]=1; v[2*p+2*b]=-1
        w = np.zeros(p*n); w[2*b*n+0]=1; w[2*b*n+1]=1
        add_term(u, v, w)
        
        # M5
        u = np.zeros(n*m); u[0*m+2]=1; u[0*m+3]=1
        v = np.zeros(m*p); v[3*p+2*b+1]=1
        w = np.zeros(p*n); w[2*b*n+0]=-1; w[2*b*n+0+n]=1
        add_term(u, v, w)
        
        # M6
        u = np.zeros(n*m); u[1*m+2]=1; u[0*m+2]=-1
        v = np.zeros(m*p); v[2*p+2*b]=1; v[2*p+2*b+1]=1
        w = np.zeros(p*n); w[2*b*n+0+n+1]=1
        add_term(u, v, w)
        
        # M7
        u = np.zeros(n*m); u[0*m+3]=1; u[1*m+3]=-1
        v = np.zeros(m*p); v[3*p+2*b]=1; v[3*p+2*b+1]=1
        w = np.zeros(p*n); w[2*b*n+0]=1
        add_term(u, v, w)
        
        # Outer Product: A2b (2x1, col 4) * B2bb (1x2, row 4)
        # C00 = A04 B4k0
        u = np.zeros(n*m); u[0*m+4]=1
        v = np.zeros(m*p); v[4*p+2*b]=1
        w = np.zeros(p*n); w[2*b*n+0]=1
        add_term(u, v, w)
        
        # C01 = A04 B4k1
        u = np.zeros(n*m); u[0*m+4]=1
        v = np.zeros(m*p); v[4*p+2*b+1]=1
        w = np.zeros(p*n); w[2*b*n+0+n]=1
        add_term(u, v, w)
        
        # C10 = A14 B4k0
        u = np.zeros(n*m); u[1*m+4]=1
        v = np.zeros(m*p); v[4*p+2*b]=1
        w = np.zeros(p*n); w[2*b*n+1]=1
        add_term(u, v, w)
        
        # C11 = A14 B4k1
        u = np.zeros(n*m); u[1*m+4]=1
        v = np.zeros(m*p); v[4*p+2*b+1]=1
        w = np.zeros(p*n); w[2*b*n+1+n]=1
        add_term(u, v, w)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
